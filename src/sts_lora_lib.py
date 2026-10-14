"""sts_lora_lib.py - pustaka eksperimen: LoRA vs full fine-tuning pada Sentence-BERT (STS).

Prinsip desain
--------------
* Satu "run" = (model, metode, rank, strategi, LR, seed). ID run deterministik (hash konfigurasi).
* Checkpoint per epoch (bobot terlatih + optimizer + scheduler + scaler + RNG) ditulis atomik ke
  folder run. Jika sesi Colab putus, memanggil run_train() lagi MELANJUTKAN dari epoch terakhir;
  jika sudah selesai, dilewati (tidak diulang).
* Satu file result.json per run (metrik validasi, parameter terlatih, memori puncak, waktu, dst.).
* Evaluasi test / transfer dilakukan terpisah (eval_run) dan prediksinya disimpan (.npy) untuk
  bootstrap berpasangan.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import pickle
import os
import platform
import random
import re
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, rankdata, spearmanr
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

# ----------------------------------------------------------------------------- konfigurasi
DIVERGE_SPEARMAN = 0.50   # run dianggap collapse bila Spearman validasi TERBAIK < 0,50 (di bawah baseline tanpa fine-tuning)
STRATEGIES = {
    "S0": "cosine-MSE loss, LR seragam, scheduler linear+warmup (baseline)",
    "S1": "cosine-MSE loss + Layer-wise LR Decay (LLRD)",
    "S2": "CoSENT loss, LR seragam, scheduler linear+warmup",
}


@dataclass
class Config:
    model_name: str = "xlm-roberta-base"
    method: str = "full"          # "full" | "lora"
    rank: int = 0                 # hanya untuk lora
    strategy: str = "S0"          # S0 | S1 | S2
    lr: float = 2e-5
    seed: int = 42
    epochs: int = 4
    batch_size: int = 32
    max_len: int = 128
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    llrd_decay: float = 0.9
    lora_dropout: float = 0.1
    cosent_scale: float = 20.0
    train_data: str = "stsb_en"
    rslora: bool = False                  # OPSIONAL (ablasi): rank-stabilized LoRA
    lora_targets: str = "all-linear"      # OPSIONAL (ablasi): "all-linear" | "qv" (hanya query & value)

    def run_id(self) -> str:
        d = asdict(self)
        # field opsional hanya masuk hash bila bukan default -> ID run lama tetap valid
        if not d["rslora"]:
            d.pop("rslora")
        if d["lora_targets"] == "all-linear":
            d.pop("lora_targets")
        name = d["model_name"].replace("/", "-")
        tag = "full" if self.method == "full" else f"lora-r{self.rank}"
        if self.method == "lora" and self.rslora:
            tag += "-rs"
        if self.method == "lora" and self.lora_targets != "all-linear":
            tag += "-" + self.lora_targets
        h = hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]
        return f"{name}_{tag}_{self.strategy}_lr{self.lr:g}_s{self.seed}_{h}"


# ----------------------------------------------------------------------------- data
def load_data(name: str, split: str) -> pd.DataFrame:
    """Kembalikan DataFrame [sentence1, sentence2, label] dengan label dinormalisasi ke 0-1.

    ID dataset (diverifikasi pada Hugging Face):
      stsb_en     : sentence-transformers/stsb      (train/validation/test; kolom score 0-1)
      semrel_en   : SemRel/SemRel2024, config eng   (train/dev/test; kolom label)
      semrel_id   : SemRel/SemRel2024, config ind   (dev/test saja)
      stsb_id     : LazarusNLP/stsb_mt_id           (validation/test; skor asli di kolom 'correlation', 0-5)
    """
    from datasets import load_dataset

    if name == "stsb_en":
        ds = load_dataset("sentence-transformers/stsb", split=split)
        df = pd.DataFrame({"sentence1": ds["sentence1"], "sentence2": ds["sentence2"],
                           "label": np.asarray(ds["score"], dtype=float)})
    elif name in ("semrel_en", "semrel_id"):
        cfg = "eng" if name == "semrel_en" else "ind"
        ds = load_dataset("SemRel/SemRel2024", cfg, split=split)
        df = pd.DataFrame({"sentence1": ds["sentence1"], "sentence2": ds["sentence2"],
                           "label": np.asarray(ds["label"], dtype=float)})
    elif name == "stsb_id":
        ds = load_dataset("LazarusNLP/stsb_mt_id", split=split)
        # PERINGATAN: kolom 'score' hanyalah nomor urut; skor kemiripan sebenarnya = 'correlation'.
        df = pd.DataFrame({"sentence1": ds["text_1"], "sentence2": ds["text_2"],
                           "label": np.asarray([float(x) for x in ds["correlation"]]) / 5.0})
    else:
        raise ValueError(name)
    df["label"] = df["label"].clip(0, 1)
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------- model
class SBERT(torch.nn.Module):
    """Siamese encoder + mean pooling (arsitektur Sentence-BERT, Reimers & Gurevych, 2019)."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def embed(self, enc):
        out = self.backbone(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        h = out.last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)


def _load_retry(path, map_location, tries=5, **kw):
    """torch.load dengan percobaan ulang: Google Drive (FUSE) kadang gagal sesaat (FileNotFoundError/OSError saat baca)."""
    for k in range(tries):
        try:
            return torch.load(path, map_location=map_location, **kw)
        except OSError as e:
            if k == tries - 1:
                raise
            print(f"  [Drive] gagal membaca {Path(path).name} ({e!r}); coba lagi {k + 1}/{tries - 1}")
            time.sleep(5 * (k + 1))


def build_model(cfg: Config):
    try:
        backbone = AutoModel.from_pretrained(cfg.model_name, add_pooling_layer=False)
    except TypeError:
        backbone = AutoModel.from_pretrained(cfg.model_name)
    if cfg.method == "lora":
        from peft import LoraConfig, get_peft_model

        targets = "all-linear" if cfg.lora_targets == "all-linear" else ["query", "value"]
        # standar: alpha = 2r (skala alpha/r = 2). rsLoRA: skala alpha/sqrt(r); alpha dikalibrasi 2*sqrt(8) agar skala sama dengan
        # LoRA standar pada r=8 (r>8 -> skala lebih kecil, r<8 -> lebih besar), sehingga ablasi menguji peran skala vs rank.
        alpha = 2 * (8 ** 0.5) if cfg.rslora else 2 * cfg.rank
        lc = LoraConfig(r=cfg.rank, lora_alpha=alpha, lora_dropout=cfg.lora_dropout,
                        target_modules=targets, bias="none", use_rslora=cfg.rslora)
        backbone = get_peft_model(backbone, lc)
    elif cfg.method != "full":
        raise ValueError(cfg.method)
    return SBERT(backbone)


def trainable_state(model: SBERT) -> dict:
    return {k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad}


def count_params(model: SBERT):
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    return tr, tot


# ----------------------------------------------------------------------------- loss & optimizer
def cosine_mse_loss(u, v, y):
    return F.mse_loss(F.cosine_similarity(u, v), y)


def cosent_loss(u, v, y, scale=20.0):
    """CoSENT (Su, 2022): log(1 + sum_{y_i>y_j} exp(scale*(cos_j - cos_i)))."""
    s = F.cosine_similarity(u, v) * scale
    diff = s[None, :] - s[:, None]                 # diff[i,j] = s_j - s_i
    mask = (y[:, None] > y[None, :]).float()       # pasangan i lebih mirip dari j
    diff = diff - (1 - mask) * 1e12
    diff = torch.cat([torch.zeros(1, device=diff.device), diff.flatten()])
    return torch.logsumexp(diff, 0)


def _layer_index(name: str, n_layers: int):
    m = re.search(r"encoder\.layer\.(\d+)\.", name)
    if m:
        return int(m.group(1))
    if "embeddings" in name:
        return -1
    return n_layers  # parameter lain (mis. pooler) diperlakukan seperti layer teratas


def build_optimizer(model: SBERT, cfg: Config):
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    nd = ("bias", "LayerNorm.weight", "layer_norm")
    groups = []
    if cfg.strategy == "S1":
        n_layers = model.backbone.config.num_hidden_layers if hasattr(model.backbone, "config") \
            else model.backbone.base_model.model.config.num_hidden_layers
        buckets = {}
        for n, p in params:
            li = _layer_index(n, n_layers)
            wd = 0.0 if any(x in n for x in nd) else cfg.weight_decay
            # LR layer teratas = lr; tiap layer ke bawah dikalikan llrd_decay; embedding terendah
            lr = cfg.lr * cfg.llrd_decay ** (n_layers - 1 - li) if li < n_layers else cfg.lr
            buckets.setdefault((lr, wd), []).append(p)
        groups = [{"params": ps, "lr": lr, "weight_decay": wd} for (lr, wd), ps in buckets.items()]
    else:
        dec = [p for n, p in params if not any(x in n for x in nd)]
        nod = [p for n, p in params if any(x in n for x in nd)]
        groups = [{"params": dec, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
                  {"params": nod, "lr": cfg.lr, "weight_decay": 0.0}]
        groups = [g for g in groups if g["params"]]
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)


# ----------------------------------------------------------------------------- util
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _io_retry(fn, what="", tries=6, wait=30):
    """Ulangi operasi berkas bila Google Drive menolak sesaat (kuota operasi/bandwidth -> OSError 'Input/output error')."""
    for k in range(tries):
        try:
            return fn()
        except OSError as e:
            if k == tries - 1:
                raise
            print(f"  [Drive] {what} gagal ({e!r}); tunggu {wait * (k + 1)} dtk lalu coba lagi ({k + 1}/{tries - 1})")
            time.sleep(wait * (k + 1))


def atomic_save(obj, path: Path):
    tmp = Path(str(path) + ".tmp")

    def go():
        torch.save(obj, tmp)
        os.replace(tmp, path)

    _io_retry(go, f"menulis {Path(path).name}")


def atomic_json(obj, path: Path):
    tmp = Path(str(path) + ".tmp")

    def go():
        tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
        os.replace(tmp, path)

    _io_retry(go, f"menulis {Path(path).name}")


def copy_to_drive(src: Path, dst: Path):
    """Salin SATU kali berkas dari disk lokal VM ke Drive (atomik, diverifikasi ukuran, dengan retry)."""
    tmp = Path(str(dst) + ".tmp")

    def go():
        shutil.copyfile(src, tmp)
        if tmp.stat().st_size != Path(src).stat().st_size:
            raise OSError("ukuran berkas hasil salin tidak sama")
        os.replace(tmp, dst)

    _io_retry(go, f"menyalin {Path(dst).name} ke Drive")


def env_info(device):
    import peft
    import transformers

    return {"torch": torch.__version__, "transformers": transformers.__version__,
            "peft": peft.__version__,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}


def pick_precision(device):
    if device.type != "cuda":
        return "fp32"
    # bf16 hanya native pada GPU Ampere ke atas (compute capability >= 8, mis. A100).
    # T4 (7.5) hanya mengemulasikan bf16 (lambat) -> gunakan fp16 + GradScaler.
    return "bf16" if torch.cuda.get_device_capability(0)[0] >= 8 else "fp16"


def _autocast(precision, device):
    if precision == "fp32":
        return torch.autocast(device_type="cpu", enabled=False)
    dt = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dt)


def _batches(df, bs, order=None):
    idx = order if order is not None else np.arange(len(df))
    for i in range(0, len(idx), bs):
        sub = df.iloc[idx[i:i + bs]]
        yield sub["sentence1"].tolist(), sub["sentence2"].tolist(), \
            torch.tensor(sub["label"].values, dtype=torch.float32)


@torch.no_grad()
def predict(model, tok, df, cfg, device, precision, bs=128):
    model.eval()
    out = []
    for s1, s2, _ in _batches(df, bs):
        e1 = tok(s1, padding=True, truncation=True, max_length=cfg.max_len, return_tensors="pt").to(device)
        e2 = tok(s2, padding=True, truncation=True, max_length=cfg.max_len, return_tensors="pt").to(device)
        with _autocast(precision, device):
            u, v = model.embed(e1), model.embed(e2)
        out.append(F.cosine_similarity(u.float(), v.float()).cpu().numpy())
    return np.concatenate(out)


def metrics(pred, y):
    return {"spearman": float(spearmanr(pred, y)[0]), "pearson": float(pearsonr(pred, y)[0])}


# ----------------------------------------------------------------------------- log komputasi & sumber daya
_LOG = {"dir": None, "stage": "unset", "session": "nosession"}


def set_log_dir(root):
    """Folder log (JSONL peristiwa komputasi) di Drive."""
    _LOG["dir"] = Path(root) / "logs"
    _LOG["dir"].mkdir(parents=True, exist_ok=True)


def set_stage(name: str):
    _LOG["stage"] = name


def log_event(**kw):
    """Tambahkan satu baris JSON ke logs/compute_log.jsonl. Tidak pernah menggagalkan pelatihan."""
    if _LOG["dir"] is None:
        return
    try:
        rec = {"ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               "session": _LOG["session"], "stage": _LOG["stage"], **kw}
        with open(_LOG["dir"] / "compute_log.jsonl", "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def log_units(remaining: float, note: str = ""):
    """Catat sisa compute units SECARA MANUAL (baca dari panel resources Colab) -> konsumsi unit nyata."""
    log_event(event="units_remaining", remaining=float(remaining), note=note)


def log_session_start(root, extra=None):
    """Catat snapshot lingkungan sesi (GPU, driver, CPU, RAM, disk, versi library)."""
    import peft
    import transformers

    _LOG["session"] = uuid.uuid4().hex[:8]
    env = {"python": platform.python_version(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
           "transformers": transformers.__version__, "peft": peft.__version__, "cpu_count": os.cpu_count()}
    try:
        import psutil
        env["ram_total_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        pass
    try:
        du = shutil.disk_usage(root)
        env["drive_total_gb"], env["drive_free_gb"] = round(du.total / 2**30), round(du.free / 2**30)
        du = shutil.disk_usage("/content") if os.path.exists("/content") else du
        env["local_disk_free_gb"] = round(du.free / 2**30)
    except Exception:
        pass
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        env.update({"gpu": p.name, "gpu_mem_gb": round(p.total_memory / 2**30, 1),
                    "gpu_capability": f"{p.major}.{p.minor}", "gpu_sm_count": p.multi_processor_count})
        try:
            import pynvml
            pynvml.nvmlInit()
            d = pynvml.nvmlSystemGetDriverVersion()
            env["driver"] = d.decode() if isinstance(d, bytes) else d
            env["power_limit_w"] = pynvml.nvmlDeviceGetEnforcedPowerLimit(pynvml.nvmlDeviceGetHandleByIndex(0)) / 1000
        except Exception:
            pass
    env.update(extra or {})
    log_event(event="session_start", env=env)
    if _LOG["dir"] is not None:
        atomic_json(env, _LOG["dir"] / f"env_snapshot_{_LOG['session']}.json")
    return env


class ResourceMonitor:
    """Sampling di thread terpisah: utilisasi/memori/daya/suhu GPU (NVML) + RAM proses & sistem (psutil).
    Energi (Wh) = integral daya terhadap waktu. Semua gagal-aman: bila NVML/psutil tidak ada, kolom kosong."""

    def __init__(self, interval=5.0, index=0):
        self.interval, self.index = interval, index
        self._nv = self._h = self._ps = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._t = None

    def start(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nv, self._h = pynvml, pynvml.nvmlDeviceGetHandleByIndex(self.index)
        except Exception:
            self._nv = None
        try:
            import psutil
            self._ps = psutil
            self._proc = psutil.Process()
        except Exception:
            self._ps = None
        self.t0 = self._last = time.time()
        self.n = 0
        self.s = {"util": 0.0, "power": 0.0}      # akumulasi berbobot waktu
        self.mx = {"util": 0.0, "power": 0.0, "temp": 0.0, "gmem": 0.0, "pram": 0.0, "sram": 0.0}
        self.energy_ws = 0.0
        self._stop.clear()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def _loop(self):
        while True:
            self._sample()
            if self._stop.wait(self.interval):
                break

    def _sample(self):
        now = time.time()
        with self._lock:
            dt = now - self._last
            self._last = now
            self.n += 1
            if self._nv is not None:
                try:
                    u = self._nv.nvmlDeviceGetUtilizationRates(self._h).gpu
                    pw = self._nv.nvmlDeviceGetPowerUsage(self._h) / 1000.0
                    tp = self._nv.nvmlDeviceGetTemperature(self._h, self._nv.NVML_TEMPERATURE_GPU)
                    gm = self._nv.nvmlDeviceGetMemoryInfo(self._h).used / 2**20
                    self.s["util"] += u * dt
                    self.s["power"] += pw * dt
                    self.energy_ws += pw * dt
                    self.mx["util"], self.mx["power"] = max(self.mx["util"], u), max(self.mx["power"], pw)
                    self.mx["temp"], self.mx["gmem"] = max(self.mx["temp"], tp), max(self.mx["gmem"], gm)
                except Exception:
                    pass
            if self._ps is not None:
                try:
                    self.mx["pram"] = max(self.mx["pram"], self._proc.memory_info().rss / 2**20)
                    self.mx["sram"] = max(self.mx["sram"], self._ps.virtual_memory().used / 2**20)
                except Exception:
                    pass

    def snapshot(self):
        with self._lock:
            wall = max(time.time() - self.t0, 1e-9)
            has = self._nv is not None
            return {"wall_seconds": wall, "n_samples": self.n, "n_sessions": 1,
                    "gpu_util_mean": self.s["util"] / wall if has else None, "gpu_util_max": self.mx["util"] if has else None,
                    "power_mean_w": self.s["power"] / wall if has else None, "power_max_w": self.mx["power"] if has else None,
                    "energy_wh": self.energy_ws / 3600.0 if has else None, "temp_max_c": self.mx["temp"] if has else None,
                    "gpu_mem_used_peak_mb": self.mx["gmem"] if has else None,
                    "proc_ram_peak_mb": self.mx["pram"] if self._ps else None,
                    "sys_ram_used_peak_mb": self.mx["sram"] if self._ps else None}

    def stop(self):
        if self._t is not None and self._t.is_alive():
            self._stop.set()
            self._t.join(timeout=10)
        return self.snapshot()


def merge_res(a, b):
    """Gabungkan ringkasan sumber daya antar-sesi (resume): jumlah untuk waktu/energi, maksimum untuk puncak,
    rata-rata berbobot waktu untuk utilisasi/daya."""
    if not a:
        return b
    out = {"wall_seconds": a["wall_seconds"] + b["wall_seconds"], "n_samples": a["n_samples"] + b["n_samples"],
           "n_sessions": a["n_sessions"] + b["n_sessions"]}
    for k in ("gpu_util_mean", "power_mean_w"):
        va, vb = a.get(k), b.get(k)
        out[k] = None if va is None and vb is None else ((va or 0) * a["wall_seconds"] + (vb or 0) * b["wall_seconds"]) / out["wall_seconds"]
    out["energy_wh"] = None if a.get("energy_wh") is None and b.get("energy_wh") is None else (a.get("energy_wh") or 0) + (b.get("energy_wh") or 0)
    for k in ("gpu_util_max", "power_max_w", "temp_max_c", "gpu_mem_used_peak_mb", "proc_ram_peak_mb", "sys_ram_used_peak_mb"):
        vs = [v for v in (a.get(k), b.get(k)) if v is not None]
        out[k] = max(vs) if vs else None
    return out


# ----------------------------------------------------------------------------- training
DIVERGE_TRANSIENT = True   # v4.6: (c) collapse sementara — Spearman validasi < ambang pada EPOCH MANA PUN (mis. 0,425 di epoch 2 lalu pulih)


def _diverged_from_hist(history, transient=None):
    """Collapse bila (a) Spearman validasi terbaik < DIVERGE_SPEARMAN, atau (b) jatuh >0,10 dari puncak sampai epoch terakhir
    (training runtuh setelah sempat membaik; mis. puncak 0,66 di epoch 1 lalu ~0,05)."""
    v = [h["val_spearman"] for h in history]
    tr = DIVERGE_TRANSIENT if transient is None else transient
    return bool(max(v) < DIVERGE_SPEARMAN or (max(v) - v[-1]) > 0.10 or (tr and min(v) < DIVERGE_SPEARMAN))


def is_diverged(result):
    return _diverged_from_hist(result["history"]) if result.get("history") else bool(result["val_spearman"] < DIVERGE_SPEARMAN)


def run_train(cfg: Config, root, train_df, val_df, device=None, log=print, ckpt_every=1, monitor_interval=5.0, scratch_root=None):
    """Latih satu konfigurasi dengan checkpoint per epoch. Aman dipanggil ulang (resume/skip).
    Mencatat waktu, memori, utilisasi, daya & energi ke result.json dan logs/compute_log.jsonl.
    scratch_root (disk lokal VM, mis. /content/sts_scratch): state optimizer per epoch dan bobot-terbaik sementara ditulis di sini,
    HANYA hasil akhir (best.pt, val_pred.npy, result.json) ditulis ke Drive -> bandwidth/operasi Drive turun >10x untuk Full FT
    (state AdamW XLM-R = 3,3 GB per epoch). Konsekuensi: run yang terputus di tengah diulang dari awal (maks. 1 run per putus sesi)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(root) / cfg.run_id()
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    work = (Path(scratch_root) / cfg.run_id()) if scratch_root else run_dir
    work.mkdir(parents=True, exist_ok=True)
    state_path, best_path, valpred_path = work / "state.pt", work / "best.pt", work / "val_pred.npy"
    legacy_state = run_dir / "state.pt"          # state lama di Drive (run yang terputus sebelum memakai scratch)

    if result_path.exists():
        log(f"[SKIP] {cfg.run_id()} sudah selesai")
        log_event(event="skip", run_id=cfg.run_id())
        return json.loads(result_path.read_text())

    precision = pick_precision(device)
    set_seed(cfg.seed)
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    model = build_model(cfg).to(device)
    n_tr, n_tot = count_params(model)
    opt = build_optimizer(model, cfg)
    steps_per_epoch = (len(train_df) + cfg.batch_size - 1) // cfg.batch_size
    total = steps_per_epoch * cfg.epochs
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"

    start_epoch, history, train_secs, peak_mem, peak_res, best_val, res_acc = 0, [], 0.0, 0.0, 0.0, -1.0, None
    load_state = state_path if state_path.exists() else (legacy_state if (scratch_root and legacy_state.exists()) else None)
    st = None
    if load_state is not None:
        try:
            st = _load_retry(load_state, "cpu", weights_only=False)
        except (EOFError, RuntimeError, pickle.UnpicklingError, ValueError) as e:
            # state.pt terpotong/rusak (mis. tulis terputus saat kuota Drive habis) -> buang, ulang run dari awal
            print(f"  [RUSAK] checkpoint {load_state.name} tidak terbaca ({e!r}); dihapus, run diulang dari epoch 0")
            for p_ in {load_state, state_path, legacy_state}:
                try: Path(p_).unlink()
                except OSError: pass
            st = None
    if st is not None:
        model.load_state_dict(st["weights"], strict=False)
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        start_epoch, history = st["epoch"], st["history"]
        train_secs, peak_mem, best_val = st["train_secs"], st["peak_mem"], st["best_val"]
        peak_res, res_acc = st.get("peak_res", 0.0), st.get("res_acc")
        random.setstate(st["rng_py"])
        np.random.set_state(st["rng_np"])
        torch.set_rng_state(st["rng_torch"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(st["rng_cuda"])
        log(f"[RESUME] {cfg.run_id()} dari epoch {start_epoch}/{cfg.epochs}")
        log_event(event="resume", run_id=cfg.run_id(), epoch=start_epoch, gpu=gpu_name)
    else:
        log(f"[START] {cfg.run_id()} | trainable={n_tr:,}/{n_tot:,} | {precision}")
        log_event(event="start", run_id=cfg.run_id(), trainable=n_tr, total=n_tot, precision=precision, gpu=gpu_name)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    mon = ResourceMonitor(monitor_interval).start()

    try:
        for epoch in range(start_epoch, cfg.epochs):
            model.train()
            t0 = time.time()
            g = np.random.default_rng(cfg.seed * 1000 + epoch)      # urutan data deterministik per epoch
            order = g.permutation(len(train_df))
            tl = 0.0
            for s1, s2, y in _batches(train_df, cfg.batch_size, order):
                e1 = tok(s1, padding=True, truncation=True, max_length=cfg.max_len, return_tensors="pt").to(device)
                e2 = tok(s2, padding=True, truncation=True, max_length=cfg.max_len, return_tensors="pt").to(device)
                y = y.to(device)
                with _autocast(precision, device):
                    u, v = model.embed(e1), model.embed(e2)
                u, v = u.float(), v.float()
                loss = cosent_loss(u, v, y, cfg.cosent_scale) if cfg.strategy == "S2" else cosine_mse_loss(u, v, y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                tl += loss.item()
            if device.type == "cuda":
                torch.cuda.synchronize()
            ep_secs = time.time() - t0
            train_secs += ep_secs
            if device.type == "cuda":
                peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / 2**20)
                peak_res = max(peak_res, torch.cuda.max_memory_reserved() / 2**20)

            pv = predict(model, tok, val_df, cfg, device, precision)
            m = metrics(pv, val_df["label"].values)
            history.append({"epoch": epoch + 1, "loss": tl / steps_per_epoch, "epoch_seconds": ep_secs,
                            "samples_per_s": len(train_df) / ep_secs, **{f"val_{k}": v for k, v in m.items()}})
            log(f"  epoch {epoch + 1}/{cfg.epochs} loss={tl / steps_per_epoch:.4f} val_spearman={m['spearman']:.4f} ({ep_secs:.0f}s)")
            log_event(event="epoch", run_id=cfg.run_id(), epoch=epoch + 1, epoch_seconds=round(ep_secs, 2),
                      val_spearman=m["spearman"], loss=tl / steps_per_epoch, peak_alloc_mb=peak_mem)
            if m["spearman"] > best_val:
                best_val = m["spearman"]
                atomic_save(trainable_state(model), best_path)
                np.save(valpred_path, pv)
            if (epoch + 1) % ckpt_every == 0 or epoch + 1 == cfg.epochs:
                atomic_save({"weights": trainable_state(model), "opt": opt.state_dict(), "sched": sched.state_dict(),
                             "scaler": scaler.state_dict(), "epoch": epoch + 1, "history": history,
                             "train_secs": train_secs, "peak_mem": peak_mem, "peak_res": peak_res, "best_val": best_val,
                             "res_acc": merge_res(res_acc, mon.snapshot()),
                             "rng_py": random.getstate(), "rng_np": np.random.get_state(),
                             "rng_torch": torch.get_rng_state(),
                             "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None},
                            state_path)
    finally:
        seg = mon.stop()

    resources = merge_res(res_acc, seg)
    best_h = max(history, key=lambda h: h["val_spearman"])
    result = {"run_id": cfg.run_id(), "stage": _LOG["stage"], "config": asdict(cfg), "best_epoch": int(best_h["epoch"]),
              "val_spearman": best_val, "val_pearson": best_h["val_pearson"],
              "diverged": _diverged_from_hist(history),
              "history": history, "trainable_params": n_tr, "total_params": n_tot,
              "peak_gpu_mem_mb": peak_mem, "peak_gpu_mem_reserved_mb": peak_res,
              "train_seconds": train_secs, "wall_seconds": resources["wall_seconds"],
              "samples_per_s": len(train_df) * cfg.epochs / max(train_secs, 1e-9),
              "checkpoint_mb": best_path.stat().st_size / 2**20, "precision": precision,
              "resources": resources, "env": env_info(device)}
    if scratch_root:                              # satu kali tulis ke Drive: bobot terbaik + prediksi validasi, lalu hasil
        copy_to_drive(best_path, run_dir / "best.pt")
        if valpred_path.exists():
            copy_to_drive(valpred_path, run_dir / "val_pred.npy")
    atomic_json(result, result_path)
    log_event(event="finish", run_id=cfg.run_id(), val_spearman=best_val, train_seconds=round(train_secs, 1),
              wall_seconds=round(resources["wall_seconds"], 1), energy_wh=resources.get("energy_wh"), gpu=gpu_name)
    state_path.unlink(missing_ok=True)       # hemat ruang: state optimizer tidak dibutuhkan lagi
    if scratch_root:
        legacy_state.unlink(missing_ok=True)
        shutil.rmtree(work, ignore_errors=True)
    return result


# ----------------------------------------------------------------------------- evaluasi
def eval_run(cfg: Config, root, df, tag, device=None, log=print):
    """Evaluasi bobot terbaik sebuah run pada dataset `df` (mis. test EN, transfer ID). Hasil di-cache."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(root) / cfg.run_id()
    out_json, out_npy = run_dir / f"eval_{tag}.json", run_dir / f"pred_{tag}.npy"
    if out_json.exists():
        return json.loads(out_json.read_text())
    precision = pick_precision(device)
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    model = build_model(cfg).to(device)
    model.load_state_dict(_load_retry(run_dir / "best.pt", device), strict=False)
    pred = predict(model, tok, df, cfg, device, precision)
    res = {"tag": tag, "n": len(df), **metrics(pred, df["label"].values)}
    np.save(out_npy, pred)
    atomic_json(res, out_json)
    log(f"[EVAL] {cfg.run_id()} {tag}: spearman={res['spearman']:.4f}")
    return res


def eval_multi(cfg: Config, root, dfs: dict, device=None, log=print):
    """Evaluasi bobot terbaik sebuah run pada BEBERAPA dataset sekaligus (model dimuat sekali). Hasil di-cache."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(root) / cfg.run_id()
    todo = [t for t in dfs if not (run_dir / f"eval_{t}.json").exists()]
    if not todo:
        return
    precision = pick_precision(device)
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    model = build_model(cfg).to(device)
    model.load_state_dict(_load_retry(run_dir / "best.pt", device), strict=False)
    for tag in todo:
        pred = predict(model, tok, dfs[tag], cfg, device, precision)
        res = {"tag": tag, "n": len(dfs[tag]), **metrics(pred, dfs[tag]["label"].values)}
        np.save(run_dir / f"pred_{tag}.npy", pred)
        atomic_json(res, run_dir / f"eval_{tag}.json")
    log_event(event="eval", run_id=cfg.run_id(), tags=todo)
    del model


# ----------------------------------------------------------------------------- statistik
def paired_bootstrap_diff(pred_a, pred_b, y, n_boot=10000, seed=0):
    """Selisih Spearman (a - b) pada contoh uji yang sama; kembalikan (selisih, CI95)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    d = np.empty(n_boot)
    for i in range(n_boot):
        ix = rng.integers(0, n, n)
        d[i] = spearmanr(pred_a[ix], y[ix])[0] - spearmanr(pred_b[ix], y[ix])[0]
    return float(spearmanr(pred_a, y)[0] - spearmanr(pred_b, y)[0]), tuple(np.percentile(d, [2.5, 97.5]))


def tost_paired(diffs, delta):
    """TOST berpasangan (Lakens, 2017) pada selisih per-seed; kembalikan p_TOST (maks dua uji satu sisi)."""
    from scipy.stats import ttest_1samp

    d = np.asarray(diffs, float)
    p_lo = ttest_1samp(d, -delta, alternative="greater").pvalue
    p_hi = ttest_1samp(d, delta, alternative="less").pvalue
    return float(max(p_lo, p_hi))


def holm(pvals):
    """Koreksi Holm-Bonferroni (Holm, 1979); kembalikan p yang disesuaikan sesuai urutan input."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def _spearman_rows(P, y):
    """Spearman tiap baris P (m x n) terhadap y (n) -- vektorisasi via peringkat."""
    rp = rankdata(P, axis=1)
    ry = rankdata(y)
    rp = rp - rp.mean(1, keepdims=True)
    ry = ry - ry.mean()
    return (rp @ ry) / (np.sqrt((rp ** 2).sum(1)) * np.sqrt((ry ** 2).sum()) + 1e-12)


def boot_mean_diff(PA, PB, y, n_boot=5000, seed=0):
    """Bootstrap atas PASANGAN KALIMAT uji: selisih rerata Spearman (A - B), dirata-ratakan lintas seed.
    PA (kA x n), PB (kB x n) = prediksi tiap seed. Kembalikan (selisih titik, (CI95 bawah, atas))."""
    PA, PB, y = np.atleast_2d(PA), np.atleast_2d(PB), np.asarray(y)
    n = len(y)
    rng = np.random.default_rng(seed)
    point = float(_spearman_rows(PA, y).mean() - _spearman_rows(PB, y).mean())
    d = np.empty(n_boot)
    for b in range(n_boot):
        ix = rng.integers(0, n, n)
        d[b] = _spearman_rows(PA[:, ix], y[ix]).mean() - _spearman_rows(PB[:, ix], y[ix]).mean()
    return point, (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def boot_drop_rel(P_en, y_en, P_id, y_id, n_boot=2000, seed=0):
    """Penurunan relatif transfer Drel = (rho_EN - rho_ID)/rho_EN pada PASANGAN PARALEL (indeks sama di EN & ID).
    Kembalikan (rho_EN, rho_ID, Drel, (CI95 bawah, atas))."""
    P_en, P_id, y_en, y_id = np.atleast_2d(P_en), np.atleast_2d(P_id), np.asarray(y_en), np.asarray(y_id)
    n = len(y_en)
    rng = np.random.default_rng(seed)

    def stat(ix):
        e, i = _spearman_rows(P_en[:, ix], y_en[ix]).mean(), _spearman_rows(P_id[:, ix], y_id[ix]).mean()
        return e, i, (e - i) / e

    e0, i0, d0 = stat(np.arange(n))
    ds = np.array([stat(rng.integers(0, n, n))[2] for _ in range(n_boot)])
    return float(e0), float(i0), float(d0), (float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)))


def tost_table(d, delta):
    """Uji ekuivalensi TOST berpasangan (Lakens, 2017) pada selisih per-seed d. Mengembalikan tabel perhitungan lengkap."""
    from scipy.stats import t as tdist
    d = np.asarray(d, float)
    n = len(d)
    m, sd = d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n)
    t1, t2 = (m + delta) / se, (m - delta) / se
    p1, p2 = 1 - tdist.cdf(t1, n - 1), tdist.cdf(t2, n - 1)
    tc = tdist.ppf(0.975, n - 1)
    return {"n": n, "mean_diff": m, "sd_diff": sd, "se": se, "ci95_lo": m - tc * se, "ci95_hi": m + tc * se,
            "t_lower": t1, "t_upper": t2, "p_lower": p1, "p_upper": p2, "p_tost": max(p1, p2)}


def collect_results(root) -> pd.DataFrame:
    rows = []
    for f in Path(root).glob("*/result.json"):
        r = json.loads(f.read_text())
        c = r["config"]
        rs = r.get("resources") or {}
        row = {"run_id": r["run_id"], "stage": r.get("stage"), "model": c["model_name"], "method": c["method"],
               "rank": c["rank"], "strategy": c["strategy"], "lr": c["lr"], "seed": c["seed"], "epochs": c["epochs"],
               "rslora": c.get("rslora", False), "lora_targets": c.get("lora_targets", "all-linear"),
               "val_spearman": r["val_spearman"], "diverged": is_diverged(r),
               "diverged_orig": (_diverged_from_hist(r["history"], transient=False) if r.get("history") else bool(r["val_spearman"] < DIVERGE_SPEARMAN)),
               "val_min": (min(h["val_spearman"] for h in r["history"]) if r.get("history") else None),
               "best_epoch": r.get("best_epoch"), "trainable_params": r["trainable_params"],
               "total_params": r.get("total_params"), "peak_gpu_mem_mb": r["peak_gpu_mem_mb"],
               "peak_gpu_mem_reserved_mb": r.get("peak_gpu_mem_reserved_mb"),
               "train_seconds": r["train_seconds"], "wall_seconds": r.get("wall_seconds", r["train_seconds"]),
               "epoch_seconds": r["train_seconds"] / c["epochs"], "samples_per_s": r.get("samples_per_s"),
               "checkpoint_mb": r["checkpoint_mb"], "precision": r["precision"], "gpu": r["env"]["gpu"],
               "gpu_util_mean": rs.get("gpu_util_mean"), "power_mean_w": rs.get("power_mean_w"),
               "energy_wh": rs.get("energy_wh"), "proc_ram_peak_mb": rs.get("proc_ram_peak_mb"),
               "n_sessions": rs.get("n_sessions")}
        for e in f.parent.glob("eval_*.json"):
            ev = json.loads(e.read_text())
            row[f"{ev['tag']}_spearman"] = ev["spearman"]
        rows.append(row)
    return pd.DataFrame(rows)