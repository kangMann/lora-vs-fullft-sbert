
## Lampiran A — Rumus matematika dan algoritma (siap disalin ke naskah)

### A.1 Rumus

**(1) Representasi kalimat dan kemiripan (SBERT; Reimers & Gurevych, 2019).** Untuk kalimat $a$ dengan keluaran encoder $h_t\in\mathbb{R}^{d}$ dan mask $m_t\in\{0,1\}$:

$$u=\frac{\sum_t m_t h_t}{\sum_t m_t},\qquad s(a,b)=\cos(u,v)=\frac{u^{\top}v}{\lVert u\rVert\,\lVert v\rVert}$$

**(2) Fungsi kerugian S0 (MSE kosinus).** Untuk *batch* berukuran $B$ dan skor acuan $y_i\in[0,1]$:

$$\mathcal{L}_{\mathrm{MSE}}=\frac{1}{B}\sum_{i=1}^{B}\bigl(s_i-y_i\bigr)^2$$

**(3) Fungsi kerugian S2 (CoSENT; Su, 2022)** dengan faktor skala $\lambda=20$:

$$\mathcal{L}_{\mathrm{CoSENT}}=\log\Bigl(1+\sum_{(i,j):\,y_i>y_j}\exp\bigl(\lambda\,(s_j-s_i)\bigr)\Bigr)$$

**(4) LoRA (Hu dkk., 2022).** Bobot beku $W_0\in\mathbb{R}^{d\times k}$ diperbarui secara *low-rank*; $B\in\mathbb{R}^{d\times r}$, $A\in\mathbb{R}^{r\times k}$, $r\ll\min(d,k)$:

$$W=W_0+\frac{\alpha}{r}BA,\qquad \#\theta_{\mathrm{LoRA}}=\sum_{\text{lapisan}} r\,(d+k),\qquad \alpha=2r\;\Rightarrow\;\frac{\alpha}{r}=2$$

**(5) rsLoRA (Kalajdzievski, 2023)** mengganti skala menjadi $\gamma_r=\alpha/\sqrt{r}$ (pada penelitian ini $\alpha=2\sqrt{8}$ sehingga $\gamma_8=2$, sama dengan LoRA standar pada $r=8$).

**(6) Layer-wise LR decay, S1 (Howard & Ruder, 2018)** untuk lapisan $\ell\in\{0,\dots,L-1\}$ dan faktor $\rho=0{,}9$:

$$\eta_\ell=\eta\,\rho^{\,L-1-\ell}$$

**(7) Optimisasi.** AdamW (Loshchilov & Hutter, 2019), $\beta=(0{,}9;0{,}999)$, *weight decay* $0{,}01$, *clipping* norma gradien $1{,}0$; laju belajar linear dengan *warmup* 10 %:

$$\eta_t=\eta\cdot\begin{cases} t/T_w, & t<T_w\\[2pt] (T-t)/(T-T_w), & t\ge T_w\end{cases}$$

**(8) Metrik.** Korelasi Spearman (metrik utama) adalah korelasi Pearson antar-peringkat; tanpa nilai kembar $\rho=1-\dfrac{6\sum_i d_i^2}{n(n^2-1)}$ dengan $d_i$ selisih peringkat.

**(9) Kriteria *collapse* dan laju divergensi.** Satu *run* disebut *collapse* bila $\max_{e}\rho^{\mathrm{val}}_{e}<0{,}50$ (di bawah baseline tanpa fine-tuning). Laju divergensi sebuah konfigurasi: $\mathrm{DR}=n_{\mathrm{collapse}}/n_{\mathrm{run}}$; *run collapse* dikeluarkan dari rerata dan dilaporkan melalui DR.

**(10) Selisih berpasangan per seed.** Untuk seed $k$ dan metrik $\rho$: $d_k=\rho^{\mathrm{LoRA}}_k-\rho^{\mathrm{Full}}_k$; rerata $\bar d$, simpangan baku $s_d$, $n$ pasangan seed.

**(11) Uji ekuivalensi TOST (Lakens, 2017)** dengan margin $\delta=0{,}01$:

$$t_L=\frac{\bar d+\delta}{s_d/\sqrt n},\quad t_U=\frac{\bar d-\delta}{s_d/\sqrt n},\quad p_{\mathrm{TOST}}=\max\bigl(P(T_{n-1}\ge t_L),\;P(T_{n-1}\le t_U)\bigr)$$

LoRA dinyatakan *ekuivalen* dengan full fine-tuning bila $p_{\mathrm{TOST}}^{\mathrm{Holm}}<0{,}05$. Interval kepercayaan 95 %: $\bar d\pm t_{0{,}975,\,n-1}\,s_d/\sqrt n$.

**(12) *Bootstrap* berpasangan atas pasangan kalimat uji.** Pada $b=1,\dots,B_{\mathrm{boot}}$, indeks kalimat diambil dengan pengembalian ($n=1379$); statistik $\Delta^{(b)}=\overline{\rho}^{\,(b)}_{\mathrm{LoRA}}-\overline{\rho}^{\,(b)}_{\mathrm{Full}}$ (rerata lintas seed); CI 95 % = persentil ke-2,5 dan 97,5.

**(13) Koreksi Holm (1979).** Untuk $m$ pengujian dalam satu famili, p-value terurut $p_{(1)}\le\dots\le p_{(m)}$:

$$\tilde p_{(i)}=\min\Bigl(1,\;\max_{j\le i}\,(m-j+1)\,p_{(j)}\Bigr)$$

**(14) Interaksi metode × strategi (RQ2).** Dengan $g^{M}_{S,k}=\rho^{M}_{S,k}-\rho^{M}_{S0,k}$ ($M\in\{\mathrm{LoRA},\mathrm{Full}\}$): $I_S=\overline{g^{\mathrm{LoRA}}_S-g^{\mathrm{Full}}_S}$; diuji dengan uji-$t$ satu sampel terhadap 0.

**(15) Penurunan relatif transfer (RQ3).**

$$\Delta_{\mathrm{rel}}=\frac{\rho_{\mathrm{EN}}-\rho_{\mathrm{ID}}}{\rho_{\mathrm{EN}}}$$

dengan CI *bootstrap* atas pasangan paralel EN–ID (indeks sama).

**(16) Sumber daya.** Energi GPU $E=\int_0^{T}P(t)\,dt\approx\dfrac{1}{3600}\sum_k P_k\,\Delta t_k$ (Wh); rasio efisiensi terhadap full fine-tuning $R_X=X_{\mathrm{LoRA}}/X_{\mathrm{Full}}$ untuk $X\in\{$waktu per epoch, memori puncak, energi, ukuran *checkpoint*, parameter terlatih$\}$.

### A.2 Algoritma

**Algoritma 1 — Protokol eksperimen keseluruhan**
```
Input : model M ∈ {XLM-R base, RoBERTa base}, metode {Full, LoRA r∈{4,8,16,32}}, strategi {S0,S1,S2}
1  Muat STS-B (train/val/test), SemRel2024 EN & ID, STS-B terjemahan ID; verifikasi pasangan EN–ID paralel
2  untuk setiap (M, metode, rank): jalankan Algoritma 3 pada S0, seed 42  → LR awal η0
3  untuk setiap (M, metode, rank, strategi): latih seed 42 dengan η0; bila collapse → turunkan LR satu langkah grid (Algoritma 3)
4  untuk setiap konfigurasi (30) dan seed k∈{1..5}: Algoritma 2  → bobot terbaik berdasarkan Spearman validasi
5  pilih Full terbaik dan LoRA terbaik per model (rerata Spearman validasi, laju divergensi ≤ 20 %)
6  tambah seed {6..10} untuk dua konfigurasi terpilih → 10 seed
7  ablasi (rsLoRA r∈{16,32}; q,v-saja r∈{8,16}) dan benchmark GPU
8  evaluasi SEKALI pada STS-B test (EN), SemRel EN, STS-B test (ID), SemRel ID
9  Algoritma 4 untuk RQ1, RQ2 (interaksi), RQ3 (Holm per famili)
Output: tabel, grafik, log sumber daya
```

**Algoritma 2 — Pelatihan SBERT (Full atau LoRA) dengan checkpoint**
```
Input : cfg (model, metode, rank, strategi, η, seed), data latih/validasi
1  id ← hash(cfg); bila result.json ada → lewati; bila checkpoint ada → lanjutkan dari epoch terakhir
2  inisialisasi encoder; bila LoRA: bekukan W0, pasang (A,B) pada semua lapisan linear
3  untuk epoch e = 1..4:
4      urutan data ← permutasi deterministik(seed·1000 + e)
5      untuk setiap batch (B=32): u,v ← SBERT(a), SBERT(b); L ← MSE (S0,S1) atau CoSENT (S2)
6          backprop; clip norma gradien 1,0; AdamW (LLRD pada S1); scheduler.step()
7      ρ_val ← Spearman(prediksi validasi); simpan bobot terlatih, optimizer, RNG, riwayat (atomik)
8  bobot terbaik ← epoch dengan ρ_val tertinggi; diverged ← (max ρ_val < 0,50)
9  catat waktu, memori puncak, daya/energi (NVML), ukuran checkpoint
```

**Algoritma 3 — Pencarian LR dengan deteksi collapse**
```
Input : grid G (urut menurun), konfigurasi (M, metode, rank, strategi)
1  untuk η ∈ G: latih (Algoritma 2, seed 42); simpan ρ_val dan flag diverged
2  η* ← argmax ρ_val di antara run yang TIDAK diverged
3  (saat screening strategi S1/S2) bila run dengan η* diverged: η* ← LR berikutnya yang lebih kecil di G; ulangi
4  bila G habis dan masih diverged → hentikan (jeda otomatis) dan laporkan
Output: LR per (model, metode, rank, strategi); catatan fallback
```

**Algoritma 4 — Perbandingan berpasangan LoRA vs Full**
```
Input : prediksi uji per seed, seed yang lolos (tidak diverged) pada kedua metode
1  d_k ← ρ_LoRA,k − ρ_Full,k untuk seed bersama (n ≥ 3)
2  hitung d̄, s_d, CI-t 95 %, p_TOST (δ = 0,01)
3  bootstrap kalimat uji (B_boot) pada rerata prediksi lintas seed → CI 95 % Δρ
4  (hanya bila n ≥ 10) uji Wilcoxon signed-rank
5  koreksi Holm dalam satu famili pengujian
6  keputusan: ekuivalen bila p_Holm < 0,05; "LoRA lebih buruk/baik" bila seluruh CI-t di luar [−δ, +δ]
```
