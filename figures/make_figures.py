"""Redraw manuscript Figures 1-4 from results/tables_v4 (run: python figures/make_figures.py)."""
import os
from pathlib import Path
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
HERE = Path(__file__).resolve().parent
T = str(HERE.parent / 'results' / 'tables_v4') + '/'
os.chdir(HERE)   # PNG files are written next to this script
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],'font.size':8,'axes.spines.top':False,'axes.spines.right':False})
MN={'xlm-roberta-base':'XLM-R base','roberta-base':'RoBERTa base'}
SC={'S0':'#4C72B0','S1':'#DD8452','S2':'#55A868'}
# Fig 1 flow
fig,ax=plt.subplots(figsize=(6.3,2.9)); ax.axis('off'); ax.set_xlim(0,100); ax.set_ylim(0,46)
def box(x,y,w,h,t,fc='#EAF1F8'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.3,rounding_size=1.2',fc=fc,ec='#33506e',lw=0.8))
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=7)
def arr(x1,y1,x2,y2): ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',lw=0.8,color='#33506e'))
row1=[('1  Data\nSTS-B train/val/test\nSemRel2024 EN, ID\nSTS-B ID (test)',2),('2  LR search\nper method and rank\n(S0, seed 42)',27),('3  Screening and\nstability probe\n(S0–S2)',52),('4  Confirmation\n30 configs × 5 seeds\n(+5 seeds, best pair)',77)]
for t,x in row1: box(x,28,21,14,t)
for i in range(3): arr(row1[i][1]+21.4,35,row1[i+1][1]-0.4,35)
row2=[('5  Collapse screen\nand weight audit\n(validation only)',77),('6  Single test\nevaluation\nEN + ID transfer',52),('7  Statistics\nTOST, CI-t, bootstrap,\nHolm',27),('8  Efficiency\nmemory, time, energy,\ncheckpoint size',2)]
for t,x in row2: box(x,4,21,14,t,'#EEF6EC')
arr(87.5,27.6,87.5,18.5)
for i in range(3): arr(row2[i][1]-0.4,11,row2[i+1][1]+21.4,11)
plt.savefig('fig1.png',dpi=300,bbox_inches='tight'); plt.close()
# Fig 2 forest
m=pd.read_csv(T+'T06_rq1_matched.csv'); b=pd.read_csv(T+'T07_rq1_best_vs_best.csv')
fig,axs=plt.subplots(1,2,figsize=(6.3,3.9),sharex=True)
for ax,(mod,nm) in zip(axs,MN.items()):
    d=m[m.model==mod].copy(); rows=[]
    bb=b[b.model==mod].iloc[0]
    rows.append((f"best pair (S2, r={int(bb['rank'])}, n={int(bb.n_seed)})",bb.mean_diff,bb.ci_lo,bb.ci_hi,'#C44E52'))
    for _,r in d.iterrows(): rows.append((f"{r.strat_lora}, r={int(r['rank'])}",r.mean_diff,r.ci_lo,r.ci_hi,SC[r.strat_lora]))
    y=np.arange(len(rows))[::-1]
    ax.axvspan(-0.01,0.01,color='#55A868',alpha=0.15,label='equivalence margin ±0.01'); ax.axvline(0,color='k',lw=0.7)
    for yy,(lab,mu,lo,hi,c) in zip(y,rows):
        ax.errorbar(mu,yy,xerr=[[mu-lo],[hi-mu]],fmt='o',color=c,ms=3,capsize=2,lw=1)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows],fontsize=6.5); ax.set_title(nm,fontsize=8)
    ax.set_xlabel('Δ Spearman (LoRA − Full), 95% CI',fontsize=7); ax.grid(axis='x',alpha=0.3)
    ax.set_xlim(-0.05,0.045)
h,l=axs[0].get_legend_handles_labels(); fig.legend(h,l,loc='lower center',fontsize=6.5,frameon=False)
plt.tight_layout(rect=(0,0.05,1,1)); plt.savefig('fig2.png',dpi=300); plt.close()
# Fig 3 efficiency
e=pd.read_csv(T+'T08_efisiensi.csv'); e=e[e.method=='lora']
fig,axs=plt.subplots(1,4,figsize=(6.5,2.2))
cols=[('R_waktu','Time per epoch'),('R_memori','Peak GPU memory'),('R_energi','Energy per run'),('R_ckpt','Checkpoint size')]
ranks=[4,8,16,32]; x=np.arange(4)
for ax,(c,t) in zip(axs,cols):
    for k,(mod,nm) in enumerate(MN.items()):
        v=[e[(e.model==mod)&(e['rank']==r)][c].iloc[0] for r in ranks]
        ax.bar(x+(k-0.5)*0.38,v,0.38,label=nm,color=['#6BAED6','#FD8D3C'][k])
    ax.axhline(1,color='k',ls='--',lw=0.7); ax.set_xticks(x); ax.set_xticklabels([f'r={r}' for r in ranks],fontsize=6.5)
    ax.set_title(t+'\n(LoRA / Full)',fontsize=7)
    if c=='R_ckpt': ax.set_yscale('log')
    ax.tick_params(labelsize=6.5)
axs[3].legend(fontsize=5.5,loc='upper left',frameon=False)
plt.tight_layout(); plt.savefig('fig3.png',dpi=300); plt.close()
# Fig 4 strategy effects
s=pd.read_csv(T+'T09_rq2_efek_strategi.csv')
fig,axs=plt.subplots(1,2,figsize=(6.3,2.6),sharey=True)
for ax,(mod,nm) in zip(axs,MN.items()):
    d=s[s.model==mod]; labs=[('full',0,'Full FT'),('lora',4,'r=4'),('lora',8,'r=8'),('lora',16,'r=16'),('lora',32,'r=32')]
    for i,(me,rk,lab) in enumerate(labs):
        for j,st in enumerate(['S1','S2']):
            r=d[(d.method==me)&(d['rank']==rk)&(d.strategy==st)].iloc[0]
            xx=i+(j-0.5)*0.28
            ax.errorbar(xx,r.gain_vs_S0,yerr=[[r.gain_vs_S0-r.ci_lo],[r.ci_hi-r.gain_vs_S0]],fmt='o',color=SC[st],ms=3.5,capsize=2,lw=1,label=st if i==0 else None)
    ax.axhline(0,color='k',lw=0.7); ax.set_xticks(range(5)); ax.set_xticklabels([l[2] for l in labs],fontsize=6.5); ax.set_title(nm,fontsize=8); ax.grid(axis='y',alpha=0.3)
axs[0].set_ylabel('Δ Spearman vs. S0 (95% CI)',fontsize=7); axs[0].legend(fontsize=6.5,frameon=False,loc='upper right')
plt.tight_layout(); plt.savefig('fig4.png',dpi=300); plt.close()
