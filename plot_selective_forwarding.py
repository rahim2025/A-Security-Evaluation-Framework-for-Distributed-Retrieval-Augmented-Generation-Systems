import argparse,os,sys
import numpy as np
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker as mt
except ImportError: print("pip install matplotlib"); sys.exit(1)
try: import pandas as pd
except ImportError: print("pip install pandas"); sys.exit(1)

C={"baseline":"#2c7bb6","attack_only":"#d7191c","attack_plus_defense":"#1a9641"}
L={"baseline":"Baseline","attack_only":"Attack only (SFA)","attack_plus_defense":"Attack+Defense (SFD)"}
M={"baseline":"o","attack_only":"s","attack_plus_defense":"^"}

def load(p):
    if not os.path.exists(p): print(f"Not found: {p}\nRun simulation first."); sys.exit(1)
    return pd.read_csv(p)

def fig1(df,od):
    strats=[s for s in["random","high_connectivity"]if s in df.strategy.values]
    fig,axes=plt.subplots(1,len(strats),figsize=(6*len(strats),5),sharey=True)
    if len(strats)==1: axes=[axes]
    fig.suptitle("Hit Rate vs Compromise Ratio",fontsize=13,fontweight="bold")
    for ax,st in zip(axes,strats):
        s=df[df.strategy.isin([st,"none"])]
        for lb in["baseline","attack_only","attack_plus_defense"]:
            r=s[s.label==lb].sort_values("attack_ratio")
            if r.empty: continue
            ye=r.hit_rate_std.values if "hit_rate_std" in r else None
            ax.errorbar(r.attack_ratio,r.hit_rate,yerr=ye,label=L[lb],color=C[lb],marker=M[lb],linewidth=2,markersize=7,capsize=4 if ye is not None else 0,linestyle="--"if lb=="baseline"else"-")
        ax.set_title({"random":"Random","high_connectivity":"Hub Targeting"}.get(st,st),fontsize=11)
        ax.set_xlabel("Compromise Ratio"); ax.set_ylabel("Hit Rate")
        ax.set_xlim(-0.02,0.57); ax.set_ylim(0.7,1.05)
        ax.yaxis.set_major_formatter(mt.FormatStrFormatter("%.2f"))
        ax.grid(True,alpha=0.3); ax.legend(fontsize=9)
    plt.tight_layout(); out=os.path.join(od,"fig1_hit_rate_vs_ratio.png"); plt.savefig(out,dpi=150,bbox_inches="tight"); plt.close(); return out

def fig2(df,od):
    if "early_hit_rate" not in df.columns: print("WARNING: no early_hit_rate col — re-run sim after patching"); return ""
    strats=[s for s in["random","high_connectivity"]if s in df.strategy.values]
    fig,axes=plt.subplots(1,len(strats),figsize=(6*len(strats),5),sharey=True)
    if len(strats)==1: axes=[axes]
    fig.suptitle("Defense Warm-up: Early vs Late Phase Hit Rate",fontsize=13,fontweight="bold")
    specs=[("attack_only","early_hit_rate","--","#d7191c","s","Attack Early"),
           ("attack_only","late_hit_rate","-","#d7191c","s","Attack Late"),
           ("attack_plus_defense","early_hit_rate","--","#1a9641","^","Defense Early"),
           ("attack_plus_defense","late_hit_rate","-","#1a9641","^","Defense Late")]
    for ax,st in zip(axes,strats):
        s=df[df.strategy.isin([st,"none"])]
        for lb,ph,ls,col,mk,lbl in specs:
            r=s[s.label==lb].sort_values("attack_ratio")
            if r.empty or ph not in r.columns: continue
            ax.plot(r.attack_ratio,r[ph],label=lbl,color=col,marker=mk,linewidth=2,markersize=7,linestyle=ls)
        b=s[s.label=="baseline"]
        if not b.empty: ax.axhline(float(b.hit_rate.values[0]),color="#2c7bb6",linestyle=":",linewidth=1.5,label="Baseline")
        ax.set_title({"random":"Random","high_connectivity":"Hub Targeting"}.get(st,st),fontsize=11)
        ax.set_xlabel("Compromise Ratio"); ax.set_ylabel("Hit Rate")
        ax.set_xlim(-0.02,0.57); ax.set_ylim(0.7,1.05)
        ax.yaxis.set_major_formatter(mt.FormatStrFormatter("%.2f"))
        ax.grid(True,alpha=0.3); ax.legend(fontsize=8,ncol=2)
    plt.tight_layout(); out=os.path.join(od,"fig2_early_vs_late.png"); plt.savefig(out,dpi=150,bbox_inches="tight"); plt.close(); return out

def fig3(df,od):
    if "defense_blacklisted" not in df.columns: return ""
    strats=[s for s in["random","high_connectivity"]if s in df.strategy.values]
    fig,axes=plt.subplots(1,len(strats),figsize=(6*len(strats),5),sharey=True)
    if len(strats)==1: axes=[axes]
    fig.suptitle("Detection Accuracy: Blacklisted vs Compromised",fontsize=13,fontweight="bold")
    for ax,st in zip(axes,strats):
        s=df[(df.label=="attack_plus_defense")&(df.strategy==st)].sort_values("attack_ratio")
        if s.empty: continue
        x=s.attack_ratio.values; comp=s.num_compromised.values; bl=s.defense_blacklisted.values
        det=np.where(comp>0,bl/comp*100,0.)
        ax2=ax.twinx()
        ax.bar(x-0.015,comp,width=0.025,label="Compromised",color="#d7191c",alpha=0.7)
        ax.bar(x+0.015,bl,width=0.025,label="Blacklisted",color="#1a9641",alpha=0.7)
        ax2.plot(x,det,"k--o",linewidth=2,markersize=6,label="Detection %")
        ax2.set_ylabel("Detection Rate (%)"); ax2.set_ylim(0,120)
        ax.set_title({"random":"Random","high_connectivity":"Hub Targeting"}.get(st,st),fontsize=11)
        ax.set_xlabel("Compromise Ratio"); ax.set_ylabel("Peers")
        ax.grid(True,alpha=0.3,axis="y")
        l1,lb1=ax.get_legend_handles_labels(); l2,lb2=ax2.get_legend_handles_labels()
        ax.legend(l1+l2,lb1+lb2,fontsize=9)
    plt.tight_layout(); out=os.path.join(od,"fig3_detection.png"); plt.savefig(out,dpi=150,bbox_inches="tight"); plt.close(); return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--csv",default="logs/selective_forwarding/results_with_defense.csv")
    p.add_argument("--outdir",default="logs/selective_forwarding/")
    a=p.parse_args(); os.makedirs(a.outdir,exist_ok=True)
    df=load(a.csv); print(f"Loaded {len(df)} rows")
    for fn,o in [("fig1",fig1(df,a.outdir)),("fig2",fig2(df,a.outdir)),("fig3",fig3(df,a.outdir))]:
        if o: print(f"{fn}: {o}")
    print("Done — open logs/selective_forwarding/ to view PNGs")

if __name__=="__main__": main()
