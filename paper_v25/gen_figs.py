import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

plt.rcParams.update({"font.size":11,"axes.titlesize":12,"axes.titleweight":"bold",
                     "figure.dpi":150,"savefig.dpi":150,"font.family":"DejaVu Sans"})

# ---------- FIG 1: System overview (flow) ----------
fig, ax = plt.subplots(figsize=(11,4.2))
ax.set_xlim(0,11); ax.set_ylim(0,4.2); ax.axis("off")
def box(x,y,w,h,text,fc,ec):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.05",
        fc=fc,ec=ec,lw=1.4))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=9.5,fontweight="bold")
def arrow(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",lw=1.6,
        color="#37474f",mutation_scale=12))
box(0.2,1.7,2.0,1.0,"Quantum / PT\nsource prior\n(RCS, Quafu/Baihua)", "#E3F2FD","#1565C0")
box(2.6,1.7,2.0,1.0,"Three training-time\ndistributions\nschedule / risk / noise", "#E8F5E9","#2E7D32")
box(5.0,2.9,2.0,1.0,"Meta-World MT10/MT50\nsimulation validation", "#FFF3E0","#EF6C00")
box(5.0,0.5,2.0,1.0,"Open X real data\n171.6 GiB · 562 shards\n2071 episodes", "#F3E5F5","#6A1B9A")
box(7.4,1.7,1.9,1.0,"Allocation head\n(865 params)\ntrained 20k steps", "#E0F7FA","#00838F")
box(9.6,1.7,1.3,1.0,"Data engine\n+ API\nservice", "#FFEBEE","#C62828")
arrow(2.2,2.2,2.55,2.2); arrow(4.6,2.3,4.95,3.1); arrow(4.6,2.0,4.95,1.0)
arrow(7.0,2.2,7.35,2.2); arrow(9.3,2.2,9.55,2.2)
ax.text(5.5,4.0,"Q-TAIL: Porter-Thomas distributional prior → long-tail embodied learning\n(simulation + real Open X Embodiment evidence, productized as a data service)",
       ha="center",fontsize=10,style="italic",color="#263238")
plt.tight_layout(); plt.savefig("paper_v25/figs/fig1_system_overview.png",bbox_inches="tight"); plt.close()

# ---------- FIG 10: Data engine source vs Q-Tail ----------
metrics=["tail_success","cvar20","tail_data_share","tail_coverage@50","overall_success"]
src=[0.4783,0.4538,0.0541,0.3235,0.6548]
qta=[0.5324,0.5094,0.3985,0.7647,0.6725]
labels=["Tail\nsuccess","CVaR@20","Tail data\nshare","Tail cov.\n@50","Overall\nsuccess"]
x=np.arange(len(metrics)); w=0.38
fig,ax=plt.subplots(figsize=(8.2,4.4))
b1=ax.bar(x-w/2,src,w,label="Source (uniform-style)",color="#90A4AE")
b2=ax.bar(x+w/2,qta,w,label="Q-Tail PT synthetic",color="#1565C0")
for i,(s,q) in enumerate(zip(src,qta)):
    ax.text(i-w/2,s+0.012,f"{s*100:.1f}%",ha="center",fontsize=8)
    ax.text(i+w/2,q+0.012,f"{q*100:.1f}%",ha="center",fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=9); ax.set_ylim(0,0.95)
ax.set_ylabel("Metric (same-budget protocol)"); ax.legend(loc="upper left",fontsize=9)
ax.set_title("Open X Data Engine: source vs Q-Tail synthetic (n=114 profiles, 34 tail tasks)")
ax.grid(axis="y",alpha=0.3)
plt.tight_layout(); plt.savefig("paper_v25/figs/fig10_data_engine.png",bbox_inches="tight"); plt.close()

# ---------- FIG 11: Open X allocation-head training ----------
fig,ax=plt.subplots(1,2,figsize=(10,4.2))
# left: tail share bar
ax[0].bar(["Source","Q-Tail"],[8.25,50.13],color=["#90A4AE","#1565C0"],width=0.55)
ax[0].text(0,9.0,"8.25%",ha="center",fontsize=10,fontweight="bold")
ax[0].text(1,52.0,"50.13%\n(6.07×)",ha="center",fontsize=10,fontweight="bold")
ax[0].set_ylabel("Tail-share of budget (%)"); ax[0].set_ylim(0,60)
ax[0].set_title("Real Open X allocation head:\ntail-share reallocation (+41.9 pp)")
ax[0].grid(axis="y",alpha=0.3)
# right: KL convergence
steps=[0,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000]
src_kl=[0.34538,1.48e-5,4.03e-6,1.70e-6,8.92e-7,4.88e-7,1.61e-6,1.14e-7,7.28e-8,4.14e-7,2.64e-7,1.43e-7,2.45e-6,1.33e-7,4.53e-8,1.34e-6,9.58e-7,2.25e-7,1.19e-7,7.10e-8,5.44e-9]
qta_kl=[0.1713,5.87e-4,3.08e-4,1.41e-4,9.02e-5,6.37e-5,1.03e-4,4.94e-5,4.61e-5,4.36e-5,4.13e-5,3.90e-5,3.71e-5,3.51e-5,3.30e-5,3.13e-5,2.95e-5,3.60e-5,2.59e-5,2.44e-5,2.42e-5]
ax[1].plot(steps,src_kl,color="#EF6C00",lw=1.6,label="Source head (KL→5.4e-9)")
ax[1].plot(steps,qta_kl,color="#1565C0",lw=1.6,label="Q-Tail head (KL→2.4e-5)")
ax[1].set_yscale("log"); ax[1].set_xlabel("Training step"); ax[1].set_ylabel("KL to target tail law (log)")
ax[1].set_title("Allocation-head convergence\n(562 shards, 20k steps)")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("paper_v25/figs/fig11_openx_training.png",bbox_inches="tight"); plt.close()
print("figures generated")
