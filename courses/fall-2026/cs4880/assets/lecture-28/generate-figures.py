"""Reproduce Lecture 28 diagrams and arithmetic checks.

Requires numpy, matplotlib. All plots are illustrative, not empirical model
results. The two genuine fitted-tree illustrations are reused from Lecture 27.
Run from any directory; figures are written beside this script.
"""
from pathlib import Path
import json, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
OUT=Path(__file__).resolve().parent
GREEN,GOLD,INK,BLUE,GRAY='#154734','#BD8B13','#1F2933','#427C9E','#75828A'
MINT,PALE,LIGHT='#E7F0EA','#FFF3D9','#EDF2F5'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.labelsize':14,
 'axes.titlesize':15,'axes.spines.top':False,'axes.spines.right':False,
 'legend.frameon':False,'figure.constrained_layout.use':True,'savefig.facecolor':'white'})
def save(fig,name):
 fig.savefig(OUT/name,dpi=190,bbox_inches='tight',pad_inches=.12);plt.close(fig)
def diagram(w=10,h=3):
 fig,ax=plt.subplots(figsize=(w,h));ax.set(xlim=(0,w),ylim=(0,h));ax.axis('off');return fig,ax
def box(ax,x,y,w,h,text,color=MINT,size=14):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.03',fc=color,ec=GREEN,lw=1.4))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,color=INK)
def arrow(ax,a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,color=GREEN,lw=1.6))
def illustrative(fig):fig.suptitle('Illustrative — not measured results',fontsize=11,color=GRAY)

# Distinguish limited flexibility, memorization, and changing input distributions.
x=np.linspace(0,1,200);rng=np.random.default_rng(28)
sx=np.linspace(.05,.95,12);sy=np.sin(2*np.pi*sx)+rng.normal(0,.12,len(sx))
fig,axs=plt.subplots(1,3,figsize=(11,2.8))
for ax in axs[:2]:
 ax.scatter(sx,sy,s=24,color=GOLD,zorder=3);ax.set(xlabel='Input',ylabel='Output');ax.grid(alpha=.15)
axs[0].plot(x,np.polyval(np.polyfit(sx,sy,1),x),color=GREEN,lw=2);axs[0].set_title('Underfitting')
axs[1].plot(x,np.sin(2*np.pi*x)+.3*np.sin(25*np.pi*x),color=GREEN,lw=2);axs[1].set_title('Overfitting (schematic)')
axs[2].hist(rng.normal(0,1,1000),bins=20,color=BLUE,alpha=.65,density=True,label='Training')
axs[2].hist(rng.normal(2,1,1000),bins=20,color=GOLD,alpha=.6,density=True,label='Deployment')
axs[2].set(xlabel='Input feature',ylabel='Density',title='Distribution shift');axs[2].legend(fontsize=10)
illustrative(fig);save(fig,'three_failures.png')

fig,ax=diagram(11,2.8)
ax.text(.1,2.4,'One recording run: adjacent frames share the same scene',fontsize=15,color=GREEN)
for i in range(8):box(ax,.3+i*1.3,1.45,1.05,.55,f'Frame {i+1}',LIGHT,12)
box(ax,.3,.15,4.8,.65,'Row split: same run on both sides',PALE,13)
box(ax,5.8,.15,4.8,.65,'Group split: entire run stays together',MINT,13)
arrow(ax,(2.8,1.4),(2.8,.85));arrow(ax,(8,1.4),(8,.85));save(fig,'leakage_groups.png')

fig,ax=plt.subplots(figsize=(5.6,2.7))
for row in range(3):
 for col in range(3):
  ax.add_patch(Rectangle((col,row),.95,.8,fc=GOLD if row==col else MINT,ec='white'))
  ax.text(col+.475,row+.4,'V' if row==col else 'T',ha='center',va='center',fontsize=18)
 ax.add_patch(Rectangle((3.35,row),1.25,.8,fc=LIGHT,ec='white'));ax.text(3.975,row+.4,'—',ha='center',va='center',fontsize=18)
ax.set(xlim=(-.05,4.7),ylim=(-.25,3),xticks=[.475+i for i in range(3)]+[3.975],xticklabels=['Run A','Run B','Run C','Test'],yticks=[i+.4 for i in range(3)],yticklabels=[f'Fold {i+1}' for i in range(3)])
ax.invert_yaxis();ax.tick_params(length=0)
for spine in ax.spines.values():spine.set_visible(False)
save(fig,'group_folds.png')

scores=np.array([.95,.9,.85,.8,.7,.6,.5,.4,.3,.2]);truth=np.array([1,1,0,1,0,1,1,0,1,0])
thresholds=np.array([.2,.4,.6,.8,.9]);precision=[];recall=[]
for threshold in thresholds:
 pred=scores>=threshold;tp=int((pred&(truth==1)).sum());fp=int((pred&(truth==0)).sum())
 precision.append(tp/(tp+fp));recall.append(tp/int(truth.sum()))
assert all(a>=b for a,b in zip(recall,recall[1:]))
fig,ax=plt.subplots(figsize=(5,3.4));ax.plot(thresholds,precision,'o-',color=BLUE,label='Precision');ax.plot(thresholds,recall,'s-',color=GREEN,label='Recall')
ax.set(xlabel='Decision threshold',ylabel='Fraction',ylim=(0,1.08));ax.grid(alpha=.15);ax.legend(fontsize=11);illustrative(fig);save(fig,'threshold_tradeoff.png')

n=np.array([20,50,100,200,400,800]);train=np.array([.02,.05,.075,.09,.105,.115]);valid=np.array([.38,.30,.235,.20,.177,.16])
fig,ax=plt.subplots(figsize=(5.4,4.1));ax.plot(n,train,'o-',color=GREEN,label='Training error');ax.plot(n,valid,'s-',color=GOLD,label='Validation error')
ax.set(xlabel='Training examples',ylabel='Error rate',ylim=(0,.43));ax.grid(alpha=.15);ax.legend();illustrative(fig);save(fig,'learning_curve.png')

names=['Shallow\ntree','Unrestricted\ntree','Random\nforest'];tr=[.80,1.,.98];val=[.78,.72,.88]
fig,ax=plt.subplots(figsize=(5.5,4));idx=np.arange(3);ax.bar(idx-.18,tr,.36,color=MINT,edgecolor=GREEN,label='Training');ax.bar(idx+.18,val,.36,color=GOLD,label='Validation')
for i in range(3):ax.text(i-.18,tr[i]+.015,f'{tr[i]:.0%}',ha='center',fontsize=11);ax.text(i+.18,val[i]+.015,f'{val[i]:.0%}',ha='center',fontsize=11)
ax.set(xticks=idx,xticklabels=names,ylabel='Accuracy',ylim=(0,1.15));ax.legend(loc='lower left',fontsize=11);illustrative(fig);save(fig,'model_validation.png')

fig,ax=diagram(11,3)
box(ax,.1,1,2.0,.8,'Training rows\nA  B  C  D',size=14)
for y,rows in [(2.2,'A A C D'),(1.2,'B B C A'),(.2,'D C C B')]:
 box(ax,3.1,y,2.2,.6,rows,LIGHT,14);box(ax,6.1,y,1.4,.6,'Tree',size=14)
 arrow(ax,(2.15,1.4),(3.05,y+.3));arrow(ax,(5.35,y+.3),(6.05,y+.3));arrow(ax,(7.55,y+.3),(8.5,1.4))
box(ax,8.55,1,2.15,.8,'Combine\npredictions',PALE,14)
ax.text(4.2,2.95,'Sample with replacement',ha='center',fontsize=13,color=GREEN);save(fig,'bagging_pipeline.png')

fig,ax=diagram(11,2.7)
labels=['Blocked','Clear','Blocked','Blocked','Clear']
for i,label in enumerate(labels):
 box(ax,.15+i*2.2,1.6,1.9,.7,f'Tree {i+1}\n{label}',PALE if label=='Blocked' else LIGHT,14)
 arrow(ax,(1.1+i*2.2,1.55),(5.5,.95))
box(ax,3.7,.1,3.6,.75,'Blocked wins: 3 to 2',size=16);save(fig,'vote_five.png')

fig,axs=plt.subplots(1,2,figsize=(10,2.3))
correlated=np.array([[0,1,0,1,0,0,1,0]]*5)
diverse=np.array([[1,1,1,0,0,0,0,0],[0,0,1,1,1,0,0,0],[0,0,0,0,1,1,1,0],[1,0,0,0,0,0,1,1],[0,1,0,1,0,1,0,0]])
assert np.all(correlated.sum(axis=1)==diverse.sum(axis=1))
assert diverse.sum(axis=0).max()<3
from matplotlib.colors import ListedColormap
for ax,data,title in zip(axs,[correlated,diverse],['Same mistakes','Different mistakes']):
 ax.imshow(data,cmap=ListedColormap([MINT,GOLD]),vmin=0,vmax=1,aspect='auto');ax.set(title=title,xlabel='Held-out case',ylabel='Tree',yticks=range(5),yticklabels=range(1,6),xticks=range(8),xticklabels=range(1,9));ax.tick_params(length=0)
fig.suptitle('Gold = error; mint = correct (illustrative)',fontsize=12,color=GRAY);save(fig,'correlated_errors.png')

fig,ax=diagram(11,3.3)
box(ax,.1,1.1,2,.8,'Training data',size=14)
for y,label in [(2.15,'Bootstrap A'),(.2,'Bootstrap B')]:
 box(ax,3,y,2.0,.7,label,LIGHT,14);arrow(ax,(2.15,1.5),(2.95,y+.35))
 box(ax,6,y,4.6,.7,'Random feature candidates\nat each split',MINT,14);arrow(ax,(5.05,y+.35),(5.95,y+.35))
ax.text(8.3,1.65,'New subset at every node',ha='center',fontsize=13,color=GREEN);save(fig,'forest_two_randomness.png')

fig,axs=plt.subplots(1,3,figsize=(11,2.7))
for ax in axs:ax.set(xlim=(0,3.4),ylim=(0,2.6));ax.axis('off')
for j in range(3):
 box(axs[0],.1+j*1.1,1.7,.95,.5,f'M{j+1}',size=12);arrow(axs[0],(.575+j*1.1,1.65),(1.7,.85))
box(axs[0],.65,.2,2.1,.6,'Average / vote',PALE,12);axs[0].set_title('Bagging: parallel fits')
for j in range(3):
 box(axs[1],.05+j*1.12,1.2,1,.6,f'M{j+1}',size=12)
 if j<2:arrow(axs[1],(1.05+j*1.12,1.5),(1.15+j*1.12,1.5))
axs[1].text(1.7,.5,'Improve current loss',ha='center',fontsize=12);axs[1].set_title('Boosting: sequential fits')
for j in range(3):
 box(axs[2],.05+j*1.12,1.7,1,.5,f'M{j+1}',size=12);arrow(axs[2],(.55+j*1.12,1.65),(1.7,.85))
box(axs[2],.5,.2,2.4,.6,'Learn a meta-model',PALE,12);axs[2].set_title('Stacking: learn combination')
save(fig,'ensemble_families.png')

fig,ax=diagram(10,2.5)
for x0,text,col in [(.15,'Train\nfit models',MINT),(3.55,'Validate\nchoose settings',PALE),(6.95,'Test\nreport frozen model',LIGHT)]:box(ax,x0,.7,2.8,1,text,col,16)
arrow(ax,(3,.7+ .5),(3.5,1.2));arrow(ax,(6.4,1.2),(6.9,1.2))
ax.text(5,.15,'Group-aware partitions; test observations never guide selection',ha='center',fontsize=13,color=GREEN);save(fig,'controlled_protocol.png')

b=np.array([1,5,10,25,50,100,200]);performance=np.array([.73,.80,.84,.87,.885,.89,.892]);time=b/10
fig,axs=plt.subplots(2,1,figsize=(5.5,4.5),sharex=True)
axs[0].plot(b,performance,'o-',color=GREEN);axs[0].set(ylabel='Validation accuracy',ylim=(.7,.93))
axs[1].plot(b,time,'o-',color=GOLD);axs[1].set(xlabel='Number of trees',ylabel='Relative fit cost')
for ax in axs:ax.grid(alpha=.15)
illustrative(fig);save(fig,'forest_runtime.png')

# Reuse genuine fitted-tree plots; do not fabricate numerical split thresholds.
for name in ['wine_tree.png','california_tree.png']:
 shutil.copyfile(OUT.parent/'lecture-27'/name,OUT/name)
TP,FP,FN,TN=9,3,6,22
metrics={'accuracy':(TP+TN)/(TP+FP+FN+TN),'precision':TP/(TP+FP),'recall':TP/(TP+FN),'f1':2*TP/(2*TP+FP+FN)}
assert np.isclose(metrics['accuracy'],.775) and np.isclose(metrics['f1'],2/3)
assert labels.count('Blocked')==3 and labels.count('Clear')==2
for B in [1,5,100]:
 assert np.isclose(1+(1-1)/B,1)
 assert np.isclose(0+(1-0)/B,1/B)
checks={'illustrative_not_empirical':['three_failures.png','threshold_tradeoff.png','learning_curve.png','model_validation.png','forest_runtime.png'], 'confusion_counts':{'TP':TP,'FP':FP,'FN':FN,'TN':TN},'metrics':metrics,'thresholds':thresholds.tolist(),'precision':precision,'recall':recall,'validation_scores':val,'tree_figures_reused_from':'../lecture-27','expected_distinct_bootstrap_fraction_N1000':float(1-(1-1/1000)**1000)}
(OUT/'figure-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print('Generated 13 reproducible figures, reused 2 tree figures, and checked slide arithmetic.')
