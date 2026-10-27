"""Reproduce CS4880 Lecture 27 figures and numerical checks.

Requires numpy, matplotlib, scikit-learn. Use --data-home for an existing
California housing cache; the dataset is downloaded if not cached. The two
instructor Python models retain their original data and max_depth=4 settings.
Slide figures show their top two levels for legibility; full trees are saved too.
"""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle
from sklearn import __version__ as sklearn_version
from sklearn.datasets import fetch_california_housing,load_wine,make_moons
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor,plot_tree
from sklearn.model_selection import train_test_split

parser=argparse.ArgumentParser();parser.add_argument('--data-home',default=None)
args=parser.parse_args();OUT=Path(__file__).resolve().parent
GREEN,GOLD,INK,BLUE,GRAY='#154734','#BD8B13','#1F2933','#427C9E','#75828A'
MINT,PALE='#E7F0EA','#FFF3D9'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.labelsize':15,
    'axes.titlesize':16,'axes.spines.top':False,'axes.spines.right':False,
    'legend.frameon':False,'figure.constrained_layout.use':True,'savefig.facecolor':'white'})
def save(fig,name):
    fig.savefig(OUT/name,dpi=210,bbox_inches='tight',pad_inches=.1);plt.close(fig)
def diagram(w=10,h=3.5):
    fig,ax=plt.subplots(figsize=(w,h));ax.set(xlim=(0,w),ylim=(0,h));ax.axis('off');return fig,ax

def box(ax,x,y,w,h,text,color=MINT,size=16):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.035',fc=color,ec=GREEN,lw=1.5))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,color=INK)
def arrow(ax,a,b,label=None,offset=(0,0)):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=16,color=GREEN,lw=1.7))
    if label:ax.text((a[0]+b[0])/2+offset[0],(a[1]+b[1])/2+offset[1],label,ha='center',va='center',fontsize=13,color=INK,bbox={'facecolor':'white','edgecolor':'none','pad':1})

def entropy(p):
    return 0. if p in [0,1] else float(-p*np.log2(p)-(1-p)*np.log2(1-p))
# A distinct lecture example, not the corridor lab's eight rows.
rows=[{'id':i+1,'load':'High','warm':'Yes' if i<3 else 'No','failed':1} for i in range(5)]
rows += [ {'id':6,'load':'Low','warm':'Yes','failed':1},
          {'id':7,'load':'Low','warm':'Yes','failed':0},
          {'id':8,'load':'Low','warm':'No','failed':0},
          {'id':9,'load':'Low','warm':'No','failed':0},
          {'id':10,'load':'Low','warm':'No','failed':0} ]
parent=entropy(sum(r['failed'] for r in rows)/len(rows));splits={}
for key in ['load','warm']:
    groups={v:[r for r in rows if r[key]==v] for v in sorted({r[key] for r in rows})}
    after=sum(len(g)/len(rows)*entropy(sum(r['failed'] for r in g)/len(g)) for g in groups.values())
    splits[key]={'counts':{k:[sum(r['failed'] for r in g),len(g)-sum(r['failed'] for r in g)] for k,g in groups.items()},'after':after,'gain':parent-after}
assert splits['load']['gain']>splits['warm']['gain']
assert rows[5]['load']==rows[6]['load'] and rows[5]['warm']==rows[6]['warm'] and rows[5]['failed']!=rows[6]['failed']

fig,axs=plt.subplots(2,1,figsize=(5,5.2),sharex=True)
x=np.linspace(0,10,200);truth=np.where(x<=5,1.,4.)
for ax,title in zip(axs,['One smooth fit','Two learned regions']):
    ax.scatter([1,2,3,4,6,7,8,9],[1,1.1,.9,1,4,3.9,4.1,4],color=GOLD,s=35)
    if title.startswith('One'):ax.plot(x,1+3/(1+np.exp(-(x-5))),color=GRAY,lw=2.5)
    else:ax.step(x,truth,where='mid',color=GREEN,lw=2.5)
    ax.set_title(title);ax.set_yticks([1,4]);ax.grid(alpha=.15)
axs[1].set_xlabel('Input');axs[0].set_ylabel('Prediction');axs[1].set_ylabel('Prediction')
save(fig,'smooth_vs_rule.png')

fig=plt.figure(figsize=(5.4,6));gs=fig.add_gridspec(2,1,height_ratios=[1.1,1])
ax=fig.add_subplot(gs[0]);ax.set(xlim=(0,5.4),ylim=(0,3));ax.axis('off')
box(ax,1.5,2.25,2.4,.5,'Size <= 6?',size=15)
box(ax,.2,1.1,2.2,.5,'Temp <= 70?',size=14);box(ax,3.5,1.1,1.65,.5,'Fail',PALE,size=15)
box(ax,.05,.05,1.8,.45,'Complete',size=13);box(ax,1.95,.05,1.6,.45,'Fail',PALE,size=13)
arrow(ax,(2.1,2.2),(1.3,1.65),'Yes',(-.3,.1));arrow(ax,(3.3,2.2),(4.3,1.65),'No',(.25,.1))
arrow(ax,(.9,1.05),(.9,.55),'Yes',(-.3,0));arrow(ax,(1.7,1.05),(2.7,.55),'No',(.2,.1))
ax=fig.add_subplot(gs[1]);ax.add_patch(Rectangle((0,30),6,40,fc=MINT,ec='none'))
ax.add_patch(Rectangle((0,70),6,30,fc=PALE,ec='none'));ax.add_patch(Rectangle((6,30),6,70,fc=PALE,ec='none'))
ax.axvline(6,color=GREEN,lw=2);ax.plot([0,6],[70,70],color=GREEN,lw=2)
ax.text(3,47,'Complete',ha='center',fontsize=14,color=GREEN)
ax.text(3,85,'Fail',ha='center',fontsize=14);ax.text(9,65,'Fail',ha='center',fontsize=14)
ax.set(xlim=(0,12),ylim=(30,100),xlabel='Job size',ylabel='Temperature');ax.set_xticks([0,6,12]);ax.set_yticks([30,70,100])
save(fig,'tree_regions.png')

p=np.linspace(.001,.999,400);fig,ax=plt.subplots(figsize=(5,4))
ax.plot(p,[-v*np.log2(v)-(1-v)*np.log2(1-v) for v in p],color=GREEN,lw=3)
ax.scatter([0,.25,.5,.75,1],[0,entropy(.25),1,entropy(.75),0],color=GOLD,s=50,zorder=3)
ax.set(xlim=(0,1),ylim=(0,1.08),xlabel='Fraction of failed jobs',ylabel='Entropy (bits)');ax.set_xticks([0,.25,.5,.75,1]);ax.grid(alpha=.15)
save(fig,'entropy_curve.png')

fig,ax=diagram(11,3.2)
for offset,key,title in [(0,'load','Ask Load?'),(5.6,'warm','Ask Warm?')]:
    box(ax,offset+1.3,2.2,2.7,.55,title,size=18)
    values=['High','Low'] if key=='load' else ['Yes','No']
    for j,value in enumerate(values):
        counts=splits[key]['counts'][value];x0=offset+.1+j*2.7
        box(ax,x0,.65,2.5,.85,f'{value}\n{counts[0]} fail / {counts[1]} complete',size=14,color=MINT if j==0 else PALE)
        arrow(ax,(offset+2.65,2.15),(x0+1.25,1.55))
    ax.text(offset+2.65,.15,f'Information gain: {splits[key]["gain"]:.3f} bits',ha='center',color=GREEN,weight='bold',fontsize=15)
save(fig,'split_comparison.png')

fig,ax=diagram(11,3.7)
box(ax,4.3,2.95,2.4,.55,'Load = High?',size=17)
box(ax,1.05,1.55,2.8,.7,'Fail\n5 fail / 0 complete',PALE,size=14)
box(ax,7.1,1.7,2.6,.55,'Warm = Yes?',size=16)
box(ax,5.45,.1,2.9,.85,'Complete (tie-break)\n1 fail / 1 complete',size=14)
box(ax,8.65,.1,2.15,.85,'Complete\n0 fail / 3 complete',size=14)
arrow(ax,(4.6,2.9),(2.45,2.3),'Yes',(-.25,.15));arrow(ax,(6.4,2.9),(8.4,2.3),'No',(.2,.15))
arrow(ax,(7.7,1.65),(6.9,1.0),'Yes',(-.2,.1));arrow(ax,(9.1,1.65),(9.7,1.0),'No',(.25,.1))
save(fig,'completed_tree.png')

fig,ax=plt.subplots(figsize=(9,2.25))
values=[2,3,4,9,11,12];midpoints=[2.5,3.5,6.5,10,11.5]
ax.scatter(values,[1]*6,color=GREEN,s=80,zorder=3)
for v in values:ax.text(v,1.12,str(v),ha='center',fontsize=16)
for v in midpoints:ax.axvline(v,ymin=.1,ymax=.63,color=GOLD,ls='--',lw=1.6)
ax.set(xlim=(1,13),ylim=(.6,1.5),xlabel='Job size (ten-page batches)',yticks=[])
ax.set_xticks(midpoints);ax.spines[['left','top','right']].set_visible(False)
save(fig,'thresholds.png')

housing=fetch_california_housing(data_home=args.data_home)
reg=DecisionTreeRegressor(max_depth=4,random_state=42).fit(housing.data,housing.target)
wine=load_wine();clf=DecisionTreeClassifier(max_depth=4,random_state=42).fit(wine.data,wine.target)

def tree_summary(model,features,filename,classes=None):
    tree=model.tree_;fig,ax=diagram(10,3.9)
    def label(node):
        name=features[tree.feature[node]]
        criterion=f'{name} <= {tree.threshold[node]:.3f}' if tree.children_left[node]>=0 else 'Leaf'
        n=int(tree.n_node_samples[node])
        if classes is None:
            extra=f'squared_error = {tree.impurity[node]:.3f}\nsamples = {n}\nvalue = {tree.value[node,0,0]:.3f}'
        else:
            vals=tree.value[node,0];counts=np.rint(vals/vals.sum()*n).astype(int)
            extra=f'gini = {tree.impurity[node]:.3f}\nsamples = {n}\nvalue = {counts.tolist()}\nclass = {classes[np.argmax(vals)]}'
        return criterion+'\n'+extra
    box(ax,3.0,2.35,4,1.4,label(0),size=14)
    for node,x0,answer in [(tree.children_left[0],.1,'True'),(tree.children_right[0],5.2,'False')]:
        box(ax,x0,.15,4.7,1.65,label(node)+'\n... further splits ...',PALE,size=13)
        arrow(ax,(4 if x0<1 else 6,2.3),(x0+2.35,1.85),answer,offset=(-.3 if x0<1 else .3,.1))
    save(fig,filename)

tree_summary(reg,housing.feature_names,'california_tree.png')
tree_summary(clf,wine.feature_names,'wine_tree.png',wine.target_names)
for model,features,filename,classes in [(reg,housing.feature_names,'california_tree_full.png',None),(clf,wine.feature_names,'wine_tree_full.png',wine.target_names)]:
    fig,ax=plt.subplots(figsize=(20,10))
    plot_tree(model,feature_names=list(features),class_names=None if classes is None else list(classes),
              filled=True,rounded=True,fontsize=10,proportion=False,ax=ax)
    save(fig,filename)

# Measured complexity/pruning comparison on a separate synthetic dataset.
X,y=make_moons(n_samples=600,noise=.30,random_state=41)
Xtr,Xrest,ytr,yrest=train_test_split(X,y,test_size=.4,stratify=y,random_state=42)
Xv,Xtest,yv,ytest=train_test_split(Xrest,yrest,test_size=.5,stratify=yrest,random_state=43)
curve=[]
for depth in range(1,13):
    model=DecisionTreeClassifier(max_depth=depth,random_state=42).fit(Xtr,ytr)
    curve.append({'depth':depth,'training_error':1-model.score(Xtr,ytr),'validation_error':1-model.score(Xv,yv)})
fig,ax=plt.subplots(figsize=(5,4))
ax.plot([r['depth'] for r in curve],[r['training_error'] for r in curve],'o-',color=GREEN,label='Training')
ax.plot([r['depth'] for r in curve],[r['validation_error'] for r in curve],'s-',color=GOLD,label='Validation')
ax.set(xlabel='Maximum tree depth',ylabel='Classification error');ax.legend();ax.grid(alpha=.15)
save(fig,'complexity_curve.png')
unrestricted=DecisionTreeClassifier(random_state=42).fit(Xtr,ytr)
pre=DecisionTreeClassifier(max_depth=4,min_samples_leaf=10,random_state=42).fit(Xtr,ytr)
path=unrestricted.cost_complexity_pruning_path(Xtr,ytr)
post_candidates=[DecisionTreeClassifier(ccp_alpha=float(a),random_state=42).fit(Xtr,ytr) for a in path.ccp_alphas]
post=max(post_candidates,key=lambda m:(m.score(Xv,yv),-m.get_n_leaves()))
models={'Unrestricted':unrestricted,'Pre-pruned':pre,'Post-pruned':post}
comparison={name:{'leaves':m.get_n_leaves(),'depth':m.get_depth(),
                   'training_accuracy':m.score(Xtr,ytr),'validation_accuracy':m.score(Xv,yv)} for name,m in models.items()}
winner=max(models,key=lambda k:(models[k].score(Xv,yv),-models[k].get_n_leaves()))
result={'versions':{'sklearn':sklearn_version,'numpy':np.__version__},
    'lecture_example':{'rows':rows,'root_entropy':parent,'splits':splits,'training_accuracy_with_stated_tie_break':.9},
    'numeric_example':{'values':values,'candidate_midpoints':midpoints},
    'housing':{'shape':list(housing.data.shape),'root_feature':housing.feature_names[reg.tree_.feature[0]],
               'root_threshold':float(reg.tree_.threshold[0]),'depth':reg.get_depth(),'leaves':reg.get_n_leaves()},
    'wine':{'shape':list(wine.data.shape),'depth':clf.get_depth(),'leaves':clf.get_n_leaves()},
    'validation_experiment':{'dataset':'make_moons(n_samples=600, noise=.30, random_state=41)',
        'counts':{'training':len(ytr),'validation':len(yv),'test':len(ytest)},'curve':curve,
        'comparison':comparison,'pre_pruning':{'max_depth':4,'min_samples_leaf':10},
        'post_pruning_alpha':float(post.ccp_alpha),'selected_model':winner,'frozen_model_test_accuracy':models[winner].score(Xtest,ytest)}}
(OUT/'chart-checks.json').write_text(json.dumps(result,indent=2,default=lambda v:v.item())+'\n')
print(json.dumps({'root_entropy':parent,'splits':splits,'comparison':comparison,'winner':winner},indent=2,default=lambda v:v.item()))
