"""Reproduce the original scikit-learn figures and measured outputs in CS 4880 Lecture 31.
Run: python generate-figures.py (scikit-learn, numpy, scipy, matplotlib required).
No external data downloads, API calls, or model training outside this script.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, load_iris, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score, accuracy_score
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC, OneClassSVM
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import mean_squared_error

ROOT=os.path.dirname(os.path.abspath(__file__))
os.makedirs(ROOT,exist_ok=True)
GREEN='#154734'; GOLD='#B58A23'; DARK='#243238'; SILVER='#B4B9B4'; BLUE='#337A96'; RED='#C05D55'; VIOLET='#856AA6'; LIGHT='#F5F7F4'
COLORS=[GREEN,GOLD,BLUE,RED,VIOLET,'#5AA99B','#BB77A1']
plt.rcParams.update({'font.size':10.4,'axes.titlesize':12,'axes.labelsize':10,'axes.edgecolor':'#708079','axes.labelcolor':DARK,'text.color':DARK,'xtick.color':DARK,'ytick.color':DARK,'savefig.facecolor':'white','figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
results={}
def save(name,fig):
    fig.savefig(os.path.join(ROOT,name+'.png'),bbox_inches='tight',dpi=195,pad_inches=.13)
    plt.close(fig)
def scatter_labels(ax,X,labels,centers=None, title=None, noise=False, alpha=.77):
    u=np.unique(labels)
    for label in u:
        mask=labels==label
        col='#969C9E' if label==-1 else COLORS[int(label)%len(COLORS)]
        ax.scatter(X[mask,0], X[mask,1], s=14 if label!=-1 else 10, color=col,alpha=.78 if label!=-1 else .42,linewidths=0,rasterized=True)
    if centers is not None:
        ax.scatter(centers[:,0],centers[:,1],marker='X',s=165,color='white',edgecolors=DARK,linewidths=1.6,zorder=6)
    if title: ax.set_title(title,fontweight='bold')
    ax.set_xlabel('Feature 1');ax.set_ylabel('Feature 2');ax.grid(alpha=.12)

# Intro / unlabeled versus post-hoc truth: labels only for demonstration, not during fit.
X_blob,y_blob=make_blobs(n_samples=450, centers=[[-5,-1],[0,3],[5,-.7]],cluster_std=[.82,1.17,.9],random_state=4880)
fig,axs=plt.subplots(1,2,figsize=(10,3.1),sharex=True,sharey=True)
axs[0].scatter(X_blob[:,0],X_blob[:,1],s=12,c='#899799',alpha=.8,rasterized=True);axs[0].set_title('Inputs available: X only',fontweight='bold')
scatter_labels(axs[1],X_blob,y_blob,title='Hidden labels shown for teaching')
for ax in axs: ax.set_xlabel('Feature 1'); ax.set_ylabel('Feature 2'); ax.grid(alpha=.12)
fig.tight_layout();save('unlabeled_vs_labels',fig)

# Scaling illustration: identical observations, raw vs scaled
rng=np.random.default_rng(4880)
feat_a=rng.normal(0,1.5,140); feat_b=70*feat_a+rng.normal(0,110,140)
X_scale=np.c_[feat_a,feat_b];Z_scale=StandardScaler().fit_transform(X_scale)
fig,ax=plt.subplots(1,2,figsize=(9.5,3))
ax[0].scatter(X_scale[:,0],X_scale[:,1],s=15,color=GREEN,alpha=.6);ax[0].set_title('Unscaled measurement units')
ax[1].scatter(Z_scale[:,0],Z_scale[:,1],s=15,color=BLUE,alpha=.6);ax[1].set_title('Standardized: mean 0, std 1')
for a in ax: a.set_xlabel('Feature 1');a.set_ylabel('Feature 2');a.grid(alpha=.14)
fig.tight_layout();save('why_scaling_matters',fig)

# KMeans deterministic demo and objective
km=KMeans(n_clusters=3,random_state=4880,n_init=10)
labels=km.fit_predict(X_blob)
res={'n_samples':len(X_blob),'inertia':round(float(km.inertia_),1),'silhouette':round(float(silhouette_score(X_blob,labels)),3),'ari_posthoc':round(float(adjusted_rand_score(y_blob,labels)),3),'centers':np.round(km.cluster_centers_,2).tolist(),'cluster_sizes':np.bincount(labels).tolist()}
results['kmeans_blobs']=res
fig,axs=plt.subplots(1,2,figsize=(10,3.25),sharex=True,sharey=True)
axs[0].scatter(X_blob[:,0],X_blob[:,1],s=13,color='#8E9999',alpha=.8);axs[0].set_title('Training data: 450 unlabeled points',fontweight='bold')
scatter_labels(axs[1],X_blob,labels,km.cluster_centers_,f'K-means: k=3; silhouette={res["silhouette"]:.3f}')
for a in axs:a.grid(alpha=.1);a.set_xlabel('Feature 1');a.set_ylabel('Feature 2')
fig.tight_layout();save('kmeans_on_blobs',fig)

# Iterative graphic simulate each assignment/update from fixed seed centers
init_centers=np.array([[-6.0,3.7],[.0,-3.4],[5.5,4.3]])
steps=[];C=init_centers.copy()
for i in range(100):
   assign=np.argmin(((X_blob[:,None,:]-C[None,:,:])**2).sum(axis=2),axis=1)
   steps.append((C.copy(),assign.copy()))
   updated=np.array([X_blob[assign==j].mean(axis=0) if np.any(assign==j) else C[j] for j in range(3)])
   if np.allclose(updated,C):
       break
   C=updated
else:
   raise RuntimeError('Illustrated K-means did not converge')
assert np.allclose(C,updated)
results['kmeans_iteration']={'assignment_steps':len(steps),'converged':True}
fig,axs=plt.subplots(1,3,figsize=(10.8,3.25),sharex=True,sharey=True)
for ax,index,title in zip(axs,[0,1,len(steps)-1],['1. Start with candidate centers','2. Assign and update','3. Repeat until stable']):
  C,A=steps[index];scatter_labels(ax,X_blob,A,C,title=title)
fig.tight_layout();save('kmeans_iteration',fig)

# Model selection diagnostics; scores are measured on same X
ks=np.arange(2,8)
inertias=[];sils=[]
for k in ks:
 m=KMeans(n_clusters=int(k),random_state=4880,n_init=10).fit(X_blob)
 inertias.append(float(m.inertia_));sils.append(float(silhouette_score(X_blob,m.labels_)))
results['k_selection']={'ks':ks.tolist(),'inertia':[round(v,1) for v in inertias],'silhouette':[round(v,3) for v in sils],'best_silhouette_k':int(ks[np.argmax(sils)])}
fig,axs=plt.subplots(1,2,figsize=(9.5,3.15))
axs[0].plot(ks,inertias,marker='o',color=GREEN,lw=2);axs[0].set_title('Inertia falls as k increases')
axs[0].set_xlabel('Number of clusters k');axs[0].set_ylabel('Within-cluster squared error')
axs[1].plot(ks,sils,marker='o',color=BLUE,lw=2);axs[1].set_title('Silhouette is not monotone')
axs[1].set_xlabel('Number of clusters k');axs[1].set_ylabel('Mean silhouette score')
for a in axs:a.set_xticks(ks);a.grid(alpha=.14)
fig.tight_layout();save('k_selection',fig)

# Nonconvex data and DBSCAN
X_moons,y_moons=make_moons(n_samples=440,noise=.07,random_state=4880)
Xm=StandardScaler().fit_transform(X_moons)
kmoon=KMeans(n_clusters=2,random_state=4880,n_init=10).fit(Xm)
db=DBSCAN(eps=.22,min_samples=6)
dblabel=db.fit_predict(Xm)
num_clusters=len(set(dblabel))-(1 if -1 in dblabel else 0)
results['moons']={'kmeans_ari_posthoc':round(float(adjusted_rand_score(y_moons,kmoon.labels_)),3),'dbscan_ari_posthoc':round(float(adjusted_rand_score(y_moons,dblabel)),3),'dbscan_clusters':num_clusters,'dbscan_noise':int(np.sum(dblabel==-1)),'kmeans_silhouette':round(float(silhouette_score(Xm,kmoon.labels_)),3),'dbscan_silhouette':round(float(silhouette_score(Xm,dblabel)),3) if len(np.unique(dblabel))>1 else None}
fig,axs=plt.subplots(1,3,figsize=(11,3.1),sharex=True,sharey=True)
axs[0].scatter(Xm[:,0],Xm[:,1],s=10,color='#929C9A',alpha=.75);axs[0].set_title('Input: curved geometry')
scatter_labels(axs[1],Xm,kmoon.labels_,title='K-means: roundish partitions')
scatter_labels(axs[2],Xm,dblabel,title='DBSCAN: density connections')
for a in axs:a.set_xlabel('Standardized feature 1');a.set_ylabel('Feature 2');a.grid(alpha=.1)
fig.tight_layout();save('moons_algorithms',fig)

# DBSCAN eps sensitivity, based on measured outcome
fig,axs=plt.subplots(1,3,figsize=(10.9,3.25),sharex=True,sharey=True)
res_eps={}
for ax,eps in zip(axs,[.12,.22,.38]):
 l=DBSCAN(eps=eps,min_samples=6).fit_predict(Xm)
 nc=len(set(l))-(1 if -1 in l else 0);nn=int((l==-1).sum())
 scatter_labels(ax,Xm,l,title=f'eps={eps:.2f}: {nc} groups, {nn} noise')
 res_eps[str(eps)]={'clusters':nc,'noise':nn}
for a in axs:a.grid(alpha=.12)
fig.tight_layout();save('dbscan_eps_sensitivity',fig)
results['dbscan_eps']=res_eps

# Hierarchical demonstration on subset small
rng=np.random.default_rng(4880)
ids=np.sort(rng.choice(np.arange(len(X_blob)),size=22,replace=False))
X_small=X_blob[ids]
Z=linkage(X_small,method='ward')
fig,ax=plt.subplots(figsize=(9.7,3.2))
dendrogram(Z,ax=ax,color_threshold=0,above_threshold_color=GREEN,leaf_rotation=0)
ax.set_xlabel('Observations (index within 22-point sample)');ax.set_ylabel('Ward linkage distance')
ax.set_title('Agglomerative clustering records a hierarchy of merges',fontweight='bold')
fig.tight_layout();save('hierarchical_dendrogram',fig)

# GMM: posterior probability on ambiguous points
X_overlap,_=make_blobs(n_samples=450,centers=[[-1.55,-.75],[1.5,-.65],[0,1.6]],cluster_std=[1.0,1.0,1.05],random_state=4880)
mg=GaussianMixture(n_components=3,random_state=4880,n_init=5).fit(X_overlap)
post=mg.predict_proba(X_overlap);uncertainty=1-post.max(axis=1)
fig,axs=plt.subplots(1,2,figsize=(10,3.2),sharex=True,sharey=True)
scatter_labels(axs[0],X_overlap,mg.predict(X_overlap),title='GMM maximum-posterior assignment')
scatter=axs[1].scatter(X_overlap[:,0],X_overlap[:,1],c=uncertainty,cmap='magma',s=13,vmin=0,vmax=2/3,alpha=.9,rasterized=True)
axs[1].set_title('Ambiguity: 1 - largest membership probability')
fig.colorbar(scatter,ax=axs[1],fraction=.046,pad=.02)
for a in axs:a.set_xlabel('Feature 1');a.set_ylabel('Feature 2');a.grid(alpha=.12)
fig.tight_layout();save('gmm_soft_membership',fig)
results['gmm']={'component_weights':np.round(mg.weights_,3).tolist(),'median_max_prob':round(float(np.median(post.max(axis=1))),3)}

# Intro PCA geometry ellipse and axes; strongly correlated 2D
rng=np.random.default_rng(4880)
u=rng.normal(size=(280,2));corr=u@np.array([[2.5,2.5],[-.6,.6]])
p2=PCA(n_components=2).fit(corr)
mu=corr.mean(axis=0)
fig,ax=plt.subplots(figsize=(7.5,3.4))
ax.scatter(corr[:,0],corr[:,1],s=12,color=BLUE,alpha=.28)
for j,c in enumerate([GREEN,GOLD]):
 direction=p2.components_[j]*np.sqrt(p2.explained_variance_[j])*2.15
 ax.annotate('',xy=mu+direction,xytext=mu,arrowprops={'arrowstyle':'-|>','lw':3,'color':c})
 ax.text(*(mu+direction*1.13),f'PC{j+1}',color=c,fontweight='bold',fontsize=13)
ax.axhline(mu[1],color='#CDD6D2',lw=.7);ax.axvline(mu[0],color='#CDD6D2',lw=.7)
ax.set_aspect('equal',adjustable='box')
ax.set_title('PCA: perpendicular directions of variation',fontweight='bold')
ax.set_xlabel('Original feature 1');ax.set_ylabel('Original feature 2');ax.grid(alpha=.1)
fig.tight_layout();save('pca_geometry',fig)
results['pca_geometry']={'variance_ratio':np.round(p2.explained_variance_ratio_,3).tolist()}

# Iris PCA all features standardized; class colors only for posthoc interpretation
iris=load_iris();Xi=StandardScaler().fit_transform(iris.data)
pca_iris=PCA(n_components=4).fit(Xi);Zi=pca_iris.transform(Xi)
ratios=pca_iris.explained_variance_ratio_
results['pca_iris']={'variance_ratio':np.round(ratios,4).tolist(),'first_two_total_pct':round(float(100*ratios[:2].sum()),1)}
fig,axs=plt.subplots(1,2,figsize=(10.7,3.2))
for j in range(3):
 m=iris.target==j
 axs[0].scatter(Zi[m,0],Zi[m,1],s=27,color=COLORS[j],alpha=.75,label=iris.target_names[j])
axs[0].set_xlabel(f'PC1 ({ratios[0]*100:.1f}% variance)');axs[0].set_ylabel(f'PC2 ({ratios[1]*100:.1f}% variance)')
axs[0].set_title('Iris: fit PCA without species labels',fontweight='bold');axs[0].legend(loc='upper right',fontsize=8,frameon=False)
axs[1].bar(np.arange(1,5),ratios,color=[GREEN,GOLD,BLUE,RED])
axs[1].plot(np.arange(1,5),np.cumsum(ratios),marker='o',color=DARK,label='Cumulative fraction')
axs[1].set_ylim(0,1.03);axs[1].set_xticks(np.arange(1,5));axs[1].set_xlabel('Principal component');axs[1].set_ylabel('Fraction of total variance')
axs[1].set_title('Explained variance: first 2 = '+str(results['pca_iris']['first_two_total_pct'])+'%',fontweight='bold')
axs[1].legend(loc='center right',fontsize=8,frameon=False)
for a in axs:a.grid(alpha=.1)
fig.tight_layout();save('iris_pca_projection_variance',fig)

# Digits original samples montage and latent reconstruction
D=load_digits();Xd=D.data;yd=D.target
fig,axes=plt.subplots(2,8,figsize=(9.9,2.6))
for ax,j in zip(axes.flat,np.arange(16)):
 ax.imshow(D.images[j],cmap='Greys',interpolation='nearest',vmin=0,vmax=16)
 ax.set_xticks([]);ax.set_yticks([])
fig.suptitle('scikit-learn digits: each image is an 8 x 8 = 64-feature vector',fontweight='bold',y=1.015)
fig.tight_layout();save('digits_samples',fig)

selected=[0,10,21,36,55,71,86,101]
fig,axes=plt.subplots(5,len(selected),figsize=(10.8,5.4))
for j,i in enumerate(selected):
 ax=axes[0,j];ax.imshow(Xd[i].reshape(8,8),cmap='Greys',vmin=0,vmax=16);ax.set_xticks([]);ax.set_yticks([])
for row,nc in enumerate([2,8,16,32],start=1):
 m=PCA(n_components=nc,svd_solver='full').fit(Xd)
 rec=m.inverse_transform(m.transform(Xd[selected]))
 for j in range(len(selected)):
  ax=axes[row,j];ax.imshow(np.clip(rec[j].reshape(8,8),0,16),cmap='Greys',vmin=0,vmax=16);ax.set_xticks([]);ax.set_yticks([])
for row,txt in enumerate(['Original (64)','2 PCs','8 PCs','16 PCs','32 PCs']): axes[row,0].set_ylabel(txt,rotation=0,ha='right',va='center',labelpad=33,fontweight='bold',fontsize=9)
fig.tight_layout();save('digits_pca_reconstruction',fig)
# rank recon MSE calculated on full data
recon={}
for n in [2,8,16,32]:
 m=PCA(n_components=n,svd_solver='full').fit(Xd)
 recon[str(n)]=round(float(mean_squared_error(Xd,m.inverse_transform(m.transform(Xd)))),2)
results['digits_reconstruction_mse']=recon

# PCA vs t-SNE on 1,150 digit samples; labels only used to color final visualization
idx=np.random.default_rng(4880).choice(len(Xd),size=1150,replace=False)
X_vis=Xd[idx];y_vis=yd[idx]
X_pca_50=PCA(n_components=40,random_state=4880).fit_transform(X_vis)
X_vis_2=X_pca_50[:,:2]
X_t=TSNE(n_components=2,perplexity=30,learning_rate='auto',init='pca',max_iter=650,random_state=4880).fit_transform(X_pca_50)
fig,axs=plt.subplots(1,2,figsize=(10.5,3.75))
for j in range(10):
 for a,Z in zip(axs,[X_vis_2,X_t]):
  mm=y_vis==j;a.scatter(Z[mm,0],Z[mm,1],s=8,alpha=.63,color=plt.cm.tab10(j),label=str(j),rasterized=True)
axs[0].set_title('PCA (linear; global variance)',fontweight='bold');axs[1].set_title('t-SNE (nonlinear; local neighbors)',fontweight='bold')
for a in axs:a.set_xlabel('Dimension 1');a.set_ylabel('Dimension 2');a.grid(alpha=.07)
axs[1].legend(loc='center left',bbox_to_anchor=(1.01,.5),title='Digit',ncol=1,frameon=False,fontsize=8)
fig.tight_layout();save('digits_pca_vs_tsne',fig)

# NMF components on the 8x8 digits (nonnegative): components are additive templates
nmf=NMF(n_components=10,init='nndsvda',random_state=4880,max_iter=1200,tol=1e-4)
W=nmf.fit_transform(Xd)
fig,axs=plt.subplots(2,5,figsize=(10.5,4.2))
for j,ax in enumerate(axs.flat):
 ax.imshow(nmf.components_[j].reshape(8,8),cmap='viridis');ax.set_title('Feature '+str(j+1),fontsize=9)
 ax.set_xticks([]);ax.set_yticks([])
fig.suptitle('Nonnegative matrix factorization: learned additive image patterns',fontweight='bold',y=1.015)
fig.tight_layout();save('nmf_digits_components',fig)
results['nmf']={'relative_frobenius_reconstruction_error':round(float(np.linalg.norm(Xd-W@nmf.components_)/np.linalg.norm(Xd)),3),'iterations':int(nmf.n_iter_)}

# Generalization: select representation on validation, test once
X_develop,X_test,y_develop,y_test=train_test_split(Xd,yd,test_size=.2,stratify=yd,random_state=4880)
X_train,X_val,y_train,y_val=train_test_split(X_develop,y_develop,test_size=.25,stratify=y_develop,random_state=4880)
steps={}; candidates={}
clf=make_pipeline(StandardScaler(),SVC(C=5,gamma='scale',kernel='rbf'))
clf.fit(X_train,y_train)
candidates['64 raw pixels']=clf
steps['64 raw pixels']=float(accuracy_score(y_val,clf.predict(X_val)))
for nc in [8,16,32,48]:
 pipe=make_pipeline(StandardScaler(),PCA(n_components=nc,random_state=4880),SVC(C=5,gamma='scale',kernel='rbf'))
 pipe.fit(X_train,y_train)
 candidates[str(nc)+' PCs']=pipe
 steps[str(nc)+' PCs']=float(accuracy_score(y_val,pipe.predict(X_val)))
selected=max(steps,key=steps.get)
# Retrain selected *architecture* on all development data, including validation, then test once.
if selected=='64 raw pixels':
 final=make_pipeline(StandardScaler(),SVC(C=5,gamma='scale',kernel='rbf'))
else:
 final=make_pipeline(StandardScaler(),PCA(n_components=int(selected.split(' ')[0]),random_state=4880),SVC(C=5,gamma='scale',kernel='rbf'))
final.fit(X_develop,y_develop)
final_test=float(accuracy_score(y_test,final.predict(X_test)))
results['digits_downstream_svc']={'validation_accuracy':{k:round(v,4) for k,v in steps.items()},'selected':selected,'final_test_accuracy':round(final_test,4),'train_n':len(X_train),'validation_n':len(X_val),'test_n':len(X_test)}
fig,ax=plt.subplots(figsize=(8.8,3.4))
keys=list(steps);vals=list(steps.values())
ax.bar(np.arange(len(keys)),vals,color=[GREEN]+[GOLD,BLUE,'#5C947D',VIOLET])
for j,v in enumerate(vals):ax.text(j,v+.006,f'{100*v:.1f}%',ha='center',va='bottom',fontweight='bold',fontsize=10)
ax.set_xticks(np.arange(len(keys)));ax.set_xticklabels(keys)
ax.set_ylim(.7,1.07);ax.set_ylabel('Validation accuracy');ax.set_title('Digits: choose the feature representation on validation',fontweight='bold');ax.grid(axis='y',alpha=.15)
fig.tight_layout();save('pca_downstream_svc_accuracy',fig)

# One Class SVM: train on clean inlier samples, evaluate with separate held-out normal + anomaly points
rng=np.random.default_rng(4880)
X_oc_train=rng.normal(0,.6,size=(240,2))
X_oc_test_normal=rng.normal(0,.6,size=(80,2))
X_oc_test_anom=rng.uniform(low=2,high=3.6,size=(50,2))*rng.choice([-1,1],size=(50,2))
oc=OneClassSVM(kernel='rbf',gamma=.4,nu=.05).fit(X_oc_train)
pred_n=oc.predict(X_oc_test_normal);pred_a=oc.predict(X_oc_test_anom)
results['one_class_svm']={'normal_acceptance':round(float((pred_n==1).mean()),3),'anomalies_flagged':int((pred_a==-1).sum()),'anomalies_total':len(pred_a),'nu':.05}
x1=np.linspace(-4,4,220);x2=np.linspace(-4,4,220)
xx,yy=np.meshgrid(x1,x2);d=oc.decision_function(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
fig,ax=plt.subplots(figsize=(7.7,3.8))
ax.contourf(xx,yy,d,levels=[-100,0,100],colors=['#F5E9E8','#EDF5EF'],alpha=.9)
ax.contour(xx,yy,d,levels=[0],colors=[GREEN],linewidths=2.3)
ax.scatter(X_oc_train[:,0],X_oc_train[:,1],s=13,color=GREEN,alpha=.43,label='Normal training data')
ax.scatter(X_oc_test_normal[:,0],X_oc_test_normal[:,1],s=22,facecolors='none',edgecolors=BLUE,alpha=.95,label='New normal observations')
ax.scatter(X_oc_test_anom[:,0],X_oc_test_anom[:,1],s=27,marker='x',color=RED,alpha=.85,label='Synthetic anomalies (test only)')
ax.set_xlim(-4,4);ax.set_ylim(-4,4);ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.1)
ax.set_title('OneClassSVM learns an inlier region without anomaly labels',fontweight='bold')
ax.legend(loc='upper left',bbox_to_anchor=(1,.95),frameon=False,fontsize=9)
fig.tight_layout();save('one_class_svm_boundary',fig)

# Figure for general ML pipeline leakage without data specifics, rendered vector-feeling schematic
fig,ax=plt.subplots(figsize=(10.7,2.55))
ax.set_xlim(0,11);ax.set_ylim(0,3);ax.axis('off')
def block(x,y,w,h,text,color=GREEN):
 from matplotlib.patches import FancyBboxPatch
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.07',edgecolor=color,facecolor='#F1F5F0',lw=1.7))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontweight='bold',fontsize=10.5)
def arr(a,b,y):ax.annotate('',xy=(b,y),xytext=(a,y),arrowprops={'arrowstyle':'-|>','color':DARK,'lw':1.5})
block(.2,1.45,1.65,.65,'Training X');arr(1.95,2.33,1.78);block(2.4,1.45,1.85,.65,'Fit scaler + PCA');arr(4.35,4.73,1.78);block(4.8,1.45,1.65,.65,'Fit classifier');arr(6.55,6.9,1.78);block(7,1.45,1.7,.65,'Frozen pipeline')
block(.2,.32,1.65,.65,'Held-out X',color=BLUE);arr(1.95,6.9,.64);ax.annotate('',xy=(7.85,1.03),xytext=(7.85,1.38),arrowprops={'arrowstyle':'-|>','color':DARK,'lw':1.5});block(7,.31,1.7,.65,'Predict',color=BLUE);arr(8.82,9.12,.64);block(9.18,.31,1.65,.65,'Compare labels',color=BLUE)
ax.text(5.2,2.53,'Fit every transform using training data only',ha='center',color=GREEN,fontweight='bold',fontsize=13)
fig.tight_layout();save('train_test_pipeline',fig)

# Save results & versions
import sklearn, scipy, platform
results['software']={'sklearn':sklearn.__version__,'numpy':np.__version__,'matplotlib':matplotlib.__version__,'scipy':scipy.__version__,'python':platform.python_version()}
with open(os.path.join(os.path.dirname(__file__),'results.json'),'w') as f:json.dump(results,f,indent=2)
print(json.dumps(results,indent=2))
print('PNG files:',len([p for p in os.listdir(ROOT) if p.endswith('.png')]))
