"""Reproduce Lecture 26 diagrams and the supplied-slide Python demonstration.

numpy, matplotlib, scikit-learn; no downloads. The diabetes example retains
BMI and the final-20 test split from the instructor's 19_480_Winter26 slides.
The arithmetic exercise is a different dataset from the robot lab.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

OUT=Path(__file__).resolve().parent
GREEN,GOLD,BLUE,INK,GRAY='#154734','#BD8B13','#427C9E','#1F2933','#75828A'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,
    'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,
    'figure.constrained_layout.use':True,'savefig.facecolor':'white'})
def save(fig,name):
    fig.savefig(OUT/name,dpi=210,bbox_inches='tight',pad_inches=.1)
    plt.close(fig)
def box(ax,x,y,w,h,label,face='#E7F0EA',size=14):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.025',
        facecolor=face,edgecolor=GREEN,lw=1.5))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=size,color=INK)
def arrow(ax,start,end,label='',offset=(0,0),curve=0):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=15,
        color=GREEN,lw=1.5,connectionstyle=f'arc3,rad={curve}'))
    if label:
        mid=((start[0]+end[0])/2+offset[0],(start[1]+end[1])/2+offset[1])
        ax.text(*mid,label,ha='center',va='center',fontsize=10.5,color=INK,
                bbox={'facecolor':'white','edgecolor':'none','pad':1.4})

fig,ax=plt.subplots(figsize=(11,4.3))
ax.set(xlim=(-.1,11),ylim=(-.3,4.6));ax.axis('off')
ax.add_patch(FancyBboxPatch((.1,0),7.6,4.1,boxstyle='round,pad=.03',
    fc='#F6F8F6',ec=GRAY,lw=1.2))
ax.text(.3,3.88,'Agent',color=GREEN,weight='bold',fontsize=14)
box(ax,.6,2.6,2.15,.65,'Critic',face='#E8F1F8')
box(ax,.6,1.35,2.15,.65,'Learning element')
box(ax,.6,.15,2.15,.65,'Problem generator',face='#FFF3D9',size=13)
box(ax,4.65,1.35,2.4,.85,'Performance\nelement')
box(ax,8.55,.55,2.2,2.8,'Environment',face='#FFF3D9')
ax.text(1.67,4.35,'Performance standard',ha='center',fontsize=12,color=INK)
arrow(ax,(1.67,4.1),(1.67,3.27))
arrow(ax,(9.65,3.4),(9.65,3.65))
arrow(ax,(9.65,3.65),(5.85,3.65),'percepts / sensors',offset=(0,.20))
arrow(ax,(5.85,3.65),(5.85,2.25))
arrow(ax,(5.8,3.65),(1.67,3.65))
arrow(ax,(1.67,3.65),(1.67,3.27))
arrow(ax,(1.67,2.57),(1.67,2.04),'feedback',offset=(.65,0))
arrow(ax,(2.8,1.9),(4.6,1.9),'changes',offset=(0,.18))
arrow(ax,(4.6,1.46),(2.8,1.46),'knowledge',offset=(0,-.17))
arrow(ax,(1.67,1.30),(1.67,.84),'learning goals',offset=(1,0))
arrow(ax,(2.8,.48),(4.65,1.35),'exploration',offset=(.15,-.18))
arrow(ax,(5.85,1.3),(5.85,.10))
arrow(ax,(5.85,.1),(9.65,.1),'actions / actuators',offset=(0,-.18))
arrow(ax,(9.65,.1),(9.65,.5))
save(fig,'learning-agent-loop.png')

fig,ax=plt.subplots(figsize=(10.5,3.5))
ax.set(xlim=(0,10.5),ylim=(0,3.5));ax.axis('off')
box(ax,.2,2.05,2.5,.95,'Labeled examples\n'+r'$\mathcal{D}=\{(\mathbf{x}_i,y_i)\}$',face='#E8F1F8')
box(ax,4,2.05,2.4,.95,'Learning algorithm\nminimize training loss')
box(ax,7.8,2.05,2.4,.95,'Fitted parameters\n'+r'$\widehat{\mathbf{w}}$',face='#FFF3D9')
arrow(ax,(2.75,2.52),(3.95,2.52))
arrow(ax,(6.45,2.52),(7.75,2.52))
box(ax,.2,.2,2.5,.95,'New observation\n'+r'$\mathbf{x}_{\mathrm{new}}$',face='#E8F1F8')
box(ax,4,.2,2.4,.95,'Fixed predictor\n'+r'$f_{\widehat{\mathbf{w}}}(\mathbf{x})$')
box(ax,7.8,.2,2.4,.95,'Prediction\n'+r'$\widehat{y}$',face='#FFF3D9')
arrow(ax,(2.75,.67),(3.95,.67))
arrow(ax,(6.45,.67),(7.75,.67))
arrow(ax,(9,2.0),(6.1,1.2),'use learned weights',offset=(.1,-.05))
ax.text(.2,3.25,'TRAINING',color=GREEN,weight='bold')
ax.text(.2,1.4,'INFERENCE',color=GREEN,weight='bold')
save(fig,'training-inference.png')

# The supplied Python example: exact data feature and split, independently run.
data=load_diabetes()
X=data.data[:,[2]];y=data.target
X_train,X_test=X[:-20],X[-20:]
y_train,y_test=y[:-20],y[-20:]
model=LinearRegression().fit(X_train,y_train)
y_pred=model.predict(X_test)
fig,ax=plt.subplots(figsize=(6,4))
ax.scatter(X_test[:,0],y_test,color=GOLD,edgecolor=INK,label='20 held-out cases')
grid=np.linspace(X_test.min(),X_test.max(),150)[:,None]
ax.plot(grid[:,0],model.predict(grid),color=GREEN,lw=2.5,label='Training-fitted line')
ax.set_xlabel('BMI (dataset-scaled feature)');ax.set_ylabel('Progression score')
ax.grid(alpha=.15);ax.legend(fontsize=11)
save(fig,'instructor-diabetes-fit.png')
benchmark={'training_count':len(y_train),'test_count':len(y_test),
    'intercept':float(model.intercept_),'slope':float(model.coef_[0]),
    'mse':float(mean_squared_error(y_test,y_pred)),
    'rmse':float(np.sqrt(mean_squared_error(y_test,y_pred))),
    'mae':float(mean_absolute_error(y_test,y_pred)),
    'r2':float(r2_score(y_test,y_pred))}

# Worked arithmetic case, distinct from the lab's crowd/delay observations.
x=np.arange(1,5,dtype=float); observed=np.array([3.,5.,4.,8.])
A=np.column_stack([np.ones(4),x]);w=np.array([.5,1.]);eta=.05
pred=A@w;resid=pred-observed
old_loss=float(np.mean(resid**2));gradient=2/len(x)*A.T@resid
updated=w-eta*gradient;new_loss=float(np.mean((A@updated-observed)**2))
assert np.allclose(gradient,[-4,-11])
assert np.allclose(updated,[.7,1.55])
assert np.isclose(old_loss,5.25) and np.isclose(new_loss,1.25875)
# Independent finite-difference audit of both gradient components.
for j in range(2):
    eps=np.zeros(2);eps[j]=1e-6
    numerical=(np.mean((A@(w+eps)-observed)**2)-np.mean((A@(w-eps)-observed)**2))/(2e-6)
    assert np.isclose(numerical,gradient[j],rtol=1e-7)
fig,axs=plt.subplots(1,2,figsize=(10,3.5))
for rate,col,title in [(.05,GREEN,'Useful rate'),(.15,GOLD,'Too large here')]:
    weights=w.copy();losses=[]
    for k in range(16):
        err=A@weights-observed;losses.append(float(np.mean(err**2)))
        weights-=rate*(2/len(x)*A.T@err)
    axs[1].plot(range(16),losses,'o-',color=col,label=f'eta={rate}')
assert losses[-1]>losses[0]
axs[1].set_yscale('log');axs[1].set_xlabel('Iteration');axs[1].set_ylabel('Training MSE (log scale)')
axs[1].legend(fontsize=11);axs[1].grid(alpha=.15)
b0=np.linspace(-.5,3,120);b1=np.linspace(.5,2,120)
B0,B1=np.meshgrid(b0,b1)
Z=np.mean((B0[...,None]+B1[...,None]*x-observed)**2,axis=-1)
axs[0].contour(B0,B1,Z,levels=[1.1,1.3,1.7,2.5,4,6,10],colors=GRAY)
weights=w.copy();path=[weights.copy()]
for k in range(12):
    weights-=eta*(2/len(x)*A.T@(A@weights-observed));path.append(weights.copy())
path=np.array(path)
axs[0].plot(path[:,0],path[:,1],'o-',color=GREEN,ms=4)
axs[0].annotate('start',path[0],xytext=(8,-12),textcoords='offset points')
axs[0].set_xlabel('Intercept w0');axs[0].set_ylabel('Slope w1')
axs[0].set_title('Loss contours and updates');axs[1].set_title('Step size changes convergence')
save(fig,'gradient-geometry.png')

# Same quadratic generating trend, varied sample size/noise; no test data used.
fig,axs=plt.subplots(1,3,figsize=(12,3.3),sharex=True)
settings=[(12,.65,23),(60,.65,23),(12,1.8,23)]
sensitivity=[]
for ax,(n,noise,seed) in zip(axs,settings):
    rng=np.random.default_rng(seed);xx=np.linspace(1,12,n)
    true=lambda z:8+.45*z+.12*z*z
    yy=true(xx)+rng.normal(0,noise,n)
    vr=np.random.default_rng(124);vx=vr.uniform(1,12,300)
    vy=true(vx)+vr.normal(0,noise,len(vx))
    degrees=[1,2,9]
    errs={}
    for d in degrees:
        poly=np.polynomial.Polynomial.fit(xx,yy,d)
        errs[d]=float(np.mean((poly(vx)-vy)**2))
    ax.plot(degrees,[errs[d] for d in degrees],'o-',color=GREEN)
    ax.set_title(f'n={n}, noise SD={noise}')
    ax.set_xlabel('Polynomial degree');ax.set_xticks(degrees);ax.grid(alpha=.15)
    sensitivity.append({'n':n,'noise_sd':noise,'validation_mse':errs,
                        'selected_degree':min(errs,key=errs.get)})
axs[0].set_ylabel('Validation MSE')
save(fig,'sample-noise-comparison.png')

# A smooth polynomial cannot uniformly erase a genuine jump.
xx=np.r_[np.linspace(0,.49,35),np.linspace(.51,1,35)]
yy=np.where(xx<.5,1.,4.)
poly=np.polynomial.Polynomial.fit(xx,yy,4)
grid=np.linspace(0,1,500)
fig,ax=plt.subplots(figsize=(6,3.5))
ax.scatter(xx,yy,color=GOLD,s=22,label='Rule-governed outcomes')
ax.plot(grid,poly(grid),color=GREEN,lw=2.5,label='Degree-4 fit')
ax.set_xlabel('Input');ax.set_ylabel('Output');ax.grid(alpha=.15);ax.legend(fontsize=11)
save(fig,'smooth-fit-limits.png')
result={'supplied_slide_example':benchmark,'gradient_example':{
    'x':x.tolist(),'y':observed.tolist(),'initial_weights':w.tolist(),'eta':eta,
    'predictions':pred.tolist(),'residuals':resid.tolist(),'initial_loss':old_loss,
    'gradient':gradient.tolist(),'updated_weights':updated.tolist(),'updated_loss':new_loss},
    'sensitivity':sensitivity}
(OUT/'foundations-checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
