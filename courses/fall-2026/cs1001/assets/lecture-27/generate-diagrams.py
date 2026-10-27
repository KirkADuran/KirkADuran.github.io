"""Reproduce the five precise diagrams for CS1001 Lecture 27 (Matplotlib)."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
OUT=Path(__file__).resolve().parent
GREEN,GOLD,INK,BLUE='#154734','#BD8B13','#1F2933','#427C9E'
MINT,PALE='#E7F0EA','#FFF3D9'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':16,
                     'figure.constrained_layout.use':True,'savefig.facecolor':'white'})
def canvas(w=5,h=6):
    fig,ax=plt.subplots(figsize=(w,h));ax.set(xlim=(0,w),ylim=(0,h));ax.axis('off');return fig,ax

def box(ax,x,y,w,h,label,color=MINT,size=16):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.035',fc=color,ec=GREEN,lw=1.5))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=size,color=INK)

def arrow(ax,a,b,label=None,offset=(0,0),curve=0):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=17,lw=1.8,color=GREEN,
                                connectionstyle=f'arc3,rad={curve}'))
    if label:
        ax.text((a[0]+b[0])/2+offset[0],(a[1]+b[1])/2+offset[1],label,
                ha='center',va='center',fontsize=15,color=INK,
                bbox={'facecolor':'white','edgecolor':'none','pad':1})
def save(fig,name):
    fig.savefig(OUT/name,dpi=210,bbox_inches='tight',pad_inches=.08);plt.close(fig)

fig,ax=canvas(5,6)
for y,inp,fn,out in [(4.45,'List','average','Number'),(2.55,'Nested list','flatten','List'),(.65,'Nested list','row_maxima','Dictionary')]:
    ax.text(.2,y+1.05,inp,fontsize=17,color=BLUE,weight='bold')
    box(ax,.25,y,.95,.65,'input',size=14)
    box(ax,1.65,y,1.8,.65,fn,size=15)
    box(ax,3.9,y,.85,.65,'output',PALE,size=13)
    arrow(ax,(1.25,y+.32),(1.6,y+.32));arrow(ax,(3.5,y+.32),(3.85,y+.32))
    ax.text(3.85,y-.38,out,ha='right',fontsize=16,color=GREEN)
save(fig,'structured-function-contracts.png')

fig,ax=canvas(5,5.6)
colors=[MINT,PALE,'#E8F1F8'];rows=[[1,2],[3,4,5],[6]]
for j,row in enumerate(rows):
    y=4.6-j*.9
    ax.text(.1,y+.22,f'Row {j}',fontsize=15,color=INK)
    for k,v in enumerate(row):box(ax,1.3+.78*k,y,.65,.5,str(v),colors[j],17)
ax.text(2.5,5.35,'Read rows in order',ha='center',weight='bold',color=GREEN)
arrow(ax,(2.5,2.55),(2.5,1.8))
for i,(v,c) in enumerate(zip([1,2,3,4,5,6],[colors[0]]*2+[colors[1]]*3+[colors[2]])):
    box(ax,.2+i*.78,1.05,.66,.55,str(v),c,18)
ax.text(2.5,.45,'New flat list',ha='center',color=GREEN,weight='bold',fontsize=18)
save(fig,'flatten-grid-to-sequence.png')

fig,ax=canvas(4.6,5.5)
ax.text(1.25,5.05,'Name',ha='center',weight='bold',color=GREEN)
ax.text(3.45,5.05,'Score',ha='center',weight='bold',color=GREEN)
for i,(name,score) in enumerate(zip(['Ada','Grace','Alan','Edsger'],[92,68,81,65])):
    y=4.05-i*.95
    ax.add_patch(FancyBboxPatch((.15,y-.1),4.3,.77,boxstyle='round,pad=.035',fc='#F7F7F5',ec='none'))
    box(ax,.4,y,1.7,.55,name,size=18)
    box(ax,2.8,y,1.3,.55,str(score),PALE,size=18)
    ax.plot([2.15,2.75],[y+.27,y+.27],color=BLUE,lw=2)
ax.text(2.3,.35,'One corresponding pair\nper iteration',ha='center',color=GREEN,fontsize=17)
save(fig,'zip-corresponding-sequences.png')

fig,ax=canvas(5.2,6)
box(ax,1.25,5.2,2.6,.6,'Test condition',size=18)
ax.add_patch(Polygon([[2.55,4.85],[3.65,4.2],[2.55,3.55],[1.45,4.2]],fc=PALE,ec=GREEN,lw=1.6))
ax.text(2.55,4.2,'True?',ha='center',va='center',fontsize=18,color=INK)
arrow(ax,(2.55,5.15),(2.55,4.88))
box(ax,.9,1.85,3.3,.9,'Execute body\n(update state)',size=18)
arrow(ax,(2.55,3.5),(2.55,2.8),'Yes',offset=(.45,0))
box(ax,3.8,3.9,1.05,.6,'Done',size=16)
arrow(ax,(3.7,4.2),(3.77,4.2))
ax.text(4.25,4.75,'No',ha='center',fontsize=16,color=INK)
ax.plot([.85,.35,.35,1.2],[2.3,2.3,5.5,5.5],color=GREEN,lw=1.7)
arrow(ax,(.7,5.5),(1.2,5.5))
ax.text(2.55,.65,'Test again after every body',ha='center',fontsize=16,color=GREEN)
save(fig,'while-loop-control-flow.png')

fig,ax=canvas(5,6.7)
box(ax,1.2,5.9,2.6,.55,'"CompSci9"',PALE,size=18)
checks=['Length >= 8','Has uppercase','Has lowercase','Has digit','No whitespace']
for i,text in enumerate(checks):
    y=4.95-i*.85
    box(ax,1.0,y,3.0,.55,text,size=16)
    ax.text(4.45,y+.27,'T',ha='center',va='center',color=GREEN,weight='bold',fontsize=18)
    arrow(ax,(.6,y+.27),(.95,y+.27));arrow(ax,(4.05,y+.27),(4.22,y+.27))
ax.plot([2.5,.6,.6],[5.85,5.85,1.8],color=GREEN,lw=1.5)
ax.plot([4.7,4.7,2.5],[5.22,.75,.75],color=GREEN,lw=1.5)
box(ax,1.0,.1,3,.5,'All five: True',PALE,size=18)
arrow(ax,(2.5,.75),(2.5,.65))
save(fig,'password-requirement-pipeline.png')
print('Generated five CS1001 structured-data diagrams.')
