#!/usr/bin/env python3
"""Reproducible local lecture figures for CS4880 Lecture 33.
Run from this file's directory. No downloads or external datasets.
Diagrams are ORIGINAL SCHEMATICS, not screenshots or benchmark reconstructions.
Randomized estimations are seeded; any claims about learned performance are
limited to the specific toy experiment printed in measured-results.txt.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, FancyArrowPatch, Ellipse
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent
ROOT.mkdir(parents=True,exist_ok=True)
SEED=4880
rng=np.random.default_rng(SEED)
GREEN='#154734'; INK='#263A32'; MINT='#168F85'; GOLD='#B88A2B'; SOFT='#EAF2ED'; GRAY='#63756B'; SKY='#4B88B3'; RED='#B34B43'; BG='#FFFFFF'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.edgecolor':'#B8CAC0','axes.labelcolor':INK,'text.color':INK,'axes.titleweight':'bold','axes.titlesize':13,'xtick.color':GRAY,'ytick.color':GRAY,'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white','savefig.bbox':'tight','savefig.pad_inches':0.1})

def fresh(w=10,h=4):
    fig,ax=plt.subplots(figsize=(w,h)); ax.set_xlim(0,10); ax.set_ylim(0,4); ax.axis('off'); return fig,ax

def box(ax,x,y,w,h,label,fc=SOFT,ec=GREEN,fs=12,weight='bold',r=.11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0.02,rounding_size={r}',facecolor=fc,edgecolor=ec,linewidth=1.9))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=fs,fontweight=weight,color=INK)

def arrow(ax,x1,y1,x2,y2,color=MINT,style='-|>',lw=2.3):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops={'arrowstyle':style,'color':color,'linewidth':lw,'shrinkA':3,'shrinkB':3})

def save(fig,name):
    fig.savefig(ROOT/name,dpi=185,transparent=False); plt.close(fig)

# 1. Table to network summary
fig,ax=fresh(); box(ax,.3,1.15,2.8,1.65,'Q-table\nstate rows × action columns');box(ax,6.6,1.15,3.0,1.65,'Neural network\nQ(s, ·; θ)',fc='#E7F3F2',ec=MINT); ax.text(5,3.15,'Same goal: choose the best action',ha='center',fontsize=15,fontweight='bold',color=GREEN); arrow(ax,3.24,2,6.42,2);ax.text(5,2.4,'function approximation',ha='center',color=GRAY); ax.text(1.7,.75,'Exact lookup in finite states',ha='center',color=GRAY,fontsize=10);ax.text(8.1,.75,'Share information across states',ha='center',color=GRAY,fontsize=10);save(fig,'table_to_network.png')

# 2. State explosion
fig,ax=plt.subplots(figsize=(10,4.1)); ax.set_yscale('log'); labs=['5×5 grid','5×5 + 12 dirt bits','84×84 grayscale image']; vals=[25,25*2**12,256**(84*84) if False else 1e25]; ax.bar(range(3),vals,color=[MINT,GOLD,GREEN],width=.62);ax.set_ylim(10,1e26); ax.set_yticks([10,1e5,1e10,1e15,1e20,1e25]); ax.set_xticks(range(3),labs);ax.set_ylabel('Distinct states (log scale)');ax.set_title('Even small observations create enormous state spaces');ax.text(2,3e25,'Image bar is a lower-bound illustration\n(actual count is $256^{7056}$)',ha='center',fontsize=10);ax.grid(axis='y',alpha=.20);save(fig,'state_explosion.png')

# 3. Atari scene original drawing
fig,ax=plt.subplots(figsize=(7.8,4.3)); ax.set_xlim(0,12);ax.set_ylim(0,8);ax.axis('off');ax.add_patch(Rectangle((.25,.25),11.5,7.5,facecolor='#12211B',edgecolor=GREEN,linewidth=3));brickcolors=['#E8754B','#F1B65A','#D9D76D','#78B77D'];
for j,color in enumerate(brickcolors):
 for i in range(11):
  if not (j==0 and i in (0,1)):
   ax.add_patch(Rectangle((.68+i*.96,6.7-j*.55),.81,.38,facecolor=color,edgecolor='#223E31',linewidth=.5))
ax.plot([2.5,4.2,6.9],[3.1,3.9,4.8],color=SKY,ls='--',lw=1.9);ax.add_patch(Circle((6.9,4.8),.14,color='white'));ax.add_patch(Rectangle((5.35,.8),2.1,.22,facecolor='#B8E4D8'));ax.text(10.7,.9,'PADDLE',ha='center',color='white',fontsize=9);ax.text(1,7.3,'BREAKOUT · illustrative',color='white',fontsize=11,fontweight='bold');save(fig,'breakout_schematic.png')

# 4. Frame stack diagram
fig,ax=fresh(); ax.text(5,3.65,'Motion needs temporal information',ha='center',fontsize=14,fontweight='bold',color=GREEN)
for i,x in enumerate([.4,2.6,4.8,7.0]):
    ax.add_patch(Rectangle((x,1.13),1.7,1.75,facecolor='#142A25',edgecolor=GRAY,linewidth=1.4));ax.add_patch(Rectangle((x+.43,1.38),.85,.09,color='#CDEAE0'));ax.add_patch(Circle((x+.26+i*.28,1.75+i*.21),.11,color='#FFCB72'));ax.text(x+.85,.85,f't−{3-i}',ha='center',fontsize=11)
for i in range(3):arrow(ax,2.12+i*2.2,2,2.50+i*2.2,2,color=GRAY,lw=1.7)
ax.text(9.2,2,'→ CNN\n  Q-values',ha='left',va='center',fontsize=12,fontweight='bold');save(fig,'frame_stack.png')

# 5. Network architecture
fig,ax=fresh(11,4.4); ax.set_xlim(0,11);ax.set_ylim(0,4.4)
box(ax,.15,1.35,1.7,1.65,'4 stacked\n84×84 frames',fs=10);arrow(ax,1.93,2.15,2.63,2.15);box(ax,2.7,1.35,1.7,1.65,'Convolutional\nfeatures',fc='#E7F3F2',ec=MINT,fs=10);arrow(ax,4.5,2.15,5.17,2.15);box(ax,5.25,1.35,1.62,1.65,'Dense\nfeatures',fs=10);arrow(ax,6.95,2.15,7.62,2.15);box(ax,7.7,.80,3.08,2.65,'Q(left) = 2.1\nQ(no-op) = 1.3\nQ(right) = 0.8',fc='#FCF4E5',ec=GOLD,fs=12);ax.text(5.5,3.85,'One forward pass estimates all discrete action values',ha='center',fontweight='bold',fontsize=14,color=GREEN);save(fig,'dqn_cnn_architecture.png')

# 6. Replay buffer
fig,ax=fresh();ax.text(5,3.7,'Collect consecutive transitions; train from randomized past samples',ha='center',fontsize=14,fontweight='bold',color=GREEN)
for j in range(6):
 x=.25+j*1.6; box(ax,x,2.05,1.25,.65,f't={j}',fc=SOFT,fs=10);
 if j<5:arrow(ax,x+1.28,2.36,x+1.57,2.36,color=GRAY,lw=1.2)
arrow(ax,5.05,1.95,5.05,1.45)
box(ax,.8,.58,8.35,.65,'Replay memory:  (s, a, r, s′, done)  × many',fc='#E7F3F2',ec=MINT,fs=12)
for x in [2.15,4.4,7.4]:arrow(ax,x,.45,x,-.07,color=GOLD,lw=1.7)
ax.text(5.1,.15,'Sample minibatches randomly (uniform replay)',ha='center',fontsize=10,color=GRAY)
save(fig,'replay_buffer.png')

# 7. Replay sampling drawn distinct
fig,ax=fresh();ax.text(5,3.55,'A minibatch is not the most recent consecutive trajectory',ha='center',fontsize=14,fontweight='bold',color=GREEN)
xs=np.arange(12); vals=[.3,.4,.5,.6,2.3,2.1,1.6,1.0,.3,.2,.7,.7];
for i,v in enumerate(vals):
 ax.add_patch(Rectangle((.35+i*.76,1.58),.52,.55,facecolor=GOLD if i in (0,4,7,11) else SOFT,edgecolor=GREEN,linewidth=1));ax.text(.61+i*.76,1.86,str(i),ha='center',va='center',fontsize=9)
ax.text(5,2.45,'Replay positions 0 ... 11',ha='center',color=GRAY);box(ax,2.6,.32,4.8,.70,'batch indices: [7, 0, 11, 4]',fc='#FCF4E5',ec=GOLD,fs=12);arrow(ax,5,1.54,5,1.06);save(fig,'random_batch.png')

# 8. Online target networks
fig,ax=fresh(); ax.text(5,3.6,'The online network learns; the target network provides a lagged reference',ha='center',fontsize=13,fontweight='bold',color=GREEN)
box(ax,.5,1.25,3.3,1.23,'Online Q(s,a; θ)\nGradient updates every batch',fc='#E7F3F2',ec=MINT,fs=11);box(ax,6.25,1.25,3.25,1.23,'Target Q(s,a; θ⁻)\nCopied every C steps',fc='#FCF4E5',ec=GOLD,fs=11);arrow(ax,3.85,1.92,6.18,1.92,color=GOLD);ax.text(5,2.38,'copy weights periodically',ha='center',fontsize=10,color=GRAY);ax.text(2.2,.83,'prediction Q(s,a)',ha='center',color=MINT,fontsize=11);ax.text(7.86,.83,'bootstrap y',ha='center',color=GOLD,fontsize=11);save(fig,'two_networks.png')

# 9. Step target network synthetic
fig,ax=plt.subplots(figsize=(10,4)); x=np.linspace(0,100,101); online=1.6+.025*x+.37*np.sin(.21*x); target=np.zeros_like(online)
for i in range(len(x)):target[i]=online[(i//18)*18]
ax.plot(x,online,color=MINT,lw=2.3,label='Online estimate (illustrative)'); ax.step(x,target,where='post',lw=2.3,color=GOLD,label='Periodically copied target');ax.set_title('Target estimates are held fixed between synchronizations');ax.set_xlabel('Gradient update number');ax.set_ylabel('Q-value for one example');ax.grid(alpha=.18);ax.legend(frameon=False);save(fig,'target_hold.png')

# 10. DQN forward and backward
fig,ax=fresh();ax.text(5,3.62,'One DQN minibatch update',ha='center',fontsize=14,fontweight='bold',color=GREEN)
box(ax,.3,1.85,2.48,.8,'Replay sample',fs=12);box(ax,3.53,2.4,2.4,.8,'Online Q(s,a; θ)',fc='#E7F3F2',ec=MINT,fs=11);box(ax,3.53,.82,2.4,.8,'Target Q(s′,·; θ⁻)',fc='#FCF4E5',ec=GOLD,fs=11);box(ax,6.88,1.62,2.82,1.0,'Loss = (y − Q)²\nBackprop to online only',fc=SOFT,fs=11);arrow(ax,2.78,2.27,3.42,2.75);arrow(ax,2.78,2.2,3.42,1.18);arrow(ax,5.95,2.76,6.77,2.42);arrow(ax,5.95,1.2,6.77,2.01);save(fig,'dqn_training_step.png')

# 11. Double small numerical example
fig,ax=plt.subplots(figsize=(9.5,3.8)); labels=['left','stay','right']; xx=np.arange(3); online=[7.0,10.0,9.0]; target=[8.0,6.0,9.5];bars=ax.bar(xx-.19,online,.36,color=MINT,label='Online: select');ax.bar(xx+.19,target,.36,color=GOLD,label='Target: evaluate');ax.set_xticks(xx,labels);ax.set_ylabel('Estimated next-state Q');ax.set_ylim(0,12);ax.set_title('Selection and evaluation can disagree');ax.legend(frameon=False);ax.text(1,10.45,'online choice',ha='center',color=MINT,fontsize=10);ax.text(2.3,10.8,'DQN target max = 9.5\nDouble target = 6.0',ha='center',fontsize=10,bbox={'facecolor':'white','edgecolor':'#D9E2DD'});ax.grid(axis='y',alpha=.16);save(fig,'double_numeric_example.png')

# 12. Overestimation MC
rng2=np.random.default_rng(SEED); samples=rng2.normal(0,1,size=(20000,6)); samples2=rng2.normal(0,1,size=(20000,6)); dqn=samples.max(axis=1); doubled=samples2[np.arange(len(samples)),samples.argmax(axis=1)]; fig,ax=plt.subplots(figsize=(10,4.2)); bins=np.linspace(-3.5,4.5,55);ax.hist(dqn,bins=bins,alpha=.62,density=True,color=GOLD,label=f'Max on same noisy estimates (mean {dqn.mean():.2f})');ax.hist(doubled,bins=bins,alpha=.48,density=True,color=MINT,label=f'Independent evaluation (mean {doubled.mean():.2f})');ax.axvline(0,color=GRAY,lw=1.5,ls='--');ax.set_title('Demonstration: the max of unbiased noisy estimates is biased upward');ax.set_xlabel('Estimated value; true value of each action = 0');ax.set_ylabel('Density');ax.legend(frameon=False,fontsize=9);save(fig,'max_bias_montecarlo.png');
(ROOT/'monte-carlo-values.txt').write_text(f'20,000 iid 6-action trials, normal(0,1) noise, seed={SEED}\nMean max estimate: {dqn.mean():.4f}\nMean independent evaluation of selected index: {doubled.mean():.4f}\n')

# 13. dueling architecture
fig,ax=fresh();ax.text(5,3.72,'Shared features split into a state-value head and an action-advantage head',ha='center',fontsize=13,fontweight='bold',color=GREEN)
box(ax,.35,1.45,2.1,1.04,'State features\nh(s)',fs=12);box(ax,3.48,2.25,2.37,.88,'Value V(s)',fc='#FCF4E5',ec=GOLD,fs=12);box(ax,3.48,.65,2.37,.88,'Advantage A(s,·)',fc='#E7F3F2',ec=MINT,fs=12);box(ax,7.12,1.38,2.55,1.26,'Mean-centered\naggregation → Q(s,·)',fs=11);arrow(ax,2.52,2.05,3.42,2.67);arrow(ax,2.52,1.87,3.42,1.08);arrow(ax,5.94,2.68,7.0,2.11);arrow(ax,5.94,1.08,7.0,1.86);save(fig,'dueling_architecture.png')

# 14. relative advantage numeric
fig,ax=plt.subplots(figsize=(9,4));actions=['left','stay','right']; a=np.array([-1.,0.,2.]); centered=a-a.mean(); V=5.; q=V+centered;xs=np.arange(3);ax.bar(xs,q,color=[GRAY,GOLD,MINT],width=.55);ax.axhline(V,ls='--',color=GREEN,lw=1.8,label='V(s)=5');ax.set_ylim(0,8.2);ax.set_xticks(xs,actions);ax.set_ylabel('Q(s, a)');ax.set_title('Illustrative dueling aggregation: V(s)=5, A=[−1, 0, 2]');ax.legend(frameon=False);[ax.text(i,qq+.15,f'{qq:.2f}',ha='center') for i,qq in enumerate(q)];ax.grid(axis='y',alpha=.14);save(fig,'dueling_numeric.png')

# 15. variant matrix infographic
fig,ax=fresh();ax.text(5,3.58,'Orthogonal modifications: target rule × network architecture',ha='center',color=GREEN,fontweight='bold',fontsize=14)
for x,title in [(.3,'Training target'),(5.2,'Q-network shape')]:
 ax.text(x+2.2,3.05,title,ha='center',color=GRAY,fontweight='bold')
box(ax,.6,2.09,4.0,.68,'DQN target or Double target',fs=11);box(ax,5.5,2.09,4.0,.68,'Single head or dueling heads',fs=11)
box(ax,2.0,.58,5.92,.8,'Dueling Double DQN = both improvements',fc='#E7F3F2',ec=MINT,fs=13)
arrow(ax,2.6,2.0,4.17,1.45);arrow(ax,7.6,2.0,5.82,1.45);save(fig,'variant_matrix.png')

# 16. Prioritized replay
fig,ax=plt.subplots(figsize=(9.4,3.7));err=np.array([.1,1.9,.3,1.25,.06,.72,2.6,.28]);xs=np.arange(1,9); ax.bar(xs,err,color=[MINT if i not in (1,6) else GOLD for i in range(8)]);ax.set_xlabel('Experience stored in replay');ax.set_ylabel('|TD error|');ax.set_title('Prioritized replay samples larger errors more often (illustrative)');ax.set_xticks(xs);ax.grid(axis='y',alpha=.14);save(fig,'prioritized_replay.png')

# 17. QWOP analogy - no game screenshot
fig,ax=fresh();ax.set_aspect('equal',adjustable='box'); ax.set_xlim(0,10);ax.set_ylim(0,4); ax.text(5,3.75,'QWOP-style coordination: discrete controls influence coupled motion',ha='center',color=GREEN,fontweight='bold',fontsize=12)
ax.plot([.5,9.5],[.45,.45],color=GRAY,lw=2);head=(4.7,2.95);ax.add_patch(Circle(head,.22,facecolor='#F1BE94',edgecolor=INK));ax.plot([4.7,4.55],[2.73,1.85],color=GREEN,lw=6);ax.plot([4.55,3.6],[1.85,1.18],color=MINT,lw=6);ax.plot([3.6,3.1],[1.18,.5],color=MINT,lw=6);ax.plot([4.55,5.4],[1.85,1.15],color=GOLD,lw=6);ax.plot([5.4,6.0],[1.15,.5],color=GOLD,lw=6);ax.plot([4.6,3.75],[2.55,2.15],color=GREEN,lw=4);ax.plot([4.6,5.45],[2.52,2.15],color=GREEN,lw=4);box(ax,.55,2.2,2.14,1.15,'Q / W\nthigh controls',fc='#E7F3F2',ec=MINT,fs=11);box(ax,7.0,2.2,2.18,1.15,'O / P\ncalf controls',fc='#FCF4E5',ec=GOLD,fs=11);save(fig,'qwop_coordination.png')

# 18. Pokemon map conceptual, not game capture
fig,ax=plt.subplots(figsize=(8.8,4.4));ax.set_aspect('equal');n=12;grid=np.zeros((7,n));
for j in range(7):
 for i in range(n):
  grid[j,i]=1 if (i+j*3)%5==0 else 0
ax.imshow(grid,cmap=ListedColormap(['#E9F3EA','#C2DEBB']),vmin=0,vmax=1,extent=[0,12,0,7],origin='lower');ax.plot([1,3,4,5,7,8,10],[1,1,2,4,4,5,6],lw=3.5,color=MINT,marker='o',markersize=6);ax.scatter([10],[6],s=290,marker='*',color=GOLD,zorder=6);ax.add_patch(Rectangle((4.65,3.7),.8,.8,facecolor=SOFT,edgecolor=GREEN));ax.text(5.05,4.08,'XP',ha='center',fontsize=10);ax.text(10.2,5.35,'gym',fontsize=11,color=GREEN);ax.set_xlim(0,12);ax.set_ylim(0,7);ax.set_xticks([]);ax.set_yticks([]);ax.set_title('Game-world reward shaping: exploration, experience, battles (schematic)');save(fig,'pokemon_reward_map.png')

# 19. Behavior cloning and expert supervision
fig,ax=fresh(); ax.text(5,3.55,'Learn the mapping from observations to expert actions',ha='center',color=GREEN,fontsize=14,fontweight='bold');box(ax,.35,1.25,2.3,1.23,'Expert trajectories\n(s, a*) pairs',fs=11);box(ax,3.9,1.25,2.28,1.23,'Classifier πθ(a|s)\nCross-entropy loss',fc='#E7F3F2',ec=MINT,fs=11);box(ax,7.42,1.25,2.25,1.23,'Imitation policy\nSelect actions',fc='#FCF4E5',ec=GOLD,fs=11);arrow(ax,2.72,1.9,3.83,1.9);arrow(ax,6.26,1.9,7.32,1.9);ax.text(5,.77,'Expert actions are labels; rewards are not required for behavior cloning',ha='center',color=GRAY,fontsize=10);save(fig,'behavior_cloning_flow.png')

# 20. Distribution shift trajectory
fig,ax=plt.subplots(figsize=(9.6,3.8)); t=np.linspace(0,1,160);center=.3*np.sin(2*np.pi*t);ax.fill_between(t,center-.35,center+.35,color=SOFT);ax.plot(t,center,color=GREEN,lw=2,label='Expert path');learner=center+np.where(t<.4,0,.18*(np.exp(5*(t-.4))-1));ax.plot(t,learner,color=RED,lw=2.2,label='Learner with small repeated error');ax.axvline(.44,color=GOLD,ls='--',lw=1.4);ax.text(.45,-.53,'First deviation → unfamiliar states',color=GRAY,fontsize=10);ax.set_ylim(-.95,2.8);ax.set_xlabel('Time along route');ax.set_ylabel('Lateral position');ax.set_title('Behavior cloning suffers from compounding errors');ax.legend(frameon=False,loc='upper left');ax.grid(alpha=.14);save(fig,'covariate_shift.png')

# 21. Dagger loop
fig,ax=fresh();ax.text(5,3.68,'DAgger: query expert on states visited by the current learner',ha='center',fontsize=14,fontweight='bold',color=GREEN)
box(ax,.25,1.38,2.18,1.05,'1. Roll out\nlearner policy',fc=SOFT,fs=11);box(ax,2.94,1.38,2.16,1.05,'2. Expert labels\nvisited states',fc='#FCF4E5',ec=GOLD,fs=11);box(ax,5.63,1.38,2.13,1.05,'3. Aggregate\ndataset',fc=SOFT,fs=11);box(ax,8.04,1.38,1.68,1.05,'4. Retrain',fc='#E7F3F2',ec=MINT,fs=11);arrow(ax,2.48,1.92,2.86,1.92);arrow(ax,5.14,1.92,5.55,1.92);arrow(ax,7.78,1.92,7.98,1.92); ax.plot([8.85,8.85,1.33,1.33],[1.34,.55,.55,1.32],color=GRAY,lw=1.6);arrow(ax,1.34,.55,1.34,1.34,color=GRAY,lw=1.5);ax.text(5,.19,'Repeat with updated learner; requires an expert oracle or annotator',ha='center',fontsize=10,color=GRAY);save(fig,'dagger_loop.png')

# 22. Imitation → RL integration
fig,ax=fresh();ax.text(5,3.62,'A common hybrid: demonstrations provide a head start, RL improves from reward',ha='center',fontsize=13,fontweight='bold',color=GREEN);box(ax,.3,1.34,2.6,1.15,'Expert demonstrations\n(s,a*) or (s,a,r,s′)',fs=11);box(ax,3.62,1.34,2.4,1.15,'BC pretraining\nor demo replay',fc='#FCF4E5',ec=GOLD,fs=11);box(ax,6.79,1.34,2.9,1.15,'RL interaction\nTD loss + replay',fc='#E7F3F2',ec=MINT,fs=11);arrow(ax,2.99,1.92,3.53,1.92);arrow(ax,6.12,1.92,6.70,1.92);ax.text(5,.77,'Different information: action labels versus scalar rewards',ha='center',color=GRAY,fontsize=10);save(fig,'imitation_to_rl.png')

# 23. Evaluation setup visual
fig,ax=fresh();ax.text(5,3.65,'Separate training behavior from policy evaluation',ha='center',color=GREEN,fontsize=14,fontweight='bold');box(ax,.3,1.54,2.76,1.15,'Training\nε-greedy + replay',fc='#E7F3F2',ec=MINT,fs=12);box(ax,3.64,1.54,2.72,1.15,'Freeze parameters\nset ε = 0',fc=SOFT,fs=12);box(ax,6.94,1.54,2.74,1.15,'Evaluate\nheld-out starts + seeds',fc='#FCF4E5',ec=GOLD,fs=12);arrow(ax,3.12,2.1,3.58,2.1);arrow(ax,6.42,2.1,6.88,2.1);ax.text(5,.81,'Report return, success, variance and sample cost; not training return alone',ha='center',color=GRAY,fontsize=10);save(fig,'evaluation_protocol.png')
print('Figures created:', len(list(ROOT.glob('*.png'))))
