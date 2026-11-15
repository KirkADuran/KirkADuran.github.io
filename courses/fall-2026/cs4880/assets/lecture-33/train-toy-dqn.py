#!/usr/bin/env python3
"""Tiny *real* DQN training demonstration (not an Atari benchmark).
Reproduces toy_dqn_results.png and toy_dqn_policy.png with fixed RNG seed.
Dependencies: numpy matplotlib torch (CPU). No external downloads.
Run: python train-toy-dqn.py
"""
from pathlib import Path
from collections import deque
import random
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SEED=4880
random.seed(SEED); np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parent
GREEN='#154734';MINT='#168F85';GOLD='#B88A2B';GRAY='#63756B';RED='#B34B43';SOFT='#EAF2ED'
N=5;WALLS={(1,1),(2,1),(3,1)};GOAL=(0,4);START=(4,0);MOVES=[(-1,0),(1,0),(0,-1),(0,1)];SYMBOLS=['↑','↓','←','→'];GAMMA=.97
valid=[(r,c) for r in range(N) for c in range(N) if (r,c) not in WALLS and (r,c)!=GOAL]

def state(x):
    v=np.zeros(N*N,dtype=np.float32);v[x[0]*N+x[1]]=1.;return v

def step(s,a):
    dr,dc=MOVES[a];ns=(s[0]+dr,s[1]+dc)
    if not (0<=ns[0]<N and 0<=ns[1]<N) or ns in WALLS:
        ns=s;r=-.35
    else:r=-.15
    done=ns==GOAL
    if done:r=10.
    return ns,r,done

class QNet(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(25,48),nn.ReLU(),nn.Linear(48,48),nn.ReLU(),nn.Linear(48,4))
    def forward(self,x):return self.net(x)

online=QNet();target=QNet();target.load_state_dict(online.state_dict());optimizer=torch.optim.Adam(online.parameters(),lr=.001)
buffer=deque(maxlen=7500);ep_returns=[];eval_scores=[];global_steps=0;EPISODES=420;BATCH=64
for ep in range(EPISODES):
    s=valid[np.random.randint(len(valid))];r_sum=0.
    eps=max(.05,1.0-ep/240*0.95)
    for t in range(30):
        if random.random()<eps:a=random.randrange(4)
        else:
            with torch.no_grad():a=int(online(torch.from_numpy(state(s))).argmax().item())
        ns,reward,done=step(s,a);buffer.append((state(s),a,reward,state(ns),float(done)))
        s=ns;r_sum+=reward;global_steps+=1
        if len(buffer)>=BATCH and global_steps%2==0:
            batch=random.sample(buffer,BATCH)
            ss=torch.from_numpy(np.stack([p[0] for p in batch]));aa=torch.tensor([p[1] for p in batch],dtype=torch.int64)
            rr=torch.tensor([p[2] for p in batch],dtype=torch.float32);nnss=torch.from_numpy(np.stack([p[3] for p in batch]));dd=torch.tensor([p[4] for p in batch],dtype=torch.float32)
            pred=online(ss).gather(1,aa[:,None]).squeeze(1)
            with torch.no_grad():y=rr+GAMMA*(1-dd)*target(nnss).max(dim=1).values
            loss=nn.functional.smooth_l1_loss(pred,y)
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(online.parameters(),5.0);optimizer.step()
        if global_steps%120==0:target.load_state_dict(online.state_dict())
        if done:break
    ep_returns.append(r_sum)
    if (ep+1)%20==0:
        wins=0;terms=[]
        for ini in valid:
            loc=ini
            for k in range(30):
                with torch.no_grad():act=int(online(torch.from_numpy(state(loc))).argmax().item())
                loc,_,ended=step(loc,act)
                if ended:
                    wins+=1;terms.append(k+1);break
        eval_scores.append((ep+1,wins/len(valid)))

policy={}
for s in valid:
    with torch.no_grad(): policy[s]=int(online(torch.from_numpy(state(s))).argmax().item())
route=[START];pos=START
for _ in range(35):
    act=policy[pos];pos,_,ended=step(pos,act);route.append(pos)
    if ended:break
success=route[-1]==GOAL
print('Episodes:',EPISODES,'Steps:',global_steps,'Replay:',len(buffer),'Final eval success fraction:',eval_scores[-1][1],'Start-to-goal successful:',success,'Greedy steps:',len(route)-1)
print('Route:',route)
(ROOT/'measured-results.txt').write_text('Toy example only (not Atari); PyTorch, seed=4880\nEpisodes: '+str(EPISODES)+'\nEnvironment interaction steps: '+str(global_steps)+'\nEvaluation success over '+str(len(valid))+' allowed starting cells: '+str(eval_scores[-1][1])+'\nFixed start '+str(START)+' reached goal: '+str(success)+' in '+str(len(route)-1)+' greedy steps\nRoute: '+str(route)+'\nTraining uses epsilon-greedy; evaluation uses epsilon=0.\n')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.edgecolor':'#BCCBC3','axes.labelcolor':'#263A32','axes.titleweight':'bold','savefig.bbox':'tight','savefig.pad_inches':0.1})
fig,axes=plt.subplots(1,2,figsize=(11.2,4.0));arr=np.array(ep_returns);window=20;ma=np.convolve(arr,np.ones(window)/window,mode='valid');axes[0].plot(np.arange(1,len(arr)+1),arr,color=GOLD,alpha=.23,lw=.9,label='Episode returns');axes[0].plot(np.arange(window,len(arr)+1),ma,color=MINT,lw=2.2,label='20-episode mean');axes[0].set(xlabel='Training episode',ylabel='Undiscounted return',title='Training with ε-greedy');axes[0].legend(frameon=False,fontsize=9);axes[0].grid(alpha=.16)
axes[1].plot([p[0] for p in eval_scores],[p[1] for p in eval_scores],marker='o',color=GREEN,lw=2);axes[1].set(xlabel='Training episode',ylabel='Greedy success fraction',title='Evaluation from all legal starts',ylim=(-.03,1.06));axes[1].grid(alpha=.16);fig.tight_layout();fig.savefig(ROOT/'toy_dqn_results.png',dpi=190);plt.close(fig)
fig,ax=plt.subplots(figsize=(6.4,5.2));ax.set_xlim(0,N);ax.set_ylim(0,N);ax.set_aspect('equal');ax.axis('off')
for r in range(N):
 for c in range(N):
  y=N-1-r; cell=(r,c);fc='#EAF2ED'
  if cell in WALLS:fc='#65776D'
  if cell==GOAL:fc='#F8EDCC'
  ax.add_patch(Rectangle((c,y),1,1,facecolor=fc,edgecolor='#A7BCB1',linewidth=1.3))
  if cell==GOAL:ax.text(c+.5,y+.5,'G',ha='center',va='center',fontsize=22,color=GREEN,fontweight='bold')
  elif cell in WALLS:ax.text(c+.5,y+.5,'×',ha='center',va='center',fontsize=20,color='white')
  else:ax.text(c+.5,y+.5,SYMBOLS[policy[cell]],ha='center',va='center',fontsize=20,color=GREEN)
xs=[c+.5 for r,c in route];ys=[N-r-.5 for r,c in route]
ax.plot(xs,ys,color=GOLD,lw=3.3,alpha=.85,marker='o',markersize=5);ax.text(.1,-.22,'S = fixed starting cell; G = goal; gold line = greedy rollout',fontsize=9,color=GRAY)
ax.text(.5,.5,'S',ha='center',va='center',fontsize=9,fontweight='bold',color=RED,bbox={'facecolor':'white','edgecolor':RED,'boxstyle':'round,pad=.08'})
ax.set_title('Learned greedy Q-network policy (toy 5×5 world)',color=GREEN,fontsize=14,fontweight='bold');fig.savefig(ROOT/'toy_dqn_policy.png',dpi=190);plt.close(fig)
