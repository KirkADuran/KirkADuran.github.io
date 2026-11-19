"""Original, reproducible teaching illustrations for CS4880 Lecture 35.
No model weights are trained; all attention maps except the exact numerical
worked example are schematic. Requires numpy and matplotlib.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

A=Path(__file__).resolve().parent
GREEN='#154734'; INK='#263A32'; MUTED='#62796A'; SOFT='#EAF2ED'; GOLD='#B88A2B'; TEAL='#168F85'; BLUE='#456A9D'; ROSE='#A85867'; WHITE='#FFFFFF'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'savefig.facecolor':'white'})

def canvas(w=11,h=4.5):
 f,ax=plt.subplots(figsize=(w,h),dpi=145); ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis('off'); return f,ax

def box(ax,x,y,w,h,text,fc=SOFT,edge=GREEN,fs=12,weight='normal',c=INK):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.7,rounding_size=2.2',facecolor=fc,edgecolor=edge,lw=1.8))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,fontweight=weight,color=c,linespacing=1.35)

def arr(ax,x1,y1,x2,y2,color=TEAL,lw=2,rad=0):
 ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=15,linewidth=lw,color=color,connectionstyle=f'arc3,rad={rad}'))

def label(ax,x,y,s,size=12,color=INK,ha='center',weight='normal'):
 ax.text(x,y,s,ha=ha,va='center',fontsize=size,color=color,fontweight=weight)

def save(f,name):
 f.savefig(A/name,bbox_inches='tight',pad_inches=.15,dpi=155); plt.close(f)

# 1 Evolution of model mechanics
f,a=canvas(12,4)
items=[('RNN','state travels\nword by word'),('Seq2seq','encoder squeezes\nsource into a code'),('Attention','decoder accesses\nall source states'),('Transformer','attention replaces\nrecurrence in the block')]
for i,(t,s) in enumerate(items):
 x=3+i*25; box(a,x,39,20,35,t+'\n'+s,fc=SOFT if i<3 else '#D7E9E2',fs=11,weight='bold' if i==3 else 'normal')
 if i<3: arr(a,x+20.5,57,x+24,57)
label(a,50,87,'The same question: where does context come from?',15,GREEN,weight='bold');save(f,'history_to_transformer.png')

# 2 Bottleneck
f,a=canvas();label(a,50,89,'Original fixed-vector seq2seq',15,GREEN,weight='bold')
for i,t in enumerate(['The','robot','read','the','sign']): box(a,4+i*11,48,10,18,t,fc='#EFF4F0',fs=11)
box(a,62,48,14,18,'one\ncontext',fc='#F7E8C9',edge=GOLD,fs=11)
box(a,82,48,15,18,'decoder',fc='#DCEAF4',edge=BLUE,fs=11)
arr(a,57,57,61,57);arr(a,77,57,81,57);label(a,47,24,'Everything must fit through a fixed-size bottleneck.',13,ROSE);save(f,'seq2seq_bottleneck.png')

# 3 Full forward path
f,a=canvas(12,4);nodes=[('Text',2,12),('Tokens',17,13),('Token IDs',33,14),('Embedding\n+ position',50,18),('Transformer\nblocks',71,14),('Logits\n/ softmax',88,11)]
for t,x,w in nodes: box(a,x,40,w,26,t,fs=10 if x==88 else 11,fc=SOFT if x!=71 else '#D5EAE4')
for i in range(len(nodes)-1):x=nodes[i][1]+nodes[i][2];x2=nodes[i+1][1];arr(a,x+.5,53,x2-.4,53)
label(a,50,84,'A decoder language model: from characters to token probabilities',14,GREEN,weight='bold');save(f,'llm_pipeline.png')

# 4 toy tokens
f,a=canvas();label(a,50,90,'Tokenization depends on a learned vocabulary',15,GREEN,weight='bold')
box(a,5,45,33,28,'unbelievable!',fs=16,fc='#EAF2ED')
arr(a,40,59,52,59)
for i,t in enumerate(['un','believ','able','!']):box(a,54+i*11,48,10,23,t,fs=12,fc=['#DDEDEA','#E9E6F2','#F6E8CE','#E9EDF4'][i])
label(a,50,22,'Illustrative segmentation: not the output of a particular tokenizer',11,MUTED);save(f,'subword_example.png')

# 5 embed lookup
f,a=canvas();label(a,50,91,'Token IDs select rows of a learned embedding matrix',15,GREEN,weight='bold')
box(a,3,43,24,28,'token ID = 2',fc='#F4EAD2',edge=GOLD,fs=15)
arr(a,29,57,41,57)
for j in range(5):
  for k in range(6):
   fc='#B9DBCB' if j==2 else ('#EEF3EF' if (j+k)%2 else '#E4EDE7')
   a.add_patch(Rectangle((43+k*6.5,31+j*10.8),6,10,facecolor=fc,edgecolor='#AFC1B7',lw=.6))
box(a,84,43,13,28,'vector\nin R^d',fc='#E3EEF7',edge=BLUE,fs=11)
arr(a,82,57,83,57)
label(a,62,16,'Each row is learned by backpropagation.',11,MUTED);save(f,'embedding_lookup.png')

# 6 word order
f,a=canvas();label(a,50,88,'An attention block needs information about order',15,GREEN,weight='bold')
for row,(line,y) in enumerate([(['dog','bites','human'],55),(['human','bites','dog'],20)]):
 for i,t in enumerate(line):box(a,14+i*26,y,23,22,t,fc=['#EAF2ED','#F5ECD7','#E1EBF3'][i],fs=14)
label(a,50,7,'The same set of token embeddings, rearranged.',11,ROSE);save(f,'word_order.png')

# 7 position heatmap
p=np.arange(22)[:,None];i=np.arange(32)[None,:];pe=np.zeros((22,32));pe[:,0::2]=np.sin(p/(10000**(np.arange(0,32,2)/32))[None,:]);pe[:,1::2]=np.cos(p/(10000**(np.arange(0,32,2)/32))[None,:]);
f,ax=plt.subplots(figsize=(10,4),dpi=160);im=ax.imshow(pe,aspect='auto',cmap='BrBG',vmin=-1,vmax=1);ax.set_xlabel('embedding coordinate');ax.set_ylabel('position in sequence');ax.set_title('Original 2017 sinusoidal positional encodings',color=GREEN,fontweight='bold');f.colorbar(im,ax=ax,shrink=.8,label='encoding value');save(f,'sinusoidal_position.png')

# 8 attention intuition schematic
f,a=canvas(12,4);label(a,50,91,'Schematic, not a measured learned attention map',12,MUTED)
words=['The','robot','dropped','the','glass','because','it','was','fragile']
for i,t in enumerate(words):
 x=2+i*10.8;box(a,x,35,9.4,22,t,fc='#F7E5C8' if t in ('it','glass') else SOFT,edge=GOLD if t in ('it','glass') else GREEN,fs=10)
arr(a,2+6*10.8+4,60,2+4*10.8+4,60,ROSE,3,rad=.55)
label(a,50,16,'A context-sensitive representation can connect “it” to “glass.”',12,GREEN);save(f,'attention_intuition.png')

# 9 Q K V
f,a=canvas();box(a,4,43,23,25,'input x_i',fs=16,fc=SOFT)
for j,(sym,col) in enumerate([('Q',BLUE),('K',TEAL),('V',GOLD)]):
 y=73-30*j;arr(a,28,55,47,y,col);box(a,50,y-11,17,22,'W_'+sym,fc=WHITE,edge=col,fs=14);arr(a,68,y,79,y,col);box(a,81,y-11,16,22,sym+'_i',fc=SOFT,edge=col,fs=15)
label(a,50,94,'Three learned linear projections of the same input',14,GREEN,weight='bold');save(f,'qkv_projections.png')

# 10 QK toy attention scores
q=np.array([1.,0.]);keys=np.array([[1.,0.],[0.,1.],[1.,1.]])
scores=keys@q/np.sqrt(2); exp=np.exp(scores-scores.max()); alpha=exp/exp.sum(); vals=np.array([[2,0],[0,2],[1,1]],dtype=float); z=alpha@vals
f,a=canvas(11,4)
for j,(t,v) in enumerate(zip(['k1=(1,0)','k2=(0,1)','k3=(1,1)'],scores)):
 box(a,3+j*33,54,30,24,t+'\nscore='+format(v,'.3f'),fc=SOFT,fs=12)
 box(a,3+j*33,17,30,23,'weight='+format(alpha[j],'.3f'),fc='#F4E9D0',edge=GOLD,fs=12)
 arr(a,18+j*33,53,18+j*33,42)
label(a,50,90,'q = (1,0), scaled by sqrt(d_k) = sqrt(2)',14,GREEN,weight='bold');save(f,'scores_to_softmax.png')

# 11 a sample attention matrix
Q=np.array([[1,0],[0,1],[1,1]],float);K=Q.copy();logits=Q@K.T/np.sqrt(2);E=np.exp(logits-logits.max(axis=1,keepdims=True));P=E/E.sum(axis=1,keepdims=True)
f,ax=plt.subplots(figsize=(6.6,4.5),dpi=155);im=ax.imshow(P,vmin=0,vmax=1,cmap='YlGn');ax.set_xticks(range(3),['key 1','key 2','key 3']);ax.set_yticks(range(3),['query 1','query 2','query 3']);ax.set_title('Exact normalized weights in a 3-token toy example',fontweight='bold',color=GREEN)
for i0 in range(3):
 for j in range(3):ax.text(j,i0,f'{P[i0,j]:.3f}',ha='center',va='center',color=INK,fontsize=12)
f.colorbar(im,ax=ax,fraction=.05,pad=.04,label='attention weight');save(f,'toy_attention_heatmap.png')

# 12 weighted average
f,a=canvas();label(a,50,92,'A weighted mixture of value vectors',15,GREEN,weight='bold')
for j,(v,weight) in enumerate(zip(vals,alpha)):
 x=3+j*27;box(a,x,42,23,29,f'v{j+1} = {tuple(map(int,v))}\nw{j+1} = {weight:.3f}',fs=12,fc=SOFT)
 arr(a,x+11.5,40,76,25,color=[TEAL,BLUE,GOLD][j],rad=(j-1)*.16)
box(a,77,15,20,24,f'output z\n({z[0]:.3f}, {z[1]:.3f})',fc='#F4EAD2',edge=GOLD,fs=11)
label(a,48,12,'z = sum_j weight_j * v_j',12,MUTED);save(f,'value_weighted_sum.png')

# 13 multi-head
f,a=canvas(11,4.5);box(a,2,45,17,22,'token\nvectors',fs=12)
for j,c in enumerate([BLUE,TEAL,GOLD]):
 y=74-27*j; arr(a,20,56,34,y,c);box(a,36,y-10,22,20,'head '+str(j+1)+'\nQ, K, V',fc=SOFT,edge=c,fs=11);arr(a,59,y,70,56,c)
box(a,72,44,14,24,'concat',fc='#F2E9D8',edge=GOLD,fs=12);arr(a,87,56,91,56);box(a,92,44,7,24,'W_O',fs=11)
label(a,50,92,'Heads learn separate projections and are recombined',14,GREEN,weight='bold');save(f,'multihead_attention.png')

# 14 causal mask
f,axes=plt.subplots(1,2,figsize=(10,4),dpi=150)
mask=np.tril(np.ones((6,6)));axes[0].imshow(np.ones((6,6)),vmin=0,vmax=1,cmap='Greens');axes[1].imshow(mask,vmin=0,vmax=1,cmap='Greens');
for ax,title in zip(axes,['Bidirectional self-attention','Causal (decoder) self-attention']):
 ax.set_xticks(range(6));ax.set_yticks(range(6));ax.set_xlabel('key position');ax.set_ylabel('query position');ax.set_title(title,color=GREEN,fontsize=12)
 for r in range(6):
  for c in range(6):ax.text(c,r,'yes' if (title.startswith('Bi') or c<=r) else '×',ha='center',va='center',fontsize=9,color=INK)
save(f,'causal_mask.png')

# 15 encoder decoder cross
f,a=canvas(11,4.5);box(a,2,48,23,28,'source tokens\nencoder stack',fs=12);box(a,41,48,23,28,'target prefix\ndecoder stack',fs=12);box(a,76,48,21,28,'next-token\nlogits',fs=12,fc='#F6E9D1',edge=GOLD)
arr(a,65,62,75,62);arr(a,25,65,40,65,color=BLUE,rad=.30);label(a,33,88,'cross-attention\nkeys and values',10,BLUE)
label(a,50,18,'The 2017 Transformer used both an encoder and a decoder.',13,GREEN);save(f,'encoder_decoder.png')

# 16 model families
f,a=canvas(11,4.5);items=[('Encoder only','bidirectional\nrepresentations','classification / tagging'),('Decoder only','causal\nnext-token prediction','generation'),('Encoder-decoder','source attention\n+ target generation','translation / summarization')]
for j,(name,work,task) in enumerate(items):
 x=2+j*33;box(a,x,28,30,48,name+'\n'+work+'\n'+task,fc=SOFT if j!=1 else '#D9EEE8',fs=11)
label(a,50,91,'Transformer refers to a family of architectural choices',14,GREEN,weight='bold');save(f,'model_families.png')

# 17 Transformer block flow
f,a=canvas(7,5.5);stages=[('Input X',5),('Self-attention',23),('Add + Norm',41),('Feed-forward MLP',59),('Add + Norm',77)]
for name,y in stages:box(a,26,y,49,14,name,fc='#F5E7C9' if 'Add' in name else SOFT,edge=GOLD if 'Add' in name else GREEN,fs=12)
for _,y in stages[:-1]:arr(a,50.5,y+14.5,50.5,y+17)
a.plot([14,14,25],[12,48,48],color=BLUE,lw=1.5);arr(a,14,48,25,48,BLUE)
a.plot([19,19,25],[48,84,84],color=BLUE,lw=1.5);arr(a,19,84,25,84,BLUE)
label(a,11,93,'residual\npaths',10,BLUE);label(a,50,97,'Schematic post-norm block (2017 paper)',13,GREEN,weight='bold');save(f,'transformer_block.png')

# 18 position wise FFN
f,a=canvas();label(a,50,88,'The same MLP is applied independently at each position',14,GREEN,weight='bold')
for j in range(4):
 x=7+j*23;box(a,x,56,18,18,'token '+str(j+1),fs=11);arr(a,x+9,55,x+9,41);box(a,x,20,18,21,'Linear\nnonlinear\nLinear',fc='#E8EEF7',edge=BLUE,fs=10)
label(a,50,8,'Attention mixes tokens; the FFN transforms features at each token.',11,MUTED);save(f,'positionwise_ffn.png')

# 19 distribution example
log=np.array([.2,.3,.9]);prob=np.exp(log-log.max());prob/=prob.sum()
f,ax=plt.subplots(figsize=(8.8,4),dpi=150);bars=ax.bar(['dog','floor','mat'],prob,color=[BLUE,TEAL,GOLD]);ax.set_ylim(0,.55);ax.set_ylabel('softmax probability');ax.set_title('Illustrative three-token vocabulary: logits [0.2, 0.3, 0.9]',color=GREEN,fontweight='bold');
for b,pv in zip(bars,prob):ax.text(b.get_x()+b.get_width()/2,pv+.018,f'{pv:.3f}',ha='center',fontsize=12)
ax.spines[['top','right']].set_visible(False);save(f,'softmax_distribution.png')

# 20 training label shift
f,a=canvas(11,4);label(a,50,90,'Teacher forcing: the target is shifted one token',15,GREEN,weight='bold')
tokens=['<BOS>','The','cat','sat','<EOS>'];targets=['The','cat','sat','<EOS>','--']
for i,(t,tgt) in enumerate(zip(tokens,targets)):
 x=2+i*19;box(a,x,51,17,19,t,fc=SOFT,fs=11);box(a,x,15,17,19,tgt,fc='#F4E6CD' if i<4 else WHITE,edge=GOLD if i<4 else MUTED,fs=11)
 if i<4: arr(a,x+8.5,50,x+8.5,36,TEAL)
label(a,2,78,'input',10,MUTED,ha='left');label(a,2,7,'target',10,MUTED,ha='left');save(f,'teacher_forcing.png')

# 21 generate loop
f,a=canvas(11,4);box(a,2,42,34,25,'prompt + previous tokens',fs=13);box(a,44,42,28,25,'causal transformer',fs=13);box(a,79,42,19,25,'sample token',fs=13,fc='#F7E8CC',edge=GOLD);arr(a,37,54,43,54);arr(a,73,54,78,54)
a.plot([88,88,18,18],[41,21,21,41],color=TEAL,lw=2);arr(a,18,22,18,41)
label(a,50,84,'Inference is sequential even when training positions are parallelized',13,GREEN,weight='bold');save(f,'autoregressive_loop.png')

# 22 attention scaling
ns=np.array([128,256,512,1024,2048,4096]);pairs=ns**2
f,ax=plt.subplots(figsize=(8,3.8),dpi=150);ax.plot(ns,pairs/1e6,'-o',color=TEAL,lw=2.5);ax.set_xlabel('context length n (tokens)');ax.set_ylabel('score matrix entries (millions)');ax.set_title('Full attention has n x n scores per head',color=GREEN,fontweight='bold');ax.grid(alpha=.25);ax.spines[['top','right']].set_visible(False);save(f,'quadratic_attention.png')

# 23 bank cases
f,a=canvas();label(a,50,88,'A token can have different representations in different contexts',14,GREEN,weight='bold')
box(a,3,50,46,25,'He deposited money\nat the bank.',fc='#E8EEF6',edge=BLUE,fs=13);box(a,51,50,46,25,'They rested beside\nthe river bank.',fc='#E8F1E9',edge=TEAL,fs=13)
box(a,22,15,25,17,'bank: finance',fc='#E7EDF7',edge=BLUE,fs=11);box(a,53,15,25,17,'bank: geography',fc='#E6F3EA',edge=TEAL,fs=11);arr(a,26,49,33,33,BLUE);arr(a,74,49,65,33,TEAL);save(f,'contextual_bank.png')

print('Figures:',len(list(A.glob('*.png'))));print('Toy scores',scores.round(4),'alpha',alpha.round(4),'z',z.round(4),'logit probs',prob.round(4))
