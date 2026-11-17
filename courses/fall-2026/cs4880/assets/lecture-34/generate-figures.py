#!/usr/bin/env python3
"""Rebuild original pedagogical figures for CS4880 Lecture 34 (NLP).
No network access or pretrained models. All numeric examples are labelled illustrative.
"""
from pathlib import Path
import math, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle, Polygon
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
ROOT.mkdir(parents=True, exist_ok=True)
G='#154734'; D='#263A32'; M='#63756B'; S='#EAF2ED'; A='#168F85'; GOLD='#B88A2B'; RED='#AD583B'; BLUE='#315D90'; PALE='#F8FAF8'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})

def fig(name, sz=(10,4.0)):
    f,ax=plt.subplots(figsize=sz);ax.set_xlim(0,10);ax.set_ylim(0,5);ax.axis('off'); return f,ax

def save(f,name):
    f.savefig(ROOT/name,bbox_inches='tight',pad_inches=.13,dpi=180);plt.close(f)

def box(ax,x,y,w,h,t,fill=S,edge=G,fs=12,weight='normal',color=D,round=.12):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0.02,rounding_size={round}',lw=1.6,edgecolor=edge,facecolor=fill)
    ax.add_patch(p);ax.text(x+w/2,y+h/2,t,ha='center',va='center',color=color,fontsize=fs,weight=weight,linespacing=1.3)

def arrow(ax,x1,y1,x2,y2,c=G,lw=2,style='-|>',rad=0):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,mutation_scale=15,lw=lw,color=c,connectionstyle=f'arc3,rad={rad}'))

def txt(ax,x,y,t,fs=12,c=D,ha='center',weight='normal'):
    ax.text(x,y,t,fontsize=fs,color=c,ha=ha,va='center',weight=weight)

# (1) motivating ambiguity
f,ax=fig('ambiguity', (10,4))
box(ax,.4,2.8,9.2,1.0,'"I saw her duck."',fill=S,fs=26,weight='bold')
box(ax,.45,.8,4.1,1.25,'duck = a bird\nI noticed the bird she owns.',fill='#F0F5F1',edge=A,fs=14)
box(ax,5.45,.8,4.1,1.25,'duck = lower her head\nI watched her move.',fill='#FFF7E9',edge=GOLD,fs=14)
arrow(ax,3.4,2.8,2.55,2.13,A);arrow(ax,6.6,2.8,7.55,2.13,GOLD)
save(f,'ambiguity.png')

# (2) historical milestones, spacious two-row timeline
f,ax=fig('timeline',(11,4.7))
steps=[('1951','Shannon\nprediction'),('1956','Dartmouth\nAI workshop'),('1957','formal syntax\n(Chomsky)'),('1966','ELIZA\npattern matching'),('1990s','corpus +\nstatistical NLP'),('2013','word2vec'),('2014-17','seq2seq, attention,\ntransformers')]
pos=[(.18,3.0),(2.73,3.0),(5.28,3.0),(7.83,3.0),(1.00,1.20),(4.00,1.20),(7.00,1.20)]
for i,((yr,description),(x,y)) in enumerate(zip(steps,pos)):
 box(ax,x,y,2.05,1.15,yr+'\n'+description,fill=S if i%2==0 else '#FFF8E9',edge=G if i%2==0 else GOLD,fs=12,weight='normal')
for i in range(3):arrow(ax,pos[i][0]+2.08,3.58,pos[i+1][0]-.05,3.58,c=A,lw=1.5)
for i in range(4,6):arrow(ax,pos[i][0]+2.08,1.78,pos[i+1][0]-.07,1.78,c=A,lw=1.5)
txt(ax,5,.38,'Chronology is simplified: symbolic and statistical research overlapped.',fs=10,c=M)
save(f,'timeline.png')

# (3) symbolic grammar pipeline
f,ax=fig('grammar_pipeline')
for i,(x,t) in enumerate([(0.2,'Words'),(2.25,'Grammar rules'),(4.45,'Parse tree'),(6.75,'Interpretation'),(8.7,'Action')]):
 box(ax,x,2.0,1.6 if i!=3 else 1.65,1.0,t,fill=S if i%2==0 else '#FFF8E9',fs=12,weight='bold')
 if i>0:arrow(ax,x-.43,2.5,x-.05,2.5,c=A)
txt(ax,5,1.1,'A symbolic system tries to turn language into structure and meaning.',fs=13)
save(f,'grammar_pipeline.png')

# (4) syntax tree
f,ax=fig('parse_tree',(10,4.6))
nodes={'S':(5,4.4),'NP':(2.75,3.55),'VP':(6.8,3.55),'Det1':(1.45,2.65),'N1':(3.35,2.65),'V':(5.5,2.65),'NP2':(8.2,2.65),'Det2':(7.2,1.75),'N2':(9.05,1.75)}
for a,b in [('S','NP'),('S','VP'),('NP','Det1'),('NP','N1'),('VP','V'),('VP','NP2'),('NP2','Det2'),('NP2','N2')]:
 x1,y1=nodes[a];x2,y2=nodes[b];ax.plot([x1,x2],[y1-.17,y2+.18],color=M,lw=1.8)
for key,(x,y) in nodes.items():
 ax.add_patch(Circle((x,y),.24,facecolor=S,edgecolor=G,lw=1.4))
 ax.text(x,y,key.replace('1','').replace('2',''),fontsize=12,ha='center',va='center',weight='bold',color=G)
for x,t in [(1.45,'the'),(3.35,'robot'),(5.5,'sees'),(7.2,'the'),(9.05,'dog')]:
 txt(ax,x,.87,t,14,c=D)
for leaf,x in [('Det1',1.45),('N1',3.35),('V',5.5),('Det2',7.2),('N2',9.05)]:
 xs,ys=nodes[leaf];ax.plot([xs,x],[ys-.25,1.07],lw=1.4,c=M)
ax.set_ylim(.5,4.9)
save(f,'parse_tree.png')

# (5) ambiguity attachment
f,ax=fig('parse_ambiguity',(11,4.6))
box(ax,.2,3.85,10,.75,'"I saw the man with a telescope."',fill=S,fs=21,weight='bold')
box(ax,.6,1.35,4.2,1.45,'Attach "with a telescope"\nto the verb phrase:\nI used a telescope to see him.',fill='#EEF6F1',edge=A,fs=13)
box(ax,5.5,1.35,4.2,1.45,'Attach "with a telescope"\nto the noun phrase:\nHe had the telescope.',fill='#FFF8E9',edge=GOLD,fs=13)
arrow(ax,3.8,3.83,2.9,2.75,c=A);arrow(ax,6.7,3.83,7.7,2.75,c=GOLD)
txt(ax,5.0,.72,'Both structures can be grammatical. Syntax alone may not resolve meaning.',fs=11)
save(f,'parse_ambiguity.png')

# (6) ELIZA rule decomposition/reassembly
f,ax=fig('eliza_flow',(11,3.8));ax.set_xlim(0,11)
box(ax,.2,2.35,2.25,1.0,'Input:\n"I feel worried"',fs=13)
box(ax,3.0,2.35,2.25,1.0,'Match pattern:\n"I feel *"',fill='#FFF8E9',edge=GOLD,fs=13)
box(ax,5.8,2.35,2.25,1.0,'Capture:\n* = worried',fs=13)
box(ax,8.55,2.35,2.25,1.0,'Reassemble:\n"Why do you feel *?"',fill='#FFF8E9',edge=GOLD,fs=12)
for a,b in [(2.45,3),(5.25,5.8),(8.05,8.55)]:arrow(ax,a,2.85,b,2.85)
box(ax,2.0,.48,7.1,.88,'Output: "Why do you feel worried?"',fs=17,fill='#ECF5F1')
arrow(ax,9.65,2.35,8.1,1.42,c=A,rad=0)
save(f,'eliza_flow.png')

# (7) eliza invented exchange
f,ax=fig('eliza_dialog',(11,4.4));ax.set_xlim(0,11)
for y,w,msg,fill,edge in [(3.4,'USER','I think my friend dislikes me.',S,G),(2.4,'ELIZA','Why do you think your friend dislikes you?','#FFF8E9',GOLD),(1.4,'USER','Because they never call.',S,G),(.4,'ELIZA','Tell me more about that.','#FFF8E9',GOLD)]:
 box(ax,.5,y,1.25,.78,w,fill=fill,edge=edge,fs=11,weight='bold')
 box(ax,1.85,y,8.7,.78,msg,fill=fill,edge=edge,fs=13)
save(f,'eliza_dialog.png')

# (8) tokenization boundaries
f,ax=fig('token_levels',(11,4.0))
txt(ax,.3,4.45,'RAW TEXT',fs=12,ha='left',weight='bold');box(ax,2.1,4.12,7.3,.65,'"Unbelievable robots!"',fs=20,weight='bold')
txt(ax,.3,3.05,'WORDS',fs=12,ha='left',weight='bold')
for x,w,t in [(2.1,4.0,'Unbelievable'),(6.2,2.2,'robots'),(8.5,.9,'!')]:box(ax,x,2.65,w,.73,t,fill=S,fs=14)
txt(ax,.3,1.6,'SUBWORDS',fs=12,ha='left',weight='bold')
for x,w,t in [(2.1,1.5,'Un'),(3.7,2.55,'believ'),(6.35,1.4,'able'),(7.85,1.2,'robot'),(9.09,.52,'s'),(9.69,.28,'!')]:box(ax,x,1.17,w,.77,t,fill='#FFF8E9',edge=GOLD,fs=12)
txt(ax,5.8,.55,'Illustrative segmentation; actual tokenizers depend on vocabulary and rules.',fs=10,c=M)
ax.set_ylim(.3,4.9);save(f,'token_levels.png')

# (9) subword unknown
f,ax=fig('oov_subwords',(10,3.7))
box(ax,.3,3.1,3.3,.9,'Known word: "robot"',fs=15)
box(ax,.3,1.4,3.3,.9,'Rare word: "robotics"',fill='#FFF8E9',edge=GOLD,fs=15)
arrow(ax,4.0,1.85,5.15,1.85,A)
for x,w,t in [(5.45,1.85,'robot'),(7.50,1.25,'ic'),(8.95,.65,'s')]:box(ax,x,1.4,w,.9,t,fill=S,fs=14)
txt(ax,7.4,3.4,'Smaller inventory can cover more word forms.',fs=14)
txt(ax,5.0,.52,'This is a didactic split, not a claimed output of a specific BPE model.',fs=11,c=M)
save(f,'oov_subwords.png')

# (10) ngrams window
f,ax=fig('ngram_windows',(10,4))
words=['the','robot','sees','the','dog']
xs=[1.15,3.0,4.85,6.7,8.55]
for x,w in zip(xs,words):box(ax,x-.7,2.55,1.4,.75,w,fs=15)
for indices,y,lab,c in [([1,2],1.8,'bigram: robot sees',A),([0,1,2],1.05,'trigram: the robot sees',GOLD)]:
 xl=xs[min(indices)]-.78;xr=xs[max(indices)]+.78
 ax.plot([xl,xl,xr,xr],[y+.18,y,xr*0+y,y+.18],color=c,lw=3)
 txt(ax,(xl+xr)/2,y-.21,lab,fs=11,c=c)
save(f,'ngram_windows.png')

# (11) bigram count table as image
corpus=['the robot sees the dog','the robot sees a cat','the robot likes the dog','the dog sees the robot','the cat likes the robot']
from collections import Counter
prev=Counter(); pairs=Counter()
for s in corpus:
 t=s.split();prev.update(t[:-1]);pairs.update(zip(t[:-1],t[1:]))
assert prev['the']==sum(n for (a,b),n in pairs.items() if a=='the')
items=[('robot',pairs['the','robot']),('dog',pairs['the','dog']),('cat',pairs['the','cat'])]
f,ax=fig('ngram_counts',(9,4.5))
box(ax,.4,3.9,8.2,.78,'A tiny 5-sentence toy corpus',fill=S,fs=19,weight='bold')
txt(ax,1.2,3.15,'After "the"',fs=14,ha='left',weight='bold')
for i,(w,cnt) in enumerate(items):
 y=2.3-.72*i
 box(ax,1.4,y,2.7,.52,w,fill='#EEF6F1',fs=13)
 box(ax,4.45,y,2.0,.52,str(cnt),fill='#FFF8E9',edge=GOLD,fs=14)
 box(ax,6.75,y,1.6,.52,f'{cnt}/{prev["the"]}',fill=S,fs=13)
txt(ax,5.35,.27,f'Count("the", next word) / Count("the") ; denominator = {prev["the"]}',fs=11)
save(f,'ngram_counts.png')

# (12) sparsity diagram
f,ax=fig('ngram_sparsity',(10,3.8))
box(ax,.3,3.15,9.4,.77,'Language has many plausible continuations, few repeated contexts.',fs=16,weight='bold')
for x,t,c in [(.55,'the robot sees','seen many times'),(3.75,'the robotic turtle','never observed'),(6.95,'a green submarine','never observed')]:
 box(ax,x,1.95,2.45,.75,t,fill=S if x==.55 else '#FFF4EE',edge=G if x==.55 else RED,fs=12)
 txt(ax,x+1.2,1.28,c,fs=11,c=G if x==.55 else RED)
txt(ax,4.9,.55,'Longer contexts improve specificity but make unseen sequences more likely.',fs=12)
save(f,'ngram_sparsity.png')

# (13) onehot matrix
f,ax=fig('one_hot',(9,5))
vocab=['cat','dog','robot','runs','sees','the'];M1=np.eye(6,dtype=int)
ax2=f.add_axes([.26,.19,.52,.65]);ax2.imshow(M1,cmap=matplotlib.colors.ListedColormap(['#F0F4F1',G]),vmin=0,vmax=1)
ax2.set_xticks(range(6));ax2.set_xticklabels(vocab,rotation=35,ha='right',fontsize=11)
ax2.set_yticks(range(6));ax2.set_yticklabels(vocab,fontsize=11)
for (i,j),v in np.ndenumerate(M1):ax2.text(j,i,str(v),ha='center',va='center',fontsize=12,color='white' if v else M)
ax2.set_title('One position per vocabulary item',color=G,fontsize=16,pad=14)
save(f,'one_hot.png')

# (14) onehot geometry vs dense geometry
f,ax=plt.subplots(1,2,figsize=(11,4.0))
for a in ax:a.set_aspect('equal');a.spines[['top','right']].set_visible(False);a.grid(alpha=.18)
ax[0].scatter([0,1,0],[0,0,1],s=110,c=[G,A,GOLD]);
for x,y,t in [(0,0,'dog'),(1,0,'cat'),(0,1,'car')]:ax[0].annotate(t,(x,y),xytext=(7,8),textcoords='offset points',fontsize=12)
ax[0].set_xlim(-.35,1.4);ax[0].set_ylim(-.35,1.4);ax[0].set_title('One-hot: equally distant basis vectors',color=G,fontsize=13)
points={'dog':(-.5,.4),'cat':(-.28,.36),'puppy':(-.52,.17),'car':(.6,-.42),'truck':(.85,-.47),'vehicle':(.64,-.64)}
for name,(x,y) in points.items():ax[1].scatter(x,y,s=110,c=A if x<0 else GOLD);ax[1].annotate(name,(x,y),xytext=(5,5),textcoords='offset points',fontsize=11)
ax[1].set_xlim(-.9,1.15);ax[1].set_ylim(-.9,.9);ax[1].set_title('Dense: possible learned neighborhoods',color=G,fontsize=13)
f.text(.5,.005,'Dense coordinates are illustrative, not measurements from a trained model.',ha='center',fontsize=10,color=M)
save(f,'geometry.png')

# (15) word2vec skipgram and cbow
f,ax=fig('word2vec_objectives',(11,4.5))
box(ax,.35,3.8,4.45,.75,'Skip-gram',fill=S,fs=18,weight='bold')
box(ax,5.2,3.8,4.45,.75,'CBOW',fill=S,fs=18,weight='bold')
box(ax,1.8,2.1,1.55,.68,'robot',fs=15)
for x,y,t in [(.6,.68,'the'),(2.05,.68,'sees'),(3.45,.68,'the')]:box(ax,x,y,1.1,.62,t,fill='#FFF8E9',edge=GOLD,fs=13);arrow(ax,2.58,2.1,x+.55,1.3,c=A)
for x,y,t in [(5.55,2.18,'the'),(7.0,2.18,'sees'),(8.45,2.18,'the')]:box(ax,x,y,1.05,.62,t,fill='#FFF8E9',edge=GOLD,fs=12);arrow(ax,x+.52,2.18,7.5,1.45,c=A)
box(ax,6.67,.53,1.7,.68,'robot',fs=15)
txt(ax,2.55,.12,'predict context from center',fs=11,c=M);txt(ax,7.5,.12,'predict center from context',fs=11,c=M)
save(f,'word2vec_objectives.png')

# (16) embedding lookup
f,ax=fig('embedding_lookup',(10,4))
box(ax,.4,2.1,1.8,1,'Token ID\n2: robot',fs=13)
arrow(ax,2.23,2.6,3.13,2.6)
for i in range(5):
 y=3.6-i*.66
 for j in range(6):
  val=np.random.default_rng(i*10+j+44).uniform(0,1)
  ax.add_patch(Rectangle((3.25+j*.53,y),.50,.52,facecolor=matplotlib.colors.to_rgba(G,alpha=.12+.72*val),edgecolor='white'))
ax.add_patch(Rectangle((3.2,2.24),3.2,.60,fill=False,edgecolor=GOLD,lw=3))
txt(ax,4.82,.55,'Embedding matrix E: vocabulary size x dimension',fs=12)
arrow(ax,6.5,2.62,7.25,2.62)
box(ax,7.37,2.15,2.25,.95,'Dense vector\n[0.2, -0.3, ...]',fs=12,fill='#FFF8E9',edge=GOLD)
save(f,'embedding_lookup.png')

# (17) polysemy failure
f,ax=fig('polysemy',(10,3.8))
box(ax,.45,3.06,3.9,.95,'"deposit money at the bank"',fs=13)
box(ax,5.6,3.06,3.9,.95,'"sit on the river bank"',fill='#FFF8E9',edge=GOLD,fs=13)
box(ax,3.98,1.38,2.05,.94,'bank\nstatic vector',fs=14,weight='bold')
arrow(ax,2.5,3.06,4.73,2.35,c=A);arrow(ax,7.5,3.06,5.27,2.35,c=GOLD)
txt(ax,5,.58,'One type-level embedding cannot choose a sense from sentence context.',fs=13)
save(f,'polysemy.png')

# (18) sequence order
f,ax=fig('sequence_order',(10,3.5))
box(ax,.35,2.85,4.25,.82,'"dog bites person"',fill=S,fs=17)
box(ax,5.4,2.85,4.25,.82,'"person bites dog"',fill='#FFF8E9',edge=GOLD,fs=17)
for x,y,t in [(1.3,1.6,'dog'),(3.2,1.6,'person'),(6.25,1.6,'person'),(8.15,1.6,'dog')]:box(ax,x,y,1.35,.67,t,fs=12)
arrow(ax,2.68,1.95,3.12,1.95,c=A);arrow(ax,7.6,1.95,8.05,1.95,c=GOLD)
txt(ax,5,.65,'Shared word inventory does not imply shared meaning.',fs=14)
save(f,'sequence_order.png')

# (19) recurrent update
f,ax=fig('rnn_chain',(11,4.0));ax.set_xlim(0,11)
xs=[1.4,4.2,7.0,9.8]
for i,x in enumerate(xs):
 box(ax,x-.55,2.2,1.1,.85,f'h{i+1}',fill=S,fs=18,weight='bold')
 arrow(ax,x,1.17,x,2.2,c=A)
 if i>0:arrow(ax,xs[i-1]+.55,2.62,x-.55,2.62,c=G)
for i,w in enumerate(['the','robot','saw','me']):txt(ax,xs[i],.95,w,fs=15)
txt(ax,5.5,4.1,'A recurrent state summarizes everything read so far.',fs=15,weight='bold')
save(f,'rnn_chain.png')

# (20) seq2seq fixed encoder decoder
f,ax=fig('seq2seq',(12,4.8));ax.set_xlim(0,12)
txt(ax,2.4,4.42,'ENCODER',fs=16,weight='bold',c=G);txt(ax,9.15,4.42,'DECODER',fs=16,weight='bold',c=G)
source=[('the',.7),('cat',1.75),('sleeps',2.8)]
for i,(w,x) in enumerate(source):
 box(ax,x,2.55,.92,.85,f'h{i+1}',fs=14)
 txt(ax,x+.46,1.95,w,fs=12)
 if i>0:arrow(ax,source[i-1][1]+.92,2.97,x,2.97)
box(ax,4.65,2.58,1.25,.79,'c',fill='#FFF8E9',edge=GOLD,fs=18,weight='bold')
arrow(ax,3.75,2.97,4.65,2.97,c=GOLD)
arrow(ax,5.9,2.97,7.05,2.97,c=GOLD)
for i,(w,x) in enumerate([('le',7.05),('chat',8.45),('dort',9.85)]):
 box(ax,x,2.55,1.02,.85,f's{i+1}',fs=14,fill='#EEF5F8',edge=BLUE)
 txt(ax,x+.51,1.95,w,fs=12)
 if i>0:arrow(ax,x-0.35,2.97,x,2.97,c=BLUE)
txt(ax,5.28,.96,'A fixed-length vector must summarize the whole source sentence.',fs=12)
save(f,'seq2seq.png')

# (21) training teacher forcing
f,ax=fig('teacher_forcing',(11,4.4));ax.set_xlim(0,11)
txt(ax,5.5,4.42,'Decoder predicts the next token, conditioned on preceding tokens.',fs=16,weight='bold')
for i,(lab,x) in enumerate([('<START>',.6),('the',3.15),('cat',5.70),('sleeps',8.25)]):
 box(ax,x,2.23,1.75,.8,lab,fill=S,fs=14)
 if i<3:arrow(ax,x+1.75,2.62,x+2.45,2.62,c=A)
for i,(lab,x) in enumerate([('predict: the',.55),('predict: cat',3.1),('predict: sleeps',5.65),('predict: <END>',8.2)]):
 box(ax,x,.70,1.85,.67,lab,fill='#FFF8E9',edge=GOLD,fs=11)
 arrow(ax,x+.9,2.2,x+.9,1.38,c=GOLD)
save(f,'teacher_forcing.png')

# (22) bottleneck
f,ax=fig('context_bottleneck',(11,4.4));ax.set_xlim(0,11)
for i in range(9):
 x=.45+i*.68
 box(ax,x,2.94,.56,.56,f'x{i+1}',fs=10)
arrow(ax,6.9,3.21,7.45,3.21)
box(ax,7.52,2.64,1.0,1.12,'c',fill='#FFF8E9',edge=GOLD,fs=26,weight='bold')
arrow(ax,8.6,3.21,9.28,3.21)
box(ax,9.4,2.86,1.2,.72,'decoder',fill='#EEF5F8',edge=BLUE,fs=11)
txt(ax,5.5,1.5,'Many source details compete for one fixed-size context.',fs=17,weight='bold')
txt(ax,5.5,.65,'For long or complex sentences, useful information may be lost.',fs=13)
save(f,'context_bottleneck.png')

# (23) attention preview
f,ax=fig('attention_preview',(11,4.6))
for i,(word,x) in enumerate([('the',.65),('tired',2.55),('student',4.45),('laughed',6.35)]):
 box(ax,x,2.92,1.65,.76,word,fill=S,fs=13)
arrow(ax,1.45,2.92,7.25,1.45,c=A,lw=1.3)
arrow(ax,3.32,2.92,7.25,1.45,c=A,lw=2.5)
arrow(ax,5.27,2.92,7.25,1.45,c=GOLD,lw=4)
arrow(ax,7.18,2.92,7.25,1.45,c=A,lw=1.5)
box(ax,6.53,.53,2.1,.75,'output word?',fill='#FFF8E9',edge=GOLD,fs=14)
arrow(ax,7.5,1.45,7.5,1.31,c=GOLD)
txt(ax,8.98,3.53,'Different source tokens\ncan matter differently',fs=11,ha='center')
save(f,'attention_preview.png')

# (24) limitations synthesis matrix
f,ax=fig('failure_ladder',(11,4.5));ax.set_xlim(0,11)
methods=[('Grammar rules','Brittle rules / ambiguity'),('N-gram counts','Sparse long contexts'),('One-hot vectors','No similarity geometry'),('word2vec','Static, context-free senses'),('Basic seq2seq','Compression + sequential cost')]
for i,(a,b) in enumerate(methods):
 y=3.9-i*.77
 box(ax,.65,y,3.55,.63,a,fill=S,fs=13,weight='bold')
 box(ax,4.5,y,5.85,.63,b,fill='#FFF8E9',edge=GOLD,fs=13)
save(f,'failure_ladder.png')

# (25) hand-crafted unigram frequencies for limitations
f,ax=plt.subplots(figsize=(9,3.5));ws=['the','robot','sees','dog','never-seen'];vals=[8,4,3,2,0]
ax.bar(ws,vals,color=[G,G,G,G,RED]);ax.set_ylabel('Toy corpus count');ax.set_ylim(0,9);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.17)
for i,n in enumerate(vals):ax.text(i,n+.15,str(n),ha='center',fontsize=12)
f.text(.5,.015,'A token absent from the training corpus receives zero raw count.',ha='center',fontsize=10,color=M)
save(f,'unseen_counts.png')

# (26) analogy motivation and caveat
f,ax=fig('word2vec_neighbors',(10,4.2))
ax.add_patch(Circle((2.7,2.7),1.30,facecolor='#EAF2ED',edgecolor=A,lw=1.6,alpha=.9))
ax.add_patch(Circle((7.3,2.7),1.30,facecolor='#FFF6E7',edgecolor=GOLD,lw=1.6,alpha=.9))
for x,y,t in [(2.0,2.65,'doctor'),(3.1,3.2,'nurse'),(2.9,2.1,'hospital'),(6.6,2.65,'river'),(7.7,3.2,'stream'),(7.55,2.1,'water')]:txt(ax,x,y,t,fs=11)
txt(ax,5,.65,'Co-occurrence can organize useful neighbors. It does not guarantee factual meaning.',fs=12)
save(f,'word2vec_neighbors.png')

# (27) full NLP task pipeline
f,ax=fig('pipeline',(11,3.8));ax.set_xlim(0,11)
parts=[(.20,1.65,'Text'),(2.1,1.65,'Tokens'),(4.2,1.65,'Vectors'),(6.3,1.65,'Sequence model'),(8.85,1.65,'Output')]
for i,(x,y,t) in enumerate(parts):
 w=1.55 if i!=3 else 2.12
 box(ax,x,y,w,.86,t,fill=S if i%2==0 else '#FFF8E9',edge=G if i%2==0 else GOLD,fs=13,weight='bold')
 if i>0:arrow(ax,x-.33,2.08,x-.05,2.08,c=A)
txt(ax,5.5,3.65,'An NLP system must choose representations AND how context is modeled.',fs=15)
save(f,'pipeline.png')
print('Generated',len(list(ROOT.glob('*.png'))),'figures at',ROOT)
print('Toy bigram denominator Count(the) =',prev['the'], '; next token counts =',items)
