# CS 4880 Lecture 33: Deep Q-Learning, Double, Dueling, and Imitation

Imported from `CS4880_Lecture33_Source_Project.zip` and integrated with the course's shared Beamer theme. The deck contains 46 slides and references 26 supplied figures. The remaining supplied recap figures are retained here as well.

## Build

From `courses/fall-2026/cs4880/lectures/`, with TinyTeX on `PATH`:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=../../build/cs4880/lecture-33-reinforcement-learning-ii-q-learning-exploration-approximate-dee \
  lecture-33-reinforcement-learning-ii-q-learning-exploration-approximate-dee.tex
```

The published PDF is stored under the same basename in `cs4880/pdfs/`.

## Reproduce the figures and demo

From this asset directory, using an environment with the dependencies in `requirements.txt`:

```sh
python generate-figures.py
python train-toy-dqn.py
```

These scripts write figures and result summaries into this directory. They use seed 4880 and require no external datasets or gameplay downloads. `previous_policy.png` and `agent_loop.png` are supplied Lecture 32 recap figures; the scripts do not regenerate them.

The Monte Carlo experiment simulates 20,000 trials of six independent standard-normal action estimates; it illustrates maximization bias rather than benchmarking an RL agent. The DQN experiment trains a PyTorch network for 420 episodes on a fixed 5-by-5 maze. The supplied images and measurements are retained unchanged.

Both scripts were rerun during integration on November 15, 2026, using Python 3.9.6, NumPy 2.0.2, Matplotlib 3.9.4, and PyTorch 2.8.0 on CPU. Both text summaries reproduced exactly:

- Monte Carlo mean maximum: 1.2689; independent evaluation mean: -0.0017.
- DQN training: 4,425 environment steps; greedy success from all 21 nonterminal starting cells.
- Fixed-start rollout: eight actions along the route recorded in `measured-results.txt`.

Results can change with numerical dependencies or hardware. Evaluation starts are the same cells available during training, so this measures control on the known maze rather than generalization to held-out maps or states. QWOP and Pokemon remain motivating cases from the supplied source; the slides do not assert unverified implementation details for their videos.

## Suggested 50-minute pace

- Motivation and pixels to Q-values: 8 minutes.
- Replay, target network, and toy DQN: 15 minutes.
- Double DQN and maximization bias: 8 minutes.
- Dueling networks and combinations: 8 minutes.
- Imitation, DAgger, and demonstrations with RL: 9 minutes.
- Synthesis: 2 minutes.
