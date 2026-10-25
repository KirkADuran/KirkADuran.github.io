"""Reproduce the single print-job case used throughout Lecture 26.

Requires numpy, matplotlib and scikit-learn. Training, validation and test
samples are fixed independently before fitting. Model choice uses validation
MSE only; the selected model is frozen before its test results are computed.
All outputs are written beside this script. No network access is needed.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

OUT = Path(__file__).resolve().parent
GREEN, GOLD, BLUE, INK, GRAY = "#154734", "#BD8B13", "#427C9E", "#1F2933", "#75828A"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 14, "axes.labelsize": 15,
    "axes.titlesize": 16, "axes.titleweight": "bold", "axes.edgecolor": INK,
    "axes.labelcolor": INK, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 12,
    "lines.linewidth": 2.5, "figure.constrained_layout.use": True,
    "savefig.facecolor": "white",
})

def save(fig, filename):
    fig.savefig(OUT / filename, dpi=210, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)

def label(ax, ylabel="Completion time (min)"):
    ax.set_xlabel("Job size (ten-page batches)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.15)

def generator(x):
    return 8 + .45*x + .12*x**2

x = np.arange(1, 13, dtype=float)
y = generator(x) + np.random.default_rng(23).normal(0, .65, len(x))
def fresh(seed):
    rng = np.random.default_rng(seed)
    z = rng.uniform(1, 12, 100)
    return z, generator(z) + rng.normal(0, .65, len(z))
xv, yv = fresh(24)
xt, yt = fresh(25)
q = np.linspace(1, 12, 500)
models = {degree: Polynomial.fit(x, y, degree) for degree in range(10)}

def metrics(model, xx, yy):
    error = yy - model(xx)
    mse = float(np.mean(error**2))
    return {"mse": mse, "rmse": float(np.sqrt(mse)),
            "mae": float(np.mean(abs(error))),
            "r2": float(1 - np.sum(error**2)/np.sum((yy-yy.mean())**2))}

training = {d: metrics(m, x, y) for d, m in models.items()}
validation = {d: metrics(m, xv, yv) for d, m in models.items()}
candidates = [0, 1, 2, 9]
selected = min(candidates, key=lambda d: validation[d]["mse"])
assert selected == 2
assert np.all(np.diff([training[d]["mse"] for d in models]) < 1e-9)
assert np.isclose(models[0](0), y.mean())
# The test sample has not been used above. Report only the frozen winner and baseline.
test = {d: metrics(models[d], xt, yt) for d in [0, selected]}
assert test[selected]["mse"] < test[0]["mse"]

fig, ax = plt.subplots(figsize=(6.2, 4))
ax.scatter(x, y, color=GOLD, edgecolor=INK, s=55, zorder=4)
label(ax)
ax.set_xticks([1, 3, 6, 9, 12])
save(fig, "case-data.png")

fig, ax = plt.subplots(figsize=(6.2, 4))
ax.scatter(x, y, color=GOLD, edgecolor=INK, s=45, zorder=4)
ax.plot(q, 8 + 1.3*q, color=GRAY, ls="--", label="Candidate A")
ax.plot(q, 4 + 2*q, color=GREEN, label="Candidate B")
label(ax)
ax.legend()
save(fig, "case-candidate-lines.png")

for show_error, filename in [(False, "case-line.png"), (True, "case-line-errors.png")]:
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.scatter(x, y, color=GOLD, edgecolor=INK, s=45, zorder=4, label="Training jobs")
    ax.plot(q, models[1](q), color=GREEN, label="Least-squares line")
    if show_error:
        for xi, yi in zip(x, y):
            ax.plot([xi, xi], [models[1](xi), yi], color=BLUE, lw=2)
    ax.legend()
    label(ax)
    save(fig, filename)

fig, ax = plt.subplots(figsize=(6.2, 4))
ax.scatter(x, models[1](x)-y, color=GREEN, s=50)
ax.axhline(0, color=GOLD, ls="--")
label(ax, "Predicted − observed (min)")
save(fig, "case-line-residuals.png")

fig, ax = plt.subplots(figsize=(6.2, 4))
ax.scatter(x, y, color=GOLD, edgecolor=INK, s=45, zorder=4)
ax.plot(q, models[1](q), color=GRAY, ls="--", label="Fitted line")
ax.plot(q, models[2](q), color=GREEN, label="Fitted quadratic")
label(ax)
ax.legend()
save(fig, "case-quadratic.png")

fig, axs = plt.subplots(1, 4, figsize=(12, 3.2), sharex=True, sharey=True)
for ax, d in zip(axs, [0, 1, 2, 9]):
    ax.scatter(x, y, color=GOLD, s=25, zorder=4)
    ax.plot(q, models[d](q), color=GREEN)
    ax.set_title(f"Degree {d}")
    ax.set_xlabel("Job size")
    ax.set_xticks([1, 6, 12])
    ax.grid(alpha=.15)
axs[0].set_ylabel("Completion time (min)")
save(fig, "case-flexibility.png")

fig, ax = plt.subplots(figsize=(6.2, 4))
degree = list(range(1, 10))
ax.plot(degree, [training[d]["mse"] for d in degree], "o-", color=GREEN, label="Training")
ax.plot(degree, [validation[d]["mse"] for d in degree], "s-", color=GOLD, label="Validation")
ax.set_xlabel("Polynomial degree")
ax.set_ylabel("MSE (minutes squared)")
ax.set_xticks([1, 2, 3, 5, 7, 9])
ax.grid(alpha=.15)
ax.legend()
save(fig, "case-validation.png")

fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.6), sharex=True, sharey=True)
for ax, d, title in zip(axs, [1, 2], ["Line", "Quadratic"]):
    ax.scatter(xv, models[d](xv)-yv, color=GREEN, s=22, alpha=.7)
    ax.axhline(0, color=GOLD, ls="--", lw=1.5)
    ax.set_title(title)
    ax.set_xlabel("Job size")
    ax.set_xticks([1, 6, 12])
    ax.grid(alpha=.15)
axs[0].set_ylabel("Validation residual: predicted − observed (min)")
save(fig, "case-residual-comparison.png")

fig, ax = plt.subplots(figsize=(6.2, 4))
ax.scatter(xt, yt, color=GRAY, s=23, alpha=.55, label="Untouched test jobs")
ax.plot(q, models[selected](q), color=GREEN, label="Frozen quadratic")
ax.axhline(y.mean(), color=BLUE, ls="--", label="Training-mean baseline")
label(ax)
ax.legend()
save(fig, "case-test.png")

result = {
    "training_x": x.tolist(), "training_y": y.tolist(),
    "training_count": len(x), "validation_count": len(xv), "test_count": len(xt),
    "generator": "8 + 0.45*x + 0.12*x**2 + N(0, 0.65**2)",
    "seeds": {"training": 23, "validation": 24, "test": 25},
    "candidate_degrees": candidates, "selected_degree": selected,
    "coefficients": {d: m.convert().coef.tolist() for d, m in models.items()},
    "training": training, "validation": validation, "test": test,
}
(OUT / "narrative-case.json").write_text(json.dumps(result, indent=2)+"\n")
print("Verified nested-family training errors, constant baseline and validation-only model choice.")
print(json.dumps({"selected_degree": selected, "test": test}, indent=2))
print("Generated 10 regression plots for one consistent case.")

# Independent classification case: pre-job printer temperature predicts failure.
# Both extremes are risky. These are different labels, not thresholded durations.
def classification_sample(seed, count):
    rng = np.random.default_rng(seed)
    temperature = rng.uniform(5, 95, count)
    probability = 1 / (1 + np.exp(-(((temperature-50)/10)**2 - 4)))
    failed = rng.binomial(1, probability)
    return temperature[:, None], failed

ct, cy = classification_sample(61, 200)
cv, cvy = classification_sample(62, 200)
ce, cey = classification_sample(63, 200)
classifiers = {}
for degree in [1, 2]:
    model = make_pipeline(PolynomialFeatures(degree, include_bias=False),
                          StandardScaler(), LogisticRegression(C=100, max_iter=2000))
    model.fit(ct, cy)
    classifiers[degree] = model

def class_metrics(model, xx, yy):
    probability = model.predict_proba(xx)[:, 1]
    predicted = probability >= .5
    clipped = np.clip(probability, 1e-15, 1-1e-15)
    return {"log_loss": float(-np.mean(yy*np.log(clipped)+(1-yy)*np.log(1-clipped))),
            "accuracy": float(np.mean(predicted == yy)),
            "tn": int(np.sum((yy == 0) & ~predicted)),
            "fp": int(np.sum((yy == 0) & predicted)),
            "fn": int(np.sum((yy == 1) & ~predicted)),
            "tp": int(np.sum((yy == 1) & predicted))}
class_validation = {d: class_metrics(m, cv, cvy) for d, m in classifiers.items()}
class_winner = min(class_validation, key=lambda d: class_validation[d]["log_loss"])
assert class_winner == 2
class_test = class_metrics(classifiers[class_winner], ce, cey)
temperature_grid = np.linspace(5, 95, 500)[:, None]
fig, ax = plt.subplots(figsize=(7, 3.5))
for d, color in [(1, GRAY), (2, GREEN)]:
    ax.plot(temperature_grid[:, 0], classifiers[d].predict_proba(temperature_grid)[:, 1],
            color=color, label=f"Degree-{d} score + sigmoid")
ax.scatter(cv[:, 0], cvy, s=12, alpha=.25, color=GOLD, label="Validation outcomes")
ax.axhline(.5, color=BLUE, ls="--", lw=1.5, label="Threshold 0.5")
ax.set_xlabel("Pre-job temperature (degrees C)")
ax.set_ylabel("Estimated failure probability")
ax.set_ylim(-.08, 1.12)
ax.grid(alpha=.15)
ax.legend(fontsize=11, loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=2)
save(fig, "case-classification.png")
classification_result = {"seeds": {"training": 61, "validation": 62, "test": 63},
    "count_per_split": 200, "validation": class_validation,
    "selected_degree": class_winner, "test": class_test,
    "threshold": .5, "regularization_C": 100,
    "generator": "Bernoulli(sigmoid(((temperature-50)/10)**2 - 4))"}
(OUT / "classification-case.json").write_text(json.dumps(classification_result, indent=2)+"\n")
print(json.dumps(classification_result, indent=2))

print("Generated one additional fitted-classifier comparison plot.")
