"""Reproduce Lecture 26's quantitative PNGs and verify its worked examples.

Run with Python 3 and numpy, matplotlib, scikit-learn installed. All outputs are
written beside this script. Synthetic examples use fixed seeds. The benchmark
data are bundled with scikit-learn; no download is needed.
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from numpy.polynomial import Polynomial
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

OUT = Path(__file__).resolve().parent
GREEN, GOLD, BLUE, INK = "#154734", "#BD8B13", "#427C9E", "#1F2933"
RED, GRAY = "#B44C49", "#75828A"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 14, "axes.labelsize": 15,
    "axes.titlesize": 16, "axes.titleweight": "bold", "axes.edgecolor": INK,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK,
    "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 12, "lines.linewidth": 2.6,
    "savefig.facecolor": "white", "figure.constrained_layout.use": True,
})


def finish(fig, filename):
    fig.savefig(OUT / filename, dpi=210, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)


def axes_labels(ax, x, y):
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(alpha=.15)


# Print-job example: deliberately distinct from the lab's calibration data.
x = np.arange(1, 5, dtype=float)
y = np.array([3., 5., 4., 8.])
X = np.column_stack([np.ones(4), x])
w = np.linalg.lstsq(X, y, rcond=None)[0]
pred = X @ w
residual = y - pred
assert np.allclose(w, [1.5, 1.4])
assert np.allclose(residual, [.1, .7, -1.7, .9])
assert np.isclose(np.mean(residual**2), 1.05)
assert np.isclose(np.sqrt(np.mean(residual**2)), 1.02469507659596)
assert np.isclose(np.mean(abs(residual)), .85)
assert np.isclose(r2_score(y, pred), .7)
initial = np.array([.5, 1.])
gradient0 = 2 / len(y) * X.T @ (X @ initial - y)
step1 = initial - .05 * gradient0
assert np.allclose(gradient0, [-4, -11])
assert np.allclose(step1, [.7, 1.55])
assert np.isclose(np.mean((y - X @ initial)**2), 5.25)
assert np.isclose(np.mean((y - X @ step1)**2), 1.25875)

for filename, show_residuals in [("slide11.png", False), ("slide12.png", True)]:
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    q = np.linspace(.8, 4.2, 100)
    ax.plot(q, w[0] + w[1] * q, color=GREEN, label="Fitted line")
    ax.scatter(x, y, s=78, color=GOLD, edgecolor=INK, zorder=4, label="Observed")
    if show_residuals:
        for xi, yi, pi, ei in zip(x, y, pred, residual):
            ax.plot([xi, xi], [pi, yi], color=BLUE if ei > 0 else RED, lw=3.2)
            ax.scatter(xi, pi, s=26, color=GREEN, zorder=5)
            ax.annotate(f"{ei:+.1f}", (xi, (yi + pi) / 2), xytext=(9, 0),
                        textcoords="offset points", fontsize=14,
                        color=BLUE if ei > 0 else RED)
    else:
        ax.legend(loc="upper left")
    ax.set_xticks(x)
    ax.set_ylim(1.8, 8.8)
    axes_labels(ax, "Job size (ten-page batches)", "Time (min)")
    finish(fig, filename)


# Convex loss contours and actual gradient-descent updates, not a schematic path.
def trajectory(eta, steps):
    ws = [initial.copy()]
    for _ in range(steps):
        grad = 2 / len(y) * X.T @ (X @ ws[-1] - y)
        ws.append(ws[-1] - eta * grad)
    ws = np.asarray(ws)
    losses = np.mean((y[:, None] - X @ ws.T)**2, axis=0)
    return ws, losses

fig, ax = plt.subplots(figsize=(5.2, 3.7))
a, b = np.meshgrid(np.linspace(.2, 2.3, 220), np.linspace(.5, 2.1, 220))
loss = np.mean((y[:, None, None] - a - x[:, None, None]*b)**2, axis=0)
ax.contour(a, b, loss, levels=[1.06, 1.1, 1.2, 1.5, 2, 3, 5], colors=GRAY, linewidths=.9)
ws, tuned_loss = trajectory(.05, 100)
ax.plot(ws[:, 0], ws[:, 1], "o-", color=GREEN, ms=3.8)
for i in [0, 1, 3, 8]:
    ax.annotate("", xy=ws[i+1], xytext=ws[i],
                arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 1.8})
ax.scatter(*w, marker="*", s=140, color=GOLD, edgecolor=INK, zorder=5)
ax.annotate("start", initial, xytext=(5, -17), textcoords="offset points", fontsize=12)
axes_labels(ax, "Intercept $w_0$", "Slope $w_1$")
assert np.all(np.diff(tuned_loss) < 0)
finish(fig, "slide20.png")

fig, ax = plt.subplots(figsize=(6, 3.9))
for eta, color, label in [(.2, RED, "Too large: 0.2"),
                          (.002, GOLD, "Slow: 0.002"),
                          (.05, GREEN, "Useful: 0.05")]:
    _, losses = trajectory(eta, 20)
    ax.plot(np.arange(21), losses, color=color, label=label)
ax.axhline(1.05, color=GRAY, ls=":", lw=1.2)
ax.set_yscale("log")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.legend(loc="upper left")
axes_labels(ax, "Update number", "Training MSE (log scale)")
finish(fig, "slide21.png")

# Synthetic curved energy use, distinct from the print-job example.
rng = np.random.default_rng(14)
speed = np.linspace(.2, 3, 22)
energy = 1 + .7*speed + .55*speed**2 + rng.normal(0, .25, len(speed))
straight = Polynomial.fit(speed, energy, 1)
q = np.linspace(.1, 3.1, 200)
fig, ax = plt.subplots(figsize=(5.8, 4))
ax.scatter(speed, energy, color=GOLD, s=34, label="Synthetic data")
ax.plot(q, straight(q), color=GREEN, label="Best straight line")
ax.plot(q, 1 + .7*q + .55*q**2, color=GRAY, ls="--", label="Generating trend")
ax.legend(fontsize=11, loc="upper left")
axes_labels(ax, "Speed (arbitrary units)", "Energy use (arbitrary units)")
finish(fig, "slide22.png")

# One fixed noisy cubic sample is reused in the capacity, validation and ridge charts.
rng = np.random.default_rng(19)
cx = np.linspace(-1, 1, 12)
truth = lambda z: .3 + .4*z - .5*z**2 + 1.4*z**3
cy = truth(cx) + rng.normal(0, .25, len(cx))
grid = np.linspace(-1, 1, 500)
fits = {degree: Polynomial.fit(cx, cy, degree) for degree in range(11)}
fig, axs = plt.subplots(1, 3, figsize=(11.8, 3), sharex=True, sharey=True)
for ax, degree in zip(axs, [1, 3, 10]):
    ax.scatter(cx, cy, color=GOLD, s=23, zorder=3)
    ax.plot(grid, truth(grid), color=GRAY, ls="--", lw=1.7)
    ax.plot(grid, fits[degree](grid), color=GREEN)
    ax.set_title(f"Degree {degree}")
    ax.set_xlabel("Input $x$")
    ax.set_xticks([-1, 0, 1])
    ax.grid(alpha=.15)
axs[0].set_ylabel("Target $y$")
fig.suptitle("Same synthetic data; dashed line = generating trend", fontsize=13)
finish(fig, "slide24.png")

vrng = np.random.default_rng(20)
vx = vrng.uniform(-1, 1, 400)
vy = truth(vx) + vrng.normal(0, .25, len(vx))
degrees = np.arange(1, 11)
train_errors = [np.mean((cy - fits[p](cx))**2) for p in degrees]
validation_errors = [np.mean((vy - fits[p](vx))**2) for p in degrees]
assert np.all(np.diff(train_errors) < 1e-10)
fig, ax = plt.subplots(figsize=(5.8, 3.9))
ax.plot(degrees, train_errors, "o-", color=GREEN, label="Training")
ax.plot(degrees, validation_errors, "s-", color=GOLD, label="Validation")
ax.set_xticks([1, 2, 4, 6, 8, 10])
ax.legend()
axes_labels(ax, "Polynomial degree", "Mean squared error")
finish(fig, "slide25.png")

lam = .01
ridge = make_pipeline(PolynomialFeatures(10, include_bias=False),
                      StandardScaler(), Ridge(alpha=len(cx)*lam))
ridge.fit(cx[:, None], cy)
fig, ax = plt.subplots(figsize=(5.8, 4))
ax.scatter(cx, cy, color=GRAY, s=30, zorder=3, label="Training points")
ax.plot(grid, fits[10](grid), color=GOLD, ls="--", label="No penalty")
ax.plot(grid, ridge.predict(grid[:, None]), color=GREEN, label=r"Ridge $\lambda=0.01$")
ax.legend(fontsize=12)
axes_labels(ax, "Input $x$", "Target $y$")
finish(fig, "slide28.png")

# Benchmark plot follows the slide's exact code and split, including its baseline.
data = load_diabetes(scaled=False)
X_tr, X_te, y_tr, y_te = train_test_split(
    data.data[:, [2]], data.target, test_size=.2, random_state=42)
model = LinearRegression().fit(X_tr, y_tr)
base = DummyRegressor(strategy="mean").fit(X_tr, y_tr)
test_pred = model.predict(X_te)
metrics = {
    "training_count": len(y_tr), "test_count": len(y_te),
    "linear_test_mse": mean_squared_error(y_te, test_pred),
    "baseline_test_mse": mean_squared_error(y_te, base.predict(X_te)),
    "linear_test_r2": r2_score(y_te, test_pred),
    "training_target_mean": float(y_tr.mean()),
    "synthetic_polynomial_degrees": degrees.tolist(),
    "synthetic_train_mse_by_degree": train_errors,
    "synthetic_validation_mse_by_degree": validation_errors,
}
assert (len(y_tr), len(y_te)) == (353, 89)
fig, ax = plt.subplots(figsize=(5.5, 4.2))
ax.scatter(X_tr[:, 0], y_tr, s=17, alpha=.25, color=GRAY, label="Training")
ax.scatter(X_te[:, 0], y_te, s=24, alpha=.85, color=GOLD, label="Test")
bmi = np.linspace(data.data[:, 2].min(), data.data[:, 2].max(), 150)
ax.plot(bmi, model.predict(bmi[:, None]), color=GREEN, label="Training fit")
ax.axhline(y_tr.mean(), color=BLUE, ls="--", label="Training mean")
ax.legend(fontsize=11, loc="upper left")
axes_labels(ax, "BMI (raw feature)", "Progression score")
finish(fig, "slide32.png")

# Diagnostic examples are synthetic, not diagnostic claims about the benchmark.
rng = np.random.default_rng(44)
dx = np.linspace(-1, 1, 55)
cloud = rng.normal(0, .18, len(dx))
curved = 1.5*(dx**2 - np.mean(dx**2)) + rng.normal(0, .1, len(dx))
fig, axs = plt.subplots(1, 2, figsize=(7, 3.4), sharey=True)
for ax, res, title in zip(axs, [cloud, curved], ["No clear pattern", "Curved pattern"]):
    ax.scatter(dx, res, color=GREEN, s=18, alpha=.85)
    ax.axhline(0, color=GOLD, ls="--", lw=1.5)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Input $x$")
    ax.set_xticks([-1, 0, 1])
    ax.grid(alpha=.15)
axs[0].set_ylabel("Residual")
finish(fig, "slide33.png")

# Exact sigmoid; the image teaches a transformation, not empirical calibration.
score = np.linspace(-6, 6, 500)
probability = 1 / (1 + np.exp(-score))
assert np.all((probability > 0) & (probability < 1))
assert np.all(np.diff(probability) > 0)
fig, ax = plt.subplots(figsize=(5.6, 3.7))
ax.plot(score, probability, color=GREEN)
ax.axhline(.5, color=GRAY, ls=":", lw=1.2)
ax.axvline(0, color=GRAY, ls=":", lw=1.2)
ax.scatter([0], [.5], color=GOLD, edgecolor=INK, s=55, zorder=4)
ax.set_yticks([0, .5, 1])
ax.set_xticks([-6, -3, 0, 3, 6])
ax.set_ylim(-.03, 1.03)
axes_labels(ax, "Linear score $z$", "Sigmoid output $p$")
finish(fig, "sigmoid-score.png")

metrics["print_job_example"] = {
    "x": x.tolist(), "y": y.tolist(), "weights": w.tolist(),
    "residuals_observed_minus_predicted": residual.tolist(),
    "mse": float(np.mean(residual**2)),
    "rmse": float(np.sqrt(np.mean(residual**2))),
    "mae": float(np.mean(abs(residual))),
    "initial_weights": initial.tolist(), "gradient": gradient0.tolist(),
    "step_weights": step1.tolist(), "step_mse": float(np.mean((y-X@step1)**2)),
}
metrics["classification_examples"] = {
    "spam_loss_at_p_0_8": float(-np.log(.8)),
    "spam_loss_at_p_0_2": float(-np.log(.2)),
    "hypothetical_confusion_matrix_at_threshold_0_5": [[64, 6], [9, 21]],
}

(OUT / "chart-checks.json").write_text(json.dumps(metrics, indent=2) + "\n")
print(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, (list, dict))}, indent=2))
print("Verified least-squares, residual, gradient and dataset-split calculations.")
print("Generated all 11 quantitative lecture figures, including the sigmoid.")
