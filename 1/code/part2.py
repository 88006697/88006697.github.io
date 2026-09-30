"""
SYDE 572 Assignment 1 - Part 2
Fit a line and a parabola to 4 points by minimizing the mean squared error.

    line     y = m*x + b
    parabola y = a*x^2 + b

MSE(theta) = (1/n) * sum (yhat_i - y_i)^2

Solved analytically by setting the partial derivatives to zero, and
numerically with Newton-Raphson optimizing one parameter at a time.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

X = np.array([0.0, 2.0, 1.0, 3.0])
Y = np.array([0.5, 3.5, 1.5, 7.5])

NAMES = {"line": ["m", "b"], "parabola": ["a", "b"]}


def design(model, x):
    """Columns of the model. Both models are linear in their parameters,
    so yhat = design @ theta."""
    x = np.asarray(x, dtype=float)
    if model == "line":
        return np.column_stack([x, np.ones_like(x)])
    elif model == "parabola":
        return np.column_stack([x ** 2, np.ones_like(x)])
    else:
        raise ValueError("unknown model " + model)


def mse(theta, model, x=X, y=Y):
    r = design(model, x) @ theta - y
    return np.mean(r ** 2)


def grad(theta, model, x=X, y=Y):
    # dMSE/dtheta = (2/n) * P^T (P theta - y)
    P = design(model, x)
    return 2 / len(y) * P.T @ (P @ theta - y)


def second(model, x=X):
    # second derivatives, constant because the MSE is quadratic
    P = design(model, x)
    return 2 / len(x) * P.T @ P


def fit_analytic(model, x=X, y=Y):
    P = design(model, x)
    A = P.T @ P
    rhs = P.T @ y
    return np.linalg.solve(A, rhs), A, rhs


def fit_one_at_a_time(model, theta0=None, x=X, y=Y, tol=1e-10, max_sweeps=5000):
    """Newton-Raphson on one parameter at a time. For parameter j:
        theta_j <- theta_j - (dMSE/dtheta_j) / (d2MSE/dtheta_j^2)
    The MSE is quadratic in each parameter on its own, so this one step lands
    on the best theta_j for the current value of the other one.
    """
    theta = np.zeros(2) if theta0 is None else np.array(theta0, dtype=float)
    H = second(model, x)
    history = [(0, None, theta.copy(), mse(theta, model, x, y))]

    for sweep in range(1, max_sweeps + 1):
        old = theta.copy()
        for j in range(2):
            g = grad(theta, model, x, y)
            theta[j] = theta[j] - g[j] / H[j, j]
            history.append((sweep, NAMES[model][j], theta.copy(),
                            mse(theta, model, x, y)))
        if np.max(np.abs(theta - old)) < tol:
            break
    return theta, history


def curve(model, theta, x):
    return design(model, x) @ theta


# ================================================

if __name__ == "__main__":

    print("data points:", [(float(a), float(b)) for a, b in zip(X, Y)])
    print("sums:  n =", len(X), " sum x =", X.sum(), " sum x^2 =", (X**2).sum(),
          " sum x^3 =", (X**3).sum(), " sum x^4 =", (X**4).sum())
    print("       sum y =", Y.sum(), " sum xy =", (X*Y).sum(),
          " sum x^2 y =", (X**2*Y).sum())

    fits = {}
    for model in ["line", "parabola"]:
        th, A, rhs = fit_analytic(model)
        th1, hist1 = fit_one_at_a_time(model)
        fits[model] = dict(theta=th, hist=hist1, mse=mse(th, model))

        print("\n" + model)
        print(" normal equations matrix:\n", A, "\n rhs:", rhs)
        print(" analytic       ",
              dict(zip(NAMES[model], [round(float(v), 6) for v in th])))
        print(" one at a time  ",
              dict(zip(NAMES[model], [round(float(v), 6) for v in th1])),
              f"after {hist1[-1][0]} sweeps")
        print(f" MSE = {mse(th, model):.8f}")

    # first few updates, to compare with the hand calculation
    for model in ["line", "parabola"]:
        print("\nfirst updates for", model)
        print(f"{'sweep':>6} {'param':>6} {NAMES[model][0]:>11} "
              f"{NAMES[model][1]:>11} {'MSE':>11}")
        for (sw, par, th, m) in fits[model]["hist"][:9]:
            print(f"{sw:6d} {str(par):>6} {th[0]:11.6f} {th[1]:11.6f} {m:11.6f}")

    # plot 1: the fits
    gx = np.linspace(-0.3, 3.3, 200)
    plt.figure(figsize=(7, 4.5))
    plt.plot(X, Y, 'ro', ms=8, label='data')
    plt.plot(gx, curve("line", fits["line"]["theta"], gx), 'k-',
             label=f"line y = 2.3x - 0.2, MSE = {fits['line']['mse']:.5f}")
    plt.plot(gx, curve("parabola", fits["parabola"]["theta"], gx), 'b--',
             label=f"parabola y = 0.7653x^2 + 0.5714, "
                   f"MSE = {fits['parabola']['mse']:.5f}")
    plt.xlabel('x'); plt.ylabel('y')
    plt.title('Least squares fits')
    plt.legend(fontsize=8); plt.grid(True)
    plt.savefig('../media/p2_fits.png', dpi=120, bbox_inches='tight')
    plt.close()

    # ------ plot 2: path on the MSE surface ------
    # start away from the answer so the staircase is visible
    starts = {"line": (0.0, 2.5), "parabola": (0.0, 3.0)}
    window = {"line": (2.5, 3.0), "parabola": (1.0, 3.0)}
    plt.figure(figsize=(11, 4.2))
    for k, model in enumerate(["line", "parabola"]):
        plt.subplot(1, 2, k + 1)
        th = fits[model]["theta"]
        wu, wv = window[model]
        u = np.linspace(th[0] - wu, th[0] + wu, 120)
        v = np.linspace(th[1] - wv, th[1] + wv, 120)
        U, V = np.meshgrid(u, v)
        Z = np.zeros_like(U)
        for i in range(U.shape[0]):
            for j in range(U.shape[1]):
                Z[i, j] = mse(np.array([U[i, j], V[i, j]]), model)
        plt.contour(U, V, Z, levels=20, colors='0.7', linewidths=0.6)

        _, hist = fit_one_at_a_time(model, theta0=starts[model], max_sweeps=8)
        path = np.array([h[2] for h in hist])
        plt.plot(path[:, 0], path[:, 1], 'b.-')
        plt.plot(starts[model][0], starts[model][1], 'go', ms=7, label='start')
        plt.plot(th[0], th[1], 'ro', ms=7, label='answer')
        plt.xlabel(NAMES[model][0]); plt.ylabel(NAMES[model][1])
        plt.title(f'MSE contours, {model}')
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('../media/p2_contours.png', dpi=120, bbox_inches='tight')
    plt.close()

    # ---------- plot 3: intermediate fits ----------
    plt.figure(figsize=(11, 4))
    for k, model in enumerate(["line", "parabola"]):
        plt.subplot(1, 2, k + 1)
        _, hist = fit_one_at_a_time(model, theta0=starts[model], max_sweeps=4)
        for i, (sw, par, th, m) in enumerate(hist[:7]):
            lab = 'start' if par is None else f'update {i}: {par}'
            plt.plot(gx, curve(model, th, gx), lw=1, label=f'{lab}, MSE = {m:.3f}')
        plt.plot(gx, curve(model, fits[model]["theta"], gx), 'k--', label='answer')
        plt.plot(X, Y, 'ro', ms=7)
        plt.xlabel('x'); plt.ylabel('y')
        plt.title(f'{model}: first updates')
        plt.legend(fontsize=7); plt.grid(True)
    plt.tight_layout()
    plt.savefig('../media/p2_steps.png', dpi=120, bbox_inches='tight')
    plt.close()

    print("\ndone, plots saved in ../media")
