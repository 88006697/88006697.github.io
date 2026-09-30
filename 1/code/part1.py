"""
SYDE 572 Assignment 1 - Part 1
Shortest distance from a point (x0, y0) to a curve y = f(x).

Square root is increasing, so minimizing the distance is the same as
minimizing the squared distance:
    D(x)  = (x - x0)^2 + (f(x) - y0)^2
    D'(x) = 2(x - x0) + 2(f(x) - y0)*f'(x)
    D''(x)= 2 + 2*f'(x)^2 + 2*(f(x) - y0)*f''(x)
"""

import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def dist_sq(x, x0, y0, f):
    return (x - x0) ** 2 + (f(x) - y0) ** 2


def d1(x, x0, y0, f, df):
    return 2 * (x - x0) + 2 * (f(x) - y0) * df(x)


def d2(x, x0, y0, f, df, ddf):
    return 2 + 2 * df(x) ** 2 + 2 * (f(x) - y0) * ddf(x)


# ----- Algorithm A: Newton-Raphson on D'(x) = 0 -----

def newton(x0, y0, f, df, ddf, guess=0.0, tol=1e-7, max_iter=100):
    x = guess
    xs = [x]
    for i in range(max_iter):
        g = d1(x, x0, y0, f, df)
        h = d2(x, x0, y0, f, df, ddf)
        if h == 0:
            print("D'' is zero, stopping")
            break
        new_x = x - g / h
        xs.append(new_x)
        if abs(new_x - x) < tol:
            x = new_x
            break
        x = new_x
    return math.sqrt(dist_sq(x, x0, y0, f)), x, xs


# --------- Algorithm B: golden section search on D(x) ---------------

PHI = (1 + math.sqrt(5)) / 2
R = 2 - PHI   # 0.381966

def golden(x0, y0, f, a, b, tol=1e-7):
    x1 = a + R * (b - a)
    x2 = b - R * (b - a)
    f1 = dist_sq(x1, x0, y0, f)
    f2 = dist_sq(x2, x0, y0, f)
    brackets = [(a, b, x1, x2)]

    while abs(b - a) > tol:
        if f1 < f2:
            # minimum must be left of x2
            b = x2
            x2, f2 = x1, f1
            x1 = a + R * (b - a)
            f1 = dist_sq(x1, x0, y0, f)
        else:
            a = x1
            x1, f1 = x2, f2
            x2 = b - R * (b - a)
            f2 = dist_sq(x2, x0, y0, f)
        brackets.append((a, b, x1, x2))

    best = (a + b) / 2
    return math.sqrt(dist_sq(best, x0, y0, f)), best, brackets


def parabola(a, b, c):
    """Returns f, f', f'' for y = a x^2 + b x + c."""
    f = lambda x: a * x ** 2 + b * x + c
    df = lambda x: 2 * a * x + b
    ddf = lambda x: 2 * a
    return f, df, ddf


# =====================================================

if __name__ == "__main__":

    f, df, ddf = parabola(1, 0, 5)
    points = [(0, 0), (-4, 0), (-8, 0), (2, 0), (6, 0)]

    # D'(0) = -2*x0 and D'(x0) has the same sign as x0, so the minimum is
    # always between 0 and x0. That is the bracket for golden section.
    def bracket(x0):
        if x0 == 0:
            return -1.0, 1.0
        return min(x0, 0.0), max(x0, 0.0)

    print("y = x^2 + 5")
    print(f"{'point':>10} {'x newton':>12} {'d newton':>12} {'it':>4} "
          f"{'x golden':>12} {'d golden':>12} {'it':>4}")

    results = []
    for (x0, y0) in points:
        dn, xn, xs = newton(x0, y0, f, df, ddf, guess=0.0)
        a, b = bracket(x0)
        dg, xg, br = golden(x0, y0, f, a, b)
        results.append(dict(pt=(x0, y0), xn=xn, dn=dn, xs=xs,
                            xg=xg, dg=dg, br=br, bracket=(a, b)))
        print(f"{str((x0,y0)):>10} {xn:12.7f} {dn:12.7f} {len(xs)-1:4d} "
              f"{xg:12.7f} {dg:12.7f} {len(br)-1:4d}")

    print("\nNewton iterations for (-4, 0), starting at x = 0")
    print(f"{'k':>3} {'x_k':>14} {'D1':>14} {'D2':>12}")
    for k, x in enumerate(results[1]["xs"]):
        print(f"{k:3d} {x:14.9f} {d1(x,-4,0,f,df):14.9f} {d2(x,-4,0,f,df,ddf):12.6f}")

    print("\nGolden section brackets for (-4, 0), first 6")
    print(f"{'k':>3} {'a':>11} {'b':>11} {'x1':>11} {'x2':>11} {'b-a':>11}")
    for k, (a, b, x1, x2) in enumerate(results[1]["br"][:6]):
        print(f"{k:3d} {a:11.6f} {b:11.6f} {x1:11.6f} {x2:11.6f} {b-a:11.6f}")

    # plot 1: the five shortest distances
    plt.figure(figsize=(7, 4.5))
    xx = np.linspace(-2.5, 2.5, 200)
    plt.plot(xx, f(xx), 'k-')
    for i, r in enumerate(results):
        x0, y0 = r["pt"]
        plt.plot([x0, r["xn"]], [y0, f(r["xn"])], 'k--', lw=0.8)
        plt.plot(x0, y0, 'ro')
        plt.plot(r["xn"], f(r["xn"]), 'bo')
        # stagger the labels or (0,0) and (2,0) sit on top of each other
        dy = -20 if i % 2 == 0 else -34
        plt.annotate(f"({x0},{y0})  d={r['dn']:.4f}", (x0, y0),
                     textcoords="offset points", xytext=(0, dy), ha='center',
                     fontsize=8)
    plt.plot([], [], 'ro', label='given point')
    plt.plot([], [], 'bo', label='closest point')
    plt.xlabel('x'); plt.ylabel('y')
    plt.title('Shortest distance to y = x^2 + 5')
    plt.legend(fontsize=8); plt.grid(True)
    plt.axis('equal'); plt.ylim(-4, 10)
    plt.savefig('../media/p1_overview.png', dpi=120, bbox_inches='tight')
    plt.close()

    # ------- plot 2: Newton steps -------
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for j, r in enumerate(results):
        x0, y0 = r["pt"]
        xs = r["xs"]

        # top row: D'(x) with the iterates on it
        ax = axes[0][j]
        lo = min(min(xs), r["xn"]) - 0.3
        hi = max(max(xs), r["xn"]) + 0.3
        gx = np.linspace(lo, hi, 200)
        ax.plot(gx, [d1(t, x0, y0, f, df) for t in gx], 'k-')
        ax.axhline(0, color='k', lw=0.6)
        ax.plot(xs, [d1(t, x0, y0, f, df) for t in xs], 'bo', ms=5)
        for k in range(min(2, len(xs))):
            ax.annotate(f"x{k}", (xs[k], 0), textcoords="offset points",
                        xytext=(0, 8), ha='center', fontsize=8)
        ax.set_title(f"D'(x), point ({x0},{y0})", fontsize=9)
        ax.set_xlabel('x'); ax.grid(True)

        # bottom row: the same iterates on the curve
        ax = axes[1][j]
        gx = np.linspace(min(x0, r["xn"]) - 1, max(x0, r["xn"]) + 1, 200)
        ax.plot(gx, f(gx), 'k-')
        for x in xs:
            ax.plot([x0, x], [y0, f(x)], 'k--', lw=0.6)
        ax.plot(xs, f(np.array(xs)), 'bo', ms=5)
        ax.plot(x0, y0, 'ro', ms=6)
        ax.set_title(f"y = x^2 + 5, d = {r['dn']:.6f}", fontsize=9)
        ax.set_xlabel('x'); ax.grid(True)
    fig.suptitle('Newton-Raphson steps, start x = 0')
    plt.tight_layout()
    plt.savefig('../media/p1_newton.png', dpi=110, bbox_inches='tight')
    plt.close()

    # plot 3: golden section brackets
    plt.figure(figsize=(7, 4.5))
    r = results[1]              # point (-4, 0)
    a0, b0 = r["bracket"]
    gx = np.linspace(a0, b0, 200)
    plt.plot(gx, [dist_sq(t, -4, 0, f) for t in gx], 'k-')
    plt.plot(r["xg"], r["dg"] ** 2, 'ro', ms=7, label='minimum')
    for k, (a, b, x1, x2) in enumerate(r["br"][:8]):
        plt.plot([a, b], [400 - 30 * k, 400 - 30 * k], color='0.6', lw=4,
                 solid_capstyle='butt')
        plt.plot([x1, x2], [400 - 30 * k, 400 - 30 * k], 'bo', ms=4)
    plt.plot([], [], color='0.6', lw=4, label='bracket [a, b]')
    plt.plot([], [], 'bo', ms=4, label='x1 and x2')
    plt.xlabel('x'); plt.ylabel('D(x)')
    plt.title('Golden section on D(x), point (-4, 0), y = x^2 + 5')
    plt.legend(fontsize=8); plt.grid(True)
    plt.savefig('../media/p1_golden.png', dpi=120, bbox_inches='tight')
    plt.close()

    # ---------- other parabolas ----------
    print("\nOther parabolas and points")
    others = [(0.5, -2, 1, 3, -2, 0.0),
              (0.5, -2, 1, -1, 4, 0.0),
              (-2, 4, 0, 4, 1, 2.0),
              (1, -1, 2, 2, 0, 1.0)]
    print(f"{'curve':>22} {'point':>10} {'x newton':>11} {'d newton':>11} {'d golden':>11}")
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, (a, b, c, px, py, guess) in zip(axes, others):
        ff, dff, ddff = parabola(a, b, c)
        dn, xn, xs = newton(px, py, ff, dff, ddff, guess=guess)
        dg, xg, br = golden(px, py, ff, -10, 10)
        name = f"y = {a}x^2 + {b}x + {c}"
        print(f"{name:>22} {str((px,py)):>10} {xn:11.7f} {dn:11.7f} {dg:11.7f}")

        gx = np.linspace(min(px, xn) - 3, max(px, xn) + 3, 300)
        ax.plot(gx, ff(gx), 'k-')
        ax.plot([px, xn], [py, ff(xn)], 'k--', lw=0.8)
        ax.plot(xs, ff(np.array(xs)), 'bo', ms=5)
        ax.plot(px, py, 'ro', ms=6)
        ax.set_title(f"{name}\npoint ({px},{py}), d = {dn:.5f}", fontsize=9)
        ax.set_xlabel('x'); ax.grid(True)
        # fix the window, or an early iterate that lands far away stretches it
        ax.set_xlim(min(px, xn) - 2.5, max(px, xn) + 2.5)
        ax.set_ylim(min(py, ff(xn)) - 2.5, max(py, ff(xn)) + 2.5)
    plt.tight_layout()
    plt.savefig('../media/p1_other.png', dpi=110, bbox_inches='tight')
    plt.close()

    # non-polynomial functions. Only f, f' and f'' change, the two
    # methods stay exactly the same.
    print("\nNon-polynomial functions")
    cases = [
        ("y = e^x", lambda x: np.exp(x), lambda x: np.exp(x), lambda x: np.exp(x),
         (2, 1), 0.0, (-3, 3)),
        ("y = ln(x)", lambda x: np.log(x), lambda x: 1 / x, lambda x: -1 / x ** 2,
         (0, 2), 1.0, (0.05, 5)),
        ("y = 1/x", lambda x: 1 / x, lambda x: -1 / x ** 2, lambda x: 2 / x ** 3,
         (3, 1), 1.0, (0.2, 6)),
        ("y = sqrt(x)", lambda x: np.sqrt(x), lambda x: 0.5 / np.sqrt(x),
         lambda x: -0.25 * x ** -1.5, (4, 0), 1.0, (0.01, 8)),
    ]
    print(f"{'curve':>20} {'point':>9} {'x newton':>11} {'d newton':>11} "
          f"{'d golden':>11}")
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, (name, ff, dff, ddff, (px, py), guess, (ga, gb)) in zip(axes, cases):
        dn, xn, xs = newton(px, py, ff, dff, ddff, guess=guess)
        dg, xg, br = golden(px, py, ff, ga, gb)
        print(f"{name:>20} {str((px,py)):>9} {xn:11.6f} {dn:11.7f} {dg:11.7f}")

        gx = np.linspace(ga, gb, 300)
        ax.plot(gx, ff(gx), 'k-')
        ax.plot([px, xn], [py, ff(xn)], 'k--', lw=0.8)
        ax.plot(xs, [ff(x) for x in xs], 'bo', ms=5)
        ax.plot(px, py, 'ro', ms=6)
        ax.set_title(f"{name}\npoint ({px},{py}), d = {dn:.5f}", fontsize=9)
        ax.set_xlabel('x'); ax.grid(True)
        ax.set_xlim(min(px, xn) - 2.5, max(px, xn) + 2.5)
        ax.set_ylim(min(py, ff(xn)) - 2.5, max(py, ff(xn)) + 2.5)
    plt.tight_layout()
    plt.savefig('../media/p1_nonpoly.png', dpi=110, bbox_inches='tight')
    plt.close()

    print("\ndone, plots saved in ../media")
