"""
SYDE 572 Assignment 2 - Part 2
Linear basis function regression with Gaussian basis functions,
centres picked from the training points.
"""

import numpy as np
import matplotlib.pyplot as plt
from part1 import load_data, normalize, fit, rms

a = 1 / 8


def gauss_design(x, centers, a):
    # phi_j(x) = exp(-a ||x - c_j||^2), plus a bias column
    d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    phi = np.exp(-a * d2)
    return np.hstack([np.ones((x.shape[0], 1)), phi])


def gaussian_regression(x_train, t_train, x_test, t_test, centers, a):
    phi_train = gauss_design(x_train, centers, a)
    phi_test = gauss_design(x_test, centers, a)
    w = fit(phi_train, t_train)
    return w, rms(phi_train, w, t_train), rms(phi_test, w, t_test)


def gaussian_regression_reg(x, t, a, lams, k=10):
    # 10-fold cv, the centres are the 90 training points of each fold
    n = len(t)
    size = n // k
    errs = []
    for lam in lams:
        e = []
        for i in range(k):
            val = np.arange(i * size, (i + 1) * size)
            tr = np.setdiff1d(np.arange(n), val)
            centers = x[tr]
            w = fit(gauss_design(x[tr], centers, a), t[tr], lam)
            e.append(rms(gauss_design(x[val], centers, a), w, t[val]))
        errs.append(np.mean(e))
    return np.array(errs)


if __name__ == '__main__':
    x, t = load_data()
    xn = normalize(x)
    x_train, x_test = xn[:100], xn[100:]
    t_train, t_test = t[:100], t[100:]

    np.random.seed(0)
    perm = np.random.permutation(100)

    # (e)
    Ks = list(range(5, 100, 10))
    train_err = []
    test_err = []
    print('(e) K, train RMS, test RMS')
    for K in Ks:
        centers = x_train[perm[:K]]
        w, e1, e2 = gaussian_regression(x_train, t_train, x_test, t_test, centers, a)
        train_err.append(e1)
        test_err.append(e2)
        print(K, e1, e2)

    plt.figure()
    plt.plot(Ks, train_err, 'ko-', label='training error')
    plt.plot(Ks, test_err, 'ks--', mfc='none', label='test error')
    plt.xlabel('number of basis functions')
    plt.ylabel('RMS error (\\$1000s)')
    plt.legend()
    plt.savefig('../media/e_gauss_error.png', dpi=150)
    plt.close()

    # (f) 90 centers, all 13 features
    lams = [0, 0.01, 0.1, 1, 10, 100, 1000]
    errs = gaussian_regression_reg(x_train, t_train, a, lams)
    print('\n(f) lambda, avg validation RMS')
    for lam, e in zip(lams, errs):
        print(lam, e)

    plt.figure()
    plt.semilogx(lams[1:], errs[1:], 'ko-', label='10-fold CV error')
    plt.axhline(errs[0], color='k', ls='--', label='lambda = 0')
    plt.legend()
    plt.xlabel('lambda')
    plt.ylabel('average validation RMS error (\\$1000s)')
    plt.savefig('../media/f_gauss_reg.png', dpi=150)
    plt.close()
