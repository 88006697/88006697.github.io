"""
SYDE 572 Assignment 2 - Part 1
Linear basis function regression with polynomial basis functions
(single-variable monomials, no cross terms) on the Boston housing data.
First 100 points are training data, the rest are test data.
"""

import numpy as np
import matplotlib.pyplot as plt

names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
         'TAX', 'PTRATIO', 'B', 'LSTAT']


def load_data():
    data = np.loadtxt('housing.data')
    return data[:, :13], data[:, 13]


def normalize(x):
    # zero mean, unit variance for each feature, using all 506 points
    return (x - x.mean(axis=0)) / x.std(axis=0)


def poly_design(x, degree):
    # bias, then x, x^2, ... for each feature, no cross terms
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    cols = [np.ones((x.shape[0], 1))]
    for d in range(1, degree + 1):
        cols.append(x ** d)
    return np.hstack(cols)


def fit(phi, t, lam=0):
    # lam = 0: pseudo-inverse solution
    # lam > 0: w = (phi^T phi + lam I)^-1 phi^T t
    if lam == 0:
        return np.linalg.lstsq(phi, t, rcond=None)[0]
    M = phi.shape[1]
    return np.linalg.solve(phi.T @ phi + lam * np.eye(M), phi.T @ t)


def rms(phi, w, t):
    return np.sqrt(np.mean((phi @ w - t) ** 2))


def cv_error(phi, t, lams, k=10):
    # k-fold cv, folds are consecutive blocks of points
    n = len(t)
    size = n // k
    errs = []
    for lam in lams:
        e = []
        for i in range(k):
            val = np.arange(i * size, (i + 1) * size)
            tr = np.setdiff1d(np.arange(n), val)
            w = fit(phi[tr], t[tr], lam)
            e.append(rms(phi[val], w, t[val]))
        errs.append(np.mean(e))
    return np.array(errs)


def polynomial_regression(x_train, t_train, x_test, t_test, degree):
    phi_train = poly_design(x_train, degree)
    phi_test = poly_design(x_test, degree)
    w = fit(phi_train, t_train)
    return w, rms(phi_train, w, t_train), rms(phi_test, w, t_test)


def polynomial_regression_1d_vis(x_train, t_train, x_test, t_test, degree, fname):
    w, e_train, e_test = polynomial_regression(x_train, t_train, x_test, t_test, degree)
    xs = np.linspace(min(x_train.min(), x_test.min()), max(x_train.max(), x_test.max()), 500)
    plt.figure()
    plt.plot(x_train, t_train, 'ko', mfc='none', label='training data')
    plt.plot(x_test, t_test, 'kx', ms=4, label='test data')
    plt.plot(xs, poly_design(xs, degree) @ w, 'k-', label='learned polynomial')
    plt.ylim(-10, 60)
    plt.xlabel('ZN (normalized)')
    plt.ylabel('MEDV (\\$1000s)')
    plt.title('Degree ' + str(degree))
    plt.legend()
    plt.savefig(fname, dpi=150)
    plt.close()
    return e_train, e_test


def polynomial_regression_reg(x, t, degree, lams):
    return cv_error(poly_design(x, degree), t, lams)


if __name__ == '__main__':
    x, t = load_data()
    xn = normalize(x)
    x_train, x_test = xn[:100], xn[100:]
    t_train, t_test = t[:100], t[100:]

    # (a)
    degrees = list(range(1, 8))
    train_err = []
    test_err = []
    print('(a) degree, train RMS, test RMS')
    for d in degrees:
        w, e1, e2 = polynomial_regression(x_train, t_train, x_test, t_test, d)
        train_err.append(e1)
        test_err.append(e2)
        print(d, e1, e2)

    plt.figure()
    plt.semilogy(degrees, train_err, 'ko-', label='training error')
    plt.semilogy(degrees, test_err, 'ks--', mfc='none', label='test error')
    plt.xlabel('polynomial degree')
    plt.ylabel('RMS error (\\$1000s, log scale)')
    plt.legend()
    plt.savefig('../media/a_poly_error.png', dpi=150)
    plt.close()

    # (b)
    w, e1, e2 = polynomial_regression(x_train, t_train, x_test, t_test, 1)
    print('\n(b) degree 1 weights')
    print('bias', w[0])
    for i in range(13):
        print('w' + str(i), names[i], w[i + 1])

    # (c) only feature 2 (ZN)
    print('\n(c) degree, train RMS, test RMS (ZN only)')
    for d in degrees:
        e1, e2 = polynomial_regression_1d_vis(x_train[:, 1], t_train, x_test[:, 1], t_test, d,
                                              '../media/c_zn_deg' + str(d) + '.png')
        print(d, e1, e2)

    # (d)
    lams = [0, 0.01, 0.1, 1, 10, 100, 1000]
    errs = polynomial_regression_reg(x_train[:, 1], t_train, 8, lams)
    print('\n(d) lambda, avg validation RMS')
    for lam, e in zip(lams, errs):
        print(lam, e)

    # lambda = 0 can't go on a log axis, reported separately
    plt.figure()
    plt.semilogx(lams[1:], errs[1:], 'ko-')
    plt.xlabel('lambda')
    plt.ylabel('average validation RMS error (\\$1000s)')
    plt.savefig('../media/d_poly_reg.png', dpi=150)
    plt.close()
