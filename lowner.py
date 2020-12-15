import numpy as np
from numpy.linalg import norm, matrix_rank, eig, inv, cholesky, svd
from numpy import sqrt
import sys
import torch
import scipy.optimize as scopt

if not sys.warnoptions:
    import os, warnings

    warnings.simplefilter("error")
    os.environ["PYTHONWARNINGS"] = "error"


## The gradient of l_p-regression at c
def grad_compute(A, x, p):
    A_ = torch.tensor(A, requires_grad=False).to(torch.device("cpu"))
    x_ = torch.tensor(x.astype(np.float64), requires_grad=True).to(torch.device("cpu"))
    y_ = torch.matmul(A_, x_).to(torch.device("cpu"))
    y_ = torch.norm(y_, p).to(torch.device("cpu"))
    y_.backward()
    return x_.grad.detach().numpy()

def argmax_v(A, F, c, p):
    opt = lambda x: -np.linalg.norm(A @ x, ord=p)
    grad_opt = lambda x: grad_compute(A, x, p)
    d = A.shape[1]
    F_inv = np.linalg.inv(F.astype(np.float64))
    constr = lambda x: d**2 * (x-c).T @ F_inv @ (x-c)
    constr_grad = lambda x: -2 * d**2 * F_inv @ x
    constraints = scopt.NonlinearConstraint(fun=constr,
                                            lb=-np.inf,
                                            ub=1,
                                            jac=constr_grad)
    constraints = {"type":"eq", "fun":lambda x: -constr(x)+1, "jac":constr_grad}
    w, v = eig(inv(F.astype(np.float64)) * d ** 2)
    v = np.vstack([v.T, -v.T])
    w = np.hstack([w, w])
    max_ = None
    max_v = None
    for ind, v in enumerate(v):
        x0 = np.real(v / np.sqrt(w[ind]) / 1.01 + c)
        val = norm(A @ x0, ord=p)
        if max_ is None or val > max_:
            max_v = x0
            max_ = val
        res = scopt.minimize(opt, x0, 
                             jac=grad_opt, method='SLSQP',
                             constraints=constraints)
        cur_x = res.x
        val = np.linalg.norm(A@cur_x, ord=p)
        if max_ is None  or (val > max_ and (constr(cur_x) - 1) <= 1e-6):
            max_v = cur_x
            max_ = val
    return max_v


def lowner(A, p, max_iter):
    iter_num = 0
    d = A.shape[1]
    # E := A ball centered around the origin which contains L
    # r = 100 / (norm(A, -2) ** (1/p))
    r = 1/norm(A, -2)
    F = r * np.identity(d)
    F_prev = F
    c = np.zeros(shape=(d,))
    n = A.shape[0]
    stop = False
    while True:
        iter_num += 1
        ## 7-12
        while norm(np.matmul(A, c), p) > 1:  ## c not in L
            grad = grad_compute(A, c, p)
            H = 1 / np.max(np.abs(grad)) * grad
            const = H.T @ F @ H
            b = 1 / sqrt(const) * np.matmul(F, H)
            c = c - 1 / (d + 1) * b
            F = (d ** 2) / (d ** 2 - 1) * (F - 2 / (d + 1) * np.outer(b, b.T))
        ## 13 - 16
        w, v = eig(inv(F.astype(np.float64)) * d ** 2)
        v = v.real.astype(np.float64)
        w = w.real.astype(np.float64)
        if not (w > 0).all():
            print("не положительно определена стала на очередной итерации")
            stop = True
        if stop:
            break
        ## 17
        v = argmax_v(A, F, c, p)
        print(norm(A@v, p))
        if  norm(A @ v, p) <= 1:
            break
        ##18..26
        grad = grad_compute(A, v, p)
        H = 1 / max(abs(grad)) * grad
        z = 1 / ((d + 1) * (d + 1))
        sigma = (d * d * d) * (d + 2) / (((d + 1) ** 3) * (d - 1))
        zeta = 1 + 1 / (2 * d * d * (d + 1) * (d + 1))
        tau = 2 / (d * (d + 1))
        const = H.T @ F @ H
        b = 1 / sqrt(const) * np.matmul(F, H)
        F_prev = F
        F = zeta * sigma * (F - tau * np.outer(b, b.T))
        c = c - z * b
        if iter_num >= max_iter:
            print("Max iteration exceeded!")
            break
    F = F_prev
    ##27..28
    G = cholesky(inv(F.astype(np.float64)))
    S, D, V = svd(G)
    D = np.diag(D)
    print("iter", iter_num)
    return D, V


def l_p_low_rank(A, k, p, max_iter=100):
    d = A.shape[1]
    D, V = lowner(A, p, max_iter)
    U = np.matmul(A, inv(np.matmul(D, V)))
    set_of_sigma = np.diag(D)
    D_k = np.diag(np.array(list(set_of_sigma[0: k]) + [0] * (d - k)))
    return U, D_k, V, set_of_sigma