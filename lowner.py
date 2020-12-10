import numpy as np
from numpy.linalg import norm, matrix_rank, eig, inv, cholesky, svd
from numpy import sqrt


## The gradient of l_p-regression at c
def grad_compute(A, c, p, d, n):
    grad = np.zeros((d,))
    factors = np.zeros((n,))
    for i in range(n): # то что в элементах градиента в знаменателях
        factors[i] = np.abs(np.dot(A[i, :], c))
    for j in range(d): # заполняем вектор который в выражении градиента
        grad[j] = np.sum(A[:, j] / factors) * c[j]
    #grad *= norm(np.matmul(A, c), p) ** (p-1) * norm(np.matmul(A, c), p-1) ** (p-1) # бесполезно, оно сокращаяется потом
    return grad


def lowner(A, p):
    
    iter_num = 0
    d = A.shape[1]
    #E := A ball centered around the origin which contains L
    r = 1/sqrt(norm(A, -2))
    F = r * np.identity(d)
    F_prev = F
    c = np.zeros(shape=(d,)) 
    n = A.shape[0]
    stop = False
    while True: 
        iter_num += 1
        ## 7-12
        while norm(np.matmul(A, c), p) > 1: ## c not in L
            grad = grad_compute(A, c, p, d, n)
            H = 1/np.max(np.abs(grad)) * grad.astype(np.longdouble)
            b = 1/sqrt(abs(np.matmul(np.transpose(H), np.matmul(F, H))) + 1e-10) * np.matmul(F, H)
            c = c - 1/(d + 1) * b
            F = (d ** 2)/(d ** 2 - 1) * (F - 2/(d + 1) * np.matmul(b, np.transpose(b)))
        ## 13 - 16
        containd = True
        w, v = eig(inv(F.astype(np.float32)) / d)
        v = v.astype(np.float64)
        w = w.astype(np.float64)
        if not (w > 0).all():
            print("не положительно определена стала на очередной итерации")
            stop = True
        if stop:
            break
        for ind, val in enumerate(w):
            v[ind] = v[ind] / sqrt(val)
            v[ind] = (v[ind])  + c
        containd = True
        for vec in v:
            if norm(np.matmul(A, vec), p) > 1:
                containd = False
        if containd:
            break
        ## 17
        max_v = v[0]
        _max = norm(np.matmul(A, v[0]), p)
        for vec in v:
            temp = norm(np.matmul(A, vec), p)
            if temp >= _max:
                max_v = vec
                _max = temp
        v = max_v
        ##18..26
        grad = grad_compute(A, v, p, d, n)
        H = 1/np.max(np.abs(grad)) * grad.astype(np.longdouble)
        z = 1/((d + 1) * (d + 1))
        sigma = (d*d*d)*(d+2)/(((d+1)**3) * (d-1))
        zeta = 1 + 1/(2 * d * d * (d + 1) * (d + 1))
        tau = 2/(d * (d + 1))
        b = 1/sqrt(abs(H.T @ F @ H) + 1e-10) * (F @ H)
        F_prev = F
        F = zeta * sigma * (F - tau * np.matmul(b, np.transpose(b))) 
        c = c - z*b
    F = F_prev
    ##27..28
    G = cholesky(inv(F.astype(np.float32)))
    S, D, V = svd(G)
    D = np.diag(D)
    print("iter", iter_num)
    return D, V

def l_p_low_rank(A, k, p):
    d = A.shape[1]
    D, V = lowner(A, p)
    U = np.matmul(A, inv(np.matmul(D, V)))
    set_of_sigma = np.diag(D)
    D_k = np.diag(np.array(list(set_of_sigma[0 : k]) + [0] * (d - k)))
    return U, D_k, V, set_of_sigma