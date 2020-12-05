import numpy as np
from numpy.linalg import norm, matrix_rank, eig, inv, cholesky, svd
from numpy import sqrt

## The gradient of l_p-regression at c
def grad_compute(A, c, p, d):
    grad = np.array(shape(d, 1))
    factors = np.array(shape(d, 1))
    for i in range(n): # то что в элементах градиента в знаменателях
        factors[i, :] = np.abs(np.matmul(np.transpose(A[i, :]), c))
    for j in range(d): # заполняем вектор который в выражении градиента
        grad[j, 0] = np.sum(A[:, j] / factors) * c[j, 0]
    grad *= norm(np.matmul(A, c), p) ** (p-1) * norm(np.matmul(A, c), p-1) ** (p-1)
    return grad


def lowner(A, p):
    d = A.shape[1]
    #E := A ball centered around the origin which contains L
    r = 1/norm(A, 2)   #3 r \leq 1/max(\sigma)
    F = r * np.identity(d) 
    c = np.zeros(shape=(d, 1)) 
    n = A.shape[0]
    while True: 
        ## 7-12
        while norm(np.matmul(A, c), p) >= 1: ## c not in L
            grad = grad_compute(A, c, p, d)
            H = 1/norm(grad, np.inf) * grad
            b = 1/sqrt(np.matmul(np.transpose(H), np.matmul(F, H))) * np.matmul(F, H)
            c = c - 1/(d + 1) * b
            F = (d ** 2)/(d ** 2 - 1) * (F - 2/(d + 1) * np.matmul(b, np.transpose(b)))
        ## 13 - 16
        containd = True
        w, v = eig(inv(F))
        for ind, val in enumerate(w):
            v[ind] /= sqrt(val)
        for vec in v:
            if norm(np.matmul(A, vec/d + c), p) >= 1:
                containd = False
        if containd:
            break
            
        ## 17
        max_v = vec[0]
        _max = norm(np.matmul(A, v), p)
        for vec in v:
            temp = norm(np.matmul(A, vec), p)
            if temp >= _max:
                max_v = vec
                _max = temp
        v = max_v
        ##18..26
        grad = grad_compute(A, v, p, d)
        H = 1/norm(grad, np.inf) * grad
        z = 1/(d + 1)/(d + 1)
        sigma = (d**3)*(d+2)/((d+1)**3)/(d-1)
        zeta = 1 + 1/2/(d**2)/((d+1) ** 2)
        tau = 2/d/(d+1)
        b = 1/sqrt(np.matmul(np.transpose(H), np.matmul(F, H))) * np.matmul(F, H)
        F = zeta * sigma * (F - tau * np.matmul(b, np.tranpose(b)))
        c = c - z*b
    ##27..28
    G = cholesky(inv(F))
    S, D, V = svd(G)
    D = np.diag(D)
    return D, V

def l_p_low_rank(A, k, p):
    d = A.shape[1]
    D, V = lowner(A, p)
    U = np.matmul(A, inv(np.matmul(D, np.transpose(V))))
    set_of_sigma = np.diag(D)
    D_k = np.diag(np.array(list(set_of_sigma[0 : k]) + [0] * (d - k)) )
    return U, D_k, V, set_of_sigma
