import numpy as np
from numpy.linalg import norm, matrix_rank


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
    #E := A ball centered around the origin which contains L
    r = 1/norm(A, 2)   #3 r \leq 1/max(\sigma)
    F = r * np.identity(d) 
    c = np.zeros(shape=(d, 1)) 
    n = A.shape[0]
    d = A.shape[1]
    while True: 
        ## 7-12
        while norm(np.matmul(A, c), p) >= 1: ## c not in L
            grad = grad_compute(A, c, p, d)
            H = 1/norm(grad, np.inf) * grad
            b = 1/sqrt(np.matmul(np.transpose(H), np.matmul(F, H))) * np.matmul(F, H)
            c = c - 1/(d + 1) * b
            F = (d ** 2)/(d ** 2 - 1) * (F - 2/(d + 1) * np.matmul(b, np.transpose(b)))
        ## 13 - 16



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
    G = np.linalg.cholesky(np.linalg.inv(F))
    S, V, D = linalg.svd(G)
    return D, V