import unittest
import numpy as np
from scipy.spatial import distance_matrix

from lowner import lowner, l_p_low_rank


class TestLowner(unittest.TestCase):
    def test1(self):
        #A = np.asarray([[1, 2, 0], [0, 1, 117], [-7, 0, 1]])
        A = np.random.rand(20, 20)
        k = 15
        U, D_k, V, sigmas = l_p_low_rank(A, k, 1, 100)
        print(np.linalg.norm(A - U @ D_k @ V))
        U, D_k, V, sigmas = l_p_low_rank(A, k, 1, 20)
        print(np.linalg.norm(A - U @ D_k @ V))
        U, D, V = np.linalg.svd(A, full_matrices=False)
        print(np.linalg.norm(A - U[:, :k] @ np.diag(D[:k]) @ V[:k, :]))

    def test2(self):
        A = np.asarray([[3.0, 5.0], [0.0, 1.0]])
        k = 2
        U, D_k, V, sigmas = l_p_low_rank(A, k, 1, 25)
        print(np.linalg.norm(A - U @ D_k @ V))
        U, D, V = np.linalg.svd(A, full_matrices=False)
        print(np.linalg.norm(A - U[:, :k] @ np.diag(D[:k]) @ V[:k, :]))


if __name__ == '__main__':
    TestLowner().test1()