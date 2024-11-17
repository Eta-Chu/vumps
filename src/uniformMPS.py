import numpy as np

from scipy.sparse.linalg import eigs, LinearOperator

Tensor = np.ndarray


class UMPS:
    def __init__(self, A: Tensor):
        self.A = A
        self.shape = A.shape()
        self.D = self.shape[0]
        self.d = self.shape[1]

    def calTransMat(self):
        E = np.tensordot(self.A, np.conj(self.A), ([1], [1]))
        E = E.transpose([0, 2, 1, 3])
        
        return E

    def normalize(self, E: Tensor) -> Tensor:
        if self.D <= 50:
            E = self.calTransMat()
            E = np.reshape(E, ([self.D*2, self.D*2]))
            norm = eigs(E, k=1, which='LM', return_eigenvectors=False)
            self.A = self.A / np.sqrt(norm)
        else:
            def matvec(x):
                y = np.tensordot(self.A, x, ([2], [0]))
                y = np.tensordot(y, np.conj(self.A), ([1, 2], [1, 2]))
                y = y.reshape(self.D*2)

                return y

            op = LinearOperator(shape=(self.D*2, self.D*2), matvec=matvec)
            norm = eigs(op, k=1, which='LM', return_eigenvectors=False)
            self.A = self.A / np.sqrt(norm)

    @staticmethod
    def random(D: int, d: int) -> "UMPS":
        A = np.random.randn(D, d, D) + 1j * np.random.randn(D, d, D)
        return UMPS(A)


if __name__ == '__main__':

    print("debug uniformMPS.py")
