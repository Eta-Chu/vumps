import numpy as np

from scipy.sparse.linalg import eigs, LinearOperator

Tensor = np.ndarray


class UMPS:
    def __init__(self, A: Tensor):
        self.A = A
        self.shape = A.shape
        self.D = self.shape[0]
        self.d = self.shape[1]

    def calTransMat(self, op=False, matrix=False):
        if op & matrix:
            raise ValueError('Can\'t output both matrix and operator.')
        if op:
            def matvec(x):
                x = x.reshape(self.D, self.D)
                y = np.tensordot(self.A, x, ([2], [0]))
                y = np.tensordot(y, np.conj(self.A), ([1, 2], [1, 2]))
                y = y.reshape(self.D**2)
                return y
            E = LinearOperator(shape=(self.D**2, self.D**2), matvec=matvec)
        else:
            E = np.tensordot(self.A, np.conj(self.A), ([1], [1]))
            E = E.transpose([0, 2, 1, 3])
            if matrix:
                E = E.reshape(self.D**2, self.D**2)
        return E

    def normalize(self) -> Tensor:
        if self.D <= 50:
            E = self.calTransMat(matrix=True)
        else:
            E = self.calTransMat(op=True)
        norm = eigs(E, k=1, which='LM', return_eigenvectors=False)
        self.A = self.A / np.sqrt(norm)

    @staticmethod
    def random(D: int, d: int) -> "UMPS":
        A = np.random.randn(D, d, D) + 1j * np.random.randn(D, d, D)
        return UMPS(A)


if __name__ == '__main__':
    x = UMPS.random(60, 2)
    x.normalize()
    E = x.calTransMat(op=True)
    val = eigs(E, k=1, which='LM', return_eigenvectors=False)
    print(val)
    print("debug uniformMPS.py")
