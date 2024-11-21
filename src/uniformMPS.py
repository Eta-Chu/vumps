import numpy as np

from scipy.sparse.linalg import eigs, LinearOperator

# todo class `TransferMatrix` contain transpose() ... 

Tensor = np.ndarray


class UMPS:
    def __init__(self, A: Tensor, isnormalize=False):
        self.A = A
        self.shape = A.shape
        self.D = self.shape[0]
        self.d = self.shape[1]
        self.isnormalize = isnormalize
        if not self.isnormalize:
            self.normalize()

    def calTransMat(self, op=False, matrix=False):
        if op & matrix:
            raise ValueError('Can\'t output both matrix and operator.')
        
        if (not op) and (not matrix):
            if self.D <= 30:
                matrix = True
            else:
                op = True

        if op:
            E = self._linearOp(self.A, self.A.conj(), left=True)
        else:
            E = np.tensordot(self.A, np.conj(self.A), ([1], [1]))
            E = E.transpose([0, 2, 1, 3])
            if matrix:
                E = E.reshape(self.D**2, self.D**2)
        return E

    def normalize(self) -> Tensor:
        if self.D <= 30:
            E = self.calTransMat(matrix=True)
        else:
            E = self.calTransMat(op=True)
        norm = eigs(E, k=1, which='LM', return_eigenvectors=False)
        self.A = self.A / np.sqrt(norm)
        self.isnormalize = True

    def calFixedPoint(self, mode='left') -> Tensor:
        if not (mode in ['left', 'right']):
            raise ValueError(f'mode should be left or right, but get {mode}')
        if mode == 'left':
            left = True
        else:
            left = False

        op = self._linearOp(self.A, self.A.conj(), left=left)
        _, f = eigs(op, k=1, which='LM')
        f = f.reshape([self.D, self.D])
        phase = np.trace(f) / np.abs(np.trace(f))
        f /= phase
        f = (f + f.conj().T) / 2
        f *= np.sign(np.trace(f))

        return f
    
    def fixedPoint(self):
        l = self.calFixedPoint(mode='left')
        r = self.calFixedPoint(mode='right')
        l = l / np.tensordot(l, r, ([0, 1], [0, 1]))
        return l, r

    @staticmethod
    def random(D: int, d: int) -> "UMPS":
        A = np.random.randn(D, d, D) + 1j * np.random.randn(D, d, D)
        return UMPS(A)

    @staticmethod
    def _linearOp(T: Tensor, B: Tensor, left=True) -> LinearOperator:
        dimlT, dimrT = T.shape[0], T.shape[2]
        dimlB, dimrB = B.shape[0], B.shape[2]
        if left:
            def matvec(x):
                x = x.reshape([dimlT, dimlB])
                y = np.tensordot(x, T, ([0], [0]))
                y = np.tensordot(y, B, ([0, 1], [0, 1]))
                y = y.reshape(dimrT*dimrB)
                return y
            shape_op = (dimrT*dimrB, dimlT*dimlB)
        else:
            def matvec(x):
                x = x.reshape([dimrT, dimrB])
                y = np.tensordot(T, x, ([2], [0]))
                y = np.tensordot(y, B, ([1, 2], [1, 2]))
                y = y.reshape(dimlT*dimlB)
                return y
            shape_op = (dimlT*dimlB, dimrT*dimrB)
        op = LinearOperator(dtype=T.dtype,
                            shape=shape_op,
                            matvec=matvec)

        return op


if __name__ == '__main__':
    x = UMPS.random(60, 2)
    E = x.calTransMat(op=True)
    val = eigs(E, k=1, which='LM', return_eigenvectors=False)
    print(val)
    print("debug uniformMPS.py")
