import numpy as np
from typing import Union
from numpy.linalg import norm
from scipy.sparse.linalg import eigs, LinearOperator
from scipy.linalg import qr, rq

# todo class `TransferMatrix` contain transpose() ... 
ts = np
Tensor = np.ndarray

__all__ = [
        'UMPS'
        ]


class UMPS:
    def __init__(self, A: Tensor, isnormalize=False):
        self.A = A
        self.shape = A.shape
        self.D = self.shape[0]
        self.d = self.shape[1]
        self.isnormalize = isnormalize
        if not self.isnormalize:
            self.normalize()
        self.canonical_form = 'Normal'
        self.l = None
        self.r = None
        self.L = None
        self.Al = None
        self.Ar = None

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
            raise ValueError(f'mode should be left or right, but get {mode}.')
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
        self.r = r
        self.l = l.T

    def leftCanonical(self, iteration=True, tol=1e-14, L0=None, maxiter=None):
        if self.canonical_form != 'left':
            if iteration:
                if maxiter is None:
                    maxiter = self.D*self.d
                if L0 is None:
                    L0 = ts.random.randn(self.D, self.D)
                L0 = L0 / norm(L0)
                Al = ts.tensordot(L0, self.A, ([1], [0]))
                Al, L = positive_qr(Al.reshape([self.D*self.d, self.D]))
                delta = norm(L - L0)
                i = 1
                while delta > tol:
                    Al = ts.tensordot(L, self.A, ([1], [0]))
                    Al, L0 = positive_qr(Al.reshape([self.D*self.d, self.D]))
                    L0 = L0 / norm(L0)
                    delta = norm(L0 - L)
                    L = L0
                    if i > maxiter:
                        raise ValueError(f'The iteration still not converge \
                                after {i} step.')
                self.Al = Al.reshape([self.D, self.d, self.D])
                self.L = L
            else:
                if self.l is None:
                    self.fixedPoint()
                val, vec = ts.linalg.eigh(self.l)
                val = ts.sqrt(val)
                self.L = ts.dot(ts.diag(val), vec.T.conj())
                self.Linverse = ts.dot(vec, ts.diag(1 / val))
                self.Al = ts.tensordot(self.L, self.A, ([1], [0]))
                self.Al = ts.tensordot(self.Al, self.Linverse, ([2], [0]))
    
            self.canonical_form = 'left'

    def rightCanonical(self, iteration=True, tol=1e-14, R0=None, maxiter=None):
        if self.canonical_form != 'right':
            if iteration:
                if maxiter is None:
                    maxiter = self.D
                if R0 is None:
                    R0 = ts.random.randn(self.D, self.D)
                R0 = R0 / norm(R0)
                Ar = ts.tensordot(self.A, R0, ([2], [0]))
                R, Ar = positive_rq(Ar.reshape([self.D, self.d*self.D]))
                delta = norm(R - R0)
                i = 1
                while delta > tol:
                    Ar = ts.tensordot(self.A, R, ([2], [0]))
                    R0, Ar = positive_rq(Ar.reshape([self.D, self.d*self.D]))
                    R0 = R0 / norm(R0)
                    delta = norm(R0 - R)
                    R = R0
                    if i > maxiter:
                        raise ValueError(f'The iteration still not converge \
                                after {i} step.')
                self.Ar = Ar.reshape([self.D, self.d, self.D])
                self.R = R
            else:
                if self.r is None:
                    self.fixedPoint()
                val, vec = ts.linalg.eigh(self.r)
                val = ts.sqrt(val)
                self.R = vec * val
                self.Rinverse = ts.dot(ts.diag(1 / val), vec.T.conj())
                self.Ar = ts.tensordot(self.Rinverse, self.A, ([1], [0]))
                self.Ar = ts.tensordot(self.Ar, self.R, ([2], [0]))
            self.canonical_form = 'right'

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


def positive_qr(a: Tensor) -> Union[Tensor, Tensor]:
    q, r = qr(a, mode='economic')

    sign = ts.diag(ts.sign(ts.diag(r)))
    q = ts.dot(q, sign)
    r = ts.dot(sign, r)
    return q, r


def positive_rq(a: Tensor) -> Union[Tensor, Tensor]:
    r, q = rq(a, mode='economic')

    sign = ts.diag(ts.sign(ts.diag(r)))
    q = ts.dot(sign, q)
    r = ts.dot(r, sign)
    return r, q
