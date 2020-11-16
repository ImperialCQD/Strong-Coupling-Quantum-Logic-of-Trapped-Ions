import numpy as np
import operator
import qutip
from qutip.fastsparse import fast_csr_matrix

__all__ = ['displace_fourier', 'FourierF', 'Sy', 'phase_space']

options = qutip.Options(atol=1e-12, rtol=1e-10, nsteps=2_500_00)

def laguerre(n: int, a: float, x: float) -> float:
    """
    Calculate the Laguerre polynomial result L_n^a(x), which is equivalent to
    Mathematica's LaguerreL[n, a, x].
    """
    if n == 0:
        return 1
    elif n == 1:
        return 1 + a - x
    # use a recurrence relation calculation for speed and accuracy
    # ref: http://functions.wolfram.com/Polynomials/LaguerreL3/17/01/01/01/
    l_2, l_1 = 1, 1 + a - x
    for m in range(2, n + 1):
        l_2, l_1 = l_1, ((a + 2*m - x - 1) * l_1 - (a + m - 1) * l_2) / m
    return l_1

def laguerre_range(n_start: int, n_end: int, a: float, x: float) -> np.ndarray:
    """
    Use the recurrence relation for nearest-neighbour in n of the Laguerre
    polynomials to calculate
        [laguerre(n_start, a, x),
         laguerre(n_start + 1, a, x),
         ...,
         laguerre(n_end - 1, a, x)]
    in linear time of `n_end` rather than quadratic.

    The time is linear in `n_end` not in the difference, because the initial
    calculation of `laguerre(n_start, a, x)` times linear time proportional to
    `n_start`, then each additional term takes another work unit.

    Reference: http://functions.wolfram.com/Polynomials/LaguerreL3/17/01/01/01/
    """
    if n_start >= n_end:
        return np.array([])
    elif n_start == n_end - 1:
        return np.array([laguerre(n_start, a, x)])
    out = np.empty((n_end - n_start, ), dtype=np.float64)
    out[0] = laguerre(n_start, a, x)
    out[1] = laguerre(n_start + 1, a, x)
    for n in range(2, n_end - n_start):
        out[n] = ((a + 2*n - x - 1) * out[n - 1] - (a + n - 1) * out[n - 2]) / n
    return out

def relative_rabi(lamb_dicke: float, n1: int, n2: int) -> float:
    """
    Get the relative Rabi frequency of a transition coupling motional levels
    `n1` and `n2` with a given Lamb--Dicke parameter.  The actual Rabi frequency
    will be the return value multiplied by the base Rabi frequency.
    """
    ldsq = lamb_dicke * lamb_dicke
    out = np.exp(-0.5 * ldsq) * (lamb_dicke ** abs(n1 - n2))
    out = out * laguerre(min(n1, n2), abs(n1 - n2), ldsq)
    fact = 1.0
    for n in range(1 + min(n1, n2), 1 + max(n1, n2)):
        fact = fact * n
    return out / np.sqrt(fact)

def relative_rabi_range(lamb_dicke: float, n_start: int, n_end: int, diff:int)\
        -> np.ndarray:
    """
    Get a range of Rabi frequencies in linear time of `n_end`.  The
    calculation of a single Rabi frequency is linear in `n`, so the naive
    version of a range is quadratic.  This method is functionally equivalent
    to
        np.array([rabi(n, n + diff) for n in range(n_start, n_end)])
    but runs in linear time."""
    if diff < 0:
        n_start = n_start + diff
        n_end = n_end + diff
        diff = -diff
    if n_start >= n_end:
        return np.array([])
    elif n_start == n_end - 1:
        return np.array([relative_rabi(lamb_dicke, n_start, n_start + diff)])
    ldsq = lamb_dicke * lamb_dicke
    const = np.exp(-0.5*ldsq) * lamb_dicke**diff
    lag = laguerre_range(n_start, n_end, diff, ldsq)
    fact = np.empty_like(lag)
    fact[0] = 1 / np.arange(n_start+1, n_start+diff+1, dtype=np.float64).prod()
    for i in range(1, n_end - n_start):
        fact[i] = fact[i - 1] * (n_start + i) / (n_start + i + diff)
    return const * lag * np.sqrt(fact)

def displace_fourier(mode, lamb_dicke, size):
    """
    Get a `qutip.Qobj` operator of the Fourier displacement operator with a
    given Lamb--Dicke parameter and `size` number of motional states.  `mode` is
    any integer, which denotes the Fourier mode.
    """
    a_mode = abs(mode)
    remove = np.exp(0.5 * lamb_dicke*lamb_dicke)
    coefficients = remove*relative_rabi_range(lamb_dicke, 0, size - a_mode, a_mode)
    coefficients = 1j**a_mode * coefficients
    indices = np.arange(max(0, -mode), min(size, size-mode), dtype=np.int32)
    indptr = np.empty(size + 1, dtype=np.int32)
    if mode < 0:
        indptr[:mode] = np.arange(1 + size - a_mode, dtype=np.int32)
        indptr[mode:] = size - a_mode
    else:
        indptr[:mode] = 0
        indptr[mode:] = np.arange(1 + size - a_mode, dtype=np.int32)
    return qutip.Qobj(fast_csr_matrix((coefficients, indices, indptr)))

Sy = qutip.tensor(qutip.sigmay(), qutip.qeye(2))\
     + qutip.tensor(qutip.qeye(2), qutip.sigmay())

gg = qutip.tensor(qutip.basis(2,1), qutip.basis(2,1))
ee = qutip.tensor(qutip.basis(2,0), qutip.basis(2,0))
ge = qutip.tensor(qutip.basis(2,1), qutip.basis(2,0))
eg = qutip.tensor(qutip.basis(2,0), qutip.basis(2,1))

def _x_op(size):
    return qutip.tensor(qutip.qeye(2), qutip.qeye(2),
                        np.sqrt(0.5)*(qutip.create(size) + qutip.destroy(size)))
def _p_op(size):
    return qutip.tensor(qutip.qeye(2), qutip.qeye(2),
                        1j/np.sqrt(2)*(qutip.create(size)-qutip.destroy(size)))
_Sy_es = Sy.eigenstates()[1][0]

def _get_ns(operator):
    while isinstance(operator, list):
        operator = operator[0]
    return operator.dims[0][-1]

def phase_space(hamiltonian, times, motion=None, states=False):
    """
    Get the `x` and `p` components of the phase space trajectory at the times
    `times`.  If the `motion` keyword argument is passed, then the system starts
    in that motional state (otherwise it starts in |g0>).
    """
    ns = _get_ns(hamiltonian)
    if motion is None:
        motion = qutip.basis(ns, 0).proj()
    if motion.isket:
        motion = motion.proj()
    _sy_state = qutip.tensor(_Sy_es.proj(), motion)
    st = qutip.mesolve(hamiltonian, _sy_state, times).states
    x, y = qutip.expect([_x_op(ns), _p_op(ns)], st)
    return ((x, y), st) if states else (x, y)


def _ensure_array(xs):
    if np.isscalar(xs):
        return np.array([xs])
    return np.array(xs)
def _distribute(f, x1, x2):
    bc = np.broadcast(x1.reshape(-1, 1), x2.reshape(1, -1))
    return np.array([f(*x) for x in bc])

def _fprod(f1, f2):
    return lambda *args: f1(*args) * f2(*args)

class FourierF:
    def __init__(self, coefficients, frequencies, decouple=None):
        self.coefficients = _ensure_array(coefficients)
        self.frequencies = _ensure_array(frequencies)
        if hasattr(decouple, '__call__'):
            self.decouple = [decouple] * len(self.coefficients)
        elif decouple is None:
            self.decouple = [(lambda *_: 1) for _ in self.coefficients]
        else:
            self.decouple = _ensure_array(decouple)
    def copy(self):
        return type(self)(self.coefficients.copy(), self.frequencies.copy(),
                          list(self.decouple))
    def _iter(self):
        return zip(self.coefficients, self.frequencies, self.decouple)
    def __call__(self, t, *_):
        return sum(d(t)*c*np.exp(1j*f*t) for c, f, d in self._iter())
    def conj(self):
        return FourierF(np.conj(self.coefficients), -self.frequencies,
                        list(self.decouple))
    def add_frequency(self, frequency):
        return FourierF(self.coefficients.copy(), self.frequencies + frequency,
                        list(self.decouple))
    def __add__(self, other):
        if other == 0:
            return self.copy()
        if not isinstance(other, FourierF):
            return NotImplemented
        return FourierF(np.concatenate([self.coefficients, other.coefficients]),
                        np.concatenate([self.frequencies, other.frequencies]),
                        self.decouple + other.decouple)
    def __radd__(self, other): return self.__add__(other)
    def __mul__(self, other):
        if np.isscalar(other):
            return FourierF(other * self.coefficients, self.frequencies.copy(),
                            self.decouple.copy())
        if not isinstance(other, FourierF):
            return NotImplemented
        coefficients = _distribute(operator.__mul__,
                                   self.coefficients, other.coefficients)
        frequencies = _distribute(operator.__add__,
                                  self.frequencies, other.frequencies)
        decouple = _distribute(_fprod, self.decouple, other.decouple)
        return FourierF(coefficients, frequencies, decouple)
    def __rmul__(self, other): return self.__mul__(other)
    def __repr__(self):
        return "".join(['FourierF(',
                        repr(list(self.coefficients)), ', ',
                        repr(list(self.frequencies)), ', ',
                        repr(self.decouple), ')'])
