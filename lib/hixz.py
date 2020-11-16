import abc
import fractions
import numbers

import attr
import numpy as np

from .sum import Summation
from .util import i_to_the_power

__all__ = ['Base', 'Conj', 'Integral', 'Mult', 'T', 'HIXZ']


class Z(abc.ABC):
    """
    The functional part of a scalar term.  This is the actual part we're able
    to pulse-shape to achieve the results in the paper.

    Nothing actually instantiates the base class, it's just used as an abstract
    base for the other components.
    """
    def __init__(self, z):
        self.z = z

    def __repr__(self):
        return ''.join([self.__class__.__name__, '(', repr(self.z), ')'])

    def integrate(self):
        return Integral(self)

    def __mul__(self, other):
        if not isinstance(other, Z):
            return NotImplemented

        return Mult([self, other])

    def __eq__(self, other): return zcmp(self, other) == 0
    def __lt__(self, other): return zcmp(self, other) <  0
    def __le__(self, other): return zcmp(self, other) <= 0
    def __gt__(self, other): return zcmp(self, other) >  0
    def __ge__(self, other): return zcmp(self, other) >= 0


class Base(Z):
    def conj(self):
        return Conj(self)


class Conj(Z):
    def conj(self):
        return self.z


class Integral(Z):
    def conj(self):
        return Integral(self.z.conj())


class Mult(Z):
    def __init__(self, z):
        super().__init__(z)
        self.z = tuple(sorted(flatten_multiply(z)))

    def __repr__(self):
        return ''.join([self.__class__.__name__,
                        '((', ', '.join(map(repr, self.z)), '))'])

    def conj(self):
        return Mult(z.conj() for z in self.z)


@attr.s(frozen=True)
class HIXZ:
    h = attr.ib(converter=tuple)
    i = attr.ib(converter=int)
    x = attr.ib(converter=fractions.Fraction)
    z = attr.ib(type=Z)

    def __attrs_post_init__(self):
        if self.i not in [0, 1]:
            sign, i = i_to_the_power(self.i)
            # Slight hack to temporarily override attr's class freezing.
            object.__setattr__(self, 'i', i)
            object.__setattr__(self, 'x', sign*self.x)

    def conj(self):
        return type(self)(self.h,
                          self.i,
                          -self.x if self.i else self.x,
                          self.z.conj())

    def integrate(self):
        return type(self)(self.h, self.i, self.x, self.z.integrate())

    def __add__(self, other):
        if isinstance(other, HIXZ) and (self.h == other.h)\
                                   and (self.i == other.i)\
                                   and (self.z == other.z):
            x = self.x + other.x
            if x == 0:
                return None
            return attr.evolve(self, x=x)
        return NotImplemented

    def __radd__(self, other): return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __neg__(self):
        return self.__mul__(-1)

    def __mul__(self, other):
        if isinstance(other, numbers.Rational):
            return HIXZ(self.h, self.i, other * self.x, self.z)
        if isinstance(other, numbers.Complex):
            if other.real and other.imag:
                return NotImplemented
            x, i = (other.real, 0) if other.imag == 0 else (other.imag, 1)
            x = fractions.Fraction(x)
            if self.i + i == 2:
                x = -x
            return HIXZ(self.h, (self.i+i) % 2, x*self.x, self.z)
        if isinstance(other, HIXZ):
            sign, i = i_to_the_power(self.i + other.i)
            x = sign * self.x * other.x
            h_ = tuple(ours + theirs for ours, theirs in zip(self.h, other.h))
            return HIXZ(h_, i, x, self.z*other.z)
        return NotImplemented

    def __rmul__(self, other): return self.__mul__(other)


def hxz_hash(hxz):
    return (hxz.h, repr(hxz.z))


class T(Summation):
    max_h = np.inf

    def __init__(self, arg=(0,), i=0, const=1, f=1, hash_=hxz_hash):
        if isinstance(arg, dict):
            self._fast_init(arg, hash_)
            return
        if isinstance(arg, HIXZ):
            arg = [arg]
        arg = list(arg)
        if arg and isinstance(arg[0], numbers.Integral):
            arg = [HIXZ(tuple(arg), i, const, Base(f))]
        arg = [a for a in arg if sum(a.h) <= self.max_h]
        super().__init__(list(arg), hash_)

    def copy(self):
        return type(self)(self._dict.copy(), hash_=self._hash)

    def integrate(self):
        return type(self)((x.integrate() for x in self), hash_=self._hash)

    def conj(self):
        return type(self)((x.conj() for x in self), hash_=self._hash)


def flatten_multiply(zs):
    for z in zs:
        if isinstance(z, Mult):
            yield from flatten_multiply(z.z)
        else:
            yield z


_num = {Base: 0, Conj: 1, Integral: 2, Mult: 3}
def zcmp(a, b):
    """
    Lazy implementation of a comparison between two functional representations
    (the `Z` class).  It's old-style `cmp` format because I couldn't be
    bothered to write 6 different comparison functions when I was just quickly
    prototyping it.
    """
    if not isinstance(a, Z) and not isinstance(b, Z):
        return (a > b) - (a < b)
    elif not isinstance(a, Z):
        return 1
    elif not isinstance(b, Z):
        return -1
    # Impose some canonical ordering to ensure a unique sort.
    an, bn = _num[type(a)], _num[type(b)]
    if an != bn:
        return an - bn
    if not isinstance(a, Mult):
        return zcmp(a.z, b.z)
    # Mult.z is sorted by the __init__ method.
    for aa, bb in zip(a.z, b.z):
        c = zcmp(aa, bb)
        if c != 0:
            return c
    return len(a.z) - len(b.z)
