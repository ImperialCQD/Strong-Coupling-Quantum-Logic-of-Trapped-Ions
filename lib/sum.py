import itertools
import functools


class Summation:
    def __init__(self, els, hash_):
        if isinstance(els, dict):
            self._fast_init(els, hash_)
            return
        self._hash = hash_
        self._dict = {}
        for el in els:
            self.insert(el)

    def _fast_init(self, dict_, hash_):
        self._dict = dict_
        self._hash = hash_

    def copy(self):
        return Summation(self._dict.copy(), self._hash)

    def insert(self, value):
        if value is None:
            return
        key = self._hash(value)
        if key in self._dict:
            self.insert(self._dict.pop(key) + value)
        else:
            self._dict[key] = value

    def is_empty(self):
        return not len(self._dict)

    def values(self):
        return list(self._dict.values())

    def __iter__(self): return iter(self._dict.values())

    def _mul_inner(self, other, mul):
        if isinstance(other, type(self)):
            pairs = itertools.product(self, other)
            return type(self)([mul(x, y) for x, y in pairs], hash_=self._hash)
        try:
            return type(self)([mul(x, other) for x in self], hash_=self._hash)
        except TypeError:
            return NotImplemented

    def __mul__(self, other): return self._mul_inner(other, lambda x, y: x*y)
    def __rmul__(self, other): return self._mul_inner(other, lambda x, y: y*x)

    def __add__(self, other):
        out = self.copy()
        if isinstance(other, Summation):
            for el in other.values():
                out.insert(el)
        else:
            try:
                out.insert(other)
            except TypeError:
                return NotImplemented
        return out

    def __radd__(self, other): return self.__add__(other)
    def __sub__(self, other): return self.__add__(-other)
    def __rsub__(self, other): return (-self).__add__(other)
    def __neg__(self): return self.__mul__(-1)

    def __repr__(self):
        return "".join([self.__class__.__name__, "(", repr(self.values()), ")"])
