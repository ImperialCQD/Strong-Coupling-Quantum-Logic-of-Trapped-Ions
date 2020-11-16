import itertools
import numbers
import numpy as np
from .sum import Summation as _Summation


def _motion_mul(left, right):
    cl, dl = left
    cr, dr = right
    if dl == 0:
        return [(1, (cl+cr, dr))]
    if cr == 0:
        return [(1, (cl, dl+dr))]
    min_ = dl if dl < cr else cr
    coeffs = [1] * (min_ + 1)
    for k in range(1, min_ + 1):
        coeffs[k] = (coeffs[k-1] * (dl - k + 1) * (cr - k + 1)) // k
    return [(coeff, (cl+cr-k, dl+dr-k)) for k, coeff in enumerate(coeffs)]


def _qubit_allowed(qubit):
    for q in qubit:
        if q not in {'1', 'x', 'y', 'z', 'P', 'Q'}:
            return False
    return True


_qubit_mul_table = {
    # 'PQ' are dirty hacks---'P' is (1.y + y.1) and 'Q' is (1.1 + y.y), and the
    # use of integers not floats makes the calculations exact.
    'PP': (1, 'Q'),
    'PQ': (4, 'P'),
    'QP': (4, 'P'),
    'QQ': (4, 'Q'),
    '11': (1.,   '1'),
    '1x': (1.,   'x'),
    '1y': (1.,   'y'),
    '1z': (1.,   'z'),
    'x1': (1.,   'x'),
    'xx': (1.,   '1'),
    'xy': (1j,   'z'),
    'xz': (-1j,  'y'),
    'y1': (1.,   'y'),
    'yx': (-1j,  'z'),
    'yy': (1.,   '1'),
    'yz': (1j,   'x'),
    'z1': (1.,   'z'),
    'zx': (1j,   'y'),
    'zy': (-1j,  'x'),
    'zz': (1.,   '1'),
}


def _qubit_mul(left, right):
    cs, ops = zip(*map(lambda x: _qubit_mul_table[x[0]+x[1]], zip(left, right)))
    return np.prod(cs), "".join(ops)


def _opatom_part_mul(a, b):
    if isinstance(a, str):
        return [_qubit_mul(a, b)]
    return _motion_mul(a, b)


def _opatom_unknown_op_msg(op):
    return "Unknown operator type: " + repr(op)


def _opatom_canonicalise(op):
    msg = None
    if isinstance(op, str):
        if len(op) > 0:
            if any(x not in '1xyzPQ' for x in op):
                msg = "Qubit operators must be one of '1xyz'."
            else:
                return op
        else:
            msg = "Qubit operators must be non-empty."
    if isinstance(op, (tuple, list)):
        if len(op) == 2 and isinstance(op[0], int) and isinstance(op[1], int):
            return tuple(op)
        msg = "Motion operators must be a 2-tuple of ints."
    if msg is None:
        # Do this last to avoid a potentially expensive call to `repr()`.
        msg = _opatom_unknown_op_msg(op)
    raise ValueError(msg)


def _opatom_part_dag(op):
    if isinstance(op, str):
        return op
    if isinstance(op, tuple):
        return op[::-1]
    raise ValueError(_opatom_unknown_op_msg(op))


def _opatom_part_spec(op):
    return f'q{len(op)}' if isinstance(op, str) else 'm'


class OpAtom:
    def __init__(self, *ops, spec=None):
        self.ops = tuple(_opatom_canonicalise(op) for op in ops)
        self.spec = spec or ''.join(map(_opatom_part_spec, self.ops))

    def copy(self):
        return type(self)(*self.ops, spec=self.spec)

    def dag(self):
        return type(self)(*map(_opatom_part_dag, self.ops), spec=self.spec)

    def __mul__(self, other):
        if not isinstance(other, type(self)) or self.spec != other.spec:
            return NotImplemented
        parts = [_opatom_part_mul(a, b) for a, b in zip(self.ops, other.ops)]
        return [(np.prod(consts), type(self)(*ops, spec=self.spec))\
                for consts, ops in map(lambda x: zip(*x), itertools.product(*parts))]

    def __hash__(self):
        return hash(self.ops)

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.spec == other.spec\
               and all(lambda a, b: a == b for a, b in zip(self.ops, other.ops))

    def __repr__(self):
        return "".join([self.__class__.__name__, "(",
                        ", ".join(map(repr, self.ops)),
                        ")"])


class _OpScalarTuple:
    def __init__(self, scalar, op=None):
        self.scalar = scalar
        self.op = op if op is not None else OpAtom()

    def copy(self):
        return type(self)(self.scalar.copy(), self.op.copy())

    def integrate(self):
        return type(self)(self.scalar.integrate(), self.op.copy())

    def dag(self):
        return type(self)(self.scalar.conj(), self.op.dag())

    def _mul_inner(self, other, mul):
        if isinstance(other, numbers.Number):
            return [type(self)(mul(self.scalar, other), self.op.copy())]
        if isinstance(other, OpAtom):
            return [type(self)(mul(self.scalar, c), op)
                    for c, op in mul(self.op, other)]
        if isinstance(other, _OpScalarTuple):
            scalar_f = mul(self.scalar, other.scalar)
            return [type(self)(mul(scalar_f, c), op)
                    for c, op in mul(self.op, other.op)]
        return NotImplemented

    def __mul__(self, other): return self._mul_inner(other, lambda x, y: x*y)
    def __rmul__(self, other): return self._mul_inner(other, lambda x, y: y*x)

    def __add__(self, other):
        if isinstance(other, _OpScalarTuple) and self.op == other.op:
            out_scalar = self.scalar + other.scalar
            if out_scalar.is_empty():
                return None
            return type(self)(out_scalar, self.op.copy())
        return NotImplemented

    def __radd__(self, other): return self.__add__(other)
    def __sub__(self, other): return self.__add__(-other)
    def __rsub__(self, other): return (-self).__add__(other)
    def __neg__(self): return self.__mul__(-1)

    def __repr__(self):
        return "".join([
            self.__class__.__name__, "(",
            repr(self.scalar), ", ",
            repr(self.op), ")"])


def _op_scalar_tuple_sum_hash(x):
    return x.op


class Op(_Summation):
    def __init__(self, arg):
        if isinstance(arg, dict):
            self._fast_init(arg, _op_scalar_tuple_sum_hash)
            return
        arg = [(x if isinstance(x, _OpScalarTuple) else _OpScalarTuple(*x))
               for x in arg]
        arg = [x for x in arg if not x.scalar.is_empty()]
        super().__init__(arg, _op_scalar_tuple_sum_hash)

    def copy(self):
        return type(self)(self._dict.copy())

    def integrate(self):
        return type(self)(x.integrate() for x in self)

    def dag(self):
        return type(self)(x.dag() for x in self)

    def _mul_inner(self, other, mul):
        if isinstance(other, type(self)):
            pairs = itertools.product(self, other)
            return type(self)([z for x, y in pairs for z in mul(x, y)])
        try:
            return type(self)([y for x in self for y in mul(x, other)])
        except TypeError:
            return NotImplemented
