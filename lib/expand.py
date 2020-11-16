import fractions
from functools import reduce
import itertools
import operator as _operator

import numpy as np

from .op import Op, OpAtom
from .hixz import T
from .util import i_to_the_power, comm

__all__ = [
    'leading_order_terms', 'next_hamiltonian', 'single_mode_dk', 'transition',
]


def leading_order_terms(operator):
    order = np.inf
    out = []
    for scalar_op_tuple in operator.values():
        cur_scalar = []
        for hixz in scalar_op_tuple.scalar.values():
            order_ = sum(hixz.h)
            if order_ > order:
                continue
            if order_ < order:
                order = order_
                out, cur_scalar = [], []
            cur_scalar.append(hixz)
        out.append((T(cur_scalar), scalar_op_tuple.op))
    return Op(out)


def next_hamiltonian(h):
    lead = leading_order_terms(h)
    generator = 1j * lead.integrate()
    factor = fractions.Fraction(1, 1)
    out = None
    for k in itertools.count(1):
        to_add = factor*(h + fractions.Fraction(-1, k)*lead)
        for _ in [None]*(k-1):
            to_add = comm(generator, to_add)
        if to_add.is_empty():
            break
        if out is None:
            out = to_add
        else:
            out = out + to_add
        factor = fractions.Fraction(1, k) * factor
    return out


def _op(t, pairs):
    return [(t, OpAtom('P', *pairs)),
            (t.conj(), OpAtom('P', *(p[::-1] for p in pairs)))]


def single_mode_dk(k):
    sign, imaginary = i_to_the_power(k)
    # Use `reduce(mul, ...)` rather than `np.prod` to ensure we stay with
    # arbitrary-precision Python ints, not numpy fixed-width ones.
    frac = fractions.Fraction(sign, reduce(_operator.mul, range(1, k+1), 1))
    out = []
    for n in range((T.max_h - k)//2 + 1):
        create, destroy = k+n, n
        eta = 2*n + k
        out.append((eta, imaginary, frac, (create, destroy)))
        frac = -frac / ((n+k+1) * (n+1))
    return out


def transition(ks, base_etas=None, id_=None):
    base_etas = base_etas if base_etas is not None else (0,) * len(ks)
    id_ = id_ if id_ is not None else (*ks, *base_etas)
    dks = [single_mode_dk(k) for k in ks]
    terms = itertools.product(*dks)
    ops = []
    for term in terms:
        etas, imaginaries, fracs, operators = zip(*term)
        etas = tuple(a + b for a, b in zip(etas, base_etas))
        if sum(etas) > T.max_h:
            continue
        imaginary, scale = 0, fractions.Fraction(1, 1)
        for imaginary_, scale_ in zip(imaginaries, fracs):
            sign, imaginary = i_to_the_power(imaginary_ + imaginary)
            scale *= sign * scale_
        ops.extend(_op(T(etas, imaginary, scale, id_), operators))
    return ops
