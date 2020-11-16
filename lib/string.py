"""
Functions for turning the output of Python symbolic representations into
Mathematica symbolic representations.
"""

from .op import Op
from .hixz import Base, Conj, Mult, Integral

__all__ = ['ms']

TONES = {}
LOOKUP = {}
ZERO_UNKNOWNS = False


def _ms_tuple(z_tup):
    """Stringify a tuple into a Mathematica-like argument list."""
    return '[' + ','.join(map(str, z_tup)) + ']'


def _ms_z(z):
    """Stringify an hixz.Z class."""
    if isinstance(z, Base):
        out = []
        tones = TONES.get(z.z, 1)
        for tone in range(1, tones + 1):
            strength = ('0' if z.z not in LOOKUP and ZERO_UNKNOWNS
                        else '\\[CapitalOmega]' + _ms_tuple(z.z + (tone,)))
            if z.z not in LOOKUP:
                frequency = 'k' + _ms_tuple(z.z + (tone,))
                if tone != 1:
                    frequency = str(tone) + frequency
            else:
                frequency = 'Times[' + str(tone) + ',' + str(LOOKUP[z.z]) + ']'
            out.append(frequency + '->{' + strength + '}')
        return 'MS`piece[{<|' + ','.join(out) + '|>},{\\[Infinity]}]'
    if isinstance(z, Conj):
        return 'MS`conjugate[' + _ms_z(z.z) + ']'
    if isinstance(z, Mult):
        return 'MS`times[' + ','.join(map(_ms_z, z.z)) + ']'
    if isinstance(z, Integral):
        return 'MS`integrate[' + _ms_z(z.z) + ']'
    raise ValueError


def _ms_scalar(f):
    """
    Stringify a scalar term into a Mathematica object.  We take care to ensure
    that the exactness of the fractions is maintained.
    """
    def mapping(f):
        frac = f'Rational[{f.x.numerator},{f.x.denominator}]'
        x = 'Complex[' + (('0,'+frac) if f.i else (frac+',0')) + ']'
        h = ','.join(f'\\[Eta]{k+1}^({h})' for k, h in enumerate(f.h))
        return ('MS`times['
                + ','.join([f'Times[{h},{x}]', _ms_z(f.z)])
                + ']')

    f = tuple(f)
    if len(f) == 0:
        return '0'
    return 'MS`plus[' + ', '.join(map(mapping, f)) + ']'


def _ms_opatom(op):
    """
    Stringify the operator part of a term.  If multiple motional modes are
    being considered, then the resultant Mathematica object will have `2n+1`
    arguments for `n` motional modes.
    """
    sy = 1 if op[0] == 'P' else 2
    pairs = ','.join(str(x) for pair in op[1:] for x in pair)
    return f'op[{sy},{pairs}]'


def _ms_opscalar_pair(op):
    return '{' + _ms_scalar(op.scalar) + ', ' + _ms_opatom(op.op.ops) + '}'


def ms(ops):
    """
    Stringify a single Op or iterable of Ops into a Mathematica representation.
    An iterable is turned into a Mathematica list.  Each pair of scalar and op
    is turned into a two-term Mathematica list
        {scalar, op}
    (it's less error-prone to pattern-match on this than on multiplication due
    to default values).

    The scalar comes out in the form defined in the attached Mathematica
    notebook, and the operator is
        op[r, p, q]
    for a term
        a^p . a^{\\dagger q} . S_y^r
    where p and q are non-negative integers and r is 1 or 2.
    """
    def mapping(term):
        return '{' + ', '.join(map(_ms_opscalar_pair, term)) + '}'

    if isinstance(ops, Op):
        return mapping(ops)
    return '{' + ','.join(map(mapping, ops)) + '}'
