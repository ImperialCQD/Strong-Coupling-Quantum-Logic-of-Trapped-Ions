__all__ = ['i_to_the_power', 'comm']


def i_to_the_power(k):
    """
    Return the sign (-1 or 1) and minimal integer power (0 or 1) of i to an
    integer power.
    """
    sign = {0: 1, 1: 1, 2: -1, 3: -1}[k % 4]
    imaginary = k % 2
    return sign, imaginary


def comm(a, b):
    """Commutator of a and b."""
    # Using `-1 * a` means everything works even when I've forgotten to
    # override __sub__ in classes.
    return a*b + (-1)*b*a
