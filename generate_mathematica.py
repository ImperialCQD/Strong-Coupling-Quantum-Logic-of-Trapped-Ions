import lib


def two_sideband_case():
    # maximum power of eta to keep - the class variable is a gross hack (it's
    # basically a global variable)
    lib.T.max_h = 3
    ops = []
    # considering sidebands 1 and 2
    for k in [1, 2]:
        # setting `base_etas` to -1 is the factor of 1/eta in equation (2).
        ops.extend(lib.transition((k,), id_=(k,), base_etas=(-1,)))
    h = lib.Op(ops)
    cur, prev = h, None
    out = []
    while cur is not None:
        cur, prev = lib.next_hamiltonian(cur), cur
        out.append(1j * lib.leading_order_terms(prev).integrate())
    # in the paper we do the first two transformations in one go, which is why
    # I join them here
    out = [out[0]+out[1]] + out[2:]

    # This bit generates Mathematica code that can be imported into the
    # notebook.  Generally you should save it to a file and import in
    # Mathematica, because directly copying/pasting will most likely hang
    # Mathematica's formatting engine.
    #
    # LOOKUP can be used to specify certain frequency components as integers,
    # which means Mathematica will handle them properly even if they result in
    # a zero-frequency integral at some point (otherwise it will catch the fact
    # there _could_ have been a zero integral, but assume it wasn't zero).
    # It's actually just another gross global variable hack.
    #
    # The string-ification assumes you're using a sum of exponential terms,
    # like in the paper, but nothing before that makes that assumption.
    lib.string.LOOKUP = {(1,): 2, (2,): 1}
    # If you want to import it into Mathematica, print it to a file instead.
    return "toEta3=" + lib.string.ms(out) + ";"


def three_sideband_case():
    lib.T.max_h = 4
    ops = []
    for k in [1, 2]:
        ops.extend(lib.transition((k,), id_=(k, 0), base_etas=(-1,)))
    # Add the other tone.
    ops.extend(lib.transition((2,), id_=(2, 1), base_etas=(0,)))
    h = lib.Op(ops)
    cur, prev = h, None
    out = []
    while cur is not None:
        cur, prev = lib.next_hamiltonian(cur), cur
        out.append(1j * lib.leading_order_terms(prev).integrate())
    out = [out[0]+out[1]] + out[2:]
    # An empty lookup generates frequencies called `k[2, 0]` and stuff like
    # that (using the id parameter above as the argument).
    lib.string.LOOKUP = {}
    return "toEta4=" + lib.string.ms(out) + ";"


if __name__ == '__main__':
    # ~5-10s runtime.
    with open("two_sideband.m", "w") as f:
        print(two_sideband_case(), file=f)
    with open("three_sideband.m", "w") as f:
        print(three_sideband_case(), file=f)
