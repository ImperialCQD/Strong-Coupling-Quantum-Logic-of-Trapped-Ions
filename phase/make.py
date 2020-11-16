import numpy as np
import matplotlib
from matplotlib import pyplot

# Ensure that the `lib` directory in this directory's parent is visible to
# Python to allow this to run.
#
# This import requires QuTiP 4.3 <= x < 5.0.
from lib import ms


# Depending on the version of QuTiP used, you might not get results equal in
# all decimal places to the values given here, but they should be within ~1e-7
# relative tolerance.


def scale_around_point(data, scale_factor, point):
    data = np.array(data)
    return scale_factor * data + (1 - scale_factor) * np.array(point)[:, None]


def h_1(ld, ns, det):
    """
    Get the QuTiP-format time-dependent Hamiltonian for the base gate.
    """
    rabi = det / (4*ld)
    op1 = -qutip.tensor(ms.Sy, ms.displace_fourier(1, ld, ns))
    f1 = ms.FourierF([rabi], [det])
    return [[op1, f1], [op1.dag(), f1.conj()]]


def h_2(ld, ns, det):
    """
    Get the QuTiP-format time-dependent Hamiltonian for the gate up to eta^3.
    """
    p = np.polynomial.Polynomial([-1, 8*ld**2 + 8*ld**4, -24*ld**6])
    rabi = det * np.sqrt(np.min(p.roots()))
    op1 = -qutip.tensor(ms.Sy, ms.displace_fourier(1, ld, ns))
    op2 = -1j * qutip.tensor(ms.Sy, ms.displace_fourier(2, ld, ns))
    f1 = ms.FourierF([rabi], [2*det])
    f2 = ms.FourierF([rabi], [det])
    return [[op1, f1], [op1.dag(), f1.conj()],
            [op2, f2], [op2.dag(), f2.conj()]]


def h_3(ld, ns, det):
    """
    Get the QuTiP-format time-dependent Hamiltonian for the gate up to eta^4.
    """
    p = np.polynomial.Polynomial([-0.125,
                                  0.2 * (2 + 2*ld*ld + ld**4),
                                  -(56/375) * ld*ld * (1 + 2*ld*ld),
                                  (382/9375) * ld**4])
    rabi = np.sqrt(np.min(p.roots())) / ld
    op1 = -qutip.tensor(ms.Sy, ms.displace_fourier(1, ld, ns))
    op2 = -1j * qutip.tensor(ms.Sy, ms.displace_fourier(2, ld, ns))
    op3 = -qutip.tensor(ms.Sy, ms.displace_fourier(3, ld, ns))
    f1 = ms.FourierF([rabi * det], [5*det])
    f2 = ms.FourierF([rabi * np.sqrt(4/5) * det,
                      rabi*rabi * np.sqrt(49/125) * det*ld*ld],
                     [2*det, -7*det])
    f3 = ms.FourierF([np.sqrt(0.6)*rabi * det], [1*det])
    return [[op1, f1], [op1.dag(), f1.conj()],
            [op2, f2], [op2.dag(), f2.conj()],
            [op3, f3], [op3.dag(), f3.conj()]]


contour_locs = np.array([0, 1/3, 2/3, 1])
scale = 0.05 * np.array([1, 1, 1])
ts = np.linspace(0, 1, 301)
levels = np.array([1, 0.5, 0.1])


def make_contours(ld, nbar, ns, hamiltonian, scale):
    motion = qutip.thermal_dm(ns, nbar)
    w_n = 201
    w_w = 5
    sy_ket = qutip.tensor(ms.Sy.eigenstates()[1][0], qutip.qeye(ns))
    h = hamiltonian(ld, ns, 2*np.pi)
    (xs, ys), states = ms.phase_space(h, ts, motion, states=True)
    contour_i = np.int64((len(xs) - 1) * contour_locs)
    figure, axes = pyplot.subplots()
    axes.axhline(0, color='grey', dashes=(5, 5), linewidth=1)
    axes.axvline(0, color='grey', dashes=(5, 5), linewidth=1)
    axes.set_aspect('equal')
    axes.set_xlabel('$\\langle\\hat x\\rangle$')
    axes.set_ylabel('$\\langle\\hat p\\rangle$')
    axes.set_title(f'$\\eta={ld}$, $\\bar n={nbar}$')
    axes.plot(xs, ys)
    contours = []
    for ii, i in enumerate(contour_i):
        centre = np.array([xs[i], ys[i]])
        w_x = np.linspace(centre[0] - w_w, centre[0] + w_w, w_n)
        w_p = np.linspace(centre[1] - w_w, centre[0] + w_w, w_n)
        w_z = qutip.wigner(sy_ket.dag() * states[i] * sy_ket, w_x, w_p)
        w_l = np.exp(-0.5 * levels**2) * np.max(w_z)
        qcs = axes.contour(*scale_around_point([w_x, w_p], scale, centre),
                           w_z, w_l,
                           colors=[matplotlib.cm.plasma(ii/4, alpha=0.5)])
        contours.append(qcs.allsegs)
    return (xs, ys), contours


def write_phase_space(xys, file):
    for xy in [np.array(x).T for x in xys]:
        for x, y in xy:
            print(f'{x: 10.9e} {y: 10.9e}', file=file)
        print(file=file)


def write_contours(contours, file):
    for time in contours:
        for contour in time:
            for x, y in contour[0]:
                print(f'{x: 10.9e} {y: 10.9e}', file=file)
            print(file=file)
        print(file=file)


def do_set(ld, nbar, scale):
    ns = 50
    hs = [h_1, h_2, h_3]
    xys = []
    all_contours = []
    for h in hs:
        (xs, ys), contours = make_contours(ld, nbar, ns, h, scale)
        xys.append((xs, ys))
        all_contours.append(contours)
    stem = f"phase-eta{ld}-nbar{nbar}"
    with open(stem + ".dat", "w") as file:
        write_phase_space(xys, file)
    for k, contours in enumerate(all_contours):
        with open(stem + f"-contours-{k+1}.dat", "w") as file:
            write_contours(contours, file)


if __name__ == '__main__':
    do_set(0.1, 0.01, 0.25)
    do_set(0.5, 2, 0.05)
