"""Map cuEST AO order and phase onto PySCF's.

cuEST orders solid harmonics 0, 1c, 1s, 2c, 2s, ... (p as z, x, y); PySCF uses
m = -l..l (p as x, y, z). Shells are paired by atom, L and exponents. Phases
come from the overlap matrix: after the permutation, each AO is flipped when
its overlap row disagrees in sign.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from saddle.basis import BasisSet

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def _pyscf_index_of_cuest_component(L: int, k: int) -> int:
    """Position in PySCF's shell of cuEST's k-th component."""
    if L == 1:
        return (2, 0, 1)[k]  # cuEST z, x, y -> PySCF x=0, y=1, z=2
    m = 0 if k == 0 else ((k + 1) // 2 if k % 2 == 1 else -(k // 2))
    return m + L


def permutation(basis: BasisSet, symbols: Sequence[str], mol: Any) -> IntArray:
    """perm such that pyscf_matrix = cuest_matrix[perm][:, perm] (before signs).

    Shells are paired by atom, angular momentum and exponents, because basis
    sources list shells in different orders (BSE vs PySCF def2-TZVP chlorine).
    """
    if not basis.pure:
        raise ValueError("only pure (spherical) bases are mapped")
    # cuEST AO offset of every Saddle shell, keyed by atom.
    ours: list[list[tuple[int, tuple[float, ...], int]]] = []
    offset = 0
    for sym in symbols:
        atom: list[tuple[int, tuple[float, ...], int]] = []
        for shell in basis.shells[sym]:
            atom.append((shell.angular_momentum, tuple(sorted(shell.exponents)), offset))
            offset += shell.n_functions
        ours.append(atom)
    to_cuest: list[int] = []
    for b in range(mol.nbas):
        if mol.bas_nctr(b) != 1:
            raise ValueError("general contractions in the PySCF basis are not mapped")
        iatom, L = int(mol.bas_atom(b)), int(mol.bas_angular(b))
        exps = tuple(sorted(float(e) for e in mol.bas_exp(b)))
        match = [
            i
            for i, (l2, e2, _) in enumerate(ours[iatom])
            if l2 == L and len(e2) == len(exps) and np.allclose(e2, exps, rtol=1e-9)
        ]
        if not match:
            raise ValueError(f"no Saddle shell matches PySCF shell {b} (atom {iatom}, L={L})")
        _, _, start = ours[iatom].pop(match[0])
        block = [0] * (2 * L + 1)
        for k in range(2 * L + 1):
            block[_pyscf_index_of_cuest_component(L, k)] = start + k
        to_cuest.extend(block)
    if any(ours_atom for ours_atom in ours):
        raise ValueError("Saddle has shells with no PySCF counterpart")
    return np.asarray(to_cuest, dtype=np.int64)


def phases(s_cuest: FloatArray, s_pyscf: FloatArray, perm: IntArray) -> FloatArray:
    """+-1 per PySCF AO, from the sign pattern of the permuted overlap."""
    sp = s_cuest[np.ix_(perm, perm)]
    n = sp.shape[0]
    sign = np.ones(n)
    for i in range(n):
        row = sp[i] * sign
        mask = np.abs(s_pyscf[i]) > 1e-6
        mask[i] = False
        if mask.any():
            agree = np.sum(np.sign(row[mask]) == np.sign(s_pyscf[i][mask]))
            if agree < mask.sum() / 2:
                sign[i] = -1.0
    return sign


def to_pyscf(matrix: FloatArray, perm: IntArray, sign: FloatArray) -> FloatArray:
    m = np.asarray(matrix)[np.ix_(perm, perm)]
    return m * sign[:, None] * sign[None, :]


def from_pyscf(matrix: FloatArray, perm: IntArray, sign: FloatArray) -> FloatArray:
    out = np.empty_like(matrix)
    signed = np.asarray(matrix) * sign[:, None] * sign[None, :]
    out[np.ix_(perm, perm)] = signed
    return out
