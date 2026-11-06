"""Build the C3 starting geometries (X- + CH3CH2X, X = F, Cl).

Atom order in every file: Nu(1) Ca(2) Cb(3) X(4) Ha(5) Ha(6) Hb(7) Hb(8) Hb(9).
anti-E2 transfers H7 (anti-periplanar to X); syn-E2 transfers H8 (syn-periplanar).
TS guesses are rough: RS-P-RFO refines them along the declared bond changes.
Distances (angstrom) are typical of published MP2/DFT TSs for these systems,
and only need to be close enough for the search to start in the right basin.

Run from this directory: python build_guesses.py
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent / "species"

PARAMS = {
    "F": {"sn2_cx": 1.85, "e2_cx": 1.75, "e2_ch": 1.35, "e2_nuh": 1.15, "cx": 1.40, "nu_r": 2.6},
    "Cl": {"sn2_cx": 2.33, "e2_cx": 2.35, "e2_ch": 1.55, "e2_nuh": 1.55, "cx": 1.80, "nu_r": 3.2},
}
CC, CH = 1.52, 1.09


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def tetra(
    center: np.ndarray, axis: np.ndarray, ref: np.ndarray, length: float, phis: list[float]
) -> list[np.ndarray]:
    """Positions at 109.5 deg from `axis` around `center`, at dihedral angles `phis` from `ref`."""
    a = unit(axis)
    e1 = unit(ref - np.dot(ref, a) * a)
    e2 = np.cross(a, e1)
    t = np.deg2rad(180.0 - 109.47)
    return [
        center
        + length
        * (np.cos(t) * a + np.sin(t) * (np.cos(np.deg2rad(p)) * e1 + np.sin(np.deg2rad(p)) * e2))
        for p in phis
    ]


def write(name: str, comment: str, symbols: list[str], xyz: list[np.ndarray]) -> None:
    lines = [str(len(symbols)), comment]
    lines += [
        f"{s:2s} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}" for s, p in zip(symbols, xyz, strict=True)
    ]
    (HERE / f"{name}.xyz").write_text("\n".join(lines) + "\n")


def ethyl_halide(x: str, p: dict[str, float]) -> list[np.ndarray]:
    """Staggered CH3CH2X: Ca origin, Cb on +x, X anti to H7."""
    ca, cb = np.zeros(3), np.array([CC, 0.0, 0.0])
    down = np.array([0.0, 0.0, -1.0])
    (xpos,) = tetra(ca, ca - cb, down, p["cx"], [0.0])
    ha = tetra(ca, ca - cb, down, CH, [120.0, 240.0])
    hb = tetra(cb, cb - ca, down, CH, [180.0, 60.0, 300.0])
    return [ca, cb, xpos, *ha, *hb]


def sn2_ts(x: str, p: dict[str, float]) -> list[np.ndarray]:
    """Nu...Ca...X collinear along z; Ca, Ha, Ha, Cb planar."""
    ca = np.zeros(3)
    xpos = np.array([0.0, 0.0, p["sn2_cx"]])
    nu = np.array([0.0, 0.0, -p["sn2_cx"]])
    cb = np.array([CC, 0.0, 0.0])
    ha = [CH * np.array([np.cos(a), np.sin(a), 0.0]) for a in np.deg2rad([120.0, 240.0])]
    hb = tetra(cb, cb - ca, np.array([0.0, 0.0, 1.0]), CH, [90.0, 210.0, 330.0])
    return [nu, ca, cb, xpos, *ha, *hb]


def e2_ts(x: str, p: dict[str, float], syn: bool) -> list[np.ndarray]:
    """Ca=Cb shortened; X leaves Ca and the transferred H leaves Cb toward Nu."""
    ca, cb = np.zeros(3), np.array([1.42, 0.0, 0.0])
    down = np.array([0.0, 0.0, -1.0])
    (xpos,) = tetra(ca, ca - cb, down, p["e2_cx"], [0.0])
    ha = tetra(ca, ca - cb, down, CH, [120.0, 240.0])
    phis = [0.0, 120.0, 240.0] if syn else [180.0, 60.0, 300.0]
    hb = tetra(cb, cb - ca, down, CH, phis)
    moved = unit(hb[0] - cb)
    hb[0] = cb + p["e2_ch"] * moved
    nu = hb[0] + p["e2_nuh"] * moved
    h7, h8, h9 = (hb[1], hb[0], hb[2]) if syn else (hb[0], hb[1], hb[2])
    return [nu, ca, cb, xpos, *ha, h7, h8, h9]


def main() -> None:
    HERE.mkdir(exist_ok=True)
    for x, p in PARAMS.items():
        lx = x.lower()
        write(f"{lx}_minus", f"{x}- nucleophile", [x], [np.zeros(3)])
        write(
            f"ethyl_{lx}",
            f"staggered CH3CH2{x}; H7 anti to {x}",
            ["C", "C", x, "H", "H", "H", "H", "H"],
            ethyl_halide(x, p),
        )
        syms = [x, "C", "C", x, "H", "H", "H", "H", "H"]
        write(f"ts_sn2_{lx}", f"{x}- + CH3CH2{x} SN2 TS guess", syms, sn2_ts(x, p))
        write(
            f"ts_anti_e2_{lx}",
            f"{x}- + CH3CH2{x} anti-E2 TS guess (H7 moves)",
            syms,
            e2_ts(x, p, False),
        )
        write(
            f"ts_syn_e2_{lx}",
            f"{x}- + CH3CH2{x} syn-E2 TS guess (H8 moves)",
            syms,
            e2_ts(x, p, True),
        )


if __name__ == "__main__":
    main()
