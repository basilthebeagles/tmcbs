"""Code registry for TMCBS.

This module gives every quantum error-correcting code used in the paper a stable,
human-readable name and a single factory that returns a :class:`Code` handle.  A
``Code`` bundles the metadata you usually want (``n``, ``k``, ``d``, family) with
the *builder object* that the circuit builders expect:

* rotated **surface codes** -> a bare ``css_code``
* **bivariate-bicycle (BB)** codes -> the ``(css_code, A_list, B_list)`` tuple

so the rest of the library never has to remember which family wants which object.

The BB polynomial parameters below are taken from Supplementary Table 1 of the
paper (equivalently the ``--l/--m/--Ax/--Ay/--Bx/--By`` arguments used in the
production runs).  Each BB code in the paper has exactly three monomials in each
of ``a(x, y)`` and ``b(x, y)``; the surface-code syndrome circuit and the BB
syndrome circuit both rely on this, so we validate it on construction.

Examples
--------
>>> from tmcbs import surface_code, bb_code, list_codes
>>> sc = surface_code(3)            # [[9, 1, 3]] rotated surface code
>>> bb = bb_code("[[18,4,4]] BB")   # smallest BB code
>>> sc.n, sc.k, sc.d
(9, 1, 3)
>>> list_codes()                    # names of everything in the registry
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from third_party.gong_sliding_window_decoder import codes_q


@dataclass(frozen=True)
class Code:
    """A handle to an instantiated QEC code plus the metadata the library needs.

    Attributes
    ----------
    name : str
        Human-readable label, e.g. ``"[[9,1,3]] SC"`` or ``"[[18,4,4]] BB"``.
    family : str
        ``"surface"`` or ``"bb"``.
    n, k, d : int
        Physical data qubits, logical qubits, and code distance *per block*.
    builder_object : Any
        Object handed verbatim to ``CircuitBuilder`` / ``CircuitBuilderSurface``.
        For surface codes this is a ``css_code``; for BB codes it is the
        ``(css_code, A_list, B_list)`` tuple.
    params : dict
        The polynomial parameters for BB codes (empty for surface codes).
    """

    name: str
    family: str
    n: int
    k: int
    d: int
    builder_object: Any = field(repr=False)
    params: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_surface(self) -> bool:
        return self.family == "surface"

    @property
    def css(self):
        """The underlying ``css_code`` (works for both families)."""
        return self.builder_object if self.is_surface else self.builder_object[0]

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.name


# --------------------------------------------------------------------------- #
# Surface codes: [[d^2, 1, d]] rotated surface codes for odd d.
# --------------------------------------------------------------------------- #
def surface_code(d: int) -> Code:
    """Return the ``[[d^2, 1, d]]`` rotated surface code for odd ``d``."""
    if d % 2 == 0:
        raise ValueError(f"Surface-code distance d must be odd, got d={d}.")
    css = codes_q.create_rotated_surface_codes(d)
    n = int(css.N)
    return Code(name=f"[[{n},1,{d}]] SC", family="surface", n=n, k=1, d=d,
                builder_object=css)


# --------------------------------------------------------------------------- #
# Bivariate-bicycle codes (Supplementary Table 1 of the paper).
# Each entry encodes a(x, y) = sum x^Ax + sum y^Ay and b(x, y) = sum y^By + sum x^Bx.
# --------------------------------------------------------------------------- #
_BB_SPECS: Dict[str, Dict[str, Any]] = {
    "[[18,4,4]] BB":    dict(l=3,  m=3, Ax=[0, 1], Ay=[1],    Bx=[0, 2],    By=[2], d=4),
    "[[36,4,6]] BB":    dict(l=3,  m=6, Ax=[1],    Ay=[2, 3], Bx=[0, 2],    By=[1], d=6),
    "[[54,4,8]] BB":    dict(l=3,  m=9, Ax=[1],    Ay=[1, 3], Bx=[0, 2],    By=[2], d=8),
    "[[90,8,10]] BB":   dict(l=15, m=3, Ax=[9],    Ay=[1, 2], Bx=[0, 2, 7], By=[],  d=10),
    "[[144,12,12]] BB": dict(l=12, m=6, Ax=[3],    Ay=[1, 2], Bx=[1, 2],    By=[3], d=12),
}


def build_bb_code(l: int, m: int, Ax: List[int], Ay: List[int],
                  Bx: List[int], By: List[int], d: int,
                  name: Optional[str] = None) -> Code:
    """Construct an arbitrary BB code from its polynomial exponents.

    ``a(x, y) = sum_p x^p (p in Ax) + sum_p y^p (p in Ay)`` and likewise for
    ``b`` with ``Bx``/``By``.  The BB syndrome-extraction circuit requires exactly
    three monomials in each polynomial, so ``len(Ax)+len(Ay) == 3`` and
    ``len(Bx)+len(By) == 3``.
    """
    if len(Ax) + len(Ay) != 3 or len(Bx) + len(By) != 3:
        raise ValueError(
            "The BB syndrome circuit expects exactly 3 monomials per polynomial; "
            f"got |a|={len(Ax) + len(Ay)}, |b|={len(Bx) + len(By)}."
        )
    obj = codes_q.create_bivariate_bicycle_codes(
        l=l, m=m, A_x_pows=Ax, A_y_pows=Ay, B_x_pows=Bx, B_y_pows=By)
    css = obj[0]
    n, k = int(css.N), int(css.K)
    params = dict(l=l, m=m, Ax=Ax, Ay=Ay, Bx=Bx, By=By)
    return Code(name=name or f"[[{n},{k},{d}]] BB", family="bb",
                n=n, k=k, d=d, builder_object=obj, params=params)


def bb_code(name: str) -> Code:
    """Return a named BB code from the paper's registry (see ``list_codes()``)."""
    if name not in _BB_SPECS:
        raise KeyError(
            f"Unknown BB code {name!r}. Known: {list(_BB_SPECS)}. "
            "Use build_bb_code(...) to construct a custom one."
        )
    spec = dict(_BB_SPECS[name])
    return build_bb_code(name=name, **spec)


# --------------------------------------------------------------------------- #
# Convenience lookups.
# --------------------------------------------------------------------------- #
#: Distances of the surface codes studied in the paper.
SURFACE_DISTANCES = (3, 5, 7, 9, 11)

#: Names of the BB codes studied in the paper, smallest first.
BB_NAMES = tuple(_BB_SPECS)


def list_codes() -> List[str]:
    """All registered code names (surface codes first, then BB)."""
    return [f"[[{d * d},1,{d}]] SC" for d in SURFACE_DISTANCES] + list(BB_NAMES)


def get_code(name: str) -> Code:
    """Look up any registered code by its label, e.g. ``"[[9,1,3]] SC"``."""
    for d in SURFACE_DISTANCES:
        if name == f"[[{d * d},1,{d}]] SC":
            return surface_code(d)
    return bb_code(name)
