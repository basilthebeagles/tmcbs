"""Code registry: metadata, construction and validation."""
import pytest

import tmcbs as t

BB_NKD = {
    "[[18,4,4]] BB": (18, 4, 4),
    "[[36,4,6]] BB": (36, 4, 6),
    "[[54,4,8]] BB": (54, 4, 8),
    "[[90,8,10]] BB": (90, 8, 10),
    "[[144,12,12]] BB": (144, 12, 12),
}


@pytest.mark.parametrize("d", [3, 5, 7, 9, 11])
def test_surface_code_metadata(d):
    sc = t.surface_code(d)
    assert (sc.n, sc.k, sc.d) == (d * d, 1, d)
    assert sc.family == "surface" and sc.is_surface
    assert sc.css.N == d * d


def test_surface_code_rejects_even_distance():
    with pytest.raises(ValueError):
        t.surface_code(4)


@pytest.mark.parametrize("name,nkd", BB_NKD.items())
def test_bb_code_metadata(name, nkd):
    c = t.bb_code(name)
    assert (c.n, c.k, c.d) == nkd
    assert c.family == "bb" and not c.is_surface
    assert c.css.N == c.n and c.css.K == c.k


def test_list_codes_and_get_roundtrip():
    names = t.list_codes()
    assert len(names) == 10
    for name in names:
        assert t.get_code(name).name == name


def test_bb_code_unknown_name():
    with pytest.raises(KeyError):
        t.bb_code("[[2,2,2]] BB")


def test_build_bb_code_requires_three_monomials_per_polynomial():
    # a(x, y) here has four monomials, which the BB syndrome circuit cannot use.
    with pytest.raises(ValueError):
        t.build_bb_code(l=3, m=3, Ax=[0, 1, 2], Ay=[1], Bx=[0, 2], By=[2], d=4)
