import pathlib

from stagpy import stagyyparsers as prs
from stagpy.stagyydata import StagyyData


def test_time_series_prs(sdat: StagyyData) -> None:
    names = ["aa", "bb", "cc"]
    data = prs.time_series(sdat.filename("time.dat"), list(names))
    assert data is not None
    assert (data.columns[:3] == names).all()
    assert (data.columns[3:] == list(map(str, range(data.shape[1] - 3)))).all()


def test_time_series_invalid_prs() -> None:
    assert prs.time_series(pathlib.Path("dummy"), []) is None


def test_rprof_prs(sdat: StagyyData) -> None:
    names = ["aa", "bb", "cc"]
    data, time = prs.rprof(sdat.filename("rprof.dat"), list(names))
    assert all((df.columns[:3] == names).all() for df in data.values())
    assert all(
        (df.columns[3:] == list(map(str, range(df.shape[1] - 3)))).all()
        for df in data.values()
    )


def test_rprof_invalid_prs() -> None:
    assert prs.rprof(pathlib.Path("dummy"), []) == ({}, None)


def test_fields_prs(sdat: StagyyData) -> None:
    parsed = prs.fields(sdat.filename("t", len(sdat.snaps) - 1))
    assert parsed is not None
    hdr, flds = parsed
    assert flds.shape[0] == 1
    assert flds.shape[4] == 1
    assert flds.shape[1:4] == tuple(hdr["nts"])


def test_field_header_prs(sdat: StagyyData) -> None:
    hdr = prs.field_header(sdat.filename("t", len(sdat.snaps) - 1))
    assert hdr is not None
    assert hdr["nts"].shape == (3,)


def test_fields_istep_prs(sdat: StagyyData) -> None:
    istep = prs.field_istep(sdat.filename("t", len(sdat.snaps) - 1))
    assert istep == sdat.snaps[-1].istep


def test_fields_invalid_prs() -> None:
    assert prs.fields(pathlib.Path("dummy")) is None
