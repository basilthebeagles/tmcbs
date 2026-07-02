"""End-to-end tests for the MPI driver ``tmcbs.run_ebit_experiment_mpi``.

These launch the driver under ``mpirun`` as a subprocess with several ranks (one
master + N workers), exactly the way it is run in production::

    mpirun -n <N> python -m mpi4py -m tmcbs.run_ebit_experiment_mpi ...

and then check that

* the process exits cleanly and writes a well-formed ``.npz``,
* the swept noise grid has the expected shape and self-consistent contents
  (``ler == errors / shots``), and
* the workers actually shared the load -- every point's shot count is a multiple
  of the number of worker ranks, which can only happen if all workers decoded.

They are skipped automatically when a usable MPI is not available (``mpirun`` not
on ``PATH`` or ``mpi4py`` cannot load ``libmpi`` -- e.g. a laptop without
OpenMPI/MPICH), so the rest of the suite still runs there. The ``mpi`` CI job
installs OpenMPI and the ``[mpi]`` extra so these run for real.
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

MODULE = "tmcbs.run_ebit_experiment_mpi"


def _mpi_available() -> bool:
    """True only if we can actually launch an MPI job on this machine."""
    if shutil.which("mpirun") is None:
        return False
    try:
        # Importing MPI triggers the libmpi dlopen; on a box without a system MPI
        # this raises even though the mpi4py wheel is installed.
        from mpi4py import MPI  # noqa: F401
    except Exception:
        return False
    return True


pytestmark = [
    pytest.mark.mpi,
    pytest.mark.skipif(
        not _mpi_available(),
        reason="needs a working MPI (mpirun on PATH + a loadable libmpi via "
               "mpi4py); install OpenMPI/MPICH and the [mpi] extra to run these",
    ),
]


def _run_driver(nranks, extra_args, tmp_path, timeout=300):
    """Run the driver under ``mpirun`` with ``nranks`` ranks; return (proc, npz_path)."""
    stem = tmp_path / "result"
    npz = tmp_path / "result.npz"  # driver writes ``<file-name>.npz``

    env = dict(os.environ)
    # Allow oversubscription on CI boxes with fewer cores than ranks. These cover
    # OpenMPI 4/5; MPICH oversubscribes by default and ignores unknown OMPI_* vars.
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    # Harmless unless the job happens to run as root in a container.
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")

    cmd = [
        "mpirun", "-n", str(nranks),
        sys.executable, "-m", "mpi4py", "-m", MODULE,
        "--file-name", str(stem),
        *extra_args,
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    return proc, npz


def _assert_point_ok(data, i, j, workers):
    """A single (phys, trans) grid cell should be finite and self-consistent."""
    ler = data["resultsArr"][i, j]
    errs = data["totalErrorsArr"][i, j]
    shots = data["numberOfShotsArr"][i, j]

    assert shots > 0
    assert np.isfinite(ler) and 0.0 <= ler <= 1.0
    assert np.isclose(ler, errs / shots), (ler, errs, shots)
    # The master dispatches `batch` shots to each of `workers` ranks per round, so
    # every accumulated shot count is an exact multiple of the worker count. This
    # only holds if all worker ranks actually contributed.
    assert int(shots) % workers == 0, (shots, workers)


def test_mpi_memory_two_workers(tmp_path):
    """3 ranks (1 master + 2 workers): a single-point memory sweep."""
    workers = 2
    proc, npz = _run_driver(
        3,
        ["--experiment", "0", "--surface-code", "-d", "3",
         "--phys-noise", "1e-2", "--decoder", "BP-OSD",
         "--needed-logical-errors", "3", "--shots-cutoff", "2000000"],
        tmp_path,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout tail:\n{proc.stdout[-2000:]}"
    assert npz.exists()

    data = np.load(npz, allow_pickle=True)
    assert data["resultsArr"].shape == (1, 1)
    assert int(data["experiment"]) == 0
    assert str(data["decoder"]) == "BP-OSD"
    _assert_point_ok(data, 0, 0, workers)
    # Near threshold (p=1e-2, d=3) a few logical errors show up well within the cap.
    assert data["totalErrorsArr"][0, 0] >= 1


def test_mpi_cnot_grid_three_workers(tmp_path):
    """4 ranks (1 master + 3 workers): a 2-point non-local-CNOT sweep with ebits."""
    workers = 3
    proc, npz = _run_driver(
        4,
        ["--experiment", "1", "--surface-code", "-d", "3",
         "--phys-noise", "1e-2,7e-3", "--trans-ratio", "1",
         "--decoder", "BP-OSD",
         "--needed-logical-errors", "3", "--shots-cutoff", "2000000"],
        tmp_path,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout tail:\n{proc.stdout[-2000:]}"
    assert npz.exists()

    data = np.load(npz, allow_pickle=True)
    # grid = (len(phys-noise), len(trans-axis)); trans-ratio => a single trans column.
    assert data["resultsArr"].shape == (2, 1)
    assert int(data["experiment"]) == 1
    assert float(data["transRatio"]) == 1.0
    for i in range(2):
        _assert_point_ok(data, i, 0, workers)


def test_mpi_requires_at_least_two_ranks(tmp_path):
    """A single rank has no worker to hand batches to, so the driver must refuse."""
    proc, npz = _run_driver(
        1,
        ["--experiment", "0", "--surface-code", "-d", "3", "--phys-noise", "1e-2"],
        tmp_path,
    )
    assert proc.returncode != 0
    assert "at least 2 MPI ranks" in proc.stderr
    assert not npz.exists()
