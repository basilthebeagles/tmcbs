#!/usr/bin/env python3
"""MPI driver for sweeping logical error rate over noise parameters.

Rank 0 (the master) walks the (physical noise, ebit/decoherence) grid and, for
each point, hands out batches of shots to the worker ranks until enough logical
errors have accumulated (or a shot cap is hit); the workers build the circuit and
decode. Circuit construction, the code registry and decoding all come from the
``tmcbs`` package, so this driver, the notebooks and ``tmcbs.estimate_ler`` agree.

Run with at least two ranks, e.g.

    mpirun -n 64 python -m mpi4py -m tmcbs.run_ebit_experiment_mpi \\
        --experiment 1 --surface-code -d 5 --phys-noise 1e-2,1e-3 \\
        --trans-ratio 1 --file-name cnot_sc_d5

Results are written to ``<file-name>.npz``; see the bottom of this file for the
stored arrays.
"""

import argparse
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Callable

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
host = MPI.Get_processor_name()

TERMINATE = "TERMINATE"
STARTUP_TAG = 1001
DEBUG_MPI = os.environ.get("TMCBS_MPI_DEBUG", "").lower() not in ("", "0", "false", "no")
_tmcbs = None
_np = None


def debug(msg):
    if DEBUG_MPI:
        print(f"[rank {rank}/{size} host={host}] {msg}", flush=True)


def get_tmcbs():
    global _tmcbs
    if _tmcbs is None:
        debug("importing tmcbs")
        import tmcbs as tmcbs_module
        _tmcbs = tmcbs_module
        debug(f"imported tmcbs={Path(_tmcbs.__file__).resolve()}")
    return _tmcbs


def get_np():
    global _np
    if _np is None:
        debug("importing numpy")
        import numpy as np_module
        _np = np_module
    return _np


def _excepthook(exc_type, exc, tb):
    # Abort the whole job on any uncaught exception so a single bad rank does not
    # leave the others hanging.
    print(f"\n### exception on rank {rank} (host {host}) ###", file=sys.stderr, flush=True)
    traceback.print_exception(exc_type, exc, tb)
    sys.stderr.flush()
    try:
        comm.Abort(1)
    except Exception:
        pass


sys.excepthook = _excepthook


def _csv(elem_type: Callable):
    """Parse a comma-separated list, e.g. '1e-3,1e-4' or '[1e-3, 1e-4]'."""
    def parse(s: str):
        s = s.strip().lstrip("[").rstrip("]")
        return [] if s == "" else [elem_type(x) for x in s.split(",")]
    return parse


parse_floats = _csv(float)
parse_ints = _csv(int)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    p.add_argument("--experiment", type=int, choices=[0, 1, 2, 3, 4, 5], required=True,
                   help="0 memory, 1 non-local CNOT, 2 teleportation, "
                        "3 local CNOT, 4 local teleportation, 5 Pauli product measurement")
    p.add_argument("--file-name", type=str, required=True,
                   help="output path without extension; results go to <file-name>.npz")
    p.add_argument("--phys-noise", type=parse_floats, required=True,
                   help="physical error rates to sweep, e.g. '1e-2,1e-3,1e-4'")

    # Ebit noise: a fixed multiple of the physical rate, or an explicit list.
    ebit = p.add_mutually_exclusive_group(required=False)
    ebit.add_argument("--trans-ratio", type=float,
                      help="ebit noise = ratio * physical noise")
    ebit.add_argument("--trans-noise", type=parse_floats,
                      help="explicit ebit error rates to sweep")

    code = p.add_mutually_exclusive_group(required=True)
    code.add_argument("--surface-code", action="store_true", help="rotated surface code")
    code.add_argument("--bicycle-code", action="store_true", help="bivariate-bicycle qLDPC code")

    p.add_argument("--distance", "-d", type=int, default=None, help="code distance")
    p.add_argument("--n", type=int, default=None,
                   help="(ignored; kept for compatibility, n is read from the code)")

    # Bivariate-bicycle polynomials a(x,y)=sum x^Ax + sum y^Ay, b likewise.
    p.add_argument("--l", type=int, help="BB parameter l")
    p.add_argument("--m", type=int, help="BB parameter m")
    p.add_argument("--Ax", type=parse_ints, help="x-powers of a(x,y), e.g. '0,1'")
    p.add_argument("--Ay", type=parse_ints, help="y-powers of a(x,y)")
    p.add_argument("--Bx", type=parse_ints, help="x-powers of b(x,y)")
    p.add_argument("--By", type=parse_ints, help="y-powers of b(x,y)")

    p.add_argument("--ebitt1t2Ratios", type=parse_floats, default=[],
                   help="sweep ebit decoherence (oldest-ebit wait time / T1) instead of "
                        "ebit noise; ebit infidelity is then set by --trans-ratio/--trans-noise")
    p.add_argument("--numEbitsPerCycle", type=int, default=-1,
                   help="ebits produced per time step for decoherence runs "
                        "(-1 = one-at-a-time; pass d for O(d) line generation)")
    p.add_argument("--ppm", type=parse_floats, default=None,
                   help="Pauli-product-measurement CNOT chain for --experiment 5, "
                        "e.g. '1,0,1' (0 local, 1 non-local; weight = length + 1)")

    p.add_argument("--decoder", type=str, default="teser",
                   help="teser (Tesseract), BP-OSD, or LSD")
    p.add_argument("--needed-logical-errors", type=int, default=25,
                   help="accumulate this many logical errors per point before stopping")
    p.add_argument("--shots-cutoff", type=int, default=2_000_000_000,
                   help="stop a point after this many shots even if the error target is unmet")
    return p


args = build_parser().parse_args()

if size < 2:
    if rank == 0:
        sys.stderr.write("error: needs at least 2 MPI ranks (1 master + >=1 worker)\n")
    sys.exit(1)

EXPERIMENT = args.experiment
fileName = args.file_name
physicalNoiseArr = args.phys_noise
neededLogicalErrors = args.needed_logical_errors
numberOfShotsCutOff = args.shots_cutoff
decoder = args.decoder
ppmInstructions = args.ppm
t1t2Ratios = args.ebitt1t2Ratios
d = args.distance if args.distance is not None else 11

# The inner sweep axis. With --ebitt1t2Ratios it is the decoherence ratios; with
# --trans-ratio it is a single placeholder (the ebit noise is derived per point);
# with --trans-noise it is the explicit ebit-noise list.
if t1t2Ratios:
    transNoiseArr = t1t2Ratios
elif args.trans_ratio is not None:
    transNoiseArr = [None]
elif args.trans_noise is not None:
    transNoiseArr = args.trans_noise
else:
    transNoiseArr = [1e-3]


def make_code():
    tmcbs = get_tmcbs()
    if args.surface_code:
        return tmcbs.surface_code(d)
    return tmcbs.build_bb_code(l=args.l, m=args.m, Ax=args.Ax or [], Ay=args.Ay or [],
                               Bx=args.Bx or [], By=args.By or [], d=d)


def adjusted_num_ebits_per_cycle(code_obj):
    # O(d) line generation reduces to the nearest divisor of n.
    num_per_cycle = args.numEbitsPerCycle
    if num_per_cycle != -1:
        while code_obj.n % num_per_cycle != 0:
            num_per_cycle -= 1
    return num_per_cycle


_code_cache = None
_num_per_cycle_cache = None


def get_code_and_num_per_cycle():
    global _code_cache, _num_per_cycle_cache
    if _code_cache is None:
        debug("building code")
        _code_cache = make_code()
        _num_per_cycle_cache = adjusted_num_ebits_per_cycle(_code_cache)
        debug(f"built code={_code_cache.name} n={_code_cache.n}")
    return _code_cache, _num_per_cycle_cache


# Keep the startup collective before tmcbs/numpy import or code construction. If
# ranks do not all arrive, fail loudly instead of leaving a job silently parked at
# "rank0 started".
def gather_startup_hosts():
    timeout = float(os.environ.get("TMCBS_MPI_STARTUP_TIMEOUT", "120"))

    if rank != 0:
        comm.isend(host, dest=0, tag=STARTUP_TAG).wait()
        return None

    requests = {worker: comm.irecv(source=worker, tag=STARTUP_TAG)
                for worker in range(1, size)}
    hosts = [host]
    deadline = time.monotonic() + timeout

    while requests:
        for worker, req in list(requests.items()):
            ready, worker_host = req.test()
            if ready:
                hosts.append(worker_host)
                del requests[worker]

        if time.monotonic() >= deadline:
            missing = ",".join(str(worker) for worker in requests)
            print(f"error: timed out after {timeout:.0f}s waiting for MPI ranks "
                  f"to start; missing ranks: {missing}. This is before tmcbs "
                  "import or circuit construction.", flush=True)
            comm.Abort(2)

        time.sleep(0.25)

    return hosts


if rank == 0:
    debug(f"rank0 started size={size}")
ready_hosts = gather_startup_hosts()

if rank == 0:
    debug(f"mpi ranks ready: {size} on {len(set(ready_hosts))} host(s)")
    code, numPerCycle = get_code_and_num_per_cycle()
    n = code.n
    print(f"code={code.name} n={n} d={d} experiment={EXPERIMENT} decoder={decoder} "
          f"ranks={size}", flush=True)
    debug(f"tmcbs={Path(get_tmcbs().__file__).resolve()}")
    debug(f"runner={Path(__file__).resolve()}")
    print(f"phys-noise={physicalNoiseArr}", flush=True)
    if t1t2Ratios:
        print(f"decoherence ratios={t1t2Ratios}  ebits/cycle={numPerCycle}", flush=True)
    elif args.trans_ratio is not None:
        print(f"ebit noise = {args.trans_ratio} x physical", flush=True)
    else:
        print(f"ebit noise={transNoiseArr}", flush=True)
    if ppmInstructions is not None:
        print(f"ppm chain={ppmInstructions}", flush=True)
    debug(f"workers ready: {size - 1}")

# Output grids live only on rank 0; workers only need the noise lists and build
# one circuit per assigned batch.
resultsArr = None
totalErrorsArr = None
numberOfShotsArr = None
timingArr = None
reachedShotLimitArr = None
if rank == 0:
    np = get_np()
    shape = (len(physicalNoiseArr), len(transNoiseArr))
    resultsArr = np.full(shape, np.nan)         # logical error rate
    totalErrorsArr = np.full(shape, np.nan)
    numberOfShotsArr = np.full(shape, np.nan)
    timingArr = np.full(shape, np.nan)          # wall-clock seconds per point
    reachedShotLimitArr = np.zeros(shape, dtype=bool)  # True if stopped on the shot cap


def decode_work_item(work):
    i, j, shots = work
    tmcbs = get_tmcbs()
    code, num_per_cycle = get_code_and_num_per_cycle()
    phys = physicalNoiseArr[i]
    if args.trans_ratio is not None:
        trans = phys * float(args.trans_ratio)
    else:
        trans = transNoiseArr[j]

    decoherence_ratio = transNoiseArr[j] if t1t2Ratios else 0.0
    circuit = tmcbs.build_for_experiment(
        code, EXPERIMENT, phys, trans,
        decoherence_ratio=decoherence_ratio,
        ebits_per_cycle=num_per_cycle,
        ppm_pattern=ppmInstructions if EXPERIMENT == 5 else None,
    )
    return tmcbs.count_logical_errors(circuit, shots, p=phys, decoder=decoder)


def gather_worker_errors(root_value=0):
    return comm.gather(root_value, root=0)[1:]


def save():
    np = get_np()
    out = Path(fileName + ".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        physicalNoiseArr=physicalNoiseArr,
        transNoiseArr=transNoiseArr,
        transRatio=args.trans_ratio,
        resultsArr=resultsArr,
        totalErrorsArr=totalErrorsArr,
        numberOfShotsArr=numberOfShotsArr,
        timingArr=timingArr,
        reachedShotLimitArr=reachedShotLimitArr,
        experiment=EXPERIMENT,
        decoder=decoder,
    )


# --------------------------------------------------------------------------- #
# Master: adaptively allocate shot batches across the noise grid.
# --------------------------------------------------------------------------- #
if rank == 0:
    workers = size - 1
    npoints = resultsArr.size
    point = 0
    for i in range(len(physicalNoiseArr)):
        for j in range(len(transNoiseArr)):
            point += 1
            t0 = time.perf_counter()
            phys = physicalNoiseArr[i]
            if t1t2Ratios:
                label = f"decoh={transNoiseArr[j]:.1e}"
            elif args.trans_ratio is not None:
                label = f"ebit={phys * args.trans_ratio:.1e}"
            else:
                label = f"ebit={transNoiseArr[j]:.1e}"
            print(f"[{point}/{npoints}] p={phys:.2e} {label}", flush=True)

            totalErrors = 0
            numberOfShots = 0
            batch = 1
            reachedShotLimit = False

            while True:
                work = [i, j, batch]
                print(f"    dispatch batch={batch} shots/worker", flush=True)
                comm.bcast(work, root=0)
                totalErrors += sum(gather_worker_errors())
                numberOfShots += workers * batch
                ler = totalErrors / numberOfShots
                print(f"    {totalErrors:>5d} errors / {numberOfShots:>12d} shots"
                      f"   LER~{ler:.2e}", flush=True)

                if totalErrors >= neededLogicalErrors:
                    break
                if numberOfShots >= numberOfShotsCutOff:
                    reachedShotLimit = True
                    break

                # Plan the next batch: grow geometrically until the first errors
                # appear, then aim straight for the remaining error budget.
                if totalErrors == 0:
                    batch *= 2
                else:
                    batch = int((neededLogicalErrors - 0.66 * totalErrors)
                                * numberOfShots / (totalErrors * workers))
                batch = min(max(batch, 1), 100_000)

            resultsArr[i, j] = totalErrors / numberOfShots
            totalErrorsArr[i, j] = totalErrors
            numberOfShotsArr[i, j] = numberOfShots
            reachedShotLimitArr[i, j] = reachedShotLimit
            timingArr[i, j] = time.perf_counter() - t0
            flag = "  [shot-limit]" if reachedShotLimit else ""
            print(f"  -> LER {resultsArr[i, j]:.3e}  ({totalErrors} errors / "
                  f"{numberOfShots} shots){flag}  [{timingArr[i, j]:.0f}s]", flush=True)
            save()

    save()
    comm.bcast(TERMINATE, root=0)


# --------------------------------------------------------------------------- #
# Workers: build the requested circuit and decode each batch.
# --------------------------------------------------------------------------- #
else:
    debug("worker entering loop")
    while True:
        work = comm.bcast(None, root=0)
        if work == TERMINATE:
            debug("worker terminating")
            break
        debug(f"worker received batch={work[2]}")
        comm.gather(decode_work_item(work), root=0)
