"""Compile-output runner for the distributed Top-K kNN CSL kernel.

The runner loads a deterministic reference case, packs the database rows into
the PE layout expected by ``pe_program.csl``, launches the device program, and
checks PE (0,0)'s final top-K output against the NumPy oracle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType  # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder  # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime  # pylint: disable=no-name-in-module

from reference import ALL_CASES


CASE_KEYS = (
    "baseline",
    "k_eq_1",
    "k_large",
    "uneven",
    "all_equal",
    "duplicates",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Compiled output directory")
    parser.add_argument("--case", required=True, help="Reference test case name")
    parser.add_argument("--cmaddr", help="IP:port for CS system")
    parser.add_argument(
        "--stats-dir",
        help="Optional directory to store per-case sim_stats snapshots",
    )
    return parser.parse_args()


def load_case(case_key: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    cases = {key: maker() for key, maker in zip(CASE_KEYS, ALL_CASES)}
    if case_key not in cases:
        valid = ", ".join(CASE_KEYS)
        raise SystemExit(f"Unknown case '{case_key}'. Valid cases: {valid}")

    case = cases[case_key]
    D_source = np.asarray(case["D"], dtype=np.float32)
    q = np.asarray(case["q"], dtype=np.float32)
    return case, D_source, q


def read_compile_params(
    name: str, case: dict[str, Any], D_source: np.ndarray
) -> tuple[int, int, int, int, int]:
    with open(Path(name) / "out.json", encoding="utf-8") as json_file:
        compile_data = json.load(json_file)

    P = int(compile_data["params"]["P"])
    d_dim = int(compile_data["params"]["d_dim"])
    rows_per_pe = int(compile_data["params"]["rows_per_pe"])
    K = int(compile_data["params"]["K"])
    D_block = int(compile_data["params"].get("D_block", 4))
    if P != case["P"] or K != case["K"] or d_dim != D_source.shape[1]:
        raise SystemExit(
            f"Compile params P={P}, d_dim={d_dim}, K={K} do not match case '{case['name']}'"
        )
    return P, d_dim, rows_per_pe, K, D_block


def pack_inputs(
    D_source: np.ndarray,
    q: np.ndarray,
    P: int,
    d_dim: int,
    rows_per_pe: int,
    D_block: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_pes = P * P
    N = D_source.shape[0]
    q_norm = np.einsum("i,i->", q, q).astype(np.float32)
    # Fold ||q||^2 into the packed row norms. This removes the repeated PE-side
    # q dot product without increasing the query broadcast payload.
    D_norms_source = (
        np.einsum("ij,ij->i", D_source, D_source).astype(np.float32) + q_norm
    )
    D_packed = np.zeros((P, P, rows_per_pe, d_dim), dtype=np.float32)
    D_norms_packed = np.zeros((P, P, rows_per_pe), dtype=np.float32)
    valid_counts = np.zeros((P, P, 1), dtype=np.uint32)

    # PE linear order matches the device index formula:
    # global_index = (py * P + px) * rows_per_pe + local_row.
    if d_dim % D_block != 0:
        raise SystemExit(
            f"Blocked D layout requires d_dim multiple of {D_block}, got d_dim={d_dim}"
        )

    for pe_linear in range(num_pes):
        py, px = divmod(pe_linear, P)
        start = pe_linear * rows_per_pe
        end = min(start + rows_per_pe, N)
        valid = max(0, end - start)
        if valid:
            shard_rows = np.zeros((rows_per_pe, d_dim), dtype=np.float32)
            shard_rows[:valid, :] = D_source[start:end]
            blocks = d_dim // D_block
            blocked = (
                shard_rows.reshape(rows_per_pe, blocks, D_block)
                .transpose(1, 0, 2)
                .reshape(rows_per_pe, d_dim)
            )
            D_packed[py, px, :, :] = blocked
            D_norms_packed[py, px, :valid] = D_norms_source[start:end]
        valid_counts[py, px, 0] = valid

    return D_packed, D_norms_packed, valid_counts


def compute_expected(
    D_source: np.ndarray, q: np.ndarray, K: int
) -> tuple[np.ndarray, np.ndarray]:
    # Keep the comparison oracle local to this runner so it uses the same
    # float32 distance path and lexicographic tie-break as the device.
    diff = D_source - q[None, :]
    all_distances = np.einsum("ij,ij->i", diff, diff).astype(np.float32)
    all_indices = np.arange(D_source.shape[0], dtype=np.uint32)
    order = np.lexsort((all_indices.astype(np.int64), all_distances))
    return all_distances[order[:K]], all_indices[order[:K]]


def memcpy_h2d_32(
    runner: SdkRuntime,
    symbol: int,
    data: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    length: int,
) -> None:
    runner.memcpy_h2d(
        symbol,
        data.ravel(),
        x,
        y,
        width,
        height,
        length,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False,
    )


def memcpy_d2h_32(
    runner: SdkRuntime,
    target: np.ndarray,
    symbol: int,
    x: int,
    y: int,
    width: int,
    height: int,
    length: int,
) -> None:
    runner.memcpy_d2h(
        target,
        symbol,
        x,
        y,
        width,
        height,
        length,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False,
    )


def copy_inputs(
    runner: SdkRuntime,
    symbols: dict[str, int],
    D_packed: np.ndarray,
    D_norms_packed: np.ndarray,
    valid_counts: np.ndarray,
    q: np.ndarray,
    P: int,
    d_dim: int,
    rows_per_pe: int,
) -> None:
    print("Copying D shards H2D...", flush=True)
    memcpy_h2d_32(runner, symbols["D"], D_packed, 0, 0, P, P, rows_per_pe * d_dim)
    print("Copying D norms H2D...", flush=True)
    memcpy_h2d_32(runner, symbols["D_norms"], D_norms_packed, 0, 0, P, P, rows_per_pe)
    print("Copying valid_count H2D...", flush=True)
    memcpy_h2d_32(runner, symbols["valid_count"], valid_counts, 0, 0, P, P, 1)
    print("Copying q to root H2D...", flush=True)
    memcpy_h2d_32(runner, symbols["q"], q, 0, 0, 1, 1, d_dim)


def read_outputs(
    runner: SdkRuntime, symbols: dict[str, int], K: int
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.zeros(K, dtype=np.float32)
    indices = np.zeros(K, dtype=np.uint32)

    print("Reading final distances D2H...", flush=True)
    memcpy_d2h_32(runner, distances, symbols["distances"], 0, 0, 1, 1, K)
    print("Reading final indices D2H...", flush=True)
    memcpy_d2h_32(runner, indices, symbols["indices"], 0, 0, 1, 1, K)
    return distances, indices


def snapshot_sim_stats(case_key: str, build_name: str, stats_dir: str | None) -> None:
    if not stats_dir:
        return

    source = Path("sim_stats.json")
    if not source.exists():
        print("sim_stats.json not found; skipping stats snapshot.", flush=True)
        return

    with open(source, encoding="utf-8") as infile:
        stats = json.load(infile)

    stats["_case"] = case_key
    stats["_build"] = build_name

    output_dir = Path(stats_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{case_key}.json"
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(stats, outfile, indent=2, sort_keys=True)
        outfile.write("\n")
    print(f"Saved sim stats snapshot: {output_path}", flush=True)


def main() -> int:
    args = parse_args()

    print(f"Loading case {args.case}...", flush=True)
    case, D_source, q = load_case(args.case)

    print("Setting up runtime...", flush=True)
    runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
    symbols = {
        "D": runner.get_id("D"),
        "D_norms": runner.get_id("D_norms"),
        "q": runner.get_id("q"),
        "valid_count": runner.get_id("valid_count"),
        "distances": runner.get_id("distances"),
        "indices": runner.get_id("indices"),
    }

    P, d_dim, rows_per_pe, K, D_block = read_compile_params(args.name, case, D_source)

    print("Loading and starting device program...", flush=True)
    runner.load()
    runner.run()

    print("Packing D shards...", flush=True)
    D_packed, D_norms_packed, valid_counts = pack_inputs(
        D_source, q, P, d_dim, rows_per_pe, D_block
    )
    expected_distances, expected_indices = compute_expected(D_source, q, K)

    copy_inputs(
        runner, symbols, D_packed, D_norms_packed, valid_counts, q, P, d_dim, rows_per_pe
    )
    print("Launching main...", flush=True)
    runner.launch("main", nonblock=False)

    distances, indices = read_outputs(runner, symbols, K)
    runner.stop()
    snapshot_sim_stats(args.case, args.name, args.stats_dir)

    print("Comparing against reference...", flush=True)
    np.testing.assert_allclose(distances, expected_distances, atol=1e-3, rtol=1e-3)
    np.testing.assert_array_equal(indices.astype(np.int32), expected_indices)
    print(f"PASS: {args.case}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
