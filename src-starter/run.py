"""Host-side entrypoint for the challenge submission.

This file is intentionally scaffolded. Build it up in the same order as the CSL
files instead of trying to complete the full runtime in one pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType  # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder  # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime  # pylint: disable=no-name-in-module

from reference import ALL_CASES, topk_reference


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
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print(f"Loading case {args.case}...", flush=True)
    cases = {key: maker() for key, maker in zip(CASE_KEYS, ALL_CASES)}
    if args.case not in cases:
        valid = ", ".join(CASE_KEYS)
        raise SystemExit(f"Unknown case '{args.case}'. Valid cases: {valid}")

    case = cases[args.case]
    D_source = np.asarray(case["D"], dtype=np.float32)
    q = np.asarray(case["q"], dtype=np.float32)
    expected_indices, expected_distances = topk_reference(D_source, q, case["K"])

    print("Setting up runtime...", flush=True)
    runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
    D_symbol = runner.get_id("D")
    D_norms_symbol = runner.get_id("D_norms")
    q_symbol = runner.get_id("q")
    valid_count_symbol = runner.get_id("valid_count")
    distances_symbol = runner.get_id("distances")
    indices_symbol = runner.get_id("indices")

    with open(Path(args.name) / "out.json", encoding="utf-8") as json_file:
        compile_data = json.load(json_file)
    P = int(compile_data["params"]["P"])
    d_dim = int(compile_data["params"]["d_dim"])
    rows_per_pe = int(compile_data["params"]["rows_per_pe"])
    K = int(compile_data["params"]["K"])
    if P != case["P"] or K != case["K"] or d_dim != D_source.shape[1]:
        raise SystemExit(
            f"Compile params P={P}, d_dim={d_dim}, K={K} do not match case '{args.case}'"
        )

    print("Loading and starting device program...", flush=True)
    runner.load()
    runner.run()

    print("Packing D shards...", flush=True)
    num_pes = P * P
    N = D_source.shape[0]
    D_norms_source = np.einsum("ij,ij->i", D_source, D_source).astype(np.float32)
    D_packed = np.zeros((P, P, rows_per_pe, d_dim), dtype=np.float32)
    D_norms_packed = np.zeros((P, P, rows_per_pe), dtype=np.float32)
    valid_counts = np.zeros((P, P, 1), dtype=np.uint32)

    for pe_linear in range(num_pes):
        py, px = divmod(pe_linear, P)
        start = pe_linear * rows_per_pe
        end = min(start + rows_per_pe, N)
        valid = max(0, end - start)
        if valid:
            D_packed[py, px, :valid, :] = D_source[start:end]
            D_norms_packed[py, px, :valid] = D_norms_source[start:end]
        valid_counts[py, px, 0] = valid

    diff = D_source - q[None, :]
    all_distances = np.einsum("ij,ij->i", diff, diff).astype(np.float32)
    all_indices = np.arange(N, dtype=np.uint32)
    order = np.lexsort((all_indices.astype(np.int64), all_distances))
    expected_distances = all_distances[order[:K]]
    expected_indices = all_indices[order[:K]]

    print("Copying D shards H2D...", flush=True)
    runner.memcpy_h2d(
        D_symbol,
        D_packed.ravel(),
        0,
        0,
        P,
        P,
        rows_per_pe * d_dim,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False,
    )
    print("Copying D norms H2D...", flush=True)
    runner.memcpy_h2d(
        D_norms_symbol,
        D_norms_packed.ravel(),
        0,
        0,
        P,
        P,
        rows_per_pe,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False,
    )
    print("Copying valid_count H2D...", flush=True)
    runner.memcpy_h2d(
        valid_count_symbol,
        valid_counts.ravel(),
        0,
        0,
        P,
        P,
        1,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False,
    )
    print("Copying q to root H2D...", flush=True)
    runner.memcpy_h2d(
        q_symbol,
        q,
        0,
        0,
        1,
        1,
        d_dim,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False,
    )
    print("Launching main...", flush=True)
    runner.launch("main", nonblock=False)

    distances = np.zeros(K, dtype=np.float32)
    indices = np.zeros(K, dtype=np.uint32)
    print("Reading final distances D2H...", flush=True)
    runner.memcpy_d2h(
        distances,
        distances_symbol,
        0,
        0,
        1,
        1,
        K,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False,
    )
    print("Reading final indices D2H...", flush=True)
    runner.memcpy_d2h(
        indices,
        indices_symbol,
        0,
        0,
        1,
        1,
        K,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False,
    )
    runner.stop()

    print("Comparing against reference...", flush=True)
    np.testing.assert_allclose(distances, expected_distances, atol=1e-3, rtol=1e-3)
    np.testing.assert_array_equal(indices.astype(np.int32), expected_indices)
    print(f"PASS: {args.case}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
