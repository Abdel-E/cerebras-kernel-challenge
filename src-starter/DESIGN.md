# DESIGN

Start time: 9:38 PM EST, April 28, 2026  
Finish time: 7:15 PM EST, May 1, 2026

## Routing Topology

The kernel runs on a `P x P` PE rectangle. PE `(px, py)` owns rows starting at `(py * P + px) * rows_per_pe`.

The host copies `D`, `D_norms`, and `valid_count` to **every** PE. `valid_count` masks padding rows. The host copies `q` **only** to PE `(0,0)`. The device broadcasts `q` across the top row on x colors `0/1`, then down each column on y colors `4/5` using `collectives_2d`.

Query broadcast topology

After compute, each row gathers local top-K Westward to the row root at `px=0`, which merges `P*K` candidates to K row winners. Column 0 then gathers row winners Northward to PE `(0,0)` for the final top-K.

Two-stage top-K reduction topology

The x entrypoints are task IDs `10/11`; y entrypoints are `12/13`. Query wavelets carry `q`; candidate wavelets carry one distance or one index. I keep distance and index gathers separate because fusing them pressured SRAM on `k_large`.

## Local Top-K Strategy

All cases first compute exact squared L2 distances:
`||D_i - q||^2 = ||D_i||^2 + ||q||^2 - 2 * dot(D_i, q)`.
The host precomputes `D_norms = ||D_i||^2 + ||q||^2`, so the PE only adds `-2 * dot(D_i, q)`. The dominant loop is distance accumulation, `O(rows_per_pe * d_dim)`: each feature uses one vector `@fmacs` over local rows. The first FMAC starts from `D_norms`, so there is no initialization sweep. When `d_dim` is divisible by 8, host packing makes the DSD stride 8.

Dominant-loop estimate:

```text
cycles/element ~= compute_task_cycles / (rows_per_pe * d_dim)
baseline ~= 25.1k / (128*32) ~= 6.1 cycles per local matrix element
```

Local selection depends on K:

- `K == 1`: single best-row tracker, `O(rows_per_pe)`.
- Small K (`K <= 16`): sorted insertion, `O(rows_per_pe * K)`. `K == 16` is specialized to reduce loop overhead.
- Full local K (`K == rows_per_pe`): keep every valid local row. For `k_large`, heap-sort the full stream, `O(rows_per_pe log rows_per_pe)`, so reducers can merge sorted streams.
- Larger non-full K: max-heap of size K, `O(rows_per_pe log K)`.

I chose this mixed policy because no single method was best for every K. Sorted insertion is cheap for small K and produces sorted streams, which makes reduction cheaper. The heap is better for large K because it avoids shifting up to K entries on each insert.

## Fabric Bandwidth

During row gather, each PE sends `K` distance wavelets and `K` index wavelets toward `px=0`. The hot edge is next to the row root because traffic from the other `P-1` PEs can cross it.

Worst case on that edge:

```text
2 * K * (P - 1) candidate wavelets
```

Column gather has the same hot-edge count for row winners. The baseline bottleneck is compute plus host memcpy, not routing. The GUI trace showed H2D movement of `D`, then the FMAC distance loop, as the largest blocks. Routing matters more for `k_large` because each PE sends `K=256` candidates, so root-side merge work and candidate movement become more visible.

## Tie-Breaking

During local top-K selection, each candidate is tagged with its global index:
`global_index = (py * P + px) * rows_per_pe + local_row`.

Every stage uses the same comparison:

```text
1. smaller distance wins
2. if distances are equal, smaller global index wins
```

This gives every candidate one deterministic rank. Ties are stable across PEs because equal-distance candidates are ordered by original row index, not by which PE produced them or when wavelets arrived.

A row outside its PE's local top-K cannot be global top-K because that PE already has K rows that rank better. Arrival order does not matter because reducers rank candidates by `(distance, global_index)`, not by when wavelets arrive.

## With 2x More Time

I would prototype a more distributed `k_large` merge to reduce root-side pressure without adding extra SRAM buffers. Since `k_large` is about 318k cycles and spends much more time moving/merging candidates than small-K cases, a reasonable target would be a 5-10% win, roughly 15k-30k cycles, if the extra communication control overhead stays low.