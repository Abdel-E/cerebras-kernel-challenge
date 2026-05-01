# Architecture And Design Interview Guide

This document is a practical walkthrough of the final kernel architecture, the
design choices behind it, and how to explain/defend those choices in an
interview.

## 1) Problem framing and constraints

Goal: exact Top-K nearest neighbors for one query vector `q` against matrix `D`
under squared L2 distance, with deterministic tie-breaking.

Hard constraints:
- Exactness: no approximations.
- Determinism: sort by `(distance, original_index)`.
- Distributed execution: `D` is sharded across a `P x P` PE grid.
- Tight PE memory budget and non-trivial collective overhead.

Core formula used on device:
`||D_i - q||^2 = ||D_i||^2 - 2*dot(D_i,q) + ||q||^2`

Host folds `||q||^2` into `D_norms`, so PE compute becomes:
`distance = D_norms[row] - 2*dot(D_row,q)`.

## 2) End-to-end dataflow

1. Host shards `D` by PE row ownership:
   `global_index = (py * P + px) * rows_per_pe + local_row`.
2. Host sends to all PEs:
   - local `D` shard
   - local `D_norms`
   - `valid_count` (for uneven padding mask)
3. Host sends `q` only to PE `(0,0)`.
4. Device broadcasts `q`:
   - x-direction on top row
   - y-direction down columns
5. Each PE computes local distances and local top-K candidates.
6. Two-stage reduction:
   - x-gather per row to row root (`x=0`), merge `P*K -> K`
   - y-gather on column 0 to `(0,0)`, merge `P*K -> K`
7. Root writes final `distances` and `indices` for host D2H.

## 3) Local compute and selection strategy

### Distance accumulation
- `scaled_q[j] = -2*q[j]` precomputed once per query.
- Dominant loop uses DSD FMAC streaming across local rows.
- Initialization is fused with first FMAC (saves one full vector sweep).
- Current best layout is blocked local `D` packing with fixed block width `8`
  in both `run.py` and `pe_program.csl`.

### Local top-K policy by case
- `K == rows_per_pe`: direct full-row path (or sorted full stream mode for
  large-K merge policy).
- `K == 1`: single best tracker.
- `K == 16` (baseline hot path): specialized insertion path.
- `K > 16` and not full-row: heap keep-best-K path.
- otherwise: generic sorted insertion.

Why mixed policy is defendable:
- Different `(P,K)` regimes have different dominant costs.
- A single generic policy was measurably slower on benchmark cases.

## 4) Merge/reduction policy

Common comparator everywhere:
`a` is better than `b` iff
- `a.distance < b.distance`, or
- equal distance and `a.index < b.index`.

This comparator is used in:
- local insertion,
- heap update logic,
- row merge,
- final merge.

Result: order of wavelet arrival does not change output; only candidate set
matters.

Merge policy:
- Small K: k-way merge of sorted streams (cursor-based).
- `P==2` path has specialized two-stream merge.
- `P==4, K==16` has an explicit 4-head merge micro-kernel to reduce control
  overhead in baseline.
- Larger K fallback uses heap-based merge where appropriate.

## 5) Fabric/routing considerations

Routing topology:
- `q` broadcast in two phases (x then y).
- Candidate traffic in two gather phases (x then y).

Main practical lessons:
- Reducing control-path overhead can matter as much as payload size.
- Some “theoretically elegant” restructures regressed due to extra callbacks or
  synchronization costs.
- Root-side work remains a key pressure point, especially for large K.

## 6) Optimization history: what worked and what did not

Worked:
- host-folded norms,
- small-K k-way merge,
- large-K heap drain improvements,
- full-row fast path,
- fused init+first FMAC,
- blocked D layout (the sweep found width `8` best so far),
- `P==2` and `P==4,K==16` merge specializations.

Rejected (important to mention):
- pair-packed distance/index transport (SRAM/control regressions),
- dynamic DSD valid-length capping,
- naive column-major repack,
- staged ownership/tree reduction variants that added control overhead.

Interview framing:
“I kept only changes that were correctness-safe and cycle-positive under
isolated measurement.”

## 7) How to defend this design

Use this structure:
1. State invariants: exactness + deterministic order.
2. Explain dominant cost model: FMAC distance loop + root reduction overhead.
3. Explain why each accepted optimization attacks one of those costs.
4. Show you rejected plausible but regressive ideas based on measured data.
5. Show next-step maturity: you know where remaining bottlenecks are and how
   you would test them safely.

If asked “why not X optimization?”:
- “We tried an isolated experiment, preserved correctness checks, and rejected
   it because measured cycle impact was negative.”

## 8) Suggested talking points for Q&A

- Why squared L2 (not sqrt): same ordering, cheaper compute.
- Why host-fold norms: saves repeated PE work, same math.
- Why multiple top-K paths: workload-specific hot paths were materially faster.
- Why deterministic tie-break is robust: single total order at every stage.
- Why measurement discipline mattered: one change at a time, case-by-case stats.

## 9) Current best-known configuration summary

- Blocked local `D` layout enabled (block width `8` best in measured sweep).
- Fused distance init with first FMAC.
- Baseline-specialized `K==16` local path and merge path.
- Existing full correctness suite still used as gate after each accepted change.

Use this as the “final architecture” narrative unless a newer measured
configuration supersedes it.
