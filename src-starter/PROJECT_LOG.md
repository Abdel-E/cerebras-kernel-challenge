## KNN CSL Kernel — Project Log (work done + next steps)

This file is the **single source of truth** for what we implemented, why, what
worked, what broke, and what we plan to do next.

### Goal / grading contract (what the kernel must do)

- **Input**: database matrix `D` (N×d), query vector `q` (d), parameters
  `P`, `d_dim`, `rows_per_pe`, `K`.
- **Output**: top-K nearest neighbors under **exact squared L2 distance**,
  sorted deterministically by:
  - primary: smaller `distance`
  - tie-break: smaller `original_index`
- **Distribution**: `D` is sharded by rows across a `P×P` PE grid in row-major
  PE order, with per-PE `valid_count` to ignore padding for uneven `N`.

### Current architecture (what exists in the codebase now)

- **Layout / symbols** (`src-starter/layout.csl`)
  - `@set_rectangle(P, P)` and `@set_tile_code(..., "pe_program.csl", ...)` for all tiles.
  - Passes compile-time params into `pe_program.csl`: `P`, `d_dim`,
    `rows_per_pe`, `K`, plus memcpy + collectives params.
  - Exports symbols for host I/O:
    - host-writable: `D`, `D_norms`, `q`, `valid_count`
    - host-readable: `distances`, `indices`
    - entrypoint: `main`

- **Host runtime** (`src-starter/run.py`)
  - Loads one of the reference test cases (via `src-starter/reference.py`).
  - Packs `D` into per-PE contiguous shards (row-major across PEs).
  - Computes and packs `D_norms` on host: `D_norms[i] = sum_j D[i,j]^2`.
  - Computes `valid_count[pe]` for uneven padding cases.
  - Copies:
    - `D` + `D_norms` + `valid_count` to **all** PEs
    - `q` to **only PE (0,0)**
  - Launches device `main`.
  - Reads back `distances`/`indices` from **PE (0,0)** and checks against the
    numpy oracle with exact tie-breaking.

- **Device program** (`src-starter/pe_program.csl`)
  - Broadcasts `q` from PE (0,0) across top row, then down columns using
    `collectives_2d`.
  - Computes all local distances, maintains local top-K per PE.
  - Hierarchical gather + merge:
    - gather local K pairs across each row to column 0
    - row roots merge `P*K → K`
    - gather row winners up column 0 to PE (0,0)
    - final root merges `P*K → K` to produce final output arrays
  - Uses a **single total order** comparator everywhere:
    `(distance, original_index)`
  - Signals completion to the host with `sys_mod.unblock_cmd_stream()` in the
    final task.

### What we did, in order (timeline of real changes)

#### 1) Plumbing + correctness scaffolding

- Built/confirmed the core shape:
  - `P×P` grid
  - host-visible symbols
  - end-to-end launch, H2D, D2H
  - deterministic comparator semantics

#### 2) First working exact baseline algorithm

- **Distance**: exact squared L2 per row.
- **Local selection**: sorted insertion into `[K]` arrays (`O(rows_per_pe*K)`).
- **Global selection**: gather all candidates to root and merge with the same
  comparator.

#### 3) Simulator performance-driven optimizations (kept exactness)

These were added because the simulator can be extremely slow; we prioritized
reducing the biggest compute hotspots first.

- **`K == rows_per_pe` fast path (k_large)**
  - When `K == rows_per_pe`, no local candidate can be dropped.
  - Skip insertion/selection and directly write out all valid local rows.

- **Root merge via repeated selection**
  - Instead of “insert with shifts”, repeatedly scan for the next best and mark
    consumed. For large `K`, this avoided heavy shifting costs.

- **DSD dot-product distance**
  - Rewrote distance computation using:
    \[
      \|D_i - q\|^2 = \|D_i\|^2 - 2 \langle D_i, q\rangle + \|q\|^2
    \]
  - Used memory DSDs with `@map` to compute dots efficiently.

- **Host-precomputed row norms**
  - Host computes `D_norms[i] = \|D_i\|^2` once and copies alongside `D`.
  - Device distance becomes: `D_norms[row] - 2*dot(D_row,q) + q_norm`.
  - Allowed because it’s derived from `D` and preserves exactness.

- **Hierarchical merge (row roots first)**
  - Reduced final-root merge input from `P*P*K` to `P*K` by reducing on row
    roots first (exact under the same comparator).

- **Heap-based merge for large K**
  - Added a fixed-size max-heap merge for `K` large, with deterministic
    output ordering.

- **Merge policy by K**
  - For small K, the heap overhead regressed.
  - Policy now is:
    - if `K <= 16`: repeated selection
    - else: heap merge

- **Column-wise DSD distance accumulation (GEMV-like)**
  - Initialize `distance_scratch[:] = D_norms[:] + q_norm`
  - For each dimension \(j\):
    - `distance_scratch[:] += D_column_j[:] * (-2*q[j])` via `@fmacs` with a
      strided column DSD.
  - This replaced “one dot per row” with “one streamed FMA per column”.

### Current measured state (last known run)

- `src-starter/sim_stats.json` currently reports:
  - `cycle_count`: **63881**
  - `sim_time`: **~93.8s**
  - `sim_stop_cause`: **Stopped due to idleness**

Note: this `sim_stats.json` is a single run; it’s not a multi-case dashboard.

### Known issues / recurring pain points

- **“Hang” on `memcpy_d2h`**: usually the device pipeline is just still running
  in the simulator (especially large K), so host blocks waiting for completion.
- **k_large runtime**: still a stress case because `K=256` forces large buffers,
  large merges, and large memory traffic even after algorithmic cleanups.

### Next steps (planned work)

These are ordered by “most likely to help overall + easiest to validate”.

1) **Baseline-focused profiling**
   - Capture `sim_stats.json` for baseline specifically and keep a history
     (e.g. `sim_stats_baseline.json`, etc.) so we can see regressions.

2) **Reduce avoidable fabric traffic**
   - Consider sending **only indices** when possible (recompute distance at the
     root from stored per-PE distances), or compress `(distance,index)` pairs
     if allowed.
   - Consider a different gather schedule (e.g. gather only indices then fetch
     distances for winners), if it fits the symbol + memcpy constraints.

3) **Specialize the baseline path (`P=4, K=16, d=32`)**
   - Hand-tune local selection and merge for K=16 (see baseline optimizations
     below).

4) **k_large-specific improvements**
   - Replace “sorted output from heap by scanning K times” with an extraction
     procedure (or a tournament / k-way merge if inputs are already sorted).
   - Explore multi-stage reduction for `P=2, K=256` so the root does less work.

5) **Memory layout tuning**
   - Explore packing `D` in a column-major or blocked format to make the
     column-wise DSD streaming even cheaper (fewer address updates / better
     burst behavior), if host packing changes are allowed.

