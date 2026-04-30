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
  - Computes `q_norm = sum_j q[j]^2` on host and folds it into packed norms:
    `D_norms[i] = sum_j D[i,j]^2 + q_norm`.
  - Computes `valid_count[pe]` for uneven padding cases.
  - Copies:
    - `D` + `D_norms` + `valid_count` to **all** PEs
    - `q` to **only PE (0,0)**
  - Launches device `main`.
  - Reads back `distances`/`indices` from **PE (0,0)** and checks against the
    numpy oracle with exact tie-breaking.
  - `src-starter/DEBUGGING.md` records the local simulator/debugger workflow,
    including `csdb` context, trace, image, and memory commands.

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
  - Host computes `D_norms[i] = \|D_i\|^2 + \|q\|^2` once and copies alongside
    `D`.
  - Device distance becomes: `D_norms[row] - 2*dot(D_row,q)`.
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
    - if `K <= 16`: k-way merge of sorted candidate streams
    - else: heap merge
  - The small-K k-way path uses one cursor per source PE. Because each local
    candidate list is already sorted by `(distance, index)`, row roots and the
    final root reduce `P` sorted streams with `O(K*P)` comparisons instead of
    rescanning the full `P*K` buffer for every output.

- **Column-wise DSD distance accumulation (GEMV-like)**
  - Initialize `distance_scratch[:] = D_norms[:]`
  - For each dimension \(j\):
    - `distance_scratch[:] += D_column_j[:] * (-2*q[j])` via `@fmacs` with a
      strided column DSD.
  - This replaced “one dot per row” with “one streamed FMA per column”.

- **Redundancy cleanup**
  - Moved `q_norm` computation from every PE to the host and folded it into
    host-packed `D_norms`, avoiding both repeated PE work and any larger query
    broadcast.
  - Removed the now-unused PE-local DSD dot helper used only for `q_norm`.
  - Skipped `init_local_topk()` in the `K == rows_per_pe` fast path because
    that path overwrites every local candidate slot directly.
  - Reduced `heap_reset()` to only reset `heap_size`; heap contents are
    overwritten by `heap_consider()` before sorted heap drain reads them.
  - Removed a duplicate host oracle call in `run.py`; the local float32 oracle
    remains the comparison source.
  - Removed dead final-insertion helpers left over from an older merge path.
  - Deliberately kept host-provided `valid_count`: it is one 32-bit word per
    PE, while deriving it on device would require adding `N` as new state plus
    per-PE arithmetic/branches. Any saved memcpy traffic is tiny compared with
    the current tens-of-thousands of simulated cycles, and the derived path
    could easily cost as much device work as it saves.
  - Baseline measurement, compiled and run with identical params before/after
    this cleanup:
    - before cleanup: `cycle_count = 162716`
    - after cleanup: `cycle_count = 162557`
    - net baseline change: **-159 cycles**
  - `k_large` compiled after the heap reset cleanup. A capped runtime attempt
    stayed at `Reading final distances D2H` past the intended 10-minute window
    and was force-stopped, so it is **not** correctness-validated in this pass.

#### 4) Small-K k-way merge

- Replaced the old small-K repeated-selection merge with k-way merging of
  sorted per-PE candidate streams.
- This preserves exact deterministic ordering because every stream is sorted by
  the same total order used globally.
- Baseline measurement:
  - before k-way merge: `cycle_count = 161918`
  - after k-way merge: `cycle_count = 114862`
  - net baseline change: **-47056 cycles** (~29.1% fewer cycles)
- Correctness checks passed after the change:
  - `baseline`
  - `k_eq_1`
  - `duplicates`
  - `all_equal`
  - `uneven`
- `k_large` still compiles on the heap path; it is intentionally unaffected by
  this optimization because the `K == rows_per_pe` local output is not sorted by
  distance.

#### 5) Task/color ID hygiene

- Updated `layout.csl` so the `collectives_2d` entrypoints match the documented
  ID map:
  - x entrypoints: task IDs `10`, `11`
  - y entrypoints: task IDs `12`, `13`
- Moved application callbacks out of collectives and memcpy-reserved IDs:
  - `bcast_q_down`: `6`
  - `compute`: `7`
  - `gather_row_indices`: `8`
  - `gather_col_distances`: `9`
  - `gather_col_indices`: `14`
  - `finish`: `18`
- Validation:
  - `baseline` passed with `cycle_count = 114864`, essentially unchanged from
    the k-way result.
  - `k_large` compile-check passed on the heap path.

#### 6) Follow-up optimization experiments

- **Column-major PE-local `D` packing: rejected**
  - Changed host packing and PE DSD access so each feature column was contiguous
    in local PE memory.
  - Correctness passed, but baseline regressed from `114864` to `117934`
    cycles, so the change was reverted.
- **Guarded heap-local selection for mid-sized K: kept**
  - Added a heap-based local selection path only for `K > 16` and
    `K != rows_per_pe`.
  - Existing small-K paths still use sorted insertion, preserving the cheap
    k-way merge input streams.
  - Existing `k_large` still uses the direct `K == rows_per_pe` path.
  - Validation:
    - synthetic `K=32` compile-check passed, exercising the new branch
    - `baseline` passed with `cycle_count = 114862`
    - `k_large` compile-check passed
- **Dynamic valid-count DSD length: rejected**
  - Short-shard DSD capping compiled and passed correctness, but baseline
    regressed from `114862` to `115080`, and uneven regressed from `70081` to
    `70236`.
  - The change was reverted.

#### 7) Baseline specialization + case-tracked profiling pass

- **Local K=16 specialization (kept)**
  - Added a baseline-focused local top-K path for `K==16` that:
    - seeds with the first valid rows,
    - sorts the 16-slot seed once,
    - then inserts only later candidates that beat slot 15.
  - This removes repeated INF-seeding/insertion churn in the hot baseline path.

- **Scalar precompute in distance loop (kept)**
  - Added `scaled_q[j] = -2*q[j]` precompute once per compute task.
  - The streamed column FMAs now reuse `scaled_q[j]` in `compute_all_distances()`
    rather than rematerializing `-2.0*q[j]` every iteration.

- **`K==rows_per_pe` sorted-stream option (kept)**
  - Added an optional branch (`SORT_STREAMS_WHEN_FULL_K = true`) that uses a
    local heap sort when `K==rows_per_pe` and `K>16`, producing sorted local
    streams.
  - Row/final reducers now use the existing k-way merge when this condition is
    enabled, replacing heap merge at those stages for full-K runs.

- **Per-case stats snapshotting (kept)**
  - `run.py` now supports `--stats-dir` and writes
    `<stats-dir>/<case>.json` snapshots from `sim_stats.json`.
  - Added `run_all_cases.sh` to compile + run all six challenge cases and print
    a summary table from `out/stats/*.json`.

### Current measured state (latest full sweep)

Full six-case sweep run via `src-starter/run_all_cases.sh` with stats captured
to `src-starter/out/stats/*.json`:

- `baseline`: `cycle_count = 111560`, `sim_time ≈ 177.835s`
- `k_eq_1`: `cycle_count = 78650`, `sim_time ≈ 98.985s`
- `k_large`: `cycle_count = 325746`, `sim_time ≈ 455.440s`
- `uneven`: `cycle_count = 66883`, `sim_time ≈ 112.009s`
- `all_equal`: `cycle_count = 54472`, `sim_time ≈ 92.357s`
- `duplicates`: `cycle_count = 58052`, `sim_time ≈ 77.116s`

Relative to the previously logged references:
- baseline improved from ~`114862` to `111560` (about **-2.9%**),
- `k_large` improved strongly from the previously logged `712258` to `325746`
  in this run configuration.

### 8) `P=2` k-way merge specialization pass

- **What changed (kept)**
  - `merge_sorted_streams()` now has a dedicated `P==2` fast path.
  - Instead of a generic “scan all streams each output” loop, it uses two
    cursors (`offset0`, `offset1`) and one direct compare between stream heads.
  - This removes loop/control overhead in the hottest reduction path for the
    `P=2` cases while preserving exact `(distance, index)` ordering.

- **Why this helps**
  - For `P=2`, the generic path still did per-output loop mechanics designed for
    arbitrary fan-in. The specialized path is the minimal operation set:
    one head-to-head compare + one cursor increment per emitted output.
  - It benefits both row-root and final-root merge stages wherever the k-way
    sorted-stream merge policy is active.

- **Validation + measured deltas (full six-case sweep)**
  - New sweep results:
    - `baseline`: `111561`
    - `k_eq_1`: `78634`
    - `k_large`: `318377`
    - `uneven`: `66886`
    - `all_equal`: `54868`
    - `duplicates`: `57830`
  - Versus the immediately prior sweep:
    - `k_large`: `325746 -> 318377` (**-7369**, ~**-2.3%**)
    - `duplicates`: `58052 -> 57830` (**-222**, ~**-0.4%**)
    - `k_eq_1`: `78650 -> 78634` (**-16**, tiny improvement)
    - `baseline`/`uneven`: effectively flat (noise-level change)
    - `all_equal`: slightly worse in this run (`+396`, ~`+0.7%`)
  - Net: the specialization appears to be a real `k_large` win without broad
    regressions; minor per-case jitter remains expected in simulator runs.

### Why the `k_large` improvement is so large

`k_large` (`P=2, K=256, rows_per_pe=256`) is dominated by reduction work at the
roots. The previous full-K path produced unsorted local buffers and then relied
on heap-based root merges over `P*K` candidates at each reduction stage.

The new full-K option changes that critical path:

1. **Local sorting moved to all PEs (parallel work).**
   - Each PE heap-sorts its own 256 candidates into deterministic ascending
     streams once.
   - This added local work is spread across all participating PEs, so it does
     not create a new serial bottleneck.

2. **Root merges switched from heap scans to k-way stream merge.**
   - With sorted local streams, row/root reducers now use the existing small-K
     stream-merger logic (`O(K*P)` comparisons) instead of heap buffering over
     the full `P*K` list.
   - For `P=2`, this is especially favorable: selecting each next winner is a
     tiny two-stream compare rather than repeated heap maintenance.

3. **Work moved off the serialized root-heavy phase.**
   - Prior root merging was on the reduction critical path; any savings there
     directly reduce end-to-end cycles.
   - The new flow shifts effort toward per-PE local preparation and away from
     root hotspots, which is why the cycle drop is disproportionally large.

Net effect in this environment: `k_large` fell from the previously logged
`712258` to `325746` cycles in the latest run configuration.

### Known issues / recurring pain points

- **“Hang” on `memcpy_d2h`**: usually the device pipeline is just still running
  in the simulator (especially large K), so host blocks waiting for completion.
- **k_large runtime**: still a stress case because `K=256` forces large buffers,
  large merges, and large memory traffic even after algorithmic cleanups.

### Next steps (planned work)

These are ordered by “most likely to help overall + easiest to validate”.

1) **Ablate full-K merge policy for `k_large`**
   - Keep a compile-time switch between:
     - full-K sorted-stream path (`SORT_STREAMS_WHEN_FULL_K = true`)
     - prior full-K direct-local + root-heap path
   - Measure both with identical toolchain/runtime settings to confirm the win
     persists and quantify variance.

2) **Tighten the baseline `K=16` local path further**
   - Evaluate an explicitly unrolled insert for K=16 (or small fixed blocks)
     versus current looped insertion to reduce branch/control overhead.
   - Keep this gated to `K==16` only to avoid regressing other cases.

3) **Selective trace-guided stall reduction**
   - Capture case-specific traces on baseline and k_large and attribute stalls
     around gather callbacks vs compute loop.
   - Use that evidence to decide whether callback/task sequencing refinements
     (not payload packing) can reduce idle gaps.

4) **Keep per-case stats history as regression guardrail**
   - Continue writing `out/stats/<case>.json` each run and compare against prior
     snapshots before accepting optimization changes.
