# DESIGN

Start time: 9:38 PM EST, April 28, 2026

## Routing Topology

The kernel uses a `P x P` rectangle. The host shards `D` by rows in PE
row-major order: PE `(px, py)` has linear id `py * P + px` and owns up to
`rows_per_pe` original rows. A per-PE `valid_count` ignores padding in uneven
cases. The host copies `q` only to PE `(0,0)`. The device broadcasts `q` across
the top row with `mpi_x.broadcast`, then down each column with
`mpi_y.broadcast`. After local compute, each row gathers candidates to column 0;
row roots reduce their row candidates to K; column 0 gathers row winners to
PE `(0,0)`, which owns final output.

```text
Host -> D shards, D_norms, valid_count -> all PEs
Host -> q -> PE(0,0)
PE(0,0) -> q broadcast across row -> top row
top row -> q broadcast down columns -> all PEs
all PEs -> local top-K -> row roots -> PE(0,0) -> Host
```

## Local Top-K Strategy

Each PE computes exact squared L2 distance using the identity
`||D_i - q||^2 = ||D_i||^2 - 2 dot(D_i, q) + ||q||^2`. The host precomputes
`||q||^2` and folds it into the packed row norms, so each PE receives
`D_norms[i] = ||D_i||^2 + ||q||^2` and does not repeat the q-norm dot product.
Distance accumulation is column-wise: initialize `distance_scratch[:]` from
`D_norms[:]`, then for each dimension stream the corresponding
strided D column through `@fmacs` with `-2*q[j]`. Local candidates are
maintained in `local_distances[K]` and `local_indices[K]`. For normal cases I
use sorted insertion, `O(rows_per_pe * K)`, because it is simple and
deterministic. For `K == rows_per_pe`, the PE writes every valid local row
directly because no local candidate can be dropped.

## Fabric Bandwidth

Query broadcast sends `d_dim` fp32 values across row and column collectives.
Each PE contributes K distances and K indices. Row roots receive `P*K` pairs,
reduce to K, then final root receives only `P*K` row winners instead of
`P*P*K` candidates. For baseline (`P=4`, `K=16`), the final root merge input is
64 pairs instead of 256. For `k_large` (`P=2`, `K=256`), row roots and final
root still process large candidate buffers, so the merge/top-K path remains the
main stress case.

## Deterministic Tie-Breaking

Every local and global comparison uses the same total order:
`(distance, original_index)`. A candidate is better if distance is smaller, or
distance is equal and original index is smaller. Because row roots and final
root both merge with this comparator, fabric arrival order cannot affect the
answer. Padding rows use sentinel `(INF, SENTINEL_INDEX)` and therefore never
beat real candidates.

## If I Had 2x More Time

I would further optimize `k_large` with a more efficient sorted-output path for
large K, such as heap extraction or a tournament merge specialized for sorted
candidate lists. I would also tune the distance kernel around DSD layout and row
packing so each PE streams row data with fewer address updates.

## Optimization Notes (Memo-Friendly)

This section summarizes the key optimizations in plain language: what changed,
how it works, and why it helps.

### 1) Host-folded norms (`D_norms = ||D||^2 + ||q||^2`)

- What changed:
  - The host precomputes `||q||^2` and adds it into every packed row norm.
- How it works:
  - Device computes `distance = D_norms[row] - 2 * dot(D_row, q)`.
  - This is mathematically identical to squared L2.
- Why it helps:
  - Removes repeated PE-side work to compute `||q||^2`.
  - Keeps query broadcast payload unchanged.

### 2) Small-K k-way merge (sorted streams)

- What changed:
  - For small K (notably `K<=16`), reducers merge sorted per-PE streams with
    per-stream cursors instead of rescanning full candidate buffers.
- How it works:
  - At each output slot, compare current heads of the input streams, emit best,
    and advance only that stream.
- Why it helps:
  - Replaces repeated full-buffer scans with cheaper incremental merging.
  - Large cycle drop in baseline because merge is on the critical path.

### 3) Large-K heap drain (`O(K log K)` output)

- What changed:
  - Heap output switched from repeated full scans to repeated heap pop.
- How it works:
  - Keep a max-heap of current best K; pop worst into output from the end.
- Why it helps:
  - Avoids `K^2`-style output cost; especially important for `k_large`.

### 4) `K==rows_per_pe` fast path

- What changed:
  - When every local row is needed, local PE skips normal top-K insertion logic.
- How it works:
  - Directly writes valid local `(distance,index)` pairs (or sorted local stream
    when full-K stream mode is enabled).
- Why it helps:
  - Removes unnecessary local selection overhead.

### 5) `P==2` merge specialization

- What changed:
  - Added a direct two-stream merge path for `P=2`.
- How it works:
  - Two cursors, one head-to-head compare per emitted output.
- Why it helps:
  - Removes generic multi-stream loop overhead.
  - Benefited `k_large` and some `P=2` cases.

### 6) Fused distance init + first FMAC

- What changed:
  - Removed separate `distance_scratch = D_norms` sweep.
- How it works:
  - First FMAC uses `norms_dsd` directly, then remaining FMACs accumulate into
    `distance_scratch`.
- Why it helps:
  - Saves one full vector pass over local rows.

### 7) Blocked DSD layout (current best: `D_block=8`)

- What changed:
  - Host packs each PE-local shard in feature blocks.
  - Kernel traverses blocked columns with a blocked DSD path.
- How it works:
  - Instead of row-major striding by `d_dim`, blocked path strides by
    `D_block` inside each feature block.
- Why it helps:
  - Better locality / lower effective stride pressure in FMAC loop.
  - Produced consistent wins in baseline, uneven, and slight win in k_large.

### 8) `P=4, K=16` merge micro-kernel

- What changed:
  - Added explicit 4-head merge path for the fixed baseline shape.
- How it works:
  - Tracks 4 stream heads directly and updates one winning stream per output.
- Why it helps:
  - Reduces generic control overhead in a very hot merge path.
  - Small but measurable win on top of blocked DSD.

### 9) Important rejected ideas (and why)

- Full pair-packed gather (`distance+index` fused transport):
  - Regressed baseline and increased SRAM pressure (k_large link failures).
- Dynamic valid-length DSD capping:
  - Regressed measured cycle counts.
- Naive full column-major repack:
  - Regressed in this simulator/hardware path.
- Staged root ownership and tree-style merge variants:
  - Added extra synchronization/control overhead; net regressions.

### 10) Correctness invariant across all optimizations

- All accepted optimizations preserve:
  - exact squared L2 distances,
  - deterministic ordering by `(distance, index)`,
  - same public host-visible symbols and I/O contract.
