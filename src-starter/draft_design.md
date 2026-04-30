# Draft Design Memo

Start time: 9:38 PM EST, April 28, 2026

This kernel computes exact top-K nearest neighbors over a `P x P` PE grid. The
host shards `D` by rows in PE row-major order. PE `(px, py)` has linear id
`py * P + px` and owns rows `[linear_id * rows_per_pe, (linear_id + 1) *
rows_per_pe)`, with `valid_count` used to ignore padding in uneven cases. The
host copies `q` only to PE `(0,0)`. The device broadcasts `q` across the top row
with `mpi_x.broadcast`, then down each column with `mpi_y.broadcast`.

```text
Host -> D shards + valid_count -> all PEs
Host -> q -> PE(0,0)
PE(0,0) -> row broadcast q -> top row
top row -> column broadcast q -> all PEs
all PEs -> local top-K -> row gather -> column gather -> PE(0,0)
PE(0,0) -> final indices/distances -> Host
```

Each PE computes exact squared L2 distances for its valid rows:

```text
dist(i) = sum_j (D[i,j] - q[j])^2
```

The local top-K algorithm uses sorted insertion into two arrays,
`local_distances[K]` and `local_indices[K]`. A candidate is better if its
distance is smaller, or if the distance is equal and its original global row
index is smaller. This is `O(rows_per_pe * K)` per PE after distance
computation. I chose sorted insertion because it is simple, deterministic, and
produces already-sorted local candidate lists, reducing tie-breaking risk.

For result movement, each PE sends `K` distances and `K` indices. Candidates
are gathered across rows to column 0, then gathered up column 0 to PE `(0,0)`.
The root PE merges all `P * P * K` candidates using the same comparator. This
makes fabric arrival order irrelevant: every candidate is compared by the total
order `(distance, original_index)`, so any arrival order produces the same final
top-K.

Worst-case candidate traffic is `P * P * K` distances plus `P * P * K` indices.
For baseline (`P=4`, `K=16`), root merges 256 candidate pairs. For `k_large`
(`P=2`, `K=256`), root merges 1024 candidate pairs. Query broadcast traffic is
`d_dim` fp32 values across each row/column broadcast step.

The main bottleneck is the scalar distance loop, especially for `d_dim=32`.
With more time, I would optimize distance computation using DSD/vector
operations or the algebraic form `||D_i - q||^2 = ||D_i||^2 - 2 dot(D_i, q) +
||q||^2` with host-precomputed row norms. For `K=256`, I would also consider
replacing sorted insertion with a fixed-size max-heap. For larger grids, I
would make the merge hierarchical so row roots reduce to K before the final
column gather.

## Devlog: `k_large` Optimization Notes

The `k_large` case is different from the other cases because `K == rows_per_pe`
(`K=256`, `rows_per_pe=256`). In that case, each PE's local top-K is simply all
valid local rows. The first implementation still used sorted insertion locally,
which performs unnecessary shifting work: every local row was inserted into a
length-256 sorted array even though none of the valid local rows can be dropped.

I added a special local path for `K == rows_per_pe`:

```text
for each local row:
  compute exact distance
  write distance directly to local_distances[row]
  write global index directly to local_indices[row]
```

Invalid padded rows still receive sentinel values `(INF, SENTINEL_INDEX)`, so
the root merge can ignore them naturally under the same comparator. This keeps
the algorithm exact while removing the local `O(rows_per_pe * K)` insertion
work for the large-K case.

I also changed the root merge from sorted insertion to repeated selection. The
root has `P * P * K` gathered candidates. Instead of inserting each candidate
into a sorted output array and shifting entries, the root repeatedly scans the
candidate array to find the next best `(distance, original_index)`, writes it
to the final output, and marks that candidate consumed. This is still simple and
exact, but avoids large shift costs when `K=256`.

These optimizations do not change the comparator or output semantics. They only
remove avoidable work in the worst `K=256` case. The next optimization target is
still distance computation: either DSD/vectorized row math or the algebraic
formula `||D_i - q||^2 = ||D_i||^2 - 2 dot(D_i, q) + ||q||^2` with precomputed
row norms.

## Devlog: DSD Dot-Product Distance

The original distance loop computed `sum_j (D[row,j] - q[j])^2` with scalar
CSL loops. I changed the distance calculation to the equivalent algebraic form:

```text
||D_i - q||^2 = dot(D_i, D_i) - 2 * dot(D_i, q) + dot(q, q)
```

The PE now computes those dot products with memory DSDs and `@map`, following
the BLAS-style dot pattern from the SDK examples. This is still exact squared
L2 distance under the same comparator, but it gives the compiler/hardware a more
vector-like memory access pattern than the manually nested scalar loop.

This adds one `q_norm = dot(q, q)` per PE after `q` broadcast, then two dot
products per candidate row: `dot(row, row)` and `dot(row, q)`. A further
optimization would be to precompute `dot(row, row)` on the host and copy row
norms alongside `D`, reducing device distance work to one dot product per row
plus a few scalar operations.

## Devlog: Host-Precomputed Row Norms

I then implemented the row-norm optimization. The host computes
`D_norms[i] = dot(D_i, D_i)` once while packing the input shards, and copies one
`rows_per_pe`-length norms array to each PE alongside `D`. The PE distance
formula becomes:

```text
distance = D_norms[row] - 2 * dot(D_row, q) + dot(q, q)
```

This removes `dot(D_row, D_row)` from the device loop, so each row now needs
only one DSD dot product with `q` instead of two row dot products. The memory
cost is small: 128 extra fp32 values per PE for baseline (512 bytes) and 256
values per PE for `k_large` (1 KB). This is allowed because row norms are
derived from `D` and do not change the exact squared-L2 semantics; they only
avoid recomputing a query-independent quantity on device.

## Devlog: Hierarchical Merge

The original gather path sent every PE's `K` local candidates all the way to
PE `(0,0)`, so the root merged `P * P * K` candidates. I changed this to a
two-level merge:

```text
1. Gather local candidates across each row to column 0.
2. Each row root merges its P*K row candidates down to K.
3. Gather only those K row winners up column 0 to PE (0,0).
4. The final root merges P*K candidates into the final K output.
```

This preserves exactness because every merge uses the same `(distance,
original_index)` comparator. Any candidate not in a row's top-K cannot be in the
global top-K: there are already K candidates in the same row group that are no
worse. The main benefit is reducing the final root's input size from
`P * P * K` to `P * K`. For `P=4`, `K=16`, this changes the final root merge
from 256 candidates to 64 candidates. For `P=2` cases, the benefit is smaller,
so the `duplicates` cycle count was essentially unchanged after this edit.

## Devlog: Heap-Based Merge

For `K=256`, even after hierarchical merge, each merge stage can still compare
many candidates. I added a fixed-size max-heap helper for row-root and final-root
merges. The heap stores the current best K candidates, with the worst candidate
at the root. A new candidate replaces the root only if it is better under the
same `(distance, original_index)` comparator. This changes merge maintenance
from repeated sorted-array shifts to `O(log K)` heap repair.

The heap itself is not sorted, so after all candidates are considered, the PE
emits sorted output by repeatedly scanning the heap for the next best candidate
and marking it consumed. This keeps the final output deterministic and simple.
For small K, like `duplicates` with `K=8`, the heap has little benefit and may
only break even. Its purpose is the large-K path, especially `k_large`.
