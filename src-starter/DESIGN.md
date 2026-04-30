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
