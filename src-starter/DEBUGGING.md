# Debugging Notes

Use this file as a quick reference when the simulator looks stuck or when a
cycle count changes unexpectedly.

## Fast Sanity Loop

```bash
./commands.sh
```

`commands.sh` compiles and runs the baseline case. A host-side pause at
`memcpy_d2h` usually means the simulator is still executing device work and has
not reached `sys_mod.unblock_cmd_stream()` in `finish()`.

After a run, check:

```bash
cat sim_stats.json
tail -n 80 sim.log
```

Useful fields in `sim_stats.json`:

- `cycle_count`: main performance number to compare across edits.
- `sim_stop_cause`: should normally be `Stopped due to idleness`.
- `idle_ce_cycles` / `idle_wavelet_cycles`: useful hints for whether time is
  being spent waiting on compute or communication.

## SDK Debugger

The SDK debugger entrypoint is `csdb`. It operates on this directory as a debug
workdir and can inspect compile contexts, trace directories, symbols, memory,
registers, and wavelets.

List available compile outputs:

```bash
csdb . context list
```

Select the output you care about before image or memory queries:

```bash
csdb . context select out/baseline
csdb . context show
```

List or select post-run traces:

```bash
csdb . trace list
csdb . trace select .
csdb . trace show
```

Inspect image metadata and symbols:

```bash
csdb . image info
csdb . image lookup --name final_distances
csdb . image lookup --name local_distances
csdb . image lookup --name distance_scratch
```

Read symbol values from the selected target/core when available:

```bash
csdb . memory read --name final_distances --stdoutput
csdb . memory read --name final_indices --stdoutput
csdb . memory read --name local_distances --stdoutput
```

The most useful cycle-debug workflow is:

1. Re-run one case and save its `sim_stats.json` under a case-specific name.
2. Use `csdb . context select <out-dir>` for the matching compile output.
3. Use `csdb . trace list` / `trace select` to point at the latest trace.
4. Inspect symbols around the suspected phase:
   `q`, `distance_scratch`, `local_distances`, `row_top_distances`,
   `final_distances`.
5. Compare `cycle_count` after one isolated edit. Do not mix algorithm changes
   and comment/docs cleanup in the same performance measurement.

