# Debugging a Rank Hang with Flight Recorder

This tutorial walks through diagnosing a **single-rank hang** in a
distributed PyTorch job using the **TorchComms Flight Recorder** and the
**Debug Server**'s periodic dump.

For full API documentation on the debug server, its
endpoints and periodic dumping see the
[torch.distributed debug HTTP server docs](https://pytorch.org/docs/main/distributed.html#torch-distributed-debug-http-server)
(source: `/home/tushar00jain/local/pytorch/docs/source/distributed.md`).


For a reference on the Flight Recorder, see [Flight Recorder Hook](https://meta-pytorch.org/torchcomms/main/hooks.html#flightrecorderhook) in torchcomms.

---

## Table of Contents

1. [Background](#background)
2. [The Scenario](#the-scenario)
3. [Running the Demo](#running-the-demo)
4. [Reading the Aggregated Text Dumps](#reading-the-aggregated-text-dumps)
5. [Running the FR CLI on Per-Rank Pickle Dumps](#running-the-fr-cli-on-per-rank-pickle-dumps)
6. [What to Look For](#what-to-look-for)
7. [Reference](#reference)

---

## Background

The **Flight Recorder** is a ring-buffer that records every collective
operation issued through a TorchComms communicator.  Each entry captures:

| Field | Description |
|---|---|
| `collective_seq_id` | Monotonically increasing sequence number (same across all ranks for a given collective) |
| `profiling_name` | e.g. `nccl:all_reduce`, `nccl:broadcast` |
| `state` | `scheduled` → `started` → `completed` |
| `input_dims` / `output_dims` | Tensor shapes |
| `traceback` | Python stack trace at the call site |

When periodic dumping is enabled on the debug server, each dump cycle
produces two kinds of output:

* **Aggregated text files** (`torchcomms_fr_trace_<ts>.txt`) — the
  frontend on rank 0 fetches FR data from all ranks and writes a
  human-readable table.
* **Per-rank pickle files** (`per_rank/rank_<N>`) — each rank's worker
  server writes its own pickle trace.  These can be fed to the
  **FR CLI** (`python -m torch.distributed.flight_recorder.fr_trace`)
  for automated cross-rank mismatch detection.

---

## The Scenario

The demo script (`verify_flight_recorder.py`) creates a two-phase workload:

* Phase 1 (all ranks): 3 all_reduce 1 broadcast operations completes normally
* Phase 2:
  * Hanging rank enters `time.sleep`
  * Other ranks issue another all_reduce that time out waiting for the hanging rank

When the timeout fires, the dump directory contains:

```
FR_DUMP_DIR/
├── torchcomms_fr_trace_<ts>.txt   ← aggregated text
└── per_rank/                      ← per-rank pickle files
    ├── rank_0
    └── rank_1
```

---

## Running the Demo

### Prerequisites

* `torchcomms` and `torch.distributed.debug` installed
* Use `TEST_BACKEND=gloo TEST_DEVICE=cpu` for CPU-only testing, or
  a CUDA host with more than 2 GPUs.

### Launch

```bash
FR_DUMP_DIR=/tmp/fr_hang_debug \
FR_DUMP_INTERVAL=3 \
COMM_TIMEOUT=15 \
TEST_BACKEND=gloo \
TEST_DEVICE=cpu \
torchrun --nproc_per_node=2 verify_flight_recorder.py
```

| Variable | Default | Description |
|---|---|---|
| `FR_DUMP_DIR` | `/tmp/fr_hang_debug` | Root dump directory |
| `FR_DUMP_INTERVAL` | `5` | Seconds between periodic dumps |
| `COMM_TIMEOUT` | `30` | Communicator timeout (seconds) |
| `HANGING_RANK` | `-1` (last rank) | Which rank to hang |
| `TEST_BACKEND` | `gloo` | Communication backend |
| `TEST_DEVICE` | `cuda` | Tensor device |

### Expected Output

```
[Rank 0/2] device=0, hanging_rank=1, timeout=15s
[Rank 1/2] device=1, hanging_rank=1, timeout=15s
[Rank 0] Debug server: http://localhost:25999
[Rank 0] Periodic dumps every 3.0s → /tmp/fr_hang_debug
[Rank 0] Per-rank pickles → /tmp/fr_hang_debug/per_rank
[Rank 0] Phase 1: Running 3 all_reduce + 1 broadcast
[Rank 0] Phase 1 complete
[Rank 0] Phase 2: all_reduce (rank 1 will NOT participate)
[Rank 0] Expecting timeout in ~15s ...
[Rank 1] Phase 1 complete
[Rank 1] >>> HANGING – entering infinite sleep <<<

... periodic mismatch warnings every 3 seconds ...

Not all ranks joining collective, sequence number: 4
collective: nccl:all_reduce
missing ranks: {1}
collective state: scheduled

... ~15 seconds pass ...

[Rank 0] Caught timeout: RuntimeError: Timed out waiting 15000ms for recv operation
[Rank 0] Pickle trace written to /tmp/fr_hang_debug/per_rank/rank_0
```

---

## Reading the Aggregated Text Dumps

The debug server writes periodic text snapshots aggregating data from
all ranks:

```bash
$ ls /tmp/fr_hang_debug/torchcomms_fr_trace_*.txt
torchcomms_fr_trace_20260401_192058.txt
torchcomms_fr_trace_20260401_192101.txt
torchcomms_fr_trace_20260401_192104.txt
...
```

Open one of the snapshots written during the hang:

```bash
cat /tmp/fr_hang_debug/torchcomms_fr_trace_20260401_192104.txt
```

The **Collectives** table shows every recorded operation:

```
--- Collectives ---
  id  group_id    pass_check  collective_seq_id  collective_name    collective_state  missing_ranks
   0  main_comm   True        0                  nccl:all_reduce    scheduled
   1  main_comm   True        1                  nccl:all_reduce    scheduled
   2  main_comm   True        2                  nccl:all_reduce    scheduled
   3  main_comm   True        3                  nccl:broadcast     scheduled
   4  main_comm   True        4                  nccl:all_reduce    scheduled         {1}    ← MISMATCH
```

The **NCCL Calls** table shows which ranks participated:

```
--- NCCL Calls ---
  id  collective_id  group_id   global_rank  collective_type
   0              0  main_comm            0  nccl:all_reduce
   1              0  main_comm            1  nccl:all_reduce
   ...
   6              3  main_comm            0  nccl:broadcast
   7              3  main_comm            1  nccl:broadcast
   8                 main_comm            0  nccl:all_reduce   ← Only rank 0!
```

The **Dump File** section confirms per-rank pickle files were written:

```
=== TorchComms FR Dump File ===
Rank 0: OK - Flight Recorder debug info written to /tmp/fr_hang_debug/per_rank/rank_0
Rank 1: OK - Flight Recorder debug info written to /tmp/fr_hang_debug/per_rank/rank_1
```

The `stacks_*.txt` files show Python tracebacks, pinpointing the
exact line each rank is stuck at:

```bash
$ cat /tmp/fr_hang_debug/stacks_20260401_192104.txt

=== Rank 0 ===
  File "verify_flight_recorder.py", line 148 in main    ← all_reduce (waiting)

=== Rank 1 ===
  File "verify_flight_recorder.py", line 140 in main    ← time.sleep (the hang!)
```

Rank 1 never issued `collective_seq_id=4`.  The stacks dump confirms
it is stuck in `time.sleep`, not in a collective.

---

## Running the FR CLI on Per-Rank Pickle Dumps

The periodic dump also triggers each rank's worker server to write a
pickle trace file into the `per_rank/` subdirectory:

```bash
$ ls /tmp/fr_hang_debug/per_rank/
rank_0  rank_1
```

### Cross-rank mismatch analysis

```bash
python -m torch.distributed.flight_recorder.fr_trace \
  /tmp/fr_hang_debug/per_rank -p rank_
```

Output:

```
Not all ranks joining collective, sequence number: 4
internal record id: 4
group info: main_comm:gloo
collective: nccl:all_reduce
missing ranks: {1}
input sizes: [[1024]]
output sizes: [[1024]]
world size: 2
expected ranks: {0, 1}
collective state: scheduled
```

The CLI detected that rank 1 never issued `collective_seq_id=4`.

### Side-by-side raw entry view

```bash
python -m torch.distributed.flight_recorder.fr_trace \
  /tmp/fr_hang_debug/per_rank -p rank_ -j
```

Output:

```
Rank 0                                             Rank 1
-------------------------------------------------  -------------------------------------------------
all_reduce(input_sizes=[[1024]], state=scheduled)  all_reduce(input_sizes=[[1024]], state=scheduled)
broadcast(input_sizes=[[1024]], state=scheduled)   broadcast(input_sizes=[[1024]], state=scheduled)
all_reduce(input_sizes=[[1024]], state=scheduled)
```

Rank 0 has 5 entries (3 `all_reduce` + 1 `broadcast` + the stuck
`all_reduce`).  Rank 1 has only 4 — the 5th `all_reduce` is missing
because rank 1 hung before issuing it.

### With stack traces

```bash
python -m torch.distributed.flight_recorder.fr_trace \
  /tmp/fr_hang_debug/per_rank -p rank_ -j --print_stack_trace
```

This adds Python stack traces to each entry, showing exactly where in
user code each collective was called.

---

## What to Look For

| Symptom | Likely cause |
|---|---|
| `missing_ranks: {N}` in the Collectives table | Rank N hung or crashed before issuing the next collective |
| Rank X's last entry is `state=started`, others are `completed` | Rank X issued the collective but is waiting for a peer that never joined |
| Mismatched `collective_name` at the same `collective_seq_id` | Code-path divergence — ranks are calling different collectives |
| Mismatched `input_sizes` / `output_sizes` | Tensor shape inconsistency across ranks |
| Stacks dump shows `time.sleep` or user code (not a collective) | The rank is stuck in compute, not in a collective |

---

### FR CLI Quick Reference

```bash
# Cross-rank mismatch analysis:
python -m torch.distributed.flight_recorder.fr_trace <dir> -p <prefix>

# Side-by-side raw entries per rank:
python -m torch.distributed.flight_recorder.fr_trace <dir> -p <prefix> -j

# With stack traces:
python -m torch.distributed.flight_recorder.fr_trace <dir> -p <prefix> -j --print_stack_trace

# Best-effort when some rank dumps are missing:
python -m torch.distributed.flight_recorder.fr_trace <dir> -p <prefix> --allow-incomplete-ranks
```
