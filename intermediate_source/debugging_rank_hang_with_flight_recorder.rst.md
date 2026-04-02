# Debugging a Rank Hang with Flight Recorder

This tutorial walks through a complete workflow for diagnosing a
**single-rank hang** in a distributed PyTorch job using the
**TorchComms Flight Recorder**.  Two complementary approaches are shown:

| | Approach A | Approach B |
|---|---|---|
| **Source** | Debug server aggregated text dumps | Per-rank pickle files + FR CLI |
| **Files** | `torchcomms_fr_trace_<ts>.txt` | `per_rank/rank_0`, `per_rank/rank_1` |
| **Reader** | Human (open in editor / `cat`) | `python -m torch.distributed.flight_recorder.fr_trace` |
| **Strength** | Single file, all ranks at a glance | Automated cross-rank mismatch detection |

---

## Table of Contents

1. [Background](#background)
2. [The Scenario](#the-scenario)
3. [Running the Demo](#running-the-demo)
4. [Approach A — Aggregated Text Dumps](#approach-a--aggregated-text-dumps)
5. [Approach B — Per-Rank Pickle Dumps + FR CLI](#approach-b--per-rank-pickle-dumps--fr-cli)
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
| `retired` | `true` once the post-hook fires |
| `input_dims` / `output_dims` | Tensor shapes |
| `traceback` | Python stack trace at the call site |

The debug server provides **two dump mechanisms**:

* **Aggregated text dumps** — the periodic dumper on rank 0 fetches FR
  data from all ranks and writes a single human-readable text file
  every *N* seconds (`torchcomms_fr_trace_<ts>.txt`).
* **Per-rank pickle dumps** — the periodic dumper also triggers each
  rank's worker server to write a pickle-format trace file (`per_rank/rank_<N>`).
  These are consumed by the **FR CLI** for automated cross-rank
  mismatch analysis.

---

## The Scenario

The demo script (`verify_flight_recorder.py`) creates a two-phase workload:

```
Phase 1 (all ranks):  3× all_reduce  +  1× broadcast   ← completes normally
Phase 2:
  • Hanging rank  →  enters time.sleep(∞)
  • Other ranks   →  issue another all_reduce  →  timeout after COMM_TIMEOUT seconds
```

When the timeout fires, the dump directory contains:

```
FR_DUMP_DIR/
├── torchcomms_fr_trace_<ts>.txt   ← Approach A: aggregated text
├── stacks_<ts>.txt                ← Python stack dumps
└── per_rank/                      ← Approach B: per-rank pickles
    ├── rank_0
    └── rank_1
```

---

## Running the Demo

### Prerequisites

* `torchcomms` and `torch.distributed.debug` installed (with deps:
  `tabulate`, `jinja2`, `aiohttp`).
* Use `TEST_BACKEND=gloo TEST_DEVICE=cpu` for CPU-only testing, or
  a CUDA host with ≥ 2 GPUs.

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

## Approach A — Aggregated Text Dumps

The debug server writes periodic text snapshots aggregating data from
all ranks:

```bash
$ ls /tmp/fr_hang_debug/torchcomms_fr_trace_*.txt
torchcomms_fr_trace_20260401_192058.txt
torchcomms_fr_trace_20260401_192101.txt
torchcomms_fr_trace_20260401_192104.txt
...
```

### Reading the dump

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

The **NCCL Calls** table confirms which ranks participated:

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

### Stacks dump

The `stacks_*.txt` files show Python tracebacks, pinpointing the
exact line each rank is stuck at:

```bash
$ cat /tmp/fr_hang_debug/stacks_20260401_192104.txt

=== Rank 0 ===
  File "verify_flight_recorder.py", line 148 in main    ← all_reduce (waiting)

=== Rank 1 ===
  File "verify_flight_recorder.py", line 140 in main    ← time.sleep (the hang!)
```

**Interpretation:** Rank 1 never issued `collective_seq_id=4`.
The stacks dump confirms it is stuck in `time.sleep`, not in a collective.

---

## Approach B — Per-Rank Pickle Dumps + FR CLI

The debug server's periodic dump also triggers each rank's worker
server to write a pickle trace file.  These land in the `per_rank/`
subdirectory:

```bash
$ ls /tmp/fr_hang_debug/per_rank/
rank_0  rank_1
```

Both files exist because each rank's worker server writes its own
trace independently — even the hanging rank has a dump.

### Cross-rank mismatch analysis

```bash
python -m torch.distributed.flight_recorder.fr_trace \
  /tmp/fr_hang_debug/per_rank -p rank_
```

**Actual output:**

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

The CLI automatically detected that **rank 1 never issued
`collective_seq_id=4`** and flagged the mismatch.

### Side-by-side raw entry view

```bash
python -m torch.distributed.flight_recorder.fr_trace \
  /tmp/fr_hang_debug/per_rank -p rank_ -j
```

**Actual output:**

```
Rank 0                                             Rank 1
-------------------------------------------------  -------------------------------------------------
all_reduce(input_sizes=[[1024]], state=scheduled)  all_reduce(input_sizes=[[1024]], state=scheduled)
broadcast(input_sizes=[[1024]], state=scheduled)   broadcast(input_sizes=[[1024]], state=scheduled)
all_reduce(input_sizes=[[1024]], state=scheduled)
```

Rank 0 has **5 entries** (3 `all_reduce` + 1 `broadcast` + the stuck
`all_reduce`).  Rank 1 has only **4 entries** — the 5th `all_reduce` is
missing because rank 1 hung before issuing it.

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

## Reference

For full API documentation on the tools used in this tutorial, see
[`torch.distributed` — Debug HTTP Server](https://pytorch.org/docs/main/distributed.html#torch-distributed-debug-http-server)
(source: `/home/tushar00jain/local/pytorch/docs/source/distributed.md`), which covers:

* `start_debug_server()` parameters (`dump_dir`, `dump_interval`, `enabled_dumps`, etc.)
* Architecture (frontend server on rank 0, per-rank worker servers)
* All frontend and worker-level HTTP endpoints
* Periodic dumping configuration and supported handlers
* Registering custom handlers (Python and C++)
* Troubleshooting common issues

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
