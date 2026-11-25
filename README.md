# ML-communications Benchmark
__Draft__

This benchmark is written in anticipation that a future system may include 
an optional, secondary interconnect, with respect to the primary interconnect
which ensures communication between the nodes. The role of the secondary 
interconnect can be in moving data between available optional accelerators 
within a node or a pod.

This benchmark attempts to validate the communication operations for the 
optional interconnect by

- moving data between different memories available to different accelerators
- exercising the compute-communication overlap functionality, anticipating 
  there will be at least two mechanisms available such that compute and 
  communication can progress simultaneously 
  (**TODO**: direct implementation through PyTorch multi-processing)

We choose PyTorch in implementing this benchmark because of this framework's 
prevalence in scientific research with active learning components. We 
anticipate that we will have an API at a sufficiently high level that 
PyTorch will be able to interact with the hardware in an appropriate fashion.

## Coverage
We are measuring the latency (**TODO**, discuss) and bandwidth of a few major 
collectives called through PyTorch, leveraging the vendor specific backend 
communication library. We anticipate that such a library might exist, and 
would be optimized for extracting better performance while using the 
optional interconnect and optional accelerators.

The communication patterns and the background compute loosely resemble a 
large language model training's forward pass and inference workload, with only 
the gradient synchronization step as an addition. Support for these 
operations with irregular shapes of the message array, sizes and memory access 
patterns, in principle, should provide enough coverage for future applications.
(**TODO** direct implementation of irregular shapes and sizes, so far we do 
fixed sizes. Discussion is needed whether we should require irregular shapes 
and sizes)

The communication operations that we aim to test are

- AllReduce -- (through Tensor Parallelism (TP), and Gradient Synchronization)
- AllGather + ReduceScatter -- (through Sequence Parallelism, SP)
- AlltoAll -- (through Ulysses Sequence Parallelism, ULSS)
- Barrier
- Broadcast (**TODO**)
- Point-to-Point: isend + irecv (**TODO**)

We have implemented these in a background of simple compute, so that this 
benchmark retains the flavor of a full application without being too complex to
execute.

## Size of the run
We expect this benchmark to be run on

- the smallest possible scalable partition, for example, using every 
  optional accelerator on a node
- the largest possible connected domain where the optional interconnect 
  can access an accelerator's memory (**TODO** We need a better framing for 
  full system runs)

## Precision types
Currently the benchmark runs on FP32, BF16, and int8 (**TODO**). We are 
flexible in our lower precision requirements, subject to accuracy estimates 
and tests. (**TODO** Discuss, if this is appropriate)

## Current memory usage in BF16 (per rank)
In each timing loop, each TP and SP rank keeps ~1.1 GB of weight matrices 
alive in memory, facilitated by sharding. ULSS keeps the full copy of the 
weights and takes ~13.4 GB. Considering the pre-allocated buffers, various 
shards, input, the gradient AllReduce buffer and intermediate output, TP and SP 
would occupy in total ~8.6 GB and ULSS would occupy ~19.2 GB. The benchmark 
reuses one layer's weights for every layer, so this is the same at any number 
of layers.

## Current message sizes for collectives in BF16
The TP AllReduce, SP AllGather and ReduceScatter have a full buffer
size of ~874 MB, 2 calls each. SP performs the collective using a ~72.9 MB 
per-peer chunk.
ULSS AlltoAll is 2 calls, each with a ~72.9 MB buffer, performed with a ~6.1 MB 
per-peer chunk.

The gradient synchronization AllReduce has a 2 GB buffer size, also called a 
bucket. The synchronization step runs once per timing loop, after all layers. 
It uses a fixed 2 GB buffer (bucket): 1 bucket for TP and SP, and 7 for ULSS. 

### Full 128 layers
For TP, SP, and ULSS, more layers repeat the same collectives. The gradient 
synchronization AllReduce keeps a fixed 2 GB bucket, so the number of buckets 
grows with layers: 72 for TP and SP, and 857 for ULSS.

## Software requirements

This code relies on PyTorch and the vendor specific collective library backend.
Our latest runs were performed with PyTorch `2.13.0`. We have used `mpiexec` as 
the primary launcher and used vendor provided environment variables for resource 
count and placement. (**TODO** elaborate more carefully, discuss PALS 
equivalent)

## Figure of Merit(?)
(**TODO**)
