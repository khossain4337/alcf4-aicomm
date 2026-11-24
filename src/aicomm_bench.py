from mpi4py import MPI  # noqa: F401  # needed for CCL_KVS_MODE=mpi
import os
import sys
import time
import math
import argparse
import logging
from collections import namedtuple

import torch
import torch.distributed as dist
import numpy as np
np.set_printoptions(suppress=True)
from torch.profiler import profile, record_function, ProfilerActivity

parser = argparse.ArgumentParser(description="parse input arguments for tensor and data  parallel partial benchmark")

parser.add_argument("-s", "--sequence_length",
                    help="Maximum sequence length. The size of the ALLGATHER buffer",
                    type=int, default=4608)
parser.add_argument("-d", "--hidden_dimension",
                    help="Hidden dimension for the matrix multiplication. Proxy for the model size.",
                    type=int, default=9216)
parser.add_argument("-it", "--iterations",
                    help="number of iterations for the timing loop",
                    type=int, default=3)
parser.add_argument("-wit", "--warmup_iterations", help="number of warmup iterations",
                    type=int, default=2)

parser.add_argument("-bucket", "--grad_allreduce_bucket", help="The largest bucket of message passed to do the allreduce over the data parallel groups (in number of elements in a message).",
                    type=float, default=5e8)

parser.add_argument("-tp_degree", "--tensor_parallel_degree", help="Tensor Parallel degree. In this context, the model is distributed across the number of (tensor parallel degree)) ranks",
                    type=int, default=None)

parser.add_argument("-sp_switch", "--sequence_parallel_switch", help="Switch sequence parallelism on or off", action='store_true')

parser.add_argument("-ulysses_enable", "--ulysses_sequence_parallel_switch", help="Switch Ulysses parallelism on or off", action='store_true')

parser.add_argument("-n_layers", "--number_of_transformer_layers", help="Number of transformer layers", type=int, default=80)

parser.add_argument("-ones", "--torch_ones", help="Fixed inputs and weights with torch.ones", action='store_true')

parser.add_argument("--init_std", help="Standard deviation for initializing weights and inputs with normal distribution",
                    type=float, default=0.01)

parser.add_argument("--init_mean", help="Mean for initializing weights and inputs with normal distribution",
                    type=float, default=0.0)

parser.add_argument("-p", "--precision", help="Data type for the elements of a tensor. float32 and bfloat16 supported.",
                    type=str, default="float32")

parser.add_argument("-dvc", "--device", help="Device type. cuda and xpu supported.",
                    type=str, default="cuda")

parser.add_argument("-f", "--log_file", help="Output file name",
                    type=str, default="tensor_parallel.log")

parser.add_argument("-dir", "--log_directory", help="Output file path",
                    type=str, default="logs/")

parser.add_argument("--no_grad_sync", help="Do NOT perform gradient synchronization over dp groups", action='store_true')
parser.add_argument("--barrier", help="Put a torch barrier before the grad sync allreduce", action='store_true')

parser.add_argument("--logging", help="Switch logging on", action='store_true')
parser.add_argument("--log_all_ranks", help="Every rank logs; default is rank 0 only", action='store_true')
parser.add_argument("--save", help="Save detail results in npy format", action='store_true') ## Generates huge files, use with caution
parser.add_argument("--trace", default=None, type=str)

args = parser.parse_args()


def trace_func(func):
    def wrapper(*args, **kwargs):
        with record_function(func.__qualname__):
            return func(*args, **kwargs)
    return wrapper


if args.precision == "float32":
    data_type = torch.float32
    data_type_multiplier = 32 ## 32 Bits = 4 Bytes
elif args.precision == "bfloat16":
    data_type = torch.bfloat16
    data_type_multiplier = 16 ## 16 Bits

dev = getattr(torch, args.device)  # torch.cuda / torch.xpu


@trace_func
def synchronize():
    dev.synchronize()


def timed(fn):
    """Run fn, sync the device, return (result, elapsed ns)."""
    start = time.perf_counter_ns()
    out = fn()
    synchronize()
    return out, time.perf_counter_ns() - start


def barrier_ms(group):
    if args.device == "xpu":
        _, ns = timed(lambda: dist.barrier(group=group, device_ids=[torch.xpu.current_device()]))
    else:
        _, ns = timed(lambda: dist.barrier())
    return ns / 1e6


def matmul_flops(input_shape, other_shape):
    """
    Calculates matmul floating point operations for torch.matmul
    """
    assert len(input_shape) > 1 #Not implemented
    assert len(other_shape) == 2 #Not implemented
    #Reference: https://math.stackexchange.com/questions/3512976/proof-of-of-flops-in-matrix-multiplication
    return np.prod(input_shape[:-1]) * other_shape[-1] * (2*other_shape[-2]-1)


rank = int(os.environ["PALS_RANKID"])
world_size = int(os.environ["PALS_WORLD_SIZE"])

log_filepath = os.path.join(args.log_directory, f"{args.log_file}_R{rank}_of_{world_size}.log")
handlers = [logging.StreamHandler(sys.stdout)]
if args.logging:
    os.makedirs(args.log_directory, exist_ok=True)
    handlers.append(logging.FileHandler(log_filepath, mode="a", delay=True))
logging.basicConfig(level="INFO" if rank == 0 or args.log_all_ranks else "WARNING", handlers=handlers)

logging.info(f"rank {rank}/{world_size}")

# No mask: all tiles visible. --gpu-bind: one tile per rank, so device_count()=1.
assert int(os.environ["PALS_LOCAL_SIZE"]) <= dev.device_count() or dev.device_count() == 1
dev.set_device(int(os.environ["PALS_LOCAL_RANKID"]) % dev.device_count())
device = f"{args.device}:{dev.current_device()}"

# MASTER_ADDR / MASTER_PORT come from mpiexec -env
dist.init_process_group(
    backend={"cuda": "nccl", "xpu": "xccl"}[args.device],
    init_method="env://",
    world_size=world_size,
    rank=rank,
)

TP = args.tensor_parallel_degree or 1
tp_group = None
for i in range(world_size // TP):
    ranks = list(range(i * TP, (i + 1) * TP))
    if rank == 0:
        logging.info(f"TP group = {ranks}")
    group = dist.new_group(ranks)
    if rank in ranks:
        tp_group = group

"""
 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11
12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35
36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47 """

dp_group = None
for i in range(TP):
    ranks = list(range(i, world_size, TP))
    if rank == 0:
        logging.info(f"DP group = {ranks}")
    group = dist.new_group(ranks)
    if rank in ranks:
        dp_group = group

"""
0,12,24,36
1,13,25,37
...
"""
S = args.sequence_length
H = args.hidden_dimension
M = 1
SP = args.sequence_parallel_switch
ULSS = args.ulysses_sequence_parallel_switch
assert not (SP and ULSS)

kw = dict(dtype=data_type, device=device)


def init_tensor(shape, ones_scale=1.0):
    if args.torch_ones:
        return torch.ones(shape, **kw) * ones_scale
    return torch.normal(mean=args.init_mean, std=torch.ones(shape, **kw) * args.init_std)


partial_input = init_tensor([S // TP, M, H])
input = init_tensor([S, M, H])
input_ulss = init_tensor([S, M, H // TP])
partial_interim2 = torch.zeros([S // TP, M, H], **kw)
partial_interim4 = torch.zeros([S // TP, M, H], **kw)
global_context_ulss = torch.zeros([S, M, H // TP], **kw)

interim1_proxy = torch.ones([S, M, H // TP], **kw)
interim2_proxy = torch.ones([S, M, H], **kw)
interim3_proxy = torch.ones([S, M, 4 * H // TP], **kw)
interim4_proxy = torch.ones([S, M, H], **kw)

wtp = 1 if ULSS else TP  # ULSS keeps full weights
attn_W_QKV = init_tensor([H // wtp, H], 1e-4)
attn_WO = init_tensor([H, H // wtp], 1e-3)
mat_h_4h = init_tensor([4 * H // wtp, H], 1e-4)
mat_4h_h = init_tensor([H, 4 * H // wtp], 1e-3)

n_layers = args.number_of_transformer_layers
number_of_total_parameters = sum(w.numel() for w in (attn_W_QKV, attn_WO, mat_h_4h, mat_4h_h)) * n_layers

highest_bucket_size = int(args.grad_allreduce_bucket)
n_iter_grad_sync = math.ceil(number_of_total_parameters / highest_bucket_size)
allreduce_grad = torch.ones([highest_bucket_size], **kw)

OPERATIONS = ["allgather_1", "QKV", "WO", "reduce_scatter_1",
              "allreduce_1", "allgather_2", "H_4H", "4H_H", "reduce_scatter_2",
              "ulss_QKV", "ulss_WO", "ulss_H_4H", "ulss_4H_H",
              "all2all_1", "all2all_2", "allreduce_2", "grad_sync",
              "timing_loop", "tp_sync", "dp_sync_begin", "dp_sync_end",
              "ulss_sync"]

TensorParallelResults = namedtuple("TensorParallelResults",
                                   ["T_dict_individual", "T_dict_total",
                                    "T_grad_sync_individual", "interim1",
                                    "interim2", "interim3", "interim4",
                                    "ulss_interim1", "ulss_interim2",
                                    "ulss_interim3", "ulss_interim4",
                                    "allreduce_grad"])


@trace_func
def tensor_parallel(N_timing_loop, n_layers, n_iter_grad_sync, warmup=False):
    """Per-op times in ns (T_ind, T_grad); totals and barriers in ms (T_tot). Warmup records zeros."""
    T_ind = {f'T_{op}': np.zeros((N_timing_loop, n_layers)) for op in OPERATIONS[:-5]}
    T_tot = {f'T_{op}': np.zeros(N_timing_loop) for op in OPERATIONS}
    T_grad = np.zeros((N_timing_loop, n_iter_grad_sync))
    interim1 = ulss_interim1 = interim1_proxy
    interim2 = ulss_interim2 = interim2_proxy
    interim3 = ulss_interim3 = interim3_proxy
    interim4 = ulss_interim4 = interim4_proxy

    def t(key, fn):
        out, ns = timed(fn)
        if not warmup:
            T_ind[key][m][i] = ns
        return out

    for m in range(N_timing_loop):
        loop_start = time.perf_counter_ns()
        if args.barrier:
            T_tot["T_tp_sync"][m] = barrier_ms(tp_group)
            if ULSS:
                T_tot["T_ulss_sync"][m] = T_tot["T_tp_sync"][m]
        for i in range(n_layers):
            if ULSS:
                ulss_interim1 = t("T_ulss_QKV", lambda: torch.matmul(partial_input, attn_W_QKV.t()))
                t("T_all2all_1", lambda: dist.all_to_all_single(global_context_ulss, input_ulss, group=tp_group))
                ulss_interim1 = ulss_interim1.reshape([S, M, H // TP])
                t("T_all2all_2", lambda: dist.all_to_all_single(ulss_interim1, global_context_ulss, group=tp_group))
                ulss_interim1 = ulss_interim1.reshape([S // TP, M, H])
                ulss_interim2 = t("T_ulss_WO", lambda: torch.matmul(ulss_interim1, attn_WO.t()))
                ulss_interim3 = t("T_ulss_H_4H", lambda: torch.matmul(ulss_interim2, mat_h_4h.t()))
                ulss_interim4 = t("T_ulss_4H_H", lambda: torch.matmul(ulss_interim3, mat_4h_h.t()))
                continue
            if SP:
                t("T_allgather_1", lambda: dist.all_gather_into_tensor(input, partial_input, group=tp_group))
            interim1 = t("T_QKV", lambda: torch.matmul(input, attn_W_QKV.t()))
            interim2 = t("T_WO", lambda: torch.matmul(interim1, attn_WO.t()))
            if SP:
                t("T_reduce_scatter_1", lambda: dist.reduce_scatter_tensor(partial_interim2, interim2, group=tp_group))
                t("T_allgather_2", lambda: dist.all_gather_into_tensor(interim2, partial_interim2, group=tp_group))
            else:
                t("T_allreduce_1", lambda: dist.all_reduce(interim2, group=tp_group))
            interim3 = t("T_H_4H", lambda: torch.matmul(interim2, mat_h_4h.t()))
            interim4 = t("T_4H_H", lambda: torch.matmul(interim3, mat_4h_h.t()))
            if SP:
                t("T_reduce_scatter_2", lambda: dist.reduce_scatter_tensor(partial_interim4, interim4, group=tp_group))
            else:
                t("T_allreduce_2", lambda: dist.all_reduce(interim4, group=tp_group))
        synchronize()
        if not args.no_grad_sync:
            if args.barrier:
                T_tot["T_dp_sync_begin"][m] = barrier_ms(dp_group)
            for k in range(n_iter_grad_sync):
                _, ns = timed(lambda: dist.all_reduce(allreduce_grad, group=dp_group))
                if not warmup:
                    T_grad[m][k] = ns
            if args.barrier:
                T_tot["T_dp_sync_end"][m] = barrier_ms(dp_group)
        T_tot["T_timing_loop"][m] = (time.perf_counter_ns() - loop_start) / 1e6

    for key in T_ind:
        T_tot[key] = T_ind[key].sum(axis=1) / 1e6
    T_tot["T_grad_sync"] = T_grad.sum(axis=1) / 1e6
    return TensorParallelResults(T_ind, T_tot, T_grad,
                                 interim1, interim2, interim3, interim4,
                                 ulss_interim1, ulss_interim2, ulss_interim3, ulss_interim4,
                                 allreduce_grad)


if rank == 0:
    logging.info("start loop")

# Untimed; also keeps warmup out of the profiler trace.
if args.warmup_iterations > 0:
    tensor_parallel(args.warmup_iterations, n_layers, n_iter_grad_sync, warmup=True)

if args.trace is not None:
    activities = [ProfilerActivity.CPU, ProfilerActivity.XPU if args.device == "xpu" else ProfilerActivity.CUDA]
    with profile(activities=activities, record_shapes=True) as prof:
        result = tensor_parallel(args.iterations, n_layers, n_iter_grad_sync)
    prof.export_chrome_trace(f"{args.log_directory}/{args.trace}-{rank}-of-{world_size}.json")
else:
    result = tensor_parallel(args.iterations, n_layers, n_iter_grad_sync)

T_dict_individual, T_dict_total, T_grad_sync_individual = result[:3]
interim1, interim2, interim3, interim4 = result[3:7]
ulss_interim1, ulss_interim2, ulss_interim3, ulss_interim4 = result[7:11]
logging.info(f"ulss_interim_1 Shape from results = {ulss_interim1.shape}")


def log_bw(name, nbytes, n, factor, t_ns):
    """nccl-tests convention: alg bw = buffer bytes / time; bus bw = alg bw * factor,
    factor = 2(n-1)/n for allreduce, (n-1)/n for allgather/reduce-scatter/all2all. bytes/ns = GB/s."""
    hi, lo = nbytes / np.min(t_ns), nbytes / np.max(t_ns)
    logging.info(f"{name}: {nbytes / 1e6:.1f} MB, n={n}, alg bw max {hi:.2f} min {lo:.2f} GB/s, "
                 f"bus bw max {hi * factor:.2f} min {lo * factor:.2f} GB/s")


def format_logging_timings(text, data, time_multiplier=1.0):
    data = data / time_multiplier
    logging.info(f"{text} takes max. {np.max(data):.4f} ms,  min. {np.min(data):.4f} ms, avg. {np.mean(data):.4f} ms")


def format_logging_flops(text, in_shape, weight_shape, layers, data):
    flops = matmul_flops(in_shape, weight_shape) * layers
    tflops = lambda t_ms: flops / (1e12 * (t_ms / 1000))
    logging.info(f"{text} TFLOPS max. {tflops(np.min(data)):.4f},  min. {tflops(np.max(data)):.4f}, avg. {tflops(np.mean(data)):.4f}")


logging.info(f"Results from RANK {rank} of {world_size}")
logging.info(f"==== Main Results ====\n")
logging.info(f"Running with {args.precision} data type")
logging.info(f"==== List of Arguments ====")
logging.info(f"Sequence Length = {args.sequence_length}")
logging.info(f"Hidden Dimension = {args.hidden_dimension}")
logging.info(f"Number of transformer layers = {args.number_of_transformer_layers}")
logging.info(f"Precision Type = {args.precision}")
logging.info(f"SP Value = {SP}")
logging.info(f"TP Degree = {TP}")
logging.info(f"ULSS Value = {ULSS}")
logging.info("==== List of Arguments ====")
logging.info("Input mean before operations = {input_mean:4f}".format(input_mean=input.mean()))
logging.info("Result mean after all  operations = {output_mean:4f}".format(output_mean=interim4.mean()))
logging.info(f"Shape of the (Q,K,V) atten. matrix = {attn_W_QKV.shape}")
logging.info(f"Shape of the WO atten. matrix = {attn_WO.shape}")
logging.info(f"Shape of the Weight matrix (H --> 4H)= {mat_h_4h.shape}")
logging.info(f"Shape of the Weight matrix (4H --> H)= {mat_4h_h.shape}")
logging.info(f"Interim 2 Size = {(ulss_interim2 if ULSS else interim2).shape}")
logging.info(f"Interim 4 Size = {(ulss_interim4 if ULSS else interim4).shape}")
logging.info(f"Parameters (per rank) = {number_of_total_parameters / 1e9} Billions")
logging.info(f"N_iter_grad_sync = {n_iter_grad_sync}")
logging.info(f"Allgather buffer size = {(args.sequence_length * args.hidden_dimension * data_type_multiplier) / 8 / 1e6} MB")
logging.info(f"Grad Sync Allreduce bucket size = {(highest_bucket_size * data_type_multiplier) / 8 / 1e6} MB")
B = data_type_multiplier // 8
DP = world_size // TP
ar = lambda n: 2 * (n - 1) / n
ag = lambda n: (n - 1) / n  # allgather, reduce-scatter, all2all
logging.info("\n==== Bandwidth per call ====")
log_bw("DP grad allreduce, per bucket", highest_bucket_size * B, DP, ar(DP), T_grad_sync_individual)
if ULSS:
    for k in (1, 2):
        log_bw(f"ULSS all2all {k}", S * (H // TP) * B, TP, ag(TP), T_dict_individual[f"T_all2all_{k}"])
elif SP:
    for op in ("allgather_1", "reduce_scatter_1", "allgather_2", "reduce_scatter_2"):
        log_bw(f"SP {op}", S * H * B, TP, ag(TP), T_dict_individual[f"T_{op}"])
else:
    for k in (1, 2):
        log_bw(f"TP allreduce {k}", S * H * B, TP, ar(TP), T_dict_individual[f"T_allreduce_{k}"])

TIMING_LABELS = [
    ("First All2All for ULSS", "T_all2all_1"),
    ("Second All2All for ULSS", "T_all2all_2"),
    ("ULSS Attention Matrix W_QKV multiplication", "T_ulss_QKV"),
    ("ULSS Attention Matrix WO multiplication", "T_ulss_WO"),
    ("ULSS H --> 4H Matrix multiplication", "T_ulss_H_4H"),
    ("ULSS 4H --> H Matrix multiplication", "T_ulss_4H_H"),
    ("First Allgather for SP", "T_allgather_1"),
    ("Column Parallel Attention Matrix W_QKV multiplication", "T_QKV"),
    ("Row Parallel Attention Matrix WO multiplication", "T_WO"),
    ("First Reduce Scatter for SP", "T_reduce_scatter_1"),
    ("First Allreduce for TP", "T_allreduce_1"),
    ("Second Allgather for SP", "T_allgather_2"),
    ("H --> 4H Matrix multiplication", "T_H_4H"),
    ("4H --> H Matrix multiplication", "T_4H_H"),
    ("Second Reduce Scatter for SP", "T_reduce_scatter_2"),
    ("Second Allreduce for TP", "T_allreduce_2"),
]
bucket_mb = highest_bucket_size * B / 1e6

logging.info("\n==== Timings per transformer layer ====")
for text, key in TIMING_LABELS:
    format_logging_timings(text, T_dict_individual[key], 1e6)
format_logging_timings(f"Grad Sync Allreduce per bucket (one {bucket_mb:.0f} MB allreduce)", T_grad_sync_individual, 1e6)
logging.info("\n==== Total Times for all transformer layers ====")
for text, key in TIMING_LABELS + [(f"Grad Sync Allreduce per timing loop ({n_iter_grad_sync} buckets)", "T_grad_sync")]:
    format_logging_timings(text, T_dict_total[key])

ALL_RESULTS = [
    ("First All2All total times from timing loop", "T_all2all_1"),
    ("Second All2All total times from timing loop", "T_all2all_2"),
    ("ULSS Attention W_QKV matrix multiplication total times from timing loop", "T_ulss_QKV"),
    ("ULSS Attention WO matrix multiplication total times from timing loop", "T_ulss_WO"),
    ("ULSS Weight matrix (H --> 4H) multiplication total times from timing loop", "T_ulss_H_4H"),
    ("ULSS Weight matrix (4H --> H) multiplication total times from timing loop", "T_ulss_4H_H"),
    ("First allgather total times from timing loop", "T_allgather_1"),
    ("First reduce scatter total times from timing loop", "T_reduce_scatter_1"),
    ("First allreduce total times from timing loop", "T_allreduce_1"),
    ("Attention W_QKV matrix multiplication total times from timing loop", "T_QKV"),
    ("Attention WO matrix multiplication total times from timing loop", "T_WO"),
    ("Weight matrix (H --> 4H) multiplication total times from timing loop", "T_H_4H"),
    ("Weight matrix (4H --> H) multiplication total times from timing loop", "T_4H_H"),
    ("Second allgather total times from timing loop", "T_allgather_2"),
    ("Second reduce scatter total times from timing loop", "T_reduce_scatter_2"),
    ("Second allreduce total times from timing loop", "T_allreduce_2"),
    ("Grad Sync Total times from timing loop", "T_grad_sync"),
    ("TP Sync times from timing loop", "T_tp_sync"),
    ("ULSS Sync times from timing loop", "T_ulss_sync"),
    ("DP Sync Barrier at the beginning times from timing loop", "T_dp_sync_begin"),
    ("DP Sync Barrier at the end times from timing loop", "T_dp_sync_end"),
]
logging.info("\n==== ALL RESULTS ====")
for text, key in ALL_RESULTS:
    logging.info(f"{text} = {T_dict_total[key]} ms")
logging.info(f"Timing loop times = {T_dict_total['T_timing_loop']}")
logging.info(f"Total time taken for {args.iterations} timing loops = {sum(T_dict_total['T_timing_loop'])} ms")

logging.info("\n==== TFLOPS per transformer layer ====")
labels = {key: text for text, key in TIMING_LABELS}
prefix = "T_ulss_" if ULSS else "T_"
matmul_inputs = (partial_input, ulss_interim1, ulss_interim2, ulss_interim3) if ULSS else (input, interim1, interim2, interim3)
for x, w, op in zip(matmul_inputs, (attn_W_QKV, attn_WO, mat_h_4h, mat_4h_h), ("QKV", "WO", "H_4H", "4H_H")):
    format_logging_flops(labels[prefix + op], x.shape, w.t().shape, n_layers, T_dict_total[prefix + op])
logging.info(f"==== Finished Running ====")

if args.save:
    result_dir = os.path.join(args.log_directory, "results")
    print(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    np.save(os.path.join(result_dir, args.log_file),
            {"T_dict_individual": T_dict_individual,
             "T_dict_total": T_dict_total,
             "T_grad_sync_individual": T_grad_sync_individual})

dist.destroy_process_group()
