#!/usr/bin/env python3
"""
training_config.py - Training memory calculator and configuration planner

Computes optimal training configuration given:
- Model architecture (from config)
- Available memory (auto-detected or specified)
- Target batch size / context length constraints

Outputs:
- Optimal batch size, micro-batch size, accumulation steps
- Memory breakdown (weights, activations, gradients, optimizer state)
- Parallelism strategy recommendations
"""

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Bytes per dtype
DTYPE_BYTES = {
    "f32": 4, "fp32": 4, "float32": 4,
    "f16": 2, "fp16": 2, "float16": 2,
    "bf16": 2, "bfloat16": 2,
    "i32": 4, "int32": 4,
    "i16": 2, "int16": 2,
    "i8": 1, "int8": 1,
}


def get_system_memory() -> Dict[str, int]:
    """Detect available system memory (RAM and GPU VRAM if available)."""
    result = {"ram_bytes": 0, "gpu_vram_bytes": 0, "gpu_count": 0}

    # Get RAM
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    # MemTotal is in kB
                    kb = int(line.split()[1])
                    result["ram_bytes"] = kb * 1024
                    break
    except:
        # Fallback: assume 16GB
        result["ram_bytes"] = 16 * 1024**3

    # Try to detect NVIDIA GPU memory
    try:
        import subprocess
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        gpus = output.split("\n")
        result["gpu_count"] = len(gpus)
        # Total VRAM across all GPUs (in MB from nvidia-smi)
        result["gpu_vram_bytes"] = sum(int(g) * 1024**2 for g in gpus if g.strip())
    except:
        pass

    return result


@dataclass
class MemoryBreakdown:
    """Memory breakdown for training."""
    # Per-component memory in bytes
    weights: int = 0              # Model weights
    weight_gradients: int = 0     # Same size as weights
    optimizer_state: int = 0      # Adam: 2x weights, SGD: 1x weights
    activations: int = 0          # Forward pass activations (per micro-batch)
    activation_gradients: int = 0 # Backward pass gradients (per micro-batch)
    scratch: int = 0              # Temporary buffers

    # Configuration
    batch_size: int = 1
    micro_batch_size: int = 1
    accumulation_steps: int = 1
    context_length: int = 512
    optimizer: str = "adamw"

    @property
    def total(self) -> int:
        return (self.weights + self.weight_gradients + self.optimizer_state +
                self.activations + self.activation_gradients + self.scratch)

    @property
    def peak_training(self) -> int:
        """Peak memory during training (forward + backward active)."""
        return (self.weights + self.weight_gradients + self.optimizer_state +
                self.activations + self.activation_gradients + self.scratch)

    def to_dict(self) -> Dict:
        return {
            "total_bytes": self.total,
            "total_gb": round(self.total / 1024**3, 2),
            "peak_training_bytes": self.peak_training,
            "peak_training_gb": round(self.peak_training / 1024**3, 2),
            "breakdown": {
                "weights_bytes": self.weights,
                "weights_gb": round(self.weights / 1024**3, 2),
                "weight_gradients_bytes": self.weight_gradients,
                "weight_gradients_gb": round(self.weight_gradients / 1024**3, 2),
                "optimizer_state_bytes": self.optimizer_state,
                "optimizer_state_gb": round(self.optimizer_state / 1024**3, 2),
                "activations_bytes": self.activations,
                "activations_gb": round(self.activations / 1024**3, 2),
                "activation_gradients_bytes": self.activation_gradients,
                "activation_gradients_gb": round(self.activation_gradients / 1024**3, 2),
                "scratch_bytes": self.scratch,
                "scratch_gb": round(self.scratch / 1024**3, 2),
            },
            "config": {
                "batch_size": self.batch_size,
                "micro_batch_size": self.micro_batch_size,
                "accumulation_steps": self.accumulation_steps,
                "context_length": self.context_length,
                "optimizer": self.optimizer,
            },
        }


@dataclass
class TrainingConfig:
    """Complete training configuration."""
    # Model dimensions
    embed_dim: int = 0
    num_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    intermediate_dim: int = 0
    num_layers: int = 0
    vocab_size: int = 0
    max_seq_len: int = 0
    dtype: str = "bf16"

    # Training parameters
    batch_size: int = 1
    micro_batch_size: int = 1
    accumulation_steps: int = 1
    context_length: int = 512
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    # Parallelism
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1

    # Memory
    available_memory_bytes: int = 0
    memory_breakdown: Optional[MemoryBreakdown] = None

    @classmethod
    def from_model_config(cls, config: Dict) -> "TrainingConfig":
        return cls(
            embed_dim=config.get("embed_dim", config.get("hidden_size", 0)),
            num_heads=config.get("num_heads", config.get("num_attention_heads", 0)),
            num_kv_heads=config.get("num_kv_heads", config.get("num_key_value_heads", 0)),
            head_dim=config.get("head_dim", 0),
            intermediate_dim=config.get("intermediate_dim", config.get("intermediate_size", 0)),
            num_layers=config.get("num_layers", config.get("num_hidden_layers", 0)),
            vocab_size=config.get("vocab_size", 0),
            max_seq_len=config.get("max_seq_len", config.get("max_position_embeddings", 2048)),
            dtype=config.get("dtype", "bf16"),
        )


def calculate_weight_memory(config: TrainingConfig) -> int:
    """Calculate total weight memory in bytes."""
    E = config.embed_dim
    H = config.num_heads
    KV = config.num_kv_heads
    D = config.head_dim
    I = config.intermediate_dim
    L = config.num_layers
    V = config.vocab_size
    elem_bytes = DTYPE_BYTES.get(config.dtype, 2)

    # Per-layer weights
    per_layer = 0
    per_layer += E  # ln1_gamma (RMSNorm)
    per_layer += H * D * E  # wq
    per_layer += KV * D * E  # wk
    per_layer += KV * D * E  # wv
    per_layer += H * E * D  # wo (output projection)
    per_layer += E  # ln2_gamma (RMSNorm)
    per_layer += 2 * I * E  # w1 (gate + up, concatenated)
    per_layer += E * I  # w2 (down projection)

    total_layer_weights = per_layer * L

    # Header/footer weights
    header_footer = 0
    header_footer += V * E  # token_emb
    header_footer += E  # final_ln_weight
    # lm_head is often tied to token_emb, so don't double count
    # If not tied: header_footer += V * E

    total_elements = total_layer_weights + header_footer
    return total_elements * elem_bytes


def calculate_activation_memory(config: TrainingConfig, batch_size: int, seq_len: int) -> int:
    """Calculate activation memory for forward pass (per micro-batch)."""
    B = batch_size
    S = seq_len
    E = config.embed_dim
    H = config.num_heads
    KV = config.num_kv_heads
    D = config.head_dim
    I = config.intermediate_dim
    L = config.num_layers
    V = config.vocab_size
    elem_bytes = DTYPE_BYTES.get(config.dtype, 2)

    # Per-layer activations (need to store for backward)
    per_layer = 0
    per_layer += B * S * E  # input
    per_layer += B * S * E  # ln1_out
    per_layer += B * S  # ln1_rstd (for backward)
    per_layer += B * H * S * D  # q
    per_layer += B * KV * S * D  # k (or KV cache size)
    per_layer += B * KV * S * D  # v
    per_layer += B * H * S * S  # attention scores (for prefill)
    per_layer += B * H * S * D  # attn_out
    per_layer += B * S * E  # proj_tmp
    per_layer += B * S * E  # residual1
    per_layer += B * S * E  # ln2_out
    per_layer += B * S  # ln2_rstd (for backward)
    per_layer += B * S * 2 * I  # fc1_out (gate + up)
    per_layer += B * S * I  # swiglu_out
    per_layer += B * S * E  # mlp_out
    per_layer += B * S * E  # output (residual2)

    total_layer_activations = per_layer * L

    # Header/footer activations
    header_footer = 0
    header_footer += B * S * E  # embedded_input
    header_footer += B * S * E  # final_output
    header_footer += B * S  # final_ln_rstd
    header_footer += B * S * V  # logits (can be large!)

    total_elements = total_layer_activations + header_footer
    return total_elements * elem_bytes


def calculate_gradient_memory(config: TrainingConfig, batch_size: int, seq_len: int) -> int:
    """Calculate gradient memory (same as activations for backward)."""
    # Activation gradients have same shape as activations
    return calculate_activation_memory(config, batch_size, seq_len)


def calculate_optimizer_memory(config: TrainingConfig, optimizer: str = "adamw") -> int:
    """Calculate optimizer state memory."""
    weight_bytes = calculate_weight_memory(config)

    if optimizer in ("adam", "adamw"):
        # Adam stores m (momentum) and v (variance) - 2x weight size
        # Usually stored in fp32 even if weights are bf16
        fp32_bytes = weight_bytes * 4 // DTYPE_BYTES.get(config.dtype, 2)
        return fp32_bytes * 2  # m + v
    elif optimizer in ("sgd", "sgd_momentum"):
        # SGD with momentum stores 1x weight size
        fp32_bytes = weight_bytes * 4 // DTYPE_BYTES.get(config.dtype, 2)
        return fp32_bytes
    else:
        # Plain SGD - no state
        return 0


def calculate_scratch_memory(config: TrainingConfig, batch_size: int, seq_len: int) -> int:
    """Calculate scratch/temporary buffer memory."""
    B = batch_size
    S = seq_len
    E = config.embed_dim
    H = config.num_heads
    elem_bytes = DTYPE_BYTES.get(config.dtype, 2)

    # Scratch buffers (rough estimate)
    scratch = 0
    scratch += B * S * E  # projection scratch
    scratch += B * H * S * S  # attention scratch (softmax temps)
    scratch += B * S * E * 2  # misc temps

    return scratch * elem_bytes


def calculate_memory_breakdown(
    config: TrainingConfig,
    batch_size: int,
    context_length: int,
    optimizer: str = "adamw",
    micro_batch_size: Optional[int] = None,
) -> MemoryBreakdown:
    """Calculate complete memory breakdown for training."""
    if micro_batch_size is None:
        micro_batch_size = batch_size

    accumulation_steps = batch_size // micro_batch_size

    breakdown = MemoryBreakdown(
        weights=calculate_weight_memory(config),
        weight_gradients=calculate_weight_memory(config),  # Same size as weights
        optimizer_state=calculate_optimizer_memory(config, optimizer),
        activations=calculate_activation_memory(config, micro_batch_size, context_length),
        activation_gradients=calculate_gradient_memory(config, micro_batch_size, context_length),
        scratch=calculate_scratch_memory(config, micro_batch_size, context_length),
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        accumulation_steps=accumulation_steps,
        context_length=context_length,
        optimizer=optimizer,
    )

    return breakdown


def find_optimal_config(
    config: TrainingConfig,
    available_memory_bytes: int,
    target_batch_size: Optional[int] = None,
    target_context_length: Optional[int] = None,
    optimizer: str = "adamw",
    memory_fraction: float = 0.9,  # Use 90% of available memory
) -> Tuple[MemoryBreakdown, Dict]:
    """
    Find optimal training configuration given memory constraints.

    Works backward from available memory to find largest feasible config.
    """
    usable_memory = int(available_memory_bytes * memory_fraction)

    # Fixed costs (weights, gradients, optimizer state)
    fixed_cost = (
        calculate_weight_memory(config) +  # weights
        calculate_weight_memory(config) +  # weight gradients
        calculate_optimizer_memory(config, optimizer)  # optimizer state
    )

    # Memory available for activations
    activation_budget = usable_memory - fixed_cost

    if activation_budget <= 0:
        return None, {"error": "Model weights + optimizer don't fit in memory",
                      "fixed_cost_gb": fixed_cost / 1024**3,
                      "available_gb": usable_memory / 1024**3}

    # Binary search for optimal batch_size and context_length
    best_config = None
    best_throughput = 0  # batch_size * context_length

    max_context = target_context_length or config.max_seq_len
    max_batch = target_batch_size or 64  # Reasonable upper bound

    # Try different configurations
    for context_length in [128, 256, 512, 1024, 2048, 4096, min(8192, max_context)]:
        if context_length > max_context:
            continue

        for micro_batch in [1, 2, 4, 8, 16, 32]:
            if target_batch_size and micro_batch > target_batch_size:
                continue

            activation_mem = calculate_activation_memory(config, micro_batch, context_length)
            gradient_mem = calculate_gradient_memory(config, micro_batch, context_length)
            scratch_mem = calculate_scratch_memory(config, micro_batch, context_length)

            total_activation_cost = activation_mem + gradient_mem + scratch_mem

            if total_activation_cost <= activation_budget:
                throughput = micro_batch * context_length
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_config = {
                        "micro_batch_size": micro_batch,
                        "context_length": context_length,
                        "activation_memory": total_activation_cost,
                    }

    if best_config is None:
        return None, {"error": "Cannot fit even minimal config (batch=1, context=128)",
                      "fixed_cost_gb": fixed_cost / 1024**3,
                      "activation_budget_gb": activation_budget / 1024**3}

    # Determine accumulation steps for target batch size
    if target_batch_size:
        accumulation_steps = max(1, target_batch_size // best_config["micro_batch_size"])
        effective_batch = best_config["micro_batch_size"] * accumulation_steps
    else:
        accumulation_steps = 1
        effective_batch = best_config["micro_batch_size"]

    breakdown = calculate_memory_breakdown(
        config,
        batch_size=effective_batch,
        context_length=best_config["context_length"],
        optimizer=optimizer,
        micro_batch_size=best_config["micro_batch_size"],
    )

    recommendations = {
        "micro_batch_size": best_config["micro_batch_size"],
        "accumulation_steps": accumulation_steps,
        "effective_batch_size": effective_batch,
        "context_length": best_config["context_length"],
        "tokens_per_step": effective_batch * best_config["context_length"],
        "memory_utilization": breakdown.total / usable_memory,
        "headroom_gb": (usable_memory - breakdown.total) / 1024**3,
    }

    return breakdown, recommendations


def compute_reduction_strategy(
    config: TrainingConfig,
    data_parallel_size: int = 1,
    tensor_parallel_size: int = 1,
) -> Dict:
    """
    Compute gradient reduction strategy based on parallelism.

    Returns ops needed for gradient synchronization.
    """
    reduction_ops = []

    if data_parallel_size > 1:
        # AllReduce gradients across data parallel workers
        reduction_ops.append({
            "op": "allreduce",
            "name": "gradient_allreduce",
            "description": f"AllReduce gradients across {data_parallel_size} data parallel workers",
            "inputs": ["all_weight_gradients"],
            "outputs": ["all_weight_gradients"],
            "attrs": {
                "reduce_op": "sum",
                "scale": 1.0 / data_parallel_size,  # Average gradients
                "group_size": data_parallel_size,
            },
        })

    if tensor_parallel_size > 1:
        # ReduceScatter for tensor parallel (column parallel layers)
        reduction_ops.append({
            "op": "reducescatter",
            "name": "tp_reducescatter",
            "description": f"ReduceScatter for tensor parallel ({tensor_parallel_size} ways)",
            "inputs": ["partial_gradients"],
            "outputs": ["local_gradients"],
            "attrs": {
                "reduce_op": "sum",
                "group_size": tensor_parallel_size,
            },
        })

    # Local reduction (within single device, across sequence dimension)
    reduction_ops.append({
        "op": "local_reduce",
        "name": "sequence_gradient_reduce",
        "description": "Sum gradients across sequence dimension (local)",
        "inputs": ["d_weight_partials"],
        "outputs": ["d_weight"],
        "attrs": {
            "reduce_op": "sum",
            "reduce_dims": [0],  # Sequence dimension
        },
    })

    return {
        "data_parallel_size": data_parallel_size,
        "tensor_parallel_size": tensor_parallel_size,
        "reduction_ops": reduction_ops,
        "communication_pattern": (
            "ring_allreduce" if data_parallel_size > 1 else
            "local_only"
        ),
    }


def emit_training_config(
    config: TrainingConfig,
    breakdown: MemoryBreakdown,
    recommendations: Dict,
    reduction_strategy: Dict,
    output_path: str,
) -> None:
    """Emit complete training configuration JSON."""
    from datetime import datetime

    output = {
        "version": 1,
        "kind": "training_config",
        "generated": datetime.utcnow().isoformat() + "Z",
        "model": {
            "embed_dim": config.embed_dim,
            "num_heads": config.num_heads,
            "num_kv_heads": config.num_kv_heads,
            "head_dim": config.head_dim,
            "intermediate_dim": config.intermediate_dim,
            "num_layers": config.num_layers,
            "vocab_size": config.vocab_size,
            "max_seq_len": config.max_seq_len,
            "dtype": config.dtype,
        },
        "training": {
            "batch_size": breakdown.batch_size,
            "micro_batch_size": breakdown.micro_batch_size,
            "accumulation_steps": breakdown.accumulation_steps,
            "context_length": breakdown.context_length,
            "optimizer": breakdown.optimizer,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "adam_beta1": config.beta1,
            "adam_beta2": config.beta2,
            "adam_epsilon": config.epsilon,
        },
        "memory": breakdown.to_dict(),
        "recommendations": recommendations,
        "parallelism": {
            "data_parallel_size": config.data_parallel_size,
            "tensor_parallel_size": config.tensor_parallel_size,
            "pipeline_parallel_size": config.pipeline_parallel_size,
        },
        "reduction": reduction_strategy,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[TRAINING] Written: {output_path}")


def print_memory_summary(breakdown: MemoryBreakdown, recommendations: Dict) -> None:
    """Print human-readable memory summary."""
    print("\n" + "=" * 60)
    print("TRAINING MEMORY ANALYSIS")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  Micro-batch size:    {breakdown.micro_batch_size}")
    print(f"  Accumulation steps:  {breakdown.accumulation_steps}")
    print(f"  Effective batch:     {breakdown.batch_size}")
    print(f"  Context length:      {breakdown.context_length}")
    print(f"  Optimizer:           {breakdown.optimizer}")

    print(f"\nMemory Breakdown:")
    d = breakdown.to_dict()["breakdown"]
    print(f"  Weights:             {d['weights_gb']:.2f} GB")
    print(f"  Weight gradients:    {d['weight_gradients_gb']:.2f} GB")
    print(f"  Optimizer state:     {d['optimizer_state_gb']:.2f} GB")
    print(f"  Activations:         {d['activations_gb']:.2f} GB")
    print(f"  Activation grads:    {d['activation_gradients_gb']:.2f} GB")
    print(f"  Scratch:             {d['scratch_gb']:.2f} GB")
    print(f"  ─────────────────────────────")
    print(f"  TOTAL:               {breakdown.to_dict()['total_gb']:.2f} GB")

    if recommendations:
        print(f"\nRecommendations:")
        print(f"  Tokens per step:     {recommendations.get('tokens_per_step', 'N/A')}")
        print(f"  Memory utilization:  {recommendations.get('memory_utilization', 0)*100:.1f}%")
        print(f"  Headroom:            {recommendations.get('headroom_gb', 0):.2f} GB")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Cluster Optimization Landscape
# ---------------------------------------------------------------------------

@dataclass
class ClusterConfig:
    """Hardware configuration for a CPU cluster."""
    num_nodes: int = 1
    cores_per_node: int = 1
    memory_per_node_gb: float = 0
    numa_domains_per_node: int = 1
    l3_cache_mb: float = 0
    memory_bandwidth_gbps: float = 0
    network_bandwidth_gbps: float = 0  # Inter-node
    network_latency_us: float = 0
    has_amx: bool = False

    @classmethod
    def detect(cls) -> "ClusterConfig":
        """Auto-detect single node configuration."""
        import os

        cores = os.cpu_count() or 1
        mem = get_system_memory()

        # Try to detect NUMA domains
        numa_domains = 1
        try:
            with open("/sys/devices/system/node/online", "r") as f:
                numa_str = f.read().strip()
                if "-" in numa_str:
                    start, end = numa_str.split("-")
                    numa_domains = int(end) - int(start) + 1
                else:
                    numa_domains = len(numa_str.split(","))
        except:
            pass

        # Try to detect AMX
        has_amx = False
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
                has_amx = "amx" in cpuinfo.lower()
        except:
            pass

        return cls(
            num_nodes=1,
            cores_per_node=cores,
            memory_per_node_gb=mem["ram_bytes"] / 1024**3,
            numa_domains_per_node=numa_domains,
            has_amx=has_amx,
        )


def get_optimization_landscape() -> Dict:
    """
    Define the optimization landscape for CPU cluster training.

    Returns a structure describing:
    - All tunable knobs (parameters)
    - What to measure (benchmark metrics)
    - Optimization strategies
    - Mathematical relationships
    """
    return {
        "description": "CPU Cluster Training Optimization Landscape",

        # ===== TUNABLE KNOBS =====
        "knobs": {
            # Memory-bound knobs
            "memory": {
                "context_length": {
                    "type": "int",
                    "range": [64, 32768],
                    "description": "Tokens per sample (S)",
                    "affects": ["activation_memory", "attention_compute", "memory_bandwidth"],
                    "tradeoff": "Higher = more tokens/sample but quadratic attention cost",
                },
                "accumulation_steps": {
                    "type": "int",
                    "range": [1, 256],
                    "description": "Samples before weight update (effective batch)",
                    "affects": ["gradient_noise", "convergence", "memory_reuse"],
                    "tradeoff": "Higher = smoother gradients but delayed updates",
                },
                "activation_checkpointing": {
                    "type": "bool",
                    "description": "Recompute activations in backward instead of storing",
                    "affects": ["activation_memory", "compute_time"],
                    "tradeoff": "Saves memory but doubles forward compute",
                },
            },

            # Compute-bound knobs
            "compute": {
                "num_threads": {
                    "type": "int",
                    "range": [1, "cores_per_node"],
                    "description": "Threads per forward/backward pass",
                    "affects": ["compute_throughput", "memory_bandwidth_pressure"],
                    "tradeoff": "More threads = faster but memory bandwidth limited",
                },
                "tile_size_m": {
                    "type": "int",
                    "values": [16, 32, 64],
                    "description": "AMX tile M dimension",
                    "affects": ["amx_efficiency", "cache_utilization"],
                },
                "tile_size_n": {
                    "type": "int",
                    "values": [16, 32, 64],
                    "description": "AMX tile N dimension",
                    "affects": ["amx_efficiency", "cache_utilization"],
                },
                "block_size_k": {
                    "type": "int",
                    "values": [64, 128, 256, 512],
                    "description": "GEMM K-dimension blocking",
                    "affects": ["cache_hits", "compute_efficiency"],
                },
            },

            # Parallelism knobs
            "parallelism": {
                "data_parallel_size": {
                    "type": "int",
                    "range": [1, "num_nodes"],
                    "description": "Replicate model, split data across nodes",
                    "affects": ["communication_volume", "scaling_efficiency"],
                    "tradeoff": "Scales well but AllReduce overhead at large scale",
                },
                "tensor_parallel_size": {
                    "type": "int",
                    "values": [1, 2, 4, 8],
                    "description": "Split individual tensors across cores/nodes",
                    "affects": ["memory_per_worker", "communication_frequency"],
                    "tradeoff": "Reduces memory but frequent point-to-point comms",
                },
                "pipeline_parallel_size": {
                    "type": "int",
                    "range": [1, "num_layers"],
                    "description": "Split layers across nodes",
                    "affects": ["memory_per_node", "pipeline_bubbles"],
                    "tradeoff": "Reduces memory but pipeline bubbles reduce efficiency",
                },
                "numa_binding": {
                    "type": "enum",
                    "values": ["none", "node", "core"],
                    "description": "NUMA memory affinity strategy",
                    "affects": ["memory_latency", "bandwidth"],
                },
            },

            # Communication knobs
            "communication": {
                "allreduce_algorithm": {
                    "type": "enum",
                    "values": ["ring", "tree", "recursive_halving", "bucket"],
                    "description": "AllReduce implementation",
                    "affects": ["allreduce_time", "bandwidth_utilization"],
                },
                "gradient_compression": {
                    "type": "enum",
                    "values": ["none", "fp16", "int8", "topk"],
                    "description": "Gradient compression for communication",
                    "affects": ["communication_volume", "gradient_accuracy"],
                },
                "overlap_compute_comm": {
                    "type": "bool",
                    "description": "Overlap backward compute with gradient communication",
                    "affects": ["total_step_time", "memory_for_double_buffering"],
                },
            },
        },

        # ===== BENCHMARK METRICS =====
        "metrics": {
            "throughput": {
                "tokens_per_second": {
                    "unit": "tokens/s",
                    "description": "Training throughput",
                    "formula": "(batch_size * context_length) / step_time",
                },
                "samples_per_second": {
                    "unit": "samples/s",
                    "description": "Samples processed per second",
                    "formula": "batch_size / step_time",
                },
                "tflops": {
                    "unit": "TFLOPS",
                    "description": "Compute throughput",
                    "formula": "total_flops / step_time / 1e12",
                },
            },
            "timing": {
                "forward_time_ms": {"unit": "ms", "description": "Forward pass time"},
                "backward_time_ms": {"unit": "ms", "description": "Backward pass time"},
                "allreduce_time_ms": {"unit": "ms", "description": "Gradient sync time"},
                "optimizer_time_ms": {"unit": "ms", "description": "Weight update time"},
                "step_time_ms": {"unit": "ms", "description": "Total step time"},
            },
            "efficiency": {
                "compute_utilization": {
                    "unit": "%",
                    "description": "AMX/compute unit utilization",
                    "target": "> 80%",
                },
                "memory_bandwidth_utilization": {
                    "unit": "%",
                    "description": "Memory bandwidth utilization",
                    "target": "< 90% (headroom for variance)",
                },
                "scaling_efficiency": {
                    "unit": "%",
                    "description": "Multi-node scaling efficiency",
                    "formula": "throughput_N_nodes / (N * throughput_1_node)",
                    "target": "> 85%",
                },
                "communication_overhead": {
                    "unit": "%",
                    "description": "Time spent in communication",
                    "formula": "allreduce_time / step_time",
                    "target": "< 15%",
                },
            },
            "memory": {
                "peak_memory_gb": {"unit": "GB", "description": "Peak memory usage"},
                "memory_efficiency": {
                    "unit": "%",
                    "description": "Useful memory / allocated memory",
                },
            },
        },

        # ===== OPTIMIZATION STRATEGY =====
        "optimization_strategy": {
            "step_1_single_node": {
                "description": "Optimize single node first",
                "actions": [
                    "Measure memory bandwidth: stream benchmark",
                    "Measure AMX peak: gemm microbenchmark",
                    "Find optimal tile/block sizes for model dimensions",
                    "Find optimal num_threads (usually cores / 2 for HT)",
                    "Determine max context_length that fits in memory",
                ],
            },
            "step_2_multi_node_weak": {
                "description": "Weak scaling test (increase data with nodes)",
                "actions": [
                    "Fix context_length, increase accumulation_steps with nodes",
                    "Measure AllReduce time vs data_parallel_size",
                    "Identify communication bottleneck point",
                    "Choose allreduce_algorithm based on cluster topology",
                ],
            },
            "step_3_multi_node_strong": {
                "description": "Strong scaling test (fixed total work)",
                "actions": [
                    "Fix total batch, distribute across nodes",
                    "Measure scaling efficiency",
                    "Consider tensor_parallel if memory-bound",
                    "Consider pipeline_parallel if compute-bound",
                ],
            },
            "step_4_overlap": {
                "description": "Optimize communication overlap",
                "actions": [
                    "Enable overlap_compute_comm",
                    "Measure effective hiding of AllReduce",
                    "Consider gradient_compression if bandwidth-limited",
                ],
            },
        },

        # ===== MATHEMATICAL RELATIONSHIPS =====
        "formulas": {
            "memory_per_node": {
                "formula": "weights + weight_grads + optimizer_state + activations + activation_grads + scratch",
                "notes": "Activation memory scales with context_length^2 (attention)",
            },
            "compute_per_step": {
                "formula": "6 * num_params * batch_size * context_length",
                "notes": "Forward ~= 2x params, backward ~= 4x params",
            },
            "allreduce_time": {
                "formula": "2 * (N-1)/N * gradient_bytes / bandwidth + latency * log2(N)",
                "notes": "Ring AllReduce with N nodes",
            },
            "optimal_batch_size": {
                "formula": "memory_available - fixed_costs / activation_memory_per_sample",
                "notes": "Work backward from memory constraint",
            },
            "scaling_efficiency": {
                "formula": "1 / (1 + communication_time / compute_time)",
                "notes": "Amdahl's law for distributed training",
            },
        },
    }


def compute_cluster_optimal_config(
    model_config: TrainingConfig,
    cluster: ClusterConfig,
    target_batch_size: Optional[int] = None,
) -> Dict:
    """
    Compute optimal training configuration for a cluster.

    This is a deterministic optimization problem:
    - Given: model size, cluster hardware, target batch size
    - Find: optimal parallelism, context length, memory allocation
    """
    # Total cluster memory
    total_memory_gb = cluster.num_nodes * cluster.memory_per_node_gb

    # Weight memory (same on all nodes for data parallel)
    weight_memory = calculate_weight_memory(model_config)
    weight_memory_gb = weight_memory / 1024**3

    # Optimizer state (always on each node)
    optimizer_memory = calculate_optimizer_memory(model_config)
    optimizer_memory_gb = optimizer_memory / 1024**3

    # Fixed cost per node
    fixed_per_node = weight_memory_gb + weight_memory_gb + optimizer_memory_gb  # weights + grads + opt

    # Memory available for activations per node
    usable_per_node = cluster.memory_per_node_gb * 0.85  # 85% usable
    activation_budget_per_node = usable_per_node - fixed_per_node

    # Find max context length that fits
    # Activation memory ~ O(context_length^2) due to attention
    # Binary search for optimal context length
    def activation_fits(ctx_len: int) -> bool:
        act_mem = calculate_activation_memory(model_config, 1, ctx_len)
        grad_mem = calculate_gradient_memory(model_config, 1, ctx_len)
        total = (act_mem + grad_mem) / 1024**3
        return total <= activation_budget_per_node

    # Binary search
    low, high = 64, model_config.max_seq_len
    optimal_ctx = low
    while low <= high:
        mid = (low + high) // 2
        if activation_fits(mid):
            optimal_ctx = mid
            low = mid + 1
        else:
            high = mid - 1

    # Compute optimal parallelism
    if target_batch_size is None:
        # Default: one sample per node per accumulation step
        target_batch_size = cluster.num_nodes * 8

    # Data parallel across nodes is most efficient for CPU clusters
    data_parallel = cluster.num_nodes
    accumulation_steps = max(1, target_batch_size // data_parallel)

    # Estimate timing (rough)
    flops_per_sample = 6 * model_config.num_layers * (
        model_config.embed_dim ** 2 * 12 +  # attention
        model_config.embed_dim * model_config.intermediate_dim * 8  # mlp
    ) * optimal_ctx

    # Rough AMX throughput estimate (bf16)
    amx_tflops = 1.0 if cluster.has_amx else 0.1  # Very rough
    compute_time_per_sample = flops_per_sample / (amx_tflops * 1e12)

    return {
        "cluster": {
            "num_nodes": cluster.num_nodes,
            "cores_per_node": cluster.cores_per_node,
            "memory_per_node_gb": cluster.memory_per_node_gb,
            "total_memory_gb": total_memory_gb,
        },
        "optimal_config": {
            "data_parallel_size": data_parallel,
            "tensor_parallel_size": 1,  # CPU: usually no TP needed
            "pipeline_parallel_size": 1,  # CPU: usually no PP needed
            "accumulation_steps": accumulation_steps,
            "effective_batch_size": data_parallel * accumulation_steps,
            "context_length": optimal_ctx,
            "tokens_per_step": data_parallel * accumulation_steps * optimal_ctx,
        },
        "memory_breakdown_per_node": {
            "weights_gb": weight_memory_gb,
            "weight_grads_gb": weight_memory_gb,
            "optimizer_gb": optimizer_memory_gb,
            "activation_budget_gb": activation_budget_per_node,
            "fixed_cost_gb": fixed_per_node,
        },
        "benchmark_required": [
            {"test": "memory_bandwidth", "tool": "stream", "expected_metric": "GB/s"},
            {"test": "amx_gemm_throughput", "tool": "bench_gemm", "expected_metric": "TFLOPS"},
            {"test": "allreduce_latency", "tool": "osu_allreduce", "expected_metric": "us"},
            {"test": "network_bandwidth", "tool": "osu_bw", "expected_metric": "GB/s"},
        ],
    }


# CLI
def main(argv: List[str]) -> int:
    if len(argv) < 1 or "--help" in argv:
        print("Usage: python training_config.py CONFIG_JSON [OPTIONS]")
        print()
        print("Options:")
        print("  --memory=GB           Available memory (auto-detect if not specified)")
        print("  --batch-size=N        Target batch size")
        print("  --context-length=N    Target context length")
        print("  --optimizer=NAME      Optimizer (adamw, sgd)")
        print("  --output=FILE         Output JSON path")
        print("  --data-parallel=N     Data parallel size")
        print("  --tensor-parallel=N   Tensor parallel size")
        return 0

    config_path = argv[0]
    with open(config_path, "r") as f:
        model_config = json.load(f)

    training_config = TrainingConfig.from_model_config(model_config)

    # Parse options
    memory_gb = None
    batch_size = None
    context_length = None
    optimizer = "adamw"
    output_path = None
    data_parallel = 1
    tensor_parallel = 1

    for arg in argv[1:]:
        if arg.startswith("--memory="):
            memory_gb = float(arg.split("=")[1])
        elif arg.startswith("--batch-size="):
            batch_size = int(arg.split("=")[1])
        elif arg.startswith("--context-length="):
            context_length = int(arg.split("=")[1])
        elif arg.startswith("--optimizer="):
            optimizer = arg.split("=")[1]
        elif arg.startswith("--output="):
            output_path = arg.split("=")[1]
        elif arg.startswith("--data-parallel="):
            data_parallel = int(arg.split("=")[1])
        elif arg.startswith("--tensor-parallel="):
            tensor_parallel = int(arg.split("=")[1])

    # Get available memory
    if memory_gb:
        available_memory = int(memory_gb * 1024**3)
    else:
        sys_mem = get_system_memory()
        available_memory = sys_mem["ram_bytes"]
        print(f"[SYSTEM] Detected RAM: {available_memory / 1024**3:.1f} GB")
        if sys_mem["gpu_vram_bytes"] > 0:
            print(f"[SYSTEM] Detected {sys_mem['gpu_count']} GPU(s) with {sys_mem['gpu_vram_bytes'] / 1024**3:.1f} GB VRAM total")

    # Find optimal config
    breakdown, recommendations = find_optimal_config(
        training_config,
        available_memory,
        target_batch_size=batch_size,
        target_context_length=context_length,
        optimizer=optimizer,
    )

    if breakdown is None:
        print(f"[ERROR] {recommendations.get('error', 'Unknown error')}")
        return 1

    # Compute reduction strategy
    training_config.data_parallel_size = data_parallel
    training_config.tensor_parallel_size = tensor_parallel
    reduction = compute_reduction_strategy(training_config, data_parallel, tensor_parallel)

    # Print summary
    print_memory_summary(breakdown, recommendations)

    # Emit config if output specified
    if output_path:
        emit_training_config(training_config, breakdown, recommendations, reduction, output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
