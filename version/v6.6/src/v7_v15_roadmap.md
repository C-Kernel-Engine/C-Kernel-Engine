# C-Kernel-Engine Long-Term Roadmap (v7-v15+)

## Vision
Build a complete, fast, open-source inference and training engine from scratch with:
- Optimized low-level kernels (SIMD, cache-aware)
- Support for multiple modalities (text, vision, audio)
- Mixture of Experts for sparse compute
- Full training capability (backpropagation)
- Distributed computing at scale

---

## v6.6: Foundation & Performance (CURRENT)
**Goal**: Fix performance regression, pass all tests, establish CI

### In Progress
- [x] Enable AVX kernels (Q5_0, Q8_0)
- [ ] Rebuild model library with fixed kernels
- [ ] Profile to verify AVX usage
- [ ] Run `make test`
- [ ] Run `make llamacpp-parity-full`
- [ ] Add flamegraph profiling to CI
- [ ] Optimize Q4_K/Q6_K kernels further

### Deliverables
- 10-20x speedup from AVX enablement
- All unit tests passing
- Parity with llama.cpp verified

---

## v7: Backpropagation & Training
**Goal**: Add automatic differentiation and training support

### Core Components
```
┌─────────────────────────────────────────────────────────────┐
│                    v7 Training Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│  Forward Pass → Store Activations → Backward Pass → Update  │
│                                                                │
│  ┌──────────┐    ┌────────────┐    ┌──────────────┐        │
│  │ Forward  │───▶│ Activation │───▶│ Loss Compute │        │
│  │  Pass    │    │  Cache     │    │              │        │
│  └──────────┘    └────────────┘    └──────┬───────┘        │
│                                           │                 │
│  ┌──────────┐    ┌────────────┐    ┌──────▼───────┐        │
│  │ Weight   │◀───│ Gradient   │◀───│  Backward    │        │
│  │ Update   │    │  Sum       │    │  Pass        │        │
│  └──────────┘    └────────────┘    └──────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### Implementation Tasks

#### 7.1 Autograd Engine
- [ ] `ck_autograd_node_t` - computation graph node structure
- [ ] `ck_autograd_create_node()` - create node with backward function
- [ ] `ck_autograd_backward()` - topological gradient computation
- [ ] `ck_autograd_zero_grad()` - clear gradients

#### 7.2 Activation Checkpointing
- [ ] `ck_checkpoint_create()` - save activations at strategic points
- [ ] `ck_checkpoint_restore()` - recompute from checkpoint
- [ ] Memory vs compute tradeoff configuration

#### 7.3 Loss Functions
- [ ] `ck_loss_cross_entropy()` - classification loss
- [ ] `ck_loss_mse()` - regression loss
- [ ] `ck_loss_kl_div()` - KL divergence for distributions

#### 7.4 Optimizers
- [ ] `ck_opt_sgd()` - stochastic gradient descent
- [ ] `ck_opt_adam()` - Adam (momentum + RMSprop)
- [ ] `ck_opt_adamw()` - Adam with weight decay
- [ ] Learning rate scheduling (constant, cosine, step)

#### 7.5 Gradient Kernels
- [ ] `gemm_grad_weights()` - weight gradient computation
- [ ] `gemm_grad_inputs()` - input gradient for GEMM
- [ ] `layernorm_backward()` - layer norm gradients
- [ ] `softmax_backward()` - softmax gradients

### Performance Targets
- Training throughput: Match PyTorch within 5x on single GPU
- Memory efficiency: Checkpointing reduces memory by 5-10x
- Batch size: Support up to 32 on 16GB GPU

---

## v8: Mixture of Experts (MoE)
**Goal**: Implement sparse MoE architecture for massive parameter efficiency

### Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                      MoE Architecture                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    Input ──┬────────────────┬────────────────┐                  │
│            │                │                │                  │
│       ┌────▼────┐      ┌────▼────┐      ┌────▼────┐            │
│       │ Expert 1│      │ Expert 2│      │ Expert N│            │
│       │ (FFN)   │      │ (FFN)   │      │ (FFN)   │            │
│       └────┬────┘      └────┬────┘      └────┬────┘            │
│            │                │                │                  │
│       ┌────▼────┐      ┌────▼────┐      ┌────▼────┐            │
│       │Gate=0.8 │      │Gate=0.1 │      │Gate=0.1 │            │
│       └────┬────┘      └────┬────┘      └────┬────┘            │
│            │                │                │                  │
│            └────────────────┼────────────────┘                  │
│                             │                                   │
│                        Weighted Sum                            │
│                        (Top-K = 2)                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Tasks

#### 8.1 Gating Network
- [ ] `ck_moe_gating()` - learnable gating function
- [ ] Top-K selection (typically K=1 or K=2)
- [ ] Load balancing loss (prevent expert collapse)
- [ ] Noisy gating for exploration

#### 8.2 Expert Implementation
- [ ] `ck_moe_expert()` - single expert (FFN block)
- [ ] Expert caching for repeated inputs
- [ ] Expert capacity limits (buffer overflow handling)
- [ ] Heterogeneous expert sizes support

#### 8.3 Efficient Dispatch
- [ ] `ck_moe_dispatch()` - route tokens to experts
- [ ] `ck_moe_combine()` - gather expert outputs
- [ ] Batched expert computation
- [ ] Expert prefetching

#### 8.4 Sparse Computation
- [ ] Zero-GEMM for inactive experts
- [ ] Expert parallelism support
- [ ] Dynamic load balancing

### Use Cases
- Switch Transformer style models
- Mixtral-style open MoE
- Goal: 10x more parameters with same compute

---

## v9: Vision Transformer (ViT)
**Goal**: Add image understanding capabilities

### Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                    Vision Transformer                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Image (HxW)                                                     │
│      │                                                           │
│      ▼                                                           │
│  ┌────────┐                                                      │
│  │ Patch  │───▶ Flattened Patches (N x D)                       │
│  │ Embed  │                                                      │
│  └────────┘                                                      │
│      │                                                           │
│      ▼                                                           │
│  ┌────────┐    ┌───────────────────────────┐                    │
│  │ CLS    │───▶│ Transformer Encoder       │                    │
│  │ Token  │    │ (L layers, D hidden)      │                    │
│  └────────┘    └───────────────────────────┘                    │
│      │                     │                                    │
│      ▼                     ▼                                    │
│  ┌────────┐          ┌────────────┐                             │
│  │ Class  │◀─────────│ Per-Patch  │                             │
│  │ Head   │          │ Outputs    │                             │
│  └────────┘          └────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Tasks

#### 9.1 Image Preprocessing
- [ ] `ck_vit_preprocess()` - resize, normalize, to tensor
- [ ] `ck_image_to_patches()` - split image into N patches
- [ ] Configurable patch size (16x16 default)
- [ ] Mean/std normalization (ImageNet stats)

#### 9.2 Positional Embeddings
- [ ] `ck_add_position_embedding()` - 2D sincos or learned
- [ ] Interpolation for variable image sizes
- [ ] Relative position biases

#### 9.3 Vision-Specific Components
- [ ] `ck_vit_attention()` - attention with image tokens
- [ ] Patch embedding projection (conv or linear)
- [ ] CLS token handling

#### 9.4 ViT Variants
- [ ] ViT-Base (12 layers, 768 dim)
- [ ] ViT-Large (24 layers, 1024 dim)
- [ ] ViT-Huge (32 layers, 1280 dim)
- [ ] DeiT (data-efficient, distillation)

#### 9.5 ImageNet Classification
- [ ] `ck_vit_classify()` - image to class logits
- [ ] Pre-trained weight loading (safetensors)
- [ ] Top-k accuracy computation

---

## v10: Vision Transformer Training
**Goal**: Train ViT on image classification tasks

### Training Pipeline
```
┌─────────────────────────────────────────────────────────────────┐
│               ViT Training Pipeline                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌────────────┐    ┌──────────────────────┐    │
│  │ Image    │───▶│ ViT        │───▶│ Classification       │    │
│  │ Batch    │    │ Forward    │    │ Loss (Cross-Entropy) │    │
│  └──────────┘    └────────────┘    └──────────┬───────────┘    │
│                                               │                 │
│  ┌──────────┐    ┌────────────┐    ┌──────────▼───────────┐    │
│  │ AdamW    │◀───│ Gradients  │◀───│ Backward Pass        │    │
│  │ Update   │    │ Accumulate │    │ (via v7 autograd)    │    │
│  └──────────┘    └────────────┘    └──────────────────────┘    │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Data Augmentation: RandomCrop, HorizontalFlip, ColorJitter│   │
│  │ Regularization: Dropout, Label Smoothing, Weight Decay    │   │
│  │ Learning Rate: Cosine Annealing with Warmup               │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Tasks

#### 10.1 Data Augmentation
- [ ] `ck_aug_random_crop()` - random aspect ratio crop
- [ ] `ck_aug_hflip()` - horizontal flip
- [ ] `ck_aug_color_jitter()` - brightness, contrast, saturation
- [ ] `ck_aug_mixup()` - MixUp augmentation
- [ ] `ck_aug_cutmix()` - CutMix augmentation

#### 10.2 Regularization
- [ ] `ck_reg_dropout()` - transformer dropout
- [ ] `ck_reg_label_smoothing()` - label smoothing
- [ ] Stochastic depth (layer dropout)

#### 10.3 Learning Rate Schedules
- [ ] `ck_lr_cosine()` - cosine annealing
- [ ] `ck_lr_warmup()` - gradual warmup
- [ ] `ck_lr_step()` - step decay
- [ ] `ck_lr_poly()` - polynomial decay

#### 10.4 Training Loop
- [ ] `ck_vit_train_epoch()` - single epoch training
- [ ] `ck_vit_eval()` - validation evaluation
- [ ] Checkpoint saving/loading
- [ ] TensorBoard logging

#### 10.5 Pretrained Weights
- [ ] Download mechanism for ImageNet pretrained weights
- [ ] Weight conversion from PyTorch safetensors
- [ ] Fine-tuning support for downstream tasks

---

## v11: Audio Transformer
**Goal**: Add speech and audio understanding

### Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                    Audio Transformer                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Audio Waveform (T samples)                                      │
│      │                                                           │
│      ▼                                                           │
│  ┌────────┐                                                      │
│  │ Mel    │───▶ Mel Spectrogram (F x T)                         │
│  │ Spectr │                                                      │
│  └────────┘                                                      │
│      │                                                           │
│      ▼                                                           │
│  ┌────────┐    ┌───────────────────────────┐                    │
│  │ Frame  │───▶│ Transformer Encoder       │                    │
│  │Embed   │    │ (Shared with ViT/v9)      │                    │
│  └────────┘    └───────────────────────────┘                    │
│      │                     │                                    │
│      ▼                     ▼                                    │
│  ┌────────┐          ┌────────────┐                             │
│  │ Task   │◀─────────│ Per-Frame  │                             │
│  │ Head   │          │ Outputs    │                             │
│  └────────┘          └────────────┘                             │
│                                                                  │
│  Tasks: ASR, Speaker ID, Audio Class, Whisper-style             │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Tasks

#### 11.1 Audio Preprocessing
- [ ] `ck_audio_load()` - load wav/mp3 files
- [ ] `ck_audio_resample()` - target sample rate (16kHz)
- [ ] `ck_audio_normalize()` - amplitude normalization

#### 11.2 Spectrogram Computation
- [ ] `ck_stft()` - Short-time Fourier Transform
- [ ] `ck_mel_filter()` - mel filterbank application
- [ ] `ck_power_spec()` - power spectrogram
- [ ] `ck_compress_spec()` - log-mel spectrogram

#### 11.3 Audio-Specific Components
- [ ] `ck_audio_position()` - relative position for audio
- [ ] `ck_audio_mask()` - padding mask for variable length
- [ ] Streaming support (chunked processing)

#### 11.4 Whisper-style Model
- [ ] Encoder-decoder architecture
- [ ] Multilingual transcription
- [ ] Timestamp prediction
- [ ] Voice activity detection

#### 11.5 Speech Tasks
- [ ] ASR (Automatic Speech Recognition)
- [ ] Speaker verification (embedding)
- [ ] Audio classification

---

## v12: Audio Transformer Training
**Goal**: Train audio models for ASR and other tasks

### Training Pipeline
```
┌─────────────────────────────────────────────────────────────────┐
│               Audio Training Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌────────────┐    ┌──────────────────────┐    │
│  │ Audio    │───▶│ Spectrogram│───▶│ Whisper/Encoder      │    │
│  │ Waveform │    │            │    │                       │    │
│  └──────────┘    └────────────┘    └───────────┬──────────┘    │
│                                                 │               │
│  ┌──────────┐    ┌────────────┐    ┌───────────▼──────────┐   │
│  │ CTC/     │◀───│ Cross-Ent  │◀───│ Decoder              │   │
│  │ Seq2Seq  │    │ Loss       │    │ (Cross-Attention)    │   │
│  └──────────┘    └────────────┘    └──────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Special: Beam Search Decoding                            │   │
│  │ Special: Token-level timestamps                          │   │
│  │ Special: Language modeling boost                         │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Tasks

#### 12.1 Data Loading
- [ ] `ck_audio_dataset()` - dataset abstraction
- [ ] Bucketing by audio length
- [ ] On-the-fly spectrogram computation
- [ ] ASR corpus support (Common Voice, LibriSpeech)

#### 12.2 Audio Augmentation
- [ ] `ck_aug_spec_mask()` - SpecAugment (time/freq masking)
- [ ] `ck_aug_time_shift()` - time shifting
- [ ] `ck_aug_speed()` - speed perturbation
- [ ] `ck_aug_noise()` - background noise injection

#### 12.3 Decoding
- [ ] `ck_decode_greedy()` - greedy decoding
- [ ] `ck_decode_beam()` - beam search (width 5-10)
- [ ] LM integration for rescoring
- [ ] Word-level timestamps

#### 12.4 Training Loop
- [ ] `ck_audio_train()` - audio training function
- [ ] SpecAugment configuration
- [ ] Gradient accumulation for long audio
- [ ] Evaluation metrics (WER, CER)

---

## v13-14: Complete Multi-Modal Transformer
**Goal**: Unified model for text, vision, and audio together

### Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│              Multi-Modal Transformer (LLaVA-style)               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐                         │
│   │  Text   │  │  Image  │  │  Audio  │                         │
│   │ Tokens  │  │ Patches │  │ Frames  │                         │
│   └────┬────┘  └────┬────┘  └────┬────┘                         │
│        │            │            │                               │
│        ▼            ▼            ▼                               │
│   ┌─────────────────────────────────────────┐                   │
│   │       Unified Projection to LLM Dim     │                   │
│   │         (Text, Image, Audio → D)        │                   │
│   └──────────────────┬──────────────────────┘                   │
│                      │                                           │
│                      ▼                                           │
│   ┌─────────────────────────────────────────┐                   │
│   │      Cross-Modal Attention Fusion       │                   │
│   │    (Image→Text, Audio→Text attention)   │                   │
│   └──────────────────┬──────────────────────┘                   │
│                      │                                           │
│                      ▼                                           │
│   ┌─────────────────────────────────────────┐                   │
│   │      Pre-trained LLM (from v6)          │                   │
│   │     (RoPE, SwiGLU, RMSNorm, GQA)        │                   │
│   └──────────────────┬──────────────────────┘                   │
│                      │                                           │
│                      ▼                                           │
│   ┌─────────────────────────────────────────┐                   │
│   │           Task Heads                    │                   │
│   │   VQA | Caption | ASR | Classification  │                   │
│   └─────────────────────────────────────────┘                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### v13: Multi-Modal Encoder

#### 13.1 Projection Layers
- [ ] `ck_proj_vision()` - image patch to LLM dim
- [ ] `ck_proj_audio()` - spectrogram to LLM dim
- [ ] `ck_proj_text()` - text to LLM dim (for consistency)

#### 13.2 Cross-Modal Attention
- [ ] `ck_cross_attn()` - attend to other modalities
- [ ] `ck_multi_head_cross()` - multi-head cross attention
- [ ] Gated cross-attention (LLaVA style)

#### 13.3 Multi-Modal Rotary
- [ ] `ck_rope_multimodal()` - extended RoPE for images/audio
- [ ] Absolute position for patches
- [ ] Relative timing for audio

#### 13.4 Vision-Language Tasks
- [ ] VQA (Visual Question Answering)
- [ ] Image Captioning
- [ ] Visual Reasoning

### v14: Unified Training & Inference

#### 14.1 Multi-Modal Data
- [ ] `ck_mm_dataset()` - mixed text/image/audio dataset
- [ ] Contrastive learning support (CLIP-style)
- [ ] Instruction tuning support

#### 14.2 Unified Inference
- [ ] Dynamic input type detection
- [ ] `ck_mm_generate()` - unified generation
- [ ] Multi-turn conversation support

#### 14.3 Efficiency
- [ ] Cache for cross-modal projections
- [ ] Prefetching for vision/audio
- [ ] KV cache sharing across modalities

---

## v15+: Distributed Computing
**Goal**: Scale to multiple GPUs and nodes

### Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                   Distributed Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    ┌─────────┐     ┌─────────┐     ┌─────────┐                 │
│    │ Node 0  │     │ Node 1  │     │ Node N  │                 │
│    │ GPU 0-2 │     │ GPU 0-2 │     │ GPU 0-2 │                 │
│    └────┬────┘     └────┬────┘     └────┬────┘                 │
│         │               │               │                       │
│         └───────────────┼───────────────┘                       │
│                         │                                       │
│                         ▼                                       │
│              ┌─────────────────────┐                            │
│              │   NCCL/PCIe/Gloo    │                            │
│              │   Communication     │                            │
│              └─────────────────────┘                            │
│                         │                                       │
│                         ▼                                       │
│              ┌─────────────────────┐                            │
│              │   Global Rank 0     │                            │
│              │   Orchestrator      │                            │
│              └─────────────────────┘                            │
│                                                                  │
│  Parallelism Strategies:                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Data Parallel: Same model, different data batches       │   │
│  │ Tensor Parallel: Split layers across GPUs               │   │
│  │ Pipeline Parallel: Sequential layers across GPUs        │   │
│  │ ZeRO: Shard optimizer states + gradients + parameters   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### v15: Distributed Infrastructure

#### 15.1 Process Group
- [ ] `ck_dist_init()` - initialize distributed
- [ ] `ck_dist_rank()` - get current rank
- [ ] `ck_dist_world_size()` - total processes
- [ ] NCCL/Gloo backend selection

#### 15.2 Communication Primitives
- [ ] `ck_dist_all_reduce()` - sum across ranks
- [ ] `ck_dist_broadcast()` - send to all ranks
- [ ] `ck_dist_gather()` - gather from all
- [ ] `ck_dist_scatter()` - scatter to all

#### 15.3 Distributed Optimizer
- [ ] `ck_dist_opt_zero()` - ZeRO optimizer
- [ ] Gradient sharding
- [ ] Automatic mixed precision (FP16/BF16)

### v16: Advanced Parallelism

#### 16.1 Pipeline Parallelism
- [ ] `ck_pipe_stage()` - define pipeline stage
- [ ] Micro-batch scheduling
- [ ] Pipeline flush and fill

#### 16.2 Tensor Parallelism
- [ ] Column-wise weight sharding
- [ ] Row-wise weight sharding
- [ ] Collective operations for all-reduce

#### 16.3 FSDP (Fully Sharded Data Parallel)
- [ ] Shard parameters at layer level
- [ ] Prefetch next layer
- [ ] Gradient checkpointing in distributed

---

## Cross-Cutting Concerns (All Versions)

### Performance Optimization
```
┌─────────────────────────────────────────────────────────────────┐
│              Continuous Performance Engineering                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ▸ SIMD Vectorization                                           │
│    - AVX-512 when available                                     │
│    - NEON for ARM                                               │
│    - RISC-V Vector extension                                    │
│                                                                  │
│  ▸ Kernel Optimization                                          │
│    - Cache blocking (loop tiling)                               │
│    - Register blocking                                          │
│    - Software pipelining                                        │
│    - Memory prefetching                                         │
│                                                                  │
│  ▸ Quantization                                                 │
│    - INT8 quantization (per-channel)                            │
│    - GPTQ/GGUF quantization                                     │
│    - SmoothQuant                                                │
│    - AWQ (Activation-aware Weight Quant)                        │
│                                                                  │
│  ▸ Code Generation                                              │
│    - Architecture-specific code gen                             │
│    - Autotuning for blocking factors                            │
│    - Profile-guided optimization                                │
│                                                                  │
│  ▸ Memory Efficiency                                            │
│    - Paged KV cache                                             │
│    - Gradient checkpointing                                     │
│    - Mixed precision memory                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Code Generation Pipeline
```
Input Model (GGUF/HuggingFace)
        │
        ▼
┌───────────────────┐
│ Graph Compilation │  ← Static op fusion, layout optimization
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   Kernel选择       │  ← Select best kernel for target
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Code Generation   │  ← Emit optimized C code with intrinsics
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   Compilation     │  ← GCC/Clang with -O3, vector extensions
└────────┬──────────┘
         │
         ▼
   Optimized .so Library
```

### Testing Strategy
```
┌─────────────────────────────────────────────────────────────────┐
│                    Testing Pyramid                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                    ┌───────────┐                                 │
│                    │ Integration│  ← End-to-end model tests     │
│                    │   Tests    │                                │
│                    └─────┬─────┘                                 │
│                          │                                       │
│                    ┌─────▼─────┐                                 │
│                    │  Parity   │  ← Compare with llama.cpp/PyTorch│
│                    │   Tests   │                                │
│                    └─────┬─────┘                                 │
│                          │                                       │
│                    ┌─────▼─────┐                                 │
│                    │  Unit     │  ← Kernel correctness tests     │
│                    │   Tests   │                                │
│                    └─────┬─────┘                                 │
│                          │                                       │
│                    ┌─────▼─────┐                                 │
│                    │ Numerical │  ← FP32 reference for comparison│
│                    │  Tests    │                                │
│                    └───────────┘                                 │
│                                                                  │
│  CI/CD: Every commit → build, test, benchmark                   │
│  Performance: Nightly benchmarks against llama.cpp              │
│  Coverage: Target 90%+ line coverage                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Version Timeline Summary

| Version | Focus | Key Features |
|---------|-------|--------------|
| v6.6 | Performance | AVX kernels, tests, profiling |
| v7 | Training | Autograd, optimizers, gradients |
| v8 | MoE | Sparse experts, gating, load balancing |
| v9 | Vision | ViT, image patches, classification |
| v10 | Vision Training | Augmentation, fine-tuning, ImageNet |
| v11 | Audio | Spectrogram, Whisper-style, ASR |
| v12 | Audio Training | Data loading, augmentation, decoding |
| v13 | Multi-Modal Encoders | Cross-modal attention, projection |
| v14 | Multi-Modal Unified | Unified inference, training |
| v15 | Distributed Base | NCCL, communication primitives |
| v16+ | Advanced Parallelism | Pipeline, tensor, ZeRO |

---

## Contributing to This Roadmap

### Immediate Needs (v6.6)
1. ✅ Enable AVX kernels (done)
2. ⏳ Rebuild model library
3. ⏳ Run full test suite
4. ⏳ Profile to verify speedup

### Near Term (v7)
1. Design autograd API (C API for maximum performance)
2. Implement gradient checkpointing
3. Add optimizer support

### Long Term (v8+)
1. MoE research review for best practices
2. Vision transformer architecture decisions
3. Audio preprocessing library integration

---

*Last Updated: 2026-01-11*
*Version: 1.0*
