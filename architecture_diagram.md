# C-Kernel-Engine Architecture

This diagram illustrates the pipeline from a HuggingFace configuration to a compiled, standalone C runtime.

```mermaid
graph TD
    subgraph Input
        Config[config.json] -->|Parse| Loader[CKModelConfig]
    end

    subgraph "Phase 1: IR Generation"
        Loader -->|Build Graph| IR[CKIRGraph]
        
        IR --> Header[Header Section<br/>(Embeds, RoPE)]
        IR --> Body[Block Section<br/>(Repeat L layers)]
        IR --> Footer[Footer Section<br/>(Norm, Head)]
        
        style Body fill:#f9f,stroke:#333,stroke-width:2px
    end

    subgraph "Phase 2: Memory Planning"
        Header & Body & Footer -->|Analyze Shapes| Planner[Memory Planner]
        Planner -->|Assign Offsets| Layout[Memory Layout<br/>ACT_BYTES / GRAD_BYTES]
    end

    subgraph "Phase 3: Code Generation"
        Layout -->|Emit| CSource["ai.c<br/>(Standalone Runtime)"]
        
        CSource -- Contains --> Main[main()]
        CSource -- Contains --> Fwd[run_model_forward]
        CSource -- Contains --> Bwd[run_model_backward]
    end

    subgraph "Phase 4: Execution"
        Fwd -->|Calls| Kernels[Optimized Kernels<br/>(src/kernels/*.c)]
        Kernels -->|CPU/AVX| Hardware[Hardware]
    end
```

## The "Website" Metaphor

The engine treats LLMs like a website generator treats pages:

| Section | Website | LLM |
| :--- | :--- | :--- |
| **Header** | `<head>`, Nav, CSS | Embeddings, Positional Encoding |
| **Block** | Blog Posts, Articles | Transformer Layers (repeated) |
| **Footer** | Copyright, Scripts | Final Norm, Language Head |

This allows us to unroll the "Block" section efficiently in C without complex control flow.
