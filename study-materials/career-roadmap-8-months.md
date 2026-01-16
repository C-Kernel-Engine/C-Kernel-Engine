# 8-Month Career Acceleration Roadmap

**Goal**: Transform from "AI engineer" to "AI systems expert in top 0.01%"

**Target Outcomes**:
- C-Kernel-Engine: 1,100+ GFLOPS (70%+ of MKL)
- YouTube: 8,000+ subscribers, $800+/month
- Job: $120K-$180K at AI infrastructure/robotics company
- Consulting: $15K-$30K in side projects

---

## Month 1-2: Foundation (Xeon + BLIS/oneDNN Study)

### Technical Goals
- [ ] Implement 16×2 microkernel in C-Kernel-Engine
- [ ] Three-level blocking (register → L1/L2 → L3)
- [ ] Profile with `perf stat`, compare against MKL baseline
- [ ] **Target**: 600-800 GFLOPS (40-50% of MKL)

### Study Materials
- [ ] Read all of `docs/study-materials/` (already created)
- [ ] Study oneDNN BRGEMM source code:
  - `oneDNN/src/cpu/x64/brgemm/jit_brgemm_kernel.cpp`
  - `oneDNN/src/cpu/x64/brgemm/brgemm_types.hpp`
- [ ] Study BLIS microkernel:
  - `blis/kernels/haswell/3/bli_gemm_haswell_asm_d6x8.c`
  - `blis/frame/3/bli_gemm.c`

### YouTube Content (4-8 videos)
1. "Why I'm building a GEMM kernel from scratch"
2. "How transformers actually work (hardware level)"
3. "Beating NumPy: My first optimized GEMM kernel"
4. "Studying oneDNN: What I learned from Intel's open source AI library"

### Networking
- [ ] Join oneDNN Slack/Discord
- [ ] Contribute to oneDNN docs (small PRs to build credibility)
- [ ] Post on Twitter/X: Weekly progress updates

---

## Month 3-4: AI-Specific Optimizations

### Technical Goals
- [ ] Implement BRGEMM (batched strided GEMM)
- [ ] Shape-specific dispatch (small-M vs large-M vs attention)
- [ ] Fused operations (bias + GELU)
- [ ] **Target**: 900-1,100 GFLOPS (60-70% of MKL)

### LLM Benchmarking
- [ ] Benchmark GPT-2 forward pass
- [ ] Benchmark BERT encoder
- [ ] Compare C-Kernel-Engine vs PyTorch (MKL backend)
- [ ] Document 2-3x speedup stories

### YouTube Content (4-8 videos)
5. "How multi-head attention actually works (in 64 bytes of cache)"
6. "I implemented Intel's BRGEMM - here's what I learned"
7. "Why small-M kernels are different (autoregressive decode)"
8. "Profiling GPT-2: Where does the time actually go?"

### Portfolio Building
- [ ] Write 2-3 blog posts on Medium/Dev.to
- [ ] Create C-Kernel-Engine README with performance charts
- [ ] Add comparative benchmarks (vs MKL, vs PyTorch)

---

## Month 5-6: Embedded Deployment (TDA4VM)

### Technical Goals
- [ ] Port kernels to ARM NEON
- [ ] Integrate with TDA4VM C71x DSP
- [ ] Deploy simple model on drone/edge device
- [ ] **Target**: 50-80 GFLOPS on TDA4VM (respectable for ARM)

### Hardware Setup
- [ ] Acquire TDA4VM dev board (SK-TDA4VM, ~$300)
- [ ] Set up cross-compilation toolchain
- [ ] Profile with ARM PMU (Performance Monitoring Unit)

### YouTube Content (4-8 videos)
9. "Porting my GEMM kernel from x86 to ARM"
10. "TDA4VM deep dive: A72, DSP, and MMA accelerator"
11. "Running a transformer on a $200 board"
12. "Drone AI: How I deployed a vision model on embedded hardware"

### Job Search Preparation
- [ ] Update LinkedIn: "AI Systems Engineer | Kernel Optimization"
- [ ] List C-Kernel-Engine prominently
- [ ] Write 1-page "brag sheet" with performance numbers
- [ ] Start reaching out to recruiters (embedded AI, robotics)

---

## Month 7-8: Advanced Optimizations + Job Search

### Technical Goals
- [ ] Prefetching and advanced cache optimizations
- [ ] Hand-written assembly for critical paths (optional)
- [ ] Low-precision support (bf16, int8 - stretch goal)
- [ ] **Target**: 1,100-1,300 GFLOPS on Xeon (70-85% of MKL)

### Production Readiness
- [ ] Edge case handling (non-divisible M/N/K)
- [ ] Threading (OpenMP for multi-core)
- [ ] API stabilization
- [ ] Documentation and examples

### YouTube Content (4-8 videos)
13. "The hardest bug I've ever debugged (cache coherency)"
14. "Why I'm NOT using AI to write my kernels"
15. "8-month journey: From 250 GFLOPS to 1200 GFLOPS"
16. "What I learned from building a GEMM library"

### Job Search Execution
- [ ] Apply to 20-30 companies:
  - **Tier 1**: Meta, Google, NVIDIA, Intel (moonshots)
  - **Tier 2**: Anthropic, Cohere, Databricks (realistic)
  - **Tier 3**: Robotics startups (most realistic)
- [ ] Prepare "portfolio presentation" (10-minute demo)
- [ ] Practice system design interviews
- [ ] Negotiate offers (aim for $140K-$180K)

### Consulting Launch
- [ ] Create consulting page on website
- [ ] Offer "GEMM optimization audit" service
- [ ] Charge $150-$200/hour for first clients
- [ ] Target 5-10 hours/week ($750-$2,000/week)

---

## Success Metrics (End of Month 8)

### Technical
- [x] C-Kernel-Engine performance: 1,100+ GFLOPS (70%+ of MKL)
- [x] Deployed on embedded system (TDA4VM)
- [x] BRGEMM implementation complete
- [x] GitHub stars: 50-100+ (shows external validation)

### Career
- [x] Job offer: $120K-$180K at AI infrastructure/robotics company
- [x] OR: Consulting income: $3K-$8K/month ($36K-$96K/year)
- [x] LinkedIn connections: 500+ (technical recruiters, engineers)
- [x] Portfolio: 3-5 blog posts, 1 GitHub repo with good docs

### YouTube/Brand
- [x] Subscribers: 8,000-15,000
- [x] Revenue: $800-$1,500/month (ads + sponsorships)
- [x] Views: 100K-200K total
- [x] Community: 100+ Discord members (optional)

### Personal
- [x] Deep understanding of cache hierarchy
- [x] Fluent in AVX-512 and ARM NEON
- [x] Confident explaining technical decisions
- [x] Part of 0.01% elite (kernel builders)

---

## Weekly Schedule (Sustainable Pace)

**Monday-Friday (Day Job + C-Kernel-Engine)**
- 6:00-7:30 AM: Study oneDNN/BLIS source code (1.5 hrs)
- 7:30-9:00 AM: Morning routine, prep for work
- 9:00-6:00 PM: Day job (learn what you can)
- 8:00-10:00 PM: Implement kernels, profile, debug (2 hrs)

**Saturday (YouTube + Portfolio)**
- 9:00-12:00 PM: Record 1-2 YouTube videos (batch)
- 1:00-3:00 PM: Edit videos, write blog post
- 4:00-6:00 PM: GitHub issues, community engagement

**Sunday (Rest + Learning)**
- 9:00-11:00 AM: Read papers, study reference libraries
- Rest of day: Actual rest (avoid burnout!)

**Total weekly commitment**: ~15-20 hours outside of day job

---

## Red Flags to Avoid

### Technical
- ❌ Don't over-engineer early (e.g., JIT compilation before basic kernel works)
- ❌ Don't ignore profiling (always measure, never assume)
- ❌ Don't claim performance you haven't measured
- ❌ Don't compare against unoptimized baselines (compare to MKL!)

### Career
- ❌ Don't accept first offer without negotiating
- ❌ Don't undersell your skills ("I'm just learning" → "I built kernels matching 70% of MKL")
- ❌ Don't ignore non-FAANG companies (robotics startups need you!)
- ❌ Don't burn out (15-20 hrs/week is sustainable, 40 is not)

### YouTube
- ❌ Don't sacrifice quality for quantity (better 1 great video than 4 mediocre)
- ❌ Don't ignore SEO (title + thumbnail + tags matter)
- ❌ Don't get discouraged by slow growth (first 1,000 subs take 6 months)
- ❌ Don't feed trolls ("Why not just use PyTorch?" → ignore or educate once)

---

## Pivot Points (When to Adjust)

**If Month 4 performance is < 700 GFLOPS:**
- [ ] Deep dive into profiling (IPC, cache misses)
- [ ] Study OpenBLAS assembly more carefully
- [ ] Consider hiring a consultant for 2-3 hours ($300-$600)

**If Month 6 YouTube is < 2,000 subs:**
- [ ] Analyze top-performing videos (watch time, CTR)
- [ ] Experiment with shorts (60-second explainers)
- [ ] Collaborate with other channels

**If Month 8 job search has no offers:**
- [ ] Lean harder into consulting (can exceed $180K/year)
- [ ] Consider contractor roles (often easier to get, $100-$150/hr)
- [ ] Geographic relocation (SF Bay, Seattle, Austin, Boston)

---

## Accountability

**Monthly check-ins:**
- Last day of each month: Review progress against goals
- Update this document with actual numbers
- Celebrate wins, analyze misses
- Adjust next month's goals

**Quarterly reviews:**
- End of Month 2, 4, 6, 8: Deep reflection
- Ask: "Am I 2-3x better than 3 months ago?"
- If no: Pivot strategy
- If yes: Double down on what's working

---

## Final Motivation

**You are not just "learning to code".**

You are building skills that put you in the **top 0.01%** of software engineers globally.

In 8 months, when you can:
- Explain why 6×8 microkernel beats 64×64 blocking
- Show 1,200 GFLOPS performance (80% of MKL)
- Deploy a transformer on a $200 embedded board
- Casually discuss roofline models and IPC optimization

**You will stand out.**

Not because you're "smart" (everyone in AI is smart).

But because you **did the work that 99.99% of people avoid**.

The companies building the future (robotics, edge AI, next-gen hardware) **desperately need you**.

Your job is to:
1. Build C-Kernel-Engine to 70%+ of MKL
2. Document the journey publicly (YouTube, blog)
3. Show up consistently for 8 months
4. Let the results speak for themselves

**In 8 months, you won't need to ask "Can I get a better job?"**

**Recruiters will be asking you: "Can we fly you out for an interview?"**

---

**Start Date**: 2025-11-23
**Target Completion**: 2025-07-23
**Status**: Month 1 (Foundation) in progress

**Last Updated**: 2025-11-23
