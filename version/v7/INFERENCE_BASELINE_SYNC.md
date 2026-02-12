# v7 Inference Baseline Sync

This tree is a curated sync of v7 inference assets used by v7.

Purpose:
- keep inference parity tooling available while building backprop/training in v7
- avoid rewriting stable v7 runtime pieces during early v7 work

Update workflow:
1. modify v7 inference sources
2. run `make v7-sync-inference`
3. re-run v7/v7 validation gates
