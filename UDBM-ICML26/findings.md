# Findings

This file records paper/code discoveries for the cleanup task.

## Paper Terminology

- Paper title: "Unifying Heterogeneous Degradations: Uncertainty-Aware Diffusion Bridge Model for All-in-One Image Restoration".
- Method name: Uncertainty-Aware Diffusion Bridge Model (UDBM).
- Stage 1 corresponds to the auxiliary restoration network / uncertainty predictor `psi(.)`.
- Stage 2 corresponds to the denoising/restoration network `N_den(.)` inside the uncertainty-aware diffusion bridge.
- Important method terms: relaxed diffusion bridge, relaxed terminal constraint, pixel-wise uncertainty, path schedule, noise schedule, dual uncertainty modulation, and single-step inference.

## Architecture Variants

- UDBM-S: Stage 1 `(width=16, blocks=[1,1,1,10])`; Stage 2 `(width=16, blocks=[1,1,1,28])`.
- UDBM-M: Stage 1 `(width=16, blocks=[1,1,1,28])`; Stage 2 `(width=24, blocks=[1,1,1,28])`.
- UDBM-L: Stage 1 `(width=32, blocks=[1,1,1,16])`; Stage 2 `(width=48, blocks=[1,1,1,28])`.

## Cleanup Targets

- Entry scripts can use paper-aligned aliases without changing checkpoint-sensitive class/module names.
- Stage-2 model files contain top-level parameter counting code, disabled FLOPs profiling blocks, and debug prints that should be removed.
- README should be reorganized around GitHub usage: install, data layout, pretrained weights, training, evaluation, metrics, and citation.
