# Task Plan

Goal: clean the UDBM-ICML26 codebase and README so the implementation is concise, paper-aligned, and GitHub-friendly.

## Phases

1. Inspect paper terminology and current code structure. Status: complete
2. Identify redundant code/comments and paper-alignment opportunities. Status: complete
3. Apply safe code cleanup and naming updates without breaking checkpoint compatibility. Status: complete
4. Rewrite README into a clearer GitHub-style document. Status: complete
5. Run syntax/path checks and summarize results. Status: complete

## Verification

- `python -m py_compile` passed for entry scripts, UDBM model files, `data/universal_dataset.py`, and metric helpers.
- Absolute path scan passed for Python, Markdown, and TeX files.
- Chinese text scan passed for public Python/Markdown files in the cleaned code package.
- Removed generated `__pycache__` directories after compilation checks.
- Runtime verification with conda environment `fcl` passed for CLI help, imports, minimal forward smoke test, and strict pretrained checkpoint loading for UDBM-S/M/L.
- Fixed a missing `matlab_functions` dependency in `metrics/metric_util.py` by making the metric utility self-contained.
- Replaced `pretrained/udbm_l/model-600.pt` with the NAFNet-48 checkpoint that strictly matches `src/model_udbm_s2_l.py`.
- Confirmed the Stage 2 inference sampler now uses the paper-aligned name `sample_uncertainty_aware_bridge`; the old `ddim_sample_uncertain_diffuir` name has no remaining references.

## Constraints

- Keep pretrained checkpoint compatibility.
- Avoid renaming model submodules that appear in checkpoint state_dict keys unless aliases preserve loading.
- Keep edits focused on the cleaned project, not the original DiffUIR-main tree.
