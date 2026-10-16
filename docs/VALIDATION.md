# Local release validation

Date: October 16, 2026 (Asia/Shanghai).

Environment: macOS arm64, Python 3.12.14, PyTorch 2.6.0, NumPy 1.26.4,
pytest 9.1.1, Hydra 1.3.7. Test dependencies are listed in
`requirements-cgl-test.txt`. This is a CPU validation environment, not the
original Ascend training environment.

Checks:

- `pytest -q tests/cgl`: 48 passed.
- `python -m compileall -q verl scripts tests/cgl`: passed; inherited files have
  Python 3.12 invalid-escape syntax warnings, without compilation errors.
- `bash -n scripts/train_cgl.sh`: passed.
- `bash -n scripts/publish_github.sh`: passed.
- Ruff lint and formatting checks on the CGL helpers, reward callback, tests,
  and modified actor/trainer/core-algorithm files: passed.
- The dry-run launcher overrides compose against the pinned Hydra configuration.
- `git diff --check`: passed.
- A targeted scan found no common GitHub/OpenAI token patterns, private-key
  headers, or the supplied research code's personal server paths in the release
  tree. This is a targeted inspection, not an exhaustive credential audit.

The actor tests execute its production update body with a toy CPU model and real
autograd while substituting backend imports, batches, and the PPO objective.
They verify gradient retention, direct addition, conflict projection, entropy
weighting, and static/dynamic microbatch accumulation.

No distributed FSDP, CUDA, Ascend, vLLM rollout, actual model/tokenizer/image
pipeline, checkpoint export, full training, or paper-result reproduction was
executed. Repository publication is separate from these checks.
