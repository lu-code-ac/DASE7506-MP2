# DASE7506 MP2

The recommended model is the original ranking MLP ensemble trained with three random seeds. Its official test accuracy is **79.65738758029978%**.

- Report: project2report.md.
- Trained checkpoint: `code/runs/model_bundle/`. No retraining is required.
- Running and training instructions: code/README.md.
- Official test score: `code/runs/test-score.json`.
- Follow-up experiments did not demonstrate an improvement, so the final package includes only the recommended model’s inference weights. Source code and result records for the other experiments are retained.

To reproduce the result, enter `code/`, install the dependencies in `requirements-inference.txt`, and run `python benchmark.py --split test`.