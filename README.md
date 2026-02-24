# PCGR (Probabilistic Concept Graph Reasoning)

This repo is a **clean, runnable** Python/PyTorch implementation of **Probabilistic Concept Graph Reasoning (PCGR)** for multimodal misinformation detection.

It follows the paper’s main pipeline:

1) **Concept growth (optional)**: propose & filter new concepts (offline heuristic or plug in your MLLM).
2) **Concept probability computation**: map a (text, image) pair to per-concept probabilities.
3) **Concept graph construction**: connect concepts across layers using semantic + Soft-PMI + (optional) NLI.
4) **Top-down hierarchical inference**: aggregate concept probabilities with attention & multiplicative “AND”-like rule.

## Quickstart (demo runs out-of-the-box)

```bash
pip install -r requirements.txt
python scripts/train_demo.py --epochs 2
```

The demo script generates a small synthetic dataset under `data/demo/`
(colored images + text), builds a layered concept graph, trains for a couple epochs,
then prints an explanation chain for a sample.

## Use your own dataset

Prepare a CSV like:

| text | image_path | label |
|------|------------|-------|
| ...  | path/to/img.jpg | 0/1 |

Then run:

```bash
python scripts/train_demo.py --csv path/to/data.csv --image_root . --epochs 5
```

## Notes on faithfulness vs. practicality

- The paper uses an MLLM to generate per-sample concept descriptions *D_i*.
  This implementation keeps that interface, but defaults to using the concept text itself
  as a lightweight stand-in (so it runs without API keys). You can plug in your own
  concept-description generator in `pcgr/growth.py`.
- NLI scoring (DeBERTa MNLI) is optional; disable it with `--no_nli` if you want
  a smaller dependency footprint or faster graph building.

## Project structure

- `pcgr/model.py` — concept probability model
- `pcgr/graph.py` — build layered DAG edges (semantic + SoftPMI + optional NLI)
- `pcgr/explain.py` — extract interpretable reasoning chains
- `pcgr/growth.py` — concept proposal + filtering (paper-style filters + pluggable generator)
- `scripts/train_demo.py` — demo training & inference

## Citation

If you use this code in research, please cite the PCGR paper and mention this repo.

