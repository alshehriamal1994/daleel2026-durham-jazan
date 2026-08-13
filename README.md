# Durham-Jazan at Daleel 2026

This repository accompanies the Durham-Jazan system description paper for the
**Daleel 2026 shared task on Arabic Argumentative Discourse Mining** (ArabicNLP 2026,
held with EMNLP 2026). It contains our code, our synthetic training data, and the
scripts behind every figure and analysis in the paper.

**Paper:** *Durham-Jazan at Daleel 2026: Domain-Routed Ensembles, LLM Augmentation, and the
Limits of Decision-Rule Tuning for Arabic Argument Mining.* Amal Saad Alshehri, Nelly Bencomo, Housam Babiker,
and Amir Atapour-Abarghouei.

<p align="center"><img src="assets/example.png" width="420" alt="A gold span annotation from the development set"></p>

The central finding of our participation is summarised in one figure. Changes to the model
or its data delivered what validation promised, while tuning of the decision rules on top
did not.

<p align="center"><img src="assets/transfer.png" width="500" alt="Promised and delivered gains for each intervention"></p>

Two further findings come from post-submission analysis. Per-label performance on
editorials tracks how much of that label's training data came from editorials, which
explains the editorial deficit every team suffered, and threshold recalibration gains
measured on the tuning half do not survive on held-out data.

<p align="center"><img src="assets/findings.png" width="760" alt="Genre skew and threshold transfer"></p>

## A short demonstration

The paper introduces *segment-concatenation generation*, a recipe that lets a large
language model produce span-annotated training data with exact character offsets and
no alignment step. The following command builds and verifies one such paragraph from
the data in this repository.

```bash
python demo_segment_concat.py
```

## Final official results

| Track | Score | Rank |
|---|---|---|
| Task 2 Closed | 0.729 overlap-F1 | 6th |
| Task 2 Open | 0.736 overlap-F1 | 2nd |
| Task 1 Closed | 0.657 macro-F1 | 5th |
| Task 1 Open | 0.669 macro-F1 | 2nd |

In the development phase we finished first on both Open leaderboards.

## Setup

```bash
pip install -r requirements.txt
```

The official task data and scoring scripts are not redistributed here. They are available
from the organisers through the task pages at
https://qatardebate.org/programs/academic-programs/daleel2026-shared-task and CodaBench.
Place `train_task_1.jsonl`, `train_task_2.jsonl`, `dev_in.jsonl`, `dev_task_1_ref.jsonl`,
`dev_task_2_ref.jsonl`, and related files under `data/`.

The scripts in `paper_analysis/` resolve their paths against the repository root, so they
can be run from any working directory and will look for their inputs in the `data/`,
`oof/`, and `preds/` layout used in the paper. The training and prediction scripts in
`src/` still carry an absolute path constant at the top (`W=...`) pointing at the
directory tree in which they were originally run, and they import the organisers' scorer
as `task2_scoring`; both need adjusting before those scripts will run elsewhere.

## Synthetic data

The `data/` directory holds the synthetic training data generated with a proprietary
large language model. All of it is machine-generated: the statistics
inside synthetic ST spans are plausible but invented, so the files are unsuitable as
factual text. It was used in training only, and thresholds and all held-out
evaluation used real data exclusively. For Task 2 the paragraphs were authored as ordered
labelled segments and joined with single spaces, so the character offsets are exact by
construction (paper, Section 3.3).

| File | Task | Contents |
|---|---|---|
| `synth_all.jsonl` | 1 | 171 multi-label paragraphs (development phase) |
| `synth_v2/` | 1 and 2 | evaluation-phase additions (Task 1 grows to 291 in total) |
| `synth2_all.jsonl` | 2 | 88 span-annotated paragraphs (development phase) |
| `synth2_v2_built.jsonl` | 2 | 70 further span-annotated paragraphs with verified offsets, giving 158 in total |

The four systems at a glance (solid boxes: Closed track; dashed: Open-track additions):

<p align="center"><img src="assets/pipeline.png" width="760" alt="Pipeline diagram of the four systems"></p>

## Post-submission analyses

The `paper_analysis/` scripts are the code behind the paper's figures and appendix
tables. Run them from anywhere, for example:

```bash
python paper_analysis/threshold_transfer.py
```

Five of them need nothing beyond this repository:

| Script | Reproduces |
|---|---|
| `synth_agreement.py` | the blind re-annotation of the synthetic data: 79.0% exact agreement, micro-F1 0.94, per-label kappa 0.61 (TE) to 1.00 (ST) (Appendix B) |
| `ranking_analysis.py` | which domain decided the shared task, from the organisers' published ranking sheet (Appendix E) |
| `transfer_plot.py` | Figure 4, promised against delivered for all seventeen audited interventions |
| `pipeline_fig.py` | Figure 2, the pipeline diagram |
| `demo_segment_concat.py` | the segment-concatenation demonstration above |

Two more need only the organisers' Task 1 files in `data/`, because the out-of-fold
predictions they run on (`oof/t1_recal_oof_closed.npy`) are included here:

| Script | Reproduces |
|---|---|
| `threshold_transfer.py` | Table 4, the controlled recalibration experiment |
| `findings_fig.py` | Figure 5, both post-submission findings |

The remainder need inputs we cannot redistribute, or a GPU:

| Script | Reproduces | Also needs |
|---|---|---|
| `analysis.py` | Table 2, per-class dev F1, and the bootstrap intervals | our dev predictions under `preds/` |
| `error_taxonomy.py` | Tables 5 and 6, the span and label error taxonomies | `preds/task2_dev_routed.jsonl` |
| `genre_gap.py` | the editorial-share analysis behind Figure 5 (left) | `preds/` dev decodes for both encoder families |
| `make_figures.py` | Figures 1, 3, 6, 7, and 9, the Arabic examples | `preds/`, plus headless Chrome and Ghostscript |
| `mine_interesting.py` | the exploratory scan that located those examples | `preds/` |
| `threshold_transfer_t2.py` | the Task 2 replication of Table 4 | `oof/t2_recal_oof.pkl` and `oof/t2_gold_all.jsonl`, the latter being organisers' gold |
| `ensemble_curve.py` | the seed-variance figures of Appendix A | a GPU; it trains eight seeds |
| `genre_transfer.py` | Table 8, the size-matched genre experiment | a GPU; it trains nine models |

The dev predictions under `preds/` are our own model outputs rather than task data, but
they are keyed to the organisers' paragraph ids and are not included here. Figure output
goes to `figs/`.

## Reproducing the final systems

| System | Key scripts |
|---|---|
| Task 2 Closed (0.729) | `train_task2_cv.py`, then `ensemble_task2_camel.py` and `ensemble_task2_marbert.py`, then `test_task2_closed_v3.py` for the domain-routed ensemble, with span compaction from `t2_closed_recal.py` |
| Task 2 Open (0.736) | the above plus synthetic spans (`predict_task2_open.py`) and the char-level blend (`t2_blend_build.py`) |
| Task 1 Closed (0.657) | `dapt.py` and `dapt_v2.py` for domain-adaptive pretraining, `backtranslate.py` and `rare_bt_gen.py` for augmentation, `dapt_rare_submit.py` for the encoder ensemble, and `t1_llm_sft.py` for the Qwen3-32B routing |
| Task 1 Open (0.669) | the above plus synthetic paragraphs (`open_submit.py`) and span fusion from Task 2 |

The `configs/` directory holds the evaluation-phase decoding thresholds and rule
configurations referenced in the paper's Appendix A: the recalibrated Task 1 sets
(`t1_recal_ths_closed.json`, `t1_recal_ths_open.json`), the routed Task 2 sets
(`task2_camel_ens.json` for editorials, `task2_marbert_deb.json` for debates), the Open
blend (`t2_blend_cfg.json`), and the fusion and per-label rules. The development-phase
Task 1 thresholds quoted in the appendix are the `BASE` constant of
`paper_analysis/threshold_transfer.py` (Closed) and the `THS` constant of
`src/test_task1_open.py` (Open). Validation utilities include `clean_holdout.py`,
`model_search.py`, `dapt_eval.py`, `rare_aug_eval.py`, and `ens_check.py`. `PROMPTS.md`
documents the fine-tuning prompt and the synthetic-data generation templates.

## Hardware

One RTX 4080 SUPER with 16 GB served all encoder work and models of up to 14B parameters.
One A100 with 80 GB ran Qwen3-32B for about six GPU hours.

## Licence

The code is released under the MIT licence and the synthetic data under CC-BY-4.0.
The official task data belongs to the Daleel 2026 organisers and is not included.

## Citation

```bibtex
@inproceedings{durham-jazan-daleel2026,
  title  = "Durham-Jazan at Daleel 2026: Domain-Routed Ensembles, LLM Augmentation, and the Limits of Decision Rule Tuning for Arabic Argument Mining",
  author = "Alshehri, Amal Saad and Bencomo, Nelly and Babiker, Housam and Atapour-Abarghouei, Amir",
  booktitle = "Proceedings of ArabicNLP 2026",
  year   = "2026"
}
```
