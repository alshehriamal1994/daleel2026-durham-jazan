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

Taken from the organisers' official results sheet. Ed and Db are the editorial and debate
columns. The entrant count excludes the organisers' baseline.

| Track | Score | Ed | Db | Rank | Winner |
|---|---|---|---|---|---|
| Task 2 Closed | 0.729 overlap-F1 | 0.610 | 0.762 | 6th of 8 | 0.755 |
| Task 2 Open | 0.736 overlap-F1 | 0.627 | 0.766 | 2nd of 2 | 0.755 |
| Task 1 Closed | 0.657 macro-F1 | 0.501 | 0.741 | 5th of 10 | 0.712 |
| Task 1 Open | 0.669 macro-F1 | 0.501 | 0.762 | 2nd of 3 | 0.712 |

In the development phase we finished first on both Open leaderboards. The evaluation-phase
ranks above are the final official standings and include submissions made after ours.

## Setup

```bash
pip install -r requirements.txt
```

The official task data and scoring scripts are not redistributed here. They are available
from the organisers through the task pages at
https://qatardebate.org/programs/academic-programs/daleel2026-shared-task and CodaBench.
Place `train_task_1.jsonl`, `train_task_2.jsonl`, `dev_in.jsonl`, `dev_task_1_ref.jsonl`,
`dev_task_2_ref.jsonl`, and related files under `data/`.

The scripts in `paper_analysis/` and `src/` resolve their paths against the repository
root, so they can be run from any working directory and will look for their inputs in the
`data/`, `oof/`, and `preds/` layout used in the paper. Set `DALEEL_ROOT` if your data
lives elsewhere:

```bash
export DALEEL_ROOT=/path/to/your/data/tree
```

Two further dependencies are on the organisers' Task 2 scorer, which we do not
redistribute. Five scripts in `src/` (`ensemble_task2.py`, `ensemble_task2_camel.py`,
`ensemble_task2_marbert.py`, `t2_open_eval.py`, `train_task2_cv.py`) import it as
`task2_scoring`, so place the organisers' `task2_scoring.py` inside `src/`. Three others
(`t2_blend_build.py`, `t2_blend_oof_exp.py`, `t2_closed_recal.py`) invoke it as a file and
read its location from `DALEEL_SCORER`:

```bash
export DALEEL_SCORER=/path/to/task2_scoring.py
```

### What is included alongside the code

`oof/` contains the model outputs behind the paper's analyses: out-of-fold probabilities
and character offsets, and the saved seed-curve and genre-transfer results. These are our
own numbers and carry no corpus text, so they are released in full.

`preds/` holds our development-set predictions. The Task 1 files are label sets only. The
Task 2 files carry labels and character offsets with the span text removed, because that
text is the organisers' corpus and is theirs to distribute. Removing it changes no result:
every analysis below works from labels and offsets.

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

The four systems at a glance (solid boxes are the Closed track, dashed the Open-track additions):

<p align="center"><img src="assets/pipeline.png" width="760" alt="Pipeline diagram of the four systems"></p>

## Post-submission analyses

The `paper_analysis/` scripts are the code behind the paper's figures and appendix
tables. Run them from anywhere, for example:

```bash
python paper_analysis/threshold_transfer.py
```

Six of them need nothing beyond this repository:

| Script | Reproduces |
|---|---|
| `synth_agreement.py` | the blind re-annotation of the synthetic data: 79.0% exact agreement, micro-F1 0.94, per-label kappa 0.61 (TE) to 1.00 (ST) (Appendix B) |
| `human_eval/score_human_eval.py` | the blind human check of 30 synthetic paragraphs: micro-F1 0.84, kappa 1.00 (ST) and 0.93 (CO), naturalness 2.93 of 3 (Appendix B). The label-hidden sheet, the filled annotation, the key, and `make_sheet_small.py` sit beside it |
| `ranking_analysis.py` | which domain decided the shared task, from the organisers' published ranking sheet (Appendix E) |
| `transfer_plot.py` | Figure 4, promised against delivered for all eighteen audited interventions |
| `pipeline_fig.py` | Figure 2, the pipeline diagram |
| `demo_segment_concat.py` | the segment-concatenation demonstration above |

Three more need only the organisers' Task 1 files in `data/`, because the out-of-fold
predictions they run on (`oof/t1_recal_oof_closed.npy`) are included here:

| Script | Reproduces |
|---|---|
| `threshold_transfer.py` | Table 7, the controlled recalibration experiment |
| `findings_fig.py` | Figure 5, both post-submission findings |
| `union_holdout_check.py` | the post-submission check of the three-seed Qwen3-32B union on the controlled holdout, from the retained seed predictions in `oof/` (Appendix A) |

Four more need the organisers' files in `data/`, together with our dev predictions,
which are now included under `preds/`:

| Script | Reproduces |
|---|---|
| `analysis.py` | Table 2, per-class dev F1, and the bootstrap intervals |
| `error_taxonomy.py` | Tables 8 and 9, the span and label error taxonomies |
| `genre_gap.py` | Table 10 and the editorial-share analysis behind Figure 5 (left) |
| `mine_interesting.py` | the exploratory scan that located the paper's Arabic examples |

The last four need more than this repository provides:

| Script | Reproduces | Also needs |
|---|---|---|
| `make_figures.py` | Figures 1, 3, 6, 7, 8, and 9, the Arabic examples | headless Chrome and Ghostscript |
| `threshold_transfer_t2.py` | the Task 2 replication of Table 7 | `oof/t2_gold_all.jsonl`, which is organisers' gold and must be regenerated |
| `ensemble_curve.py` | the seed-variance figures of Appendix A | a GPU, and trains eight seeds |
| `genre_transfer.py` | Table 11, the size-matched genre experiment | a GPU, and trains nine models |

The dev predictions under `preds/` are our own model outputs rather than task data. The
Task 1 files are label sets. The Task 2 files keep labels and character offsets but not the
span text, which belongs to the organisers' corpus. No analysis here reads that field, and
removing it leaves every reported number unchanged. Figure output goes to `figs/`, and the
Arabic examples to `paper_analysis/`.

## Reproducing the final systems

| System | Key scripts |
|---|---|
| Task 2 Closed (0.729) | `dapt_v2.py` and `dapt_v2_marbert.py` for the encoders, `train_task2_cv.py` with `ensemble_task2_camel.py` and `ensemble_task2_marbert.py` for the per-domain thresholds, then `test_task2_closed_v3.py` for the domain-routed decode, which it writes uncompacted. The span compaction that earned the submitted score is the decoder inside `t2_closed_recal.py` |
| Task 2 Open (0.736) | the same encoders, then `t2_open_recal_apply.py` to add the synthetic spans and save test probabilities, then `t2_blend_build.py` for the char-level blend and the compaction |
| Task 1 Closed (0.657) | `dapt_v2.py` for domain-adaptive pretraining, `backtranslate.py`, `backtranslate_multi.py` and `dev_bt_gen.py` for the back-translations, `t1_recal.py closed` for the encoder ensemble and its recalibrated thresholds, `t1_marbert_route.py` for the debate probability blend, and `t1_llm_sft.py` for one Qwen3-32B seed |
| Task 1 Open (0.669) | the same, with `t1_recal.py open` adding the synthetic paragraphs, and per-class span fusion from Task 2 on the debates |

The two Task 1 entries cannot be reproduced here. Their editorial predictions are the union
of three Qwen3-32B seeds, and `t1_llm_sft.py` trains and predicts one seed at a fixed seed
value, so the seed runs, the union, and the routing step are not released. The Task 1 Open
span fusion is released as rule configurations only. The union rule can still be checked on
the retained holdout predictions with `paper_analysis/union_holdout_check.py`.

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

The code is released under the MIT licence (LICENSE) and the synthetic data under CC-BY-4.0 (LICENSE-DATA). The shared-task data released by the organisers is not redistributed here and remains subject to their terms.
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
