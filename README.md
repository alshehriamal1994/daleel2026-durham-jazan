# Durham-Jazan at Daleel 2026

Code and LLM-generated synthetic data for the Durham-Jazan system description paper at the
**Daleel 2026 shared task on Arabic Argumentative Discourse Mining** (ArabicNLP 2026, co-located with EMNLP 2026).

**Paper:** *Durham-Jazan at Daleel 2026: Domain-Routed Ensembles, LLM Augmentation, and the
Limits of Decision-Layer Tuning for Arabic Argument Mining* — Amal Saad Alshehri, Nelly Bencomo,
Amir Atapour-Abarghouei.

## Final official results

| Track | Score | Rank |
|---|---|---|
| Task 2 Closed | 0.729 overlap-F1 | 6th |
| Task 2 Open | 0.736 overlap-F1 | 2nd |
| Task 1 Closed | 0.657 macro-F1 | 5th |
| Task 1 Open | 0.669 macro-F1 | 2nd |

All four development-phase leaderboards: 1st.

## Setup

```bash
pip install -r requirements.txt
```

The official task data and scoring scripts are **not** redistributed here — obtain them from the
organizers (https://qatardebate.org/programs/academic-programs/daleel2026-shared-task and the
task CodaBench pages) and place `train_task_1.jsonl`, `train_task_2.jsonl`, `dev_in.jsonl`, etc.
under `data/`. Scripts expect the working-directory layout used in the paper
(`data/`, `models/`, `oof/`, `preds/`); adjust the path constants at the top of each script.

## Synthetic data (`data/`)

Synthetic training data generated with a proprietary LLM, used **in training only**; thresholds and
all held-out evaluation use real data exclusively. Task 2 paragraphs were authored as ordered labeled
segments and joined with single spaces, so character offsets are exact by construction
(*segment-concatenation generation*; paper §3.3).

| File | Task | Contents |
|---|---|---|
| `synth_all.jsonl` | 1 | 171 multi-label paragraphs (development phase) |
| `synth_v2/` | 1, 2 | evaluation-phase additions (Task 1 → 291 total; Task 2 segment batches) |
| `synth2_all.jsonl` | 2 | 88 span-annotated paragraphs (development phase) |
| `synth2_v2_built.jsonl` | 2 | 158 span-annotated paragraphs, offsets verified (evaluation phase) |

## Reproducing the final systems (`src/`)

| System | Key scripts |
|---|---|
| Task 2 Closed (0.729) | `train_task2_cv.py` → `ensemble_task2_camel.py` / `ensemble_task2_marbert.py` → `test_task2_closed_v3.py` (domain-routed 8+8 seeds) + span compaction (grid in `t2_closed_recal.py`) |
| Task 2 Open (0.736) | above + synthetic spans (`predict_task2_open.py`) + cross-family char-level blend (`t2_blend_build.py`) |
| Task 1 Closed (0.657) | `dapt.py` / `dapt_v2.py` (domain-adaptive pretraining), `backtranslate.py` + `rare_bt_gen.py` (augmentation), `dapt_rare_submit.py` (encoder ensemble), `t1_llm_sft.py` (Qwen3-32B QLoRA; editorial routing) |
| Task 1 Open (0.669) | above + synthetic paragraphs (`open_submit.py`) + Task-2→Task-1 span fusion |

Validation utilities: `clean_holdout.py`, `model_search.py`, `dapt_eval.py`, `rare_aug_eval.py`, `ens_check.py`.

`paper_analysis/` contains the scripts behind the paper's transfer plot (Fig. 2), per-class table,
bootstrap analyses, and Arabic example figures. `PROMPTS.md` documents the Qwen3 fine-tuning
prompt and the synthetic-data generation templates (including the segment-concatenation format).

## Hardware

RTX 4080 SUPER 16 GB (all encoder work, LLMs ≤14B); one A100 80 GB via SLURM for Qwen3-32B
(~6 GPU-hours).

## License

Code: MIT. Synthetic data: CC-BY-4.0. The official task data belongs to the Daleel 2026
organizers and is not included.

## Citation

```bibtex
@inproceedings{durham-jazan-daleel2026,
  title  = "Durham-Jazan at Daleel 2026: Domain-Routed Ensembles, LLM Augmentation, and the Limits of Decision-Layer Tuning for Arabic Argument Mining",
  author = "Alshehri, Amal Saad and Bencomo, Nelly and Atapour-Abarghouei, Amir",
  booktitle = "Proceedings of ArabicNLP 2026",
  year   = "2026"
}
```
