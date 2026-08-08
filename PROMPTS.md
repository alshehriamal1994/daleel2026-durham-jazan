# Prompt Templates

## 1. Qwen3 fine-tuning prompt (Task 1, generative classification)

Used verbatim in `src/t1_llm_sft.py`. The Arabic original is there and an English translation is in the paper appendix.
Structure: task instruction naming the paragraph type (editorial/debate) + the official
annotation-guideline label definitions + output-format constraint + the paragraph.

```
صنّف الفقرة التالية من نوع (مقال افتتاحي | مناظرة) حسب أنواع الأدلة الحجاجية الموجودة فيها.
التعريفات: AS (افتراض): افتراضات أو استنتاجات أو آراء أو أحكام أو ادعاءات تحتاج إلى دعم.
AN (واقعة): دليل عبر تجربة شخصية أو قصة أو حدث واقعي أو مثال ملموس.
ST (إحصائية): دليل كمي أو دراسات أو نسب وأرقام، ولا يُشترط ذكر المصدر.
TE (شهادة): اقتباس أو استشهاد بخبراء أو جهات أو مصادر محددة؛ وفي المناظرات، إعادة ذكر ادعاءات الفريق الخصم تُعد شهادة.
CO (مسلّمة): معرفة متفق عليها أو حقيقة بديهية أو شرح موضوعي لكيفية عمل إجراء ما.
OT (أخرى): لا يسهم إسهاماً حقيقياً في الخطاب الحجاجي (تحيات، تنظيم، انتقالات).
أخرج فقط قائمة الرموز الموجودة مفصولة بفواصل، بدون أي شرح.

الفقرة:
{text}
```

## 2. Synthetic-data generation template (Tasks 1 and 2, proprietary LLM)

Generation was run in batches through LLM agents. Each batch prompt was assembled from the
components below. The batch wording varied by topic and register assignment, and this is the template:

- **Role/task**: write realistic Arabic paragraphs in the style of (news editorial | televised
  debate turn), as training data for argumentative evidence-type classification.
- **Label scheme**: the six official definitions above, verbatim.
- **Few-shot grounding**: three to five real training examples of the target classes (paragraph + gold labels,
  or paragraph + labelled spans for Task 2).
- **Targeting**: instructions to include the starved classes (ST, CO, and editorial OT), with
  realistic invented figures/statistics for ST, and to vary topics widely (economy, health,
  education, environment, technology, sport, law, ...) across both registers.
- **Task 1 output format**: JSONL, `{"text": ..., "labels": [...], "type": "editorial"|"debate"}`,
  multi-label annotations required to be internally consistent with the definitions.
- **Task 2 output format** (*segment-concatenation generation*): the paragraph is authored as an
  **ordered list of labelled segments** `{"type": ..., "segments": [{"t": <text>, "l": <label|O>}]}`.
  Unlabelled connective glue (for example «و») is allowed between spans with label `O`. The builder
  (`src/t2_open_eval.py` / `src/predict_task2_open.py`) joins segments with single spaces, so
  character offsets are exact by construction. Segment counts were steered to match the real
  distribution of about 4.9 spans per paragraph.
- **Validation**: generated batches were de-duplicated, offset-verified (Task 2), and label-sanity
  checked before use, and all synthetic data is used in training only.
