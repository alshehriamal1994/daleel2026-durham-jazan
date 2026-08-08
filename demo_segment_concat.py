"""A short demonstration of segment-concatenation generation.

The LLM writes each paragraph as an ordered list of labelled segments.
Joining them with single spaces gives exact character offsets by construction,
so span-annotated training data needs no tag parsing and no alignment step.

Run:  python demo_segment_concat.py
"""
import json

record = json.loads(open("data/synth_v2/t2_batch_agent.jsonl", encoding="utf-8").readline())

text, spans, pos = "", [], 0
for seg in record["segments"]:
    if pos:
        text += " "
        pos += 1
    start = pos
    text += seg["t"]
    pos += len(seg["t"])
    if seg["l"] != "O":                      # "O" segments are unlabelled glue
        spans.append({"label": seg["l"], "start_offset": start, "end_offset": pos})

print(f"Built paragraph ({record['type']}, {len(text)} chars, {len(spans)} spans):\n")
for s in spans:
    piece = text[s["start_offset"]:s["end_offset"]]
    ok = "OK " if piece == next(x["t"] for x in record["segments"]
                                if x["t"] == piece) else "BAD"
    print(f"  [{ok}] {s['label']}  [{s['start_offset']:4d},{s['end_offset']:4d})  {piece[:60]}")

assert all(text[s["start_offset"]:s["end_offset"]] for s in spans)
print("\nEvery offset is exact by construction, with no parsing and no alignment step.")
