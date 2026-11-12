"""Rebuild the validation and test conversations from human-reviewed sentence variants.

Addresses the paraphrased, human-reviewed part of "Recommended next steps", item 1, in
report/transcript-classifier-comparison.md. The original split is read and never modified.

Conditions written to data/benchmark_1000_reworded/:
  para        : every evidence-bearing stock sentence replaced by one reviewed variant (tier 1)
  orig_near   : original wording, plus reviewed near-miss sentences (tier 2)
  para_near   : both
  para_curly  : as para, with straight apostrophes set as typographic ones; a punctuation control
Gold labels are carried over unchanged: a tier-1 variant preserves the attribute it replaces, and a
tier-2 near miss for attribute X is only inserted into a conversation whose gold for X is 0.

Usage, from the repository root (standard library only):
  python3 scripts/build_reworded_split.py
  python3 scripts/build_reworded_split.py <split.json> <variants.json> <out_dir>
"""
from __future__ import annotations
import hashlib, json, random, re, sys
from pathlib import Path

SEED = 20260919
NEAR_PER_CONVERSATION = 2
SPLIT_RE = re.compile(r'(?<=[.?!”"])\s+')
LINE_RE = re.compile(r"\[(Agent|Caller)\]: (.*)")

def key_and_slots(sentence: str):
    """Return (normalised key, slots) so numbers and quoted messages can be put back."""
    slots = {}
    s = sentence
    m = re.search(r"\$(\d+)", s)
    if m: slots["N"] = m.group(1); s = s.replace(m.group(0), "$N")
    m = re.search(r"ending in (\d+)", s)
    if m: slots["N"] = m.group(1); s = s.replace(m.group(0), "ending in N")
    m = re.search(r"about (\d+) Mbps", s)
    if m: slots["N"] = m.group(1); s = s.replace(m.group(0), "about N Mbps")
    m = re.fullmatch(r"I keep receiving the message “(.+?)\.”", s)
    if m: slots["X"] = m.group(1); s = "I keep receiving the message “X.”"
    return s, slots

def fill(template: str, slots: dict) -> str:
    out = template
    if "N" in slots: out = re.sub(r"\bN\b", slots["N"], out)
    if "X" in slots: out = out.replace("“X.”", f"“{slots['X']}.”").replace("“X”", f"“{slots['X']}”")
    return out

def parse(conv: str):
    turns = []
    for line in conv.split("\n"):
        role, text = LINE_RE.match(line).groups()
        turns.append([role, [p for p in SPLIT_RE.split(text) if p]])
    return turns

def render(turns) -> str:
    return "\n".join(f"[{role}]: {' '.join(sents)}" for role, sents in turns)

def main():
    root = Path(__file__).resolve().parents[1]
    defaults = [root / "data/benchmark_1000/split.json", root / "data/benchmark_1000_reworded/variants.json",
                root / "data/benchmark_1000_reworded"]
    split_path, variants_path, out_dir = [Path(a) for a in sys.argv[1:4]] if len(sys.argv) == 4 else defaults
    out_dir.mkdir(parents=True, exist_ok=True)
    split = json.loads(split_path.read_text())
    V = json.loads(variants_path.read_text())
    tier1 = {(e["role"], e["original"]): e for e in V["tier1"]}
    tier2 = V["tier2"]
    stats = {"variants_sha256": hashlib.sha256(variants_path.read_bytes()).hexdigest(), "seed": SEED, "conditions": {}}
    for cond in ("para", "orig_near", "para_near", "para_curly"):
        rows_out, n_repl, n_near, n_changed = [], 0, 0, 0
        for r in split["rows"]:
            if r["split"] not in ("validation", "test"):
                continue
            rng = random.Random(f"{SEED}:{r['id']}")  # same draws in every condition
            turns = parse(r["conversation"])
            choices, used = {}, []
            for ti, (role, sents) in enumerate(turns):
                for si, s in enumerate(sents):
                    k, slots = key_and_slots(s)
                    e = tier1.get((role, k))
                    pick = rng.randrange(3)
                    if e is not None:
                        choices[(ti, si)] = fill(e["variants"][pick], slots)
                        used.append(f"{e['id']}.v{pick + 1}")
            eligible = [n for n in tier2 if int(r["truth"][n["near"]]) == 0]
            rng2 = random.Random(f"{SEED}:near:{r['id']}")
            rng2.shuffle(eligible)
            seen, near = set(), []
            for n in eligible:
                if n["near"] in seen: continue
                seen.add(n["near"]); near.append(n)
                if len(near) == NEAR_PER_CONVERSATION: break
            if cond in ("para", "para_near", "para_curly"):
                for (ti, si), text in choices.items():
                    turns[ti][1][si] = text.replace("'", "’") if cond == "para_curly" else text; n_repl += 1
            if cond in ("orig_near", "para_near"):
                caller_turns = [i for i, t in enumerate(turns) if t[0] == "Caller"]
                agent_turns = [i for i, t in enumerate(turns) if t[0] == "Agent"]
                for n in near:
                    if n["role"] == "Caller":
                        ti = caller_turns[min(1, len(caller_turns) - 1)]      # the symptoms turn
                    else:
                        ti = agent_turns[max(0, len(agent_turns) - 2)]        # the agent's last substantive turn
                    turns[ti][1].append(n["sentence"]); n_near += 1
            conv = render(turns)
            n_changed += int(conv != r["conversation"])
            rows_out.append({"id": r["id"], "split": r["split"], "conversation": conv, "truth": r["truth"],
                             "near_misses": [n["id"] for n in near] if cond in ("orig_near", "para_near") else [],
                             "variants_used": used if cond != "orig_near" else []})
        payload = {"metadata": {**split["metadata"], "condition": cond, "source_seed": split["metadata"]["seed"], "rebuild_seed": SEED,
                                "variants_sha256": stats["variants_sha256"]}, "rows": rows_out}
        (out_dir / f"split_{cond}.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
        stats["conditions"][cond] = {"rows": len(rows_out), "conversations_changed": n_changed, "sentences_replaced": n_repl, "near_misses_inserted": n_near}
    (out_dir / "rebuild_stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats, indent=1))

if __name__ == "__main__":
    main()
