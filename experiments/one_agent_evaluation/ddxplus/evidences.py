import json

# 加载 evidences 文件
# 请修改路径
with open(
    "/mnt/e/project/LLM/baseline/ddxplus/data/release_evidences.json",
    "r",
    encoding="utf‑8",
) as f:
    evidences_def = json.load(f)

def decode_evidence(eid):
    if "_@_" in eid:
        ecode, vcode = eid.split("_@_")
    else:
        ecode = eid
        vcode = None

    if ecode not in evidences_def:
        return f"[Unknown evidence {eid}]"

    entry = evidences_def[ecode]
    question = entry.get("question_en", entry.get("question_fr", f"Evidence {ecode}"))

    if vcode:
        # find value meaning
        vm = entry.get("value_meaning", {}).get(vcode)
        if vm:
            # pick english if exists
            val_text = vm.get("en", vm.get("fr", vcode))
        else:
            val_text = vcode
        return f"{question} Value: {val_text}."
    else:
        # binary evidence
        return f"{question}"
