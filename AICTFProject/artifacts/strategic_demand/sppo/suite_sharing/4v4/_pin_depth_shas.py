import json
from pathlib import Path
sp = Path(r"k:\MultiAgentUAV\AICTFProject\artifacts\strategic_demand\sppo")
spec_path = sp / "SUITE_SHARING_4V4_CROSSOVER_EVAL_SPEC.json"
spec = json.loads(spec_path.read_text(encoding="utf-8"))
for arm in ("share_backbone", "share_macro"):
    frozen = json.loads((sp / f"suite_sharing/4v4/{arm}/STUDENT_FROZEN.json").read_text(encoding="utf-8"))
    spec["ARMS"][arm]["sha256"] = frozen["sha256"]
    print(f"pinned {arm} {frozen['sha256'][:12]}...")
spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
print("SPEC sha pins written")
