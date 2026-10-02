r"""Read-only suite bar for the 4v4 dual-branch pipeline (same shape as 2v2).

Does not start or touch training. Newline bars so Get-Content -Wait shows them.

    Get-Content 4v4\suite_tqdm.log -Wait -Tail 1
"""
from __future__ import annotations

import re
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
SCALE = "4v4"
LOG = PROJ / SCALE / "suite_tqdm.log"
MANIFESTS = PROJ / SCALE / "manifests"
SUITE_LOG = PROJ / SCALE / "dual_branch_4v4.log"
WAIT_2V2 = PROJ / "2v2" / "manifests" / "phase6_package.json"
TRAIN = {
    "A": PROJ / SCALE / "dual_branch_train_A.log.err",
    "B": PROJ / SCALE / "dual_branch_train_B.log.err",
}
SMOKE = {
    "A": PROJ / SCALE / "dual_branch_smoke_A.log.err",
    "B": PROJ / SCALE / "dual_branch_smoke_B.log.err",
}
FINAL = {
    "A": PROJ / "artifacts/scale_4v4_specialists/pi_A_specialist_4v4_dual_branch_v1/ckpts/final_pi_A_specialist_4v4_dual_branch_v1.zip",
    "B": PROJ / "artifacts/scale_4v4_specialists/pi_B_specialist_4v4_dual_branch_v1/ckpts/final_pi_B_specialist_4v4_dual_branch_v1.zip",
}
SD = PROJ / "artifacts/strategic_demand/sppo"
STUDENTS = (
    SD / "suite_sharing_std/4v4_stage4/share_encoder/STUDENT_FROZEN.json",
    SD / "suite_sharing_std/4v4_stage4/fully_shared_z_r/STUDENT_FROZEN.json",
    SD / "suite_sharing_std/4v4_stage4/role_only/STUDENT_FROZEN.json",
)
EVALS = (
    "TOP50_4V4_STAGE4_SHARE_ENCODER_CROSSOVER_EVAL_RESULT.json",
    "TOP50_4V4_STAGE4_FULLY_SHARED_ZR_CROSSOVER_EVAL_RESULT.json",
    "TOP50_4V4_STAGE4_ROLE_ONLY_CROSSOVER_EVAL_RESULT.json",
)
PHASES = (
    ("wait_2v2", 0),
    ("smoke_A", 5),
    ("smoke_B", 5),
    ("train_A", 200),
    ("train_B", 200),
    ("export_seal", 8),
    ("diagnostic", 40),
    ("dataset", 40),
    ("students", 90),
    ("student_evals", 60),
    ("package", 2),
)
TOTAL = sum(w for _, w in PHASES)  # 650, same as 2v2 work phases
STEP_RE = re.compile(r"(\d+)\s*/\s*200000")
PPO_RE = re.compile(r"PPO:\s+\d+%\|[^\n\r]*\|\s*\d+/200000\s+\[[^\]]+\]")


def read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16", "replace")
    return raw.decode("utf-8", "replace")


def tail(path: Path, n: int = 16000) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as fh:
        fh.seek(0, 2)
        fh.seek(max(0, fh.tell() - n))
        raw = fh.read()
    if b"\x00" in raw[:80]:
        return raw.decode("utf-16", "replace")
    return raw.decode("utf-8", "replace")


def clean(s: str) -> str:
    return " ".join(s.replace("\r", " ").split())


def ppo_progress(pol: str) -> tuple[float, str]:
    if FINAL[pol].is_file():
        return 1.0, f"PPO: 100%|{'#' * 40}| 200000/200000 done"
    text = tail(TRAIN[pol]).replace("\r", "\n")
    hits = STEP_RE.findall(text)
    if not hits:
        return 0.0, f"PPO {pol} waiting"
    step = min(200_000, int(hits[-1]))
    bars = PPO_RE.findall(text)
    detail = clean(bars[-1] if bars else f"PPO: {step}/200000")
    return step / 200_000, detail


def smoke_done(pol: str) -> bool:
    if (MANIFESTS / f"phase1_smoke_{pol}.json").is_file():
        return True
    suite = read_text(SUITE_LOG)
    if f"{pol} smoke PASS" in suite:
        return True
    return FINAL[pol].is_file() or TRAIN[pol].is_file()


def frac(name: str) -> tuple[float, str]:
    if name == "wait_2v2":
        if WAIT_2V2.is_file():
            return 1.0, "2v2 suite done"
        return 0.0, "waiting for 2v2/manifests/phase6_package.json"
    if name == "smoke_A":
        return (1.0, "smoke A done") if smoke_done("A") else (0.0, "smoke A")
    if name == "smoke_B":
        return (1.0, "smoke B done") if smoke_done("B") else (0.0, "smoke B")
    if name == "train_A":
        return ppo_progress("A")
    if name == "train_B":
        return ppo_progress("B")
    if name == "export_seal":
        exp = (MANIFESTS / "phase1_export.json").is_file()
        seal = (MANIFESTS / "phase1_technical_seal.json").is_file()
        return ((0.5 if exp else 0.0) + (0.5 if seal else 0.0),
                "sealed" if seal else ("export" if exp else "export+seal"))
    if name == "diagnostic":
        if (MANIFESTS / "phase2_diagnostic.json").is_file() or (
            MANIFESTS / "phase2_historical_top50_SKIPPED.json"
        ).is_file():
            return 1.0, "diagnostic done/skipped"
        if (MANIFESTS / "matched128.json").is_file():
            return 1.0, "matched128 done"
        text = tail(PROJ / SCALE / "dual_branch_matched128.log.err").replace("\r", "\n")
        hits = re.findall(r"(\d+)\s*/\s*(\d+)", text)
        if hits:
            a, b = int(hits[-1][0]), max(1, int(hits[-1][1]))
            return min(1.0, a / b), f"matched128 {a}/{b}"
        text = tail(PROJ / SCALE / "dual_branch_eval.log.err").replace("\r", "\n")
        hits = re.findall(r"(\d+)\s*/\s*(\d+)", text)
        if hits:
            a, b = int(hits[-1][0]), max(1, int(hits[-1][1]))
            return min(1.0, a / b), f"diagnostic {a}/{b}"
        return 0.0, "matched128/diagnostic"
    if name == "dataset":
        if (MANIFESTS / "phase3_dataset.json").is_file():
            return 1.0, "dataset done"
        text = tail(PROJ / SCALE / "dual_branch_stage4_collect.log.err").replace("\r", "\n")
        hits = re.findall(r"(\d+)\s*/\s*(\d+)", text)
        if hits:
            a, b = int(hits[-1][0]), max(1, int(hits[-1][1]))
            return min(1.0, a / b), f"dataset {a}/{b}"
        return 0.0, "dataset"
    if name == "students":
        n = sum(1 for p in STUDENTS if p.is_file())
        return n / 3, f"students {n}/3"
    if name == "student_evals":
        n = sum(1 for name_ in EVALS if (SD / name_).is_file())
        if (MANIFESTS / "phase5_evals.json").is_file():
            return 1.0, "student evals done"
        return n / 3, f"student evals {n}/3"
    if name == "package":
        return (1.0, "zip done") if (MANIFESTS / "phase6_package.json").is_file() else (0.0, "package")
    return 0.0, name


def bar(done: float) -> str:
    width = 40
    filled = max(0, min(width, int(round(width * done / TOTAL))))
    pct = 100.0 * done / TOTAL
    return f"SUITE {SCALE}: {pct:5.1f}%|{'#' * filled}{'-' * (width - filled)}| {done:.0f}/{TOTAL}"


def current_detail(parts: list[tuple[str, float, str]]) -> str:
    active = [p for p in parts if 0.0 < p[1] < 1.0]
    if active:
        name, _f, detail = active[-1]
        return f"{name}  {detail}"
    for name, f, detail in parts:
        if f < 1.0:
            return f"{name}  {detail}"
    return "suite complete"


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    while True:
        parts = []
        done = 0.0
        for name, weight in PHASES:
            f, detail = frac(name)
            f = max(0.0, min(1.0, f))
            parts.append((name, f, detail))
            done += weight * f
        line = clean(f"{bar(done)}  {current_detail(parts)}")
        print(line, flush=True)
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        if (MANIFESTS / "phase6_package.json").is_file():
            break
        time.sleep(5)


if __name__ == "__main__":
    main()
