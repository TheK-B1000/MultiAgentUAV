# Pyquaticus 2v2 training (RLlib PPO)

Run all commands from the **project root** (the repo that contains `research_pyquaticus/`, `pyquaticus/`, and `AICTFProject/`). Use the env where Pyquaticus and Ray are installed (e.g. `env-full`).

---

## 2v2 training commands

### 1. Curriculum + league (default: ours vs all baselines)

Blue = **our PPO learner**. Red = curriculum (OP1 → OP2 → OP3), then **league**: mix of scripted OP3, species (rusher/camper/balanced), and **up to 5 snapshot opponents** (past checkpoints). Best for strong, diverse opponents.

```bash
python research_pyquaticus/run_experiment.py
```

Or explicitly:

```bash
python research_pyquaticus/run_experiment.py --config research_pyquaticus/configs/aquaticus_2v2.yaml
```

- **Config:** `research_pyquaticus/configs/aquaticus_2v2.yaml`  
- **Mode:** `CURRICULUM_LEAGUE`  
- **Opponents:** OP1 → OP2 → OP3 (curriculum), then OP3 + species + snapshots (league, after passing OP3 gate).

---

### 2. Curriculum only, no league (ours vs OP1 / OP2 / OP3 only)

Blue = **our PPO learner**. Red = scripted only: OP1, then OP2, then OP3. No snapshots, no species. Good baseline and ablations.

```bash
python research_pyquaticus/run_experiment.py --config research_pyquaticus/configs/aquaticus_2v2_no_league.yaml
```

- **Config:** `research_pyquaticus/configs/aquaticus_2v2_no_league.yaml`  
- **Mode:** `CURRICULUM_NO_LEAGUE`  
- **Opponents:** Scripted OP1 → OP2 → OP3 only.

---

### 3. Self-play (ours vs past snapshots of ourselves)

Blue = **our PPO learner**. Red = **past snapshot(s)** of the learner (up to 5 kept), with fallback to scripted OP3 when no snapshots exist yet.

```bash
python research_pyquaticus/run_experiment.py --config research_pyquaticus/configs/aquaticus_2v2_self_play.yaml
```

- **Config:** `research_pyquaticus/configs/aquaticus_2v2_self_play.yaml`  
- **Mode:** `SELF_PLAY`  
- **Opponents:** Snapshot policies (frozen past checkpoints), else OP3.

---

## After training

- **Results:** `research_pyquaticus/results/<exp_name>_<timestamp>/`
  - `checkpoints/` – RLlib checkpoints
  - `final_checkpoint.txt` – path to last checkpoint
  - `eval/` – evaluation summary and CSVs
  - `config_used.yaml` – exact config used
- **View in viewer:**  
  `python AICTFProject/ctfviewer.py --checkpoint <path_from_final_checkpoint.txt>`

---

## 4v4 / 8v8

Same pattern with 4v4 or 8v8 configs:

```bash
python research_pyquaticus/run_experiment.py --config research_pyquaticus/configs/aquaticus_4v4_league.yaml
python research_pyquaticus/run_experiment.py --config research_pyquaticus/configs/aquaticus_8v8_league.yaml
```

No-league and self-play configs exist for 4v4/8v8 as well (e.g. `aquaticus_4v4_no_league.yaml`, `aquaticus_4v4_self_play.yaml`).
