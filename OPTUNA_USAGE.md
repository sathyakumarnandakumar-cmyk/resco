# Optuna Usage Guide

## Results Analysis

**Current Study Status:**
- Study name: `resco_optuna`
- Total trials attempted: 7
- Successful: 0 ❌
- Failed: 6 (SUMO/TraCI parallel conflicts)
- Interrupted: 1 (Trial #6)

**Why All Failed:**
All trials failed due to SUMO/TraCI conflicts when running with `n_jobs > 1`. The failures occurred during simulation, not during hyperparameter selection, so **no performance data was collected**.

**Parameters Explored (but not evaluated):**
```
Trial 0: BATCH=32,  GAMMA=0.95, EPS_END=0.0330, DECAY=500, TARGET=2000
Trial 1: BATCH=32,  GAMMA=0.99, EPS_END=0.0389, DECAY=500, TARGET=4500
Trial 2: BATCH=256, GAMMA=0.98, EPS_END=0.0192, DECAY=500, TARGET=5000
Trial 3: BATCH=128, GAMMA=0.9,  EPS_END=0.0040, DECAY=500, TARGET=4000
Trial 4: BATCH=32,  GAMMA=0.95, EPS_END=0.0133, DECAY=500, TARGET=4500
Trial 5: BATCH=64,  GAMMA=0.9,  EPS_END=0.0256, DECAY=220, TARGET=3500
Trial 6: BATCH=256, GAMMA=0.95, EPS_END=0.0415, DECAY=100, TARGET=5000 (interrupted)
```

**Conclusion:** No good parameters found yet - need to run with `n_jobs=1` to collect actual performance data.

---

## How to Resume vs Start New

### Option 1: RESUME Existing Study ✅ Recommended if you want to continue

Keeps the existing database and continues from where it left off. Optuna will:
- Skip the 6 failed trials (they remain in history)
- Mark interrupted Trial #6 as failed
- Start new trials from #7 onwards

```bash
cd /home/sathya/resco-for-malaysia/resco_benchmark
source ../.venv/bin/activate

# Resume with sequential execution (stable)
python main-o.py \
    --agent IDQN \
    --net mlp \
    --n_trials 20 \
    --n_jobs 1 \
    --seed 42 \
    --study_name resco_optuna

# The existing database will be loaded automatically
# It will run trials 7-26 (20 new trials)
```

**Pros:**
- Preserves history of what was attempted
- Optuna's sampler learns from all trials (including failed ones)
- Can analyze what parameter combinations were tried

**Cons:**
- Database contains failed trial clutter
- Can't easily distinguish "failed due to bug" vs "failed due to bad params"

---

### Option 2: START NEW (Clean Slate) ✅ Recommended for fresh start

Delete old database and start completely fresh:

```bash
cd /home/sathya/resco-for-malaysia/resco_benchmark
source ../.venv/bin/activate

# Delete old database
rm optuna_resco.db

# Start fresh study
python main-o.py \
    --agent IDQN \
    --net mlp \
    --n_trials 20 \
    --n_jobs 1 \
    --seed 42 \
    --study_name resco_optuna_v2

# This will create optuna_resco.db with trials 0-19
```

**Pros:**
- Clean database with only valid results
- Easy to analyze - no failed trials
- Fresh start with lessons learned

**Cons:**
- Loses record of what was tried before

---

### Option 3: START NEW STUDY (Different Name) - Parallel Studies

Keep old database, create new study with different name:

```bash
cd /home/sathya/resco-for-malaysia/resco_benchmark
source ../.venv/bin/activate

python main-o.py \
    --agent IDQN \
    --net mlp \
    --n_trials 20 \
    --n_jobs 1 \
    --seed 42 \
    --study_name resco_optuna_sequential \
    --storage sqlite:///optuna_sequential.db

# Creates new database: optuna_sequential.db
```

**Pros:**
- Both studies exist side-by-side
- Can compare different approaches
- Old data preserved for reference

---

## Recommended: Start Fresh with Sequential

Since all previous trials failed without data, start clean:

```bash
cd /home/sathya/resco-for-malaysia/resco_benchmark
source ../.venv/bin/activate

# Clean up old failed study
rm optuna_resco.db

# Start new sequential study (STABLE)
python main-o.py \
    --agent IDQN \
    --net mlp \
    --n_trials 20 \
    --n_jobs 1 \
    --seed 42 \
    2>&1 | tee optuna_sequential_run.log &

# View dashboard (separate terminal)
source ../.venv/bin/activate
optuna-dashboard sqlite:///optuna_resco.db
```

---

## Monitor Progress

### Check study status:
```bash
cd /home/sathya/resco-for-malaysia/resco_benchmark
source ../.venv/bin/activate

python -c "
import optuna
study = optuna.load_study(study_name='resco_optuna', storage='sqlite:///optuna_resco.db')
completed = [t for t in study.trials if t.state.name=='COMPLETE']
print(f'Completed: {len(completed)}/{len(study.trials)}')
if completed:
    best = study.best_trial
    print(f'Best avg_delay: {best.value:.4f}')
    print(f'Best params: {best.params}')
"
```

### Watch log file:
```bash
tail -f optuna_run.log
```

### View in browser:
```bash
optuna-dashboard sqlite:///optuna_resco.db
# Open: http://127.0.0.1:8080
```

---

## Tips for Success

1. **Always use `n_jobs=1`** - SUMO/TraCI doesn't support parallel execution in same process
2. **Run in tmux/screen** - Trials take hours, protect against disconnection
3. **Each trial = 60 episodes** - Plan for long runtime (hours per trial)
4. **Sequential is stable** - Python/SUMO/Neptune still use multiple threads internally
5. **Best models auto-saved** to `/home/sathya/resco-for-malaysia/models/optuna/`
