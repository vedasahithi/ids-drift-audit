#!/usr/bin/env bash
# Complete experiment, in the order used. Results: results/final (override with DRIFTIDS_RESULTS).
set -e
cd "$(dirname "$0")/.."
export DRIFTIDS_RESULTS="${DRIFTIDS_RESULTS:-$PWD/results/final}" PYTHONWARNINGS=ignore
J=${JOBS:-22}
python -m unittest discover -s tests
export OMP_NUM_THREADS=1 DRIFTIDS_THREADS=$J
for ds in unsw cic_s1a cic_s1b_ddos cic_s1b_infil ton_s1c; do
  python phase0.py screen --config configs/phase0_$ds.json      # time split, leakage and shortcut checks
done
for ds in cic_s1a ton_s1c; do
  python phase1.py --config configs/phase0_$ds.json --detector lightgbm --windows 10 --reps 30 --jobs $J --interval posterior3
done
for ds in unsw cic_s1a ton_s1c; do
  python phase2a_attribution.py --config configs/phase0_$ds.json --detector lightgbm --null 100 --windows 10 --jobs $J
done
for ds in cic_s1a cic_s1b_ddos ton_s1c cic_s1b_infil; do
  python phase2b_streams.py --config configs/phase0_$ds.json --detector lightgbm --reps 10 --jobs $J
done
for pair in "unsw cic_s1a" "unsw ton_s1c" "cic_s1a ton_s1c" "ton_s1c cic_s1a"; do
  set -- $pair
  python phase2b_streams.py --config configs/phase0_$1.json --deploy-config configs/phase0_$2.json --detector lightgbm --reps 10 --jobs $J
done
export OMP_NUM_THREADS=2 DRIFTIDS_THREADS=2
for ds in cic_s1a ton_s1c; do
  python phase2b_policies.py --config configs/phase0_$ds.json --detector lightgbm --seeds 10 --jobs 10
done
EXTRA="$DRIFTIDS_RESULTS/../final_extra_seeds"
for ds in cic_s1a ton_s1c; do
  DRIFTIDS_POLICIES=dad,periodic DRIFTIDS_RESULTS="$EXTRA" python phase2b_policies.py --config configs/phase0_$ds.json \
    --detector lightgbm --seeds 10 --seed-start 10 --jobs 10 --no-post
  python tools/merge_seeds.py "$DRIFTIDS_RESULTS/phase2b/policies_$ds" "$EXTRA/phase2b/policies_$ds"
done
python analysis_hypotheses.py --s3 cic_s1a ton_s1c --s1 cic_s1a cic_s1b_ddos ton_s1c \
  --s2 unsw_to_cic_s1a unsw_to_ton_s1c cic_s1a_to_ton_s1c ton_s1c_to_cic_s1a --policy cic_s1a ton_s1c \
  --s3-suffix _s3_posterior3_lightgbm --label final
