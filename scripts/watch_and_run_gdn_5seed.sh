#!/usr/bin/env bash
# Watcher: wait until the 4 GPUs are free of the external user's load, then
# launch GDN G-DeltaUQ seeds 1,2,3,100 (one per GPU). seed42 already exists as
# the canonical reference and is reused.
#
# "Free" = fewer than 5 external (quinn-torch-env) python procs AND every GPU
# has >= 35 GB free. Re-checks every 2 min. Safe to leave running for hours.
set +e
cd /mnt/datassd3/rashinda/CF_Uncertainity_for_STGNN

free_enough() {
  # external procs cleared?
  local ext
  ext=$(ps -eo cmd 2>/dev/null | grep -v grep | grep -c quinn-torch-env)
  [ "$ext" -ge 5 ] && return 1
  # every GPU has >= 35000 MiB free?
  local freem
  while read -r f; do
    [ "$f" -lt 35000 ] && return 1
  done < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
  return 0
}

echo "[watch] $(date) waiting for GPUs to free up..."
while ! free_enough; do
  sleep 120
done
echo "[watch] $(date) GPUs free -> launching GDN 5-seed"

declare -A GPU=( [1]=0 [2]=1 [3]=2 [42]=3 [100]=3 )
for S in 1 2 3 42 100; do
  tmux new-session -d -s gdn_s$S "bash scripts/run_gdn_one_seed.sh $S ${GPU[$S]} > /tmp/gdn_seed$S.log 2>&1; exec bash"
  echo "[watch] launched gdn_s$S on GPU ${GPU[$S]}"
  sleep 5
done
echo "[watch] $(date) all 5 GDN seeds launched in tmux gdn_s1/s2/s3/s42/s100"
echo "GDN_5SEED_LAUNCHED"
