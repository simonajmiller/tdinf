#!/usr/bin/env bash
function get_id() {
    if [[ "$1" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo "sbatch failed"
        exit 1
    fi
}

cd /mnt/home/smiller/src/time-domain-gw-inference/review/GW150914_IMRPhenomXPHM/output/250619_output
runid=$(get_id "$(sbatch -t 3-0 -p cca -n 1 -c 128 disBatch /mnt/home/smiller/src/time-domain-gw-inference/review/GW150914_IMRPhenomXPHM/output/250619_output/tasks_run.txt)")
waveid=$(get_id "$(sbatch -t 3-0 --dependency=afterok:$runid -p cca -n 8 -c 128 disBatch /mnt/home/smiller/src/time-domain-gw-inference/review/GW150914_IMRPhenomXPHM/output/250619_output/tasks_waveforms.txt)")
cd -
