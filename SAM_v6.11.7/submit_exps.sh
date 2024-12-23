#!/bin/bash
group=2
for wn in {6..7}; do
	cd RCE_noisywave_wn${wn}
	sleep 5
	job_script="resub_$group.ens"
	job_id=$(sbatch "$job_script" | awk '{print $4}')
	sleep 5

	echo "Waiting for job $job_id to start..."
        while true; do
            job_status=$(squeue -j "$job_id" -h -o "%T")  # Get job status
            if [[ "$job_status" == "RUNNING" ]]; then
                echo "Job $job_id is now running."
                break
            fi
            sleep 5  # Check status every 5 seconds
        done
	sleep 30

	cd ..
	sleep 5
done
