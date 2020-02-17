#!/bin/sh
#SBATCH --job-name=job_test    		# Job name (any name)
#SBATCH --ntasks=1                  	# Run on a single CPU
#SBATCH --time=02:00:00             	# Time limit hrs:min:sec
#SBATCH --output=docker_output_%j.out   # Standard output and error log name format
#SBATCH --cpus-per-task=1		# CPUs per task
#SBATCH --gres=gpu:1			# Request number of GPUs (1,2,3,4).
#SBATCH --mem=32GB			# Requested memory (you can opt for smaller memory if dataset is small, so that rest of the memory will be allocated to others)

nvidia-docker run -t ${USE_TTY}  --name $SLURM_JOB_ID --user $(id -u $USER):$(id -g $USER) --rm -v /home/mohit/diabetic-retinopathy-detection/:/workspace/ -v /etc/passwd:/etc/passwd -v /etc/group:/etc/group -v /etc/shadow:/etc/shadow tensorflow_keras:latest python network.py
