#!/bin/bash
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -t 20:00:00
#SBATCH -p gpu2
#SBATCH --gres=gpu:1
#SBATCH -A loni_depedlab11

module load cuda

cd /home/xzha135/work/projects_ws/DAC/STAFI

#./test_gradient_all_tensors.sh
./test_rank_bits_from_previous_json.sh op_097/weights_candidates_097_top500.json
