#!/bin/bash

#$-l rt_G.large=1
#$-l h_rt=36:00:00
#$-j y
#$-cwd
# ジョブ名
#$-N job_name
#$-m a
#$-m b
#$-m e
#$-v GPU_COMPUTE_MODE=1

# 標準出力先
#$-o logs/stdout.txt

# qrsh -g gcc50441 -l rt_G.small=1 -l rt_F=1 -l h_rt=1:00:00 

source /etc/profile.d/modules.sh
module load cuda/11.5/11.5.2
module load cudnn/8.3/8.3.3
module load gcc/11.2.0
module load python/3.8/3.8.13

echo "python version is"
python3 -V

pip3 install -r requirements.txt

cd /home/acd14210nv/shinra-bert/code

echo "working directory is {$PWD}"
python3 train.py --config_file abci.yml