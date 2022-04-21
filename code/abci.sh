#!/bin/bash

#$-l rt_F=1
#$-l h_rt=36:00:00
#$-j y
#$-cwd
# ジョブ名
#$-N job_name
#$-m a
#$-m b
#$-m e

# 標準出力先
#$-o logs/stdout.txt

source /etc/profile.d/modules.sh
module load cuda/11.0/11.0.3
module load cudnn/8.2/8.2.4
module load gcc/11.2.0
module load python/3.7/3.7.13

echo "python version is"
python3 -V

cd /home/acd14210nv/shinra-bert/code

echo "working directory is {$PWD}"
pip3 install -r requirements.txt
python3 train.py --config_file abci.yml
