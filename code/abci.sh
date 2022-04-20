#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=36:00:00
#$ -j y
#$ -cwd
# ジョブ名
#$ -N job_name
#$ -m a
#$ -m b
#$ -m e
# 標準出力先
#$ -o logs/stdout.txt

source /etc/profile.d/modules.sh
module load cuda/11.0/11.0.3
module load cudnn/8.2/8.2.4
module load gcc/11.2.0
module load python/3.7/3.7.13

cd /home/acd14210nv/shinra-bert/code

# CUDA_VISIBLE_DEVICES=0 taskset -c 0-19 train.py --debug --data_path /data/trial_en/trial_en/en/ --file_label_name en_ENEW_LIST.json --file_data_name en-trial-wiki-20190121-cirrussearch-content.json.gz


qsub -g gcc50441 -l rt_F train.py --debug --data_path /data/trial_en/trial_en/en/ --file_label_name en_ENEW_LIST.json --file_data_name en-trial-wiki-20190121-cirrussearch-content.json.gz
