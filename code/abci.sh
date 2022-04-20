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
module load cudnn/8.2/8.2.1
module load gcc/9.3.0
module load python/3.7/3.7.10

cd 動かしたいコードのあるパス
 
CUDA_VISIBLE_DEVICES=0 taskset -c 0-19 exec

CUDA_VISIBLE_DEVICES=1 taskset -c 20-39 train.py --debug --data_path /data/trial_en/trial_en/en/ --file_label_name en_ENEW_LIST.json --file_data_name en-trial-wiki-20190121-cirrussearch-content.json.gz


# =====================
function waitfunction() {
  while :
  do
    RET=$(jobs | grep Running | wc -l)
    if [ $RET -eq 0 ]; then
      return
    else
      echo "$RET job left."
      echo "=============="
      jobs
      echo "=============="
      sleep 300
    fi
  done
}
 
# バックグラウンドジョブが消えるまで動かす
waitfunction
