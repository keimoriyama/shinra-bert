#!/bin/bash

#$ -S /bin/sh
#$ -P g-liat
#$ -jc gpu-container_g4
#$ -cwd
#$ -e logs/logs.txt
#$ -o logs/stdout.txt
#$ -m b
#$ -m e
#$ -m a
#$ -ac d=nvcr-pytorch-2205

. /fefs/opt/dgx/env_set/nvcr-pytorch-2205.sh

export MY_PROXY_URL="http://10.1.10.1:8080/"
export HTTP_PROXY=$MY_PROXY_URL
export HTTPS_PROXY=$MY_PROXY_URL
export FTP_PROXY=$MY_PROXY_URL
export http_proxy=$MY_PROXY_URL
export https_proxy=$MY_PROXY_URL
export ftp_proxy=$MY_PROXY_URL

pip install -r requirements.txt

echo "python version is"
python -V

echo "working directory is {$PWD}"
python train.py --config_file abci.yml
