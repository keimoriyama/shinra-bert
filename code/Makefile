local:
	python train.py --config_file local.yml

abci_job:
	rm -rf logs/stdout.txt
	qsub -g gcc50441 ./abci.sh

activate:
	source env/bin/activate