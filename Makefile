local:
	python train.py --config_file local.yml

abci_job:
	rm -rf logs/stdout.txt
	qsub -g gcc50441 ./abci.sh

raiden_job:
	rm -rf logs/logs.txt
	rm -rf logs/stdout.txt
	qsub raiden.sh

activate:
	source env/bin/activate