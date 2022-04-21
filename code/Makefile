local:
	python train.py --config_file local.yml

run_abci:
	qsub -g gcc50441 ./abci.sh

abci:
	python train.py --config_file abci.yml

activate:
	source env/bin/activate