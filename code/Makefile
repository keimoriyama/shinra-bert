local:
	python train.py --config_file local.yml

run_abci:
	qsub -g gcc50441 ./abci.sh

abci:
	./abci.sh

activate:
	source env/bin/activate