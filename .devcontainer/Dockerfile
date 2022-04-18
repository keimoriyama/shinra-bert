FROM pytorch/pytorch


RUN pip3 install \
    Janome==0.4.1 \
    subword-nmt==0.3.7 \
    hydra-core==1.0 \
    transformers==3.0.1 \
    tensorboard==2.4.1 \
    tensorboardX==2.2 \
    scikit-learn==0.24.1 \
    tqdm==4.41.1 \
    autopep8 \
    flake8 \
    mlflow \
    pytorch-lightning

RUN apt-get update && apt-get install -y build-essential git