from preprocess import preprocess
from model import MyBertSequenceClassification
from transformers import BertConfig, BertTokenizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import argparse
from omegaconf import OmegaConf
import os

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import MLFlowLogger


num_devices = torch.cuda.device_count()
device_ids = list(range(num_devices))

print(f"{num_devices} GPUs available")

bert_version = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(bert_version)

def tokenize_text(text):
    return tokenizer(text,
                    return_tensors='pt',
                    padding='max_length',
                    truncation = True)

class ShinraDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        text = self.data[index][0]
        label = self.data[index][1]
        return text, label

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    texts, labels = list(zip(*batch))
    texts = list(texts)
    label = torch.tensor(labels)
    texts = tokenize_text(texts)
    return texts, label

parser = argparse.ArgumentParser()


def main():
    seed_everything(10)
    parser.add_argument("--config_file")
    args = parser.parse_args()
    config = OmegaConf.load("./config/" + args.config_file)

    debug = config.debug
    data_path = config.data.data_path
    file_label_name = config.data.file_label_name
    file_data_name = config.data.file_data_name
    batch_size = config.data.batch_size
    data_type = config.data.data_type 
    num_workers = int(os.cpu_count()/ num_devices)
    epoch = config.train.epoch
    lr = config.optim.learning_rate
    exp_name = config.exp_name
    if not debug:
        mlf_logger = MLFlowLogger(experiment_name = exp_name)
        mlf_logger.log_hyperparams(config.data)
        mlf_logger.log_hyperparams(config.train)
    cfg = BertConfig.from_pretrained(bert_version)
    data, label_index_dict = preprocess(debug, data_path, file_data_name, file_label_name,bert_version, data_type)

    class_num = max(label_index_dict.keys())
    criterion = torch.nn.CrossEntropyLoss()
    train_data, test_data = train_test_split(data)
    test_data, val_data = train_test_split(test_data)
    train_dataset = ShinraDataset(train_data)
    test_dataset = ShinraDataset(test_data)
    val_dataset = ShinraDataset(val_data)
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  collate_fn=collate_fn,
                                  num_workers=num_wokers,
                                  persistent_workers = True, 
                                  shuffle=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                collate_fn=collate_fn,
                                num_workers=num_workers,
                                persistent_workers = True,
                                shuffle=False)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 collate_fn=collate_fn,
                                 num_workers=num_workers,
                                 persistent_workers = True,
                                 shuffle=False)

    model = MyBertSequenceClassification(cfg, class_num, criterion, lr)
    if debug:
        trainer = Trainer(max_epochs = 1)
        trainer.fit(model, train_dataloader)
    else:
        trainer = Trainer(max_epochs = epoch,
                            accelerator="gpu", 
                            devices = num_devices, 
                            strategy="ddp_find_unused_parameters_false",
                            logger = mlf_logger
                            )
        trainer.fit(model, train_dataloader, val_dataloader)
        trainer.test(model, test_dataloader)
    

if __name__ == '__main__':
    main()
