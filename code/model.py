from transformers import BertModel, BertConfig, BertTokenizer
import torch.nn as nn
import torch
import pytorch_lightning as pl
from sklearn.metrics import f1_score
import ipdb


class MyBertSequenceClassification(pl.LightningModule):
    def __init__(self, cfg, class_num, criterion, lr) -> None:
        super().__init__()
        self.model = BertModel(cfg)
        self.in_features = self.model.pooler.dense.out_features
        self.out_features = class_num
        self.linear = nn.Linear(self.in_features, self.out_features)
        self.criterion = criterion
        self.lr = lr

    def forward(self, x):
        out = self.model(**x)[1]
        out = self.linear(out)
        return out
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, batch, _):
        text, label = batch
        output = self(text)
        loss = self.criterion(output, label)
        # self.log("train loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def training_epoch_end(self, train_loss):
        train_loss = [i['loss'].item() for i in train_loss]
        loss = sum(train_loss)/len(train_loss)
        self.log("train loss", loss)

    def validation_step(self, batch, _):
        text, label = batch
        output = self(text)
        loss = self.criterion(output, label)
        # self.log("validation loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_epoch_end(self, val_loss):
        #ipdb.set_trace()
        val_loss = [i.item() for i in val_loss]
        loss = sum(val_loss)/len(val_loss)
        self.log("validation loss", loss)
    
    def test_step(self, batch, _):
        text,label = batch
        output = self(text)
        loss = self.criterion(output, label)
        pred = torch.argmax(output, dim=1)
        acc = torch.sum(pred == label).item() / len(pred)
        label = label.to('cpu').detach().numpy().copy()
        pred = pred.to('cpu').detach().numpy().copy()
        microf1 = f1_score(label, pred, average='micro')
        self.log("testing loss", loss,prog_bar=True, on_epoch=True)
        self.log("accuracy", acc,prog_bar=True, on_epoch=True)
        self.log("micro f1", microf1, prog_bar = True, on_epoch=True)
        
def main():
    y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_pred = [0, 1, 1, 1, 1, 0, 0, 0, 1, 1]
    print(f1_score(y_true, y_pred, average = "micro"))



if __name__ == '__main__':
    main()
