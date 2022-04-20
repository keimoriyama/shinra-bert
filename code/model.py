from transformers import BertModel, BertConfig, BertTokenizer
import torch.nn as nn
import torch
import pytorch_lightning as pl


class MyBertSequenceClassification(pl.LightningModule):
    def __init__(self, cfg, class_num, criterion) -> None:
        super().__init__()
        self.model = BertModel(cfg)
        self.in_features = self.model.pooler.dense.out_features
        self.out_features = class_num
        self.linear = nn.Linear(self.in_features, self.out_features)
        self.criterion = criterion

    def forward(self, x):
        out = self.model(**x)[1]
        out = self.linear(out)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        return optimizer

    def training_step(self, batch):
        text, label = batch
        output = self(text)
        loss = self.criterion(output, label)
        self.log("train_loss", loss.item())
        return loss

    def validation_step(self, batch, _):
        text, label = batch
        # print(batch, sample)
        output = self(text)
        with torch.no_grad():
            loss = self.criterion(output, label)
        self.log("validation loss", loss.item())
        return loss

    def test_step(self, batch, _):
        text, label = batch
        with torch.no_grad():
            output = self(text)
        loss = self.criterion(output, label)
        pred = torch.argmax(output, dim=1)
        acc = torch.sum(pred == label).item() / len(pred)
        self.log_dict({"test loss": loss.item(), "test acc": acc})


def main():
    bert_version = "bert-base-cased"
    cfg = BertConfig.from_pretrained(bert_version)
    bert = BertModel.from_pretrained(bert_version)
    class_num = 10
    model = MyBertSequenceClassification(bert, class_num)
    print(model)
    tokenizer = BertTokenizer.from_pretrained(bert_version)
    text = ["I have a pen", "I like a cat"]
    text = tokenizer(text, return_tensors="pt")
    print(text)
    loss = model(text)
    print("loss", loss.size())


if __name__ == '__main__':
    main()
