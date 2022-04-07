from transformers import AutoTokenizer, BertModel
import torch.nn as nn


def main():
    pass


class MyBertSequenceClassification(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        tokenizer = AutoTokenizer()
        model = BertModel()


if __name__ == '__main__':
    main()
