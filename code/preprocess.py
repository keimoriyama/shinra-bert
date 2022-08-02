import os
from transformers import BertTokenizer
from tqdm import tqdm
import json

from data.shinra_utils import FileUtils, label_preprocess, wikidata_preprocess
import pandas as pd


class Tokenizer:
    def __init__(self, bert_version):
        self.tokenizer = BertTokenizer.from_pretrained(bert_version)

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def token2ids(self, text):
        return self.tokenizer(
            text,
            is_split_into_words=True,
            return_tensors="np",
            padding="max_length",
            truncation=True,
        )

    def decode(self, text):
        ids = self.tokenizer.convert_tokens_to_ids(text)
        return self.tokenizer.decode(ids)


def split_list(target, max_len):
    res = [target[i : i + max_len] for i in range(0, len(target), max_len)]
    return res


def extract_label(label_df):
    label_df.drop_duplicates(subset="pageid")
    labels = (
        label_df[~label_df.duplicated(subset="ENE_name")]
        .reset_index()
        .drop(columns=["ENE_id", "title", "index", "pageid"])
    )
    label_index = [i for i in range(len(labels))]
    labels["label"] = label_index

    label_df = pd.merge(label_df, labels, on="ENE_name")
    return label_df, labels


def generate_train_data(wiki_df, label_df, data_type, tokenizer, max_len):
    tokenized_texts = []
    append = tokenized_texts.append
    with tqdm(total=len(wiki_df), desc="extracting text") as t:
        for index in wiki_df.index:
            text = wiki_df["text"][index]
            id = wiki_df["id"][index]
            tokenized_text = tokenizer.tokenize(text)
            if data_type == "entire":
                tokenized_texts_list = split_list(tokenized_text, max_len)
                for text in tokenized_texts_list:
                    ids = tokenizer.token2ids(text)
                    pair = {
                        "id": id,
                        "input_ids": ids["input_ids"],
                        "attention_mask": ids["attention_mask"],
                        "token_type_ids": ids["token_type_ids"],
                    }
                    append(pair)
            elif data_type == "all":
                pair = {"id": id, "text": tokenized_text}
                append(pair)
            t.update(1)
    wiki_data = pd.DataFrame(tokenized_texts)
    label_df, labels = extract_label(label_df)
    df = pd.merge(wiki_data, label_df, left_on="id", right_on="pageid")
    return df, labels


def read_data(
    debug, data_path, file_data_name, file_label_name, data_type, tokenizer, max_len=512
):
    utils = FileUtils(data_path, file_label_name, file_data_name)
    train_data_dir = f"./data/json/{data_type}/"
    train_data_name = "train_data.jsonl"
    df = pd.DataFrame()
    if debug:
        train_data_dir += str(debug) + "/"
    if not os.path.exists(train_data_dir):
        os.makedirs(train_data_dir)

    train_data_path = train_data_dir + train_data_name

    if os.path.isfile(train_data_path):
        data = []
        with open(train_data_path, "r") as f, tqdm(desc="reading data") as t:
            for line in f:
                file = json.loads(line)
                data.append(file)
                t.update()
        df = pd.DataFrame(data)
        labels = df.filter(items=["ENE_name", "label"])
        df = df.filter(items=["input_ids", "attention_mask", "token_type_ids", "label"])
    else:
        wiki_data = utils.load_wikidata()
        label_data = utils.load_labeldata()
        wiki_df = wikidata_preprocess(wiki_data)
        label_df = label_preprocess(label_data)
        df, labels = generate_train_data(
            wiki_df, label_df, data_type, tokenizer, max_len
        )
        df.to_json(train_data_path, orient="records", lines=True)
        df = df.filter(items=["input_ids", "attention_mask", "token_type_ids" "label"])

    return df, labels


def make_input(df):
    df = df.filter(items=["input_ids", "attention_mask", "token_type_ids", "label"])
    text_ans_pair = df.values.tolist()
    return text_ans_pair


def make_label_index_pair(df):
    df = df.filter(items=["label", "ENE_name"])
    pair = {}
    label_dict = {}
    for i in df.index:
        index = df["label"][i]
        label = df["ENE_name"][i]
        if index in label_dict.keys():
            continue
        pair = {index: label}
        label_dict.update(pair)
    return label_dict


def preprocess(
    debug, data_path, file_label_name, file_data_name, bert_version, data_type
):
    tokenizer = Tokenizer(bert_version=bert_version)
    df, labels = read_data(
        debug, data_path, file_data_name, file_label_name, data_type, tokenizer
    )
    dataset = make_input(df)
    label_dict = make_label_index_pair(labels)
    return dataset, label_dict


def main():
    # preprocess(True, "/data/", "en_ENEW_LIST.json", "en-trial-wiki-20190121-cirrussearch-content.json.gz", "bert-base-cased", "all")
    preprocess(
        True,
        "/data/",
        "en-trial-wiki-20190121-cirrussearch-content.json.gz",
        "en_ENEW_LIST.json",
        "bert-base-cased",
        "entire",
    )


if __name__ == "__main__":
    main()
