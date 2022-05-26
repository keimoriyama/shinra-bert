import os
from posixpath import split
from transformers import BertTokenizer
from tqdm import tqdm

from data.shinra_utils import FileUtils, label_preprocess, wikidata_preprocess
import pandas as pd

class Tokenizer():
    def __init__(self, bert_version):
        self.tokenizer = BertTokenizer.from_pretrained(bert_version)

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)
    
    def decode(self, text):
        ids = self.tokenizer.convert_tokens_to_ids(text)
        return self.tokenizer.decode(ids)



def gen_data(debug, data_path, file_data_name, file_label_name):
    label_path = "/data/csv/labels.csv"
    wiki_path = "/data/csv/wiki.csv"
    base = "."
    label_path = base + label_path
    wiki_path = base + wiki_path

    base_path = base + data_path
    label_name = file_label_name
    wiki_name = file_data_name
    files = FileUtils(base_path, label_name, wiki_name)
    
    if os.path.isfile(label_path):
        label_df = pd.read_csv(label_path)
    else:
        labels = files.load_labeldata()
        label_df = label_preprocess(labels)
        label_df.to_csv(label_path, index=False)
    if os.path.isfile(wiki_path):
        wiki_df = pd.read_csv(wiki_path)
        wiki_df["text"] = wiki_df["text"].astype(str)
    else:
        wikidata = files.load_wikidata(debug)
        wiki_df = wikidata_preprocess(wikidata)
        wiki_df.to_csv(wiki_path, index=False)
    return label_df, wiki_df


def split_list(target, max_len):
    res = [target[i:i+max_len] for i in range(0, len(target), max_len)]
    return res

def read_data(debug, data_path, file_data_name, file_label_name, tokenizer,max_len=512):
    label_df, wiki_df = gen_data(debug, data_path, file_data_name,file_label_name)
    tokenized_texts = []
    if debug:
        wiki_df = wiki_df[:5]
    with tqdm(total=len(wiki_df), desc="extracting text") as t:
        for index in wiki_df.index:
            text = wiki_df['text'][index]
            tokenized_text = tokenizer.tokenize(text)
            tokenized_texts_list = split_list(tokenized_text, max_len)
            id = wiki_df['id'][index]
            for text in tokenized_texts_list:
                pretokenized_text = tokenizer.decode(text)
                import ipdb; ipdb.set_trace()

                pair = {"id": id, "text": pretokenized_text}
                tokenized_texts.append(pair)
            t.update(1)
    wiki_data = pd.DataFrame(tokenized_texts)
    
    label_df.drop_duplicates(subset="pageid")
    labels = label_df[~label_df.duplicated(
        subset='ENE_name')].reset_index().drop(
            columns=['ENE_id', 'title', 'index', 'pageid'])
    label_index = [i for i in range(len(labels))]
    labels['label'] = label_index

    label_df = pd.merge(label_df, labels, on="ENE_name")
    df = pd.merge(wiki_data, label_df, left_on='id', right_on='pageid')
    df.to_csv("./data/csv/data.csv", index=False)

    df = df.drop(columns=["pageid", "id", "ENE_id"])
    return df, labels

def make_input(df):
    df = df.filter(items=["text", "label"])
    text_ans_pair = df.values.tolist()
    return text_ans_pair


def make_label_index_pair(df):
    df = df.filter(items=["label", "ENE_name"])
    pair = {}
    label_dict = {}
    for i in df.index:
        index = df['label'][i]
        label = df['ENE_name'][i]
        if index in label_dict.keys():
            continue
        pair = {index: label}
        label_dict.update(pair)
    return label_dict


def preprocess(debug, data_path, file_data_name, file_label_name, bert_version):
    tokenizer = Tokenizer(bert_version=bert_version)
    df, labels = read_data(debug, data_path, file_data_name, file_label_name, tokenizer)
    dataset = make_input(df)
    label_dict = make_label_index_pair(labels)
    return dataset, label_dict


def main():
    preprocess(True, "/data/", "en_ENEW_LIST.json", "en-trial-wiki-20190121-cirrussearch-content.json.gz", "bert-base-cased")


if __name__ == "__main__":
    main()
