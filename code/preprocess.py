import os
import argparse
from data.shinra_utils import FileUtils, label_preprocess, wikidata_preprocess
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm


def load_arg():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def gen_data():
    label_path = "/data/csv/labels.csv"
    wiki_path = "/data/csv/wiki.csv"
    cwd = os.getcwd()
    base = "."
    if '/workspaces/shinra-bert' == cwd:
        base = "./code/"
    dir_path = base + "/data/csv/"
    label_path = base + label_path
    wiki_path = base + wiki_path

    if not os.listdir(dir_path):
        base_path = base + "/data/trial_en/trial_en/en/"
        label_name = "en_ENEW_LIST.json"
        wiki_name = "en-trial-wiki-20190121-cirrussearch-content.json"
        files = FileUtils(base_path, label_name, wiki_name)
        labels = files.load_labeldata(False)
        wikidata = files.load_wikidata(True)
        label_df = label_preprocess(labels)
        label_df.to_csv(label_path, index=False)
        del label_df, labels
        wiki_df = wikidata_preprocess(wikidata)
        wiki_df.to_csv(wiki_path, index=False)
        del wikidata, wiki_df

    label_df = pd.read_csv(label_path)
    wiki_df = pd.read_csv(wiki_path)
    return label_df, wiki_df


def preprocess():

    label_df, wiki_df = gen_data()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    tokenized_texts = []
    with tqdm(total=len(wiki_df), desc="tokenizing text") as t:
        for index in wiki_df.index:
            text = wiki_df['text'][index]
            id = wiki_df['id'][index]
            pair = {
                "id": id,
                "tokenized_text": text
            }
            tokenized_texts.append(pair)
            t.update(1)
    wiki_data = pd.DataFrame(tokenized_texts)
    label_df.drop_duplicates(subset="pageid")
    labels = label_df[~label_df.duplicated(subset='ENE_name')].reset_index().drop(columns=['ENE_id', 'title', 'index', 'pageid'])
    label_index = [i for i in range(len(labels))]
    labels['label'] = label_index
    label_df = pd.merge(label_df, labels, on="ENE_name")
    df = pd.merge(wiki_data, label_df, left_on='id', right_on='pageid')

    df = df.drop(columns=["pageid", "id", "ENE_id"])

    df.to_csv("./data/csv/data.csv", index=False)
    return df


def main():
    preprocess()


if __name__ == "__main__":
    main()
