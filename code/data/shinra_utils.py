import json
import tqdm
import os
import pandas as pd


class FileUtils:
    def __init__(self, base_path: str, label_name: str, wiki_name: str, limit=1000):
        self.base_path = base_path
        self.label_path = self.base_path + label_name
        self.wiki_path = self.base_path + wiki_name
        self.limit = limit

    def load_labeldata(self, limit):
        return self.load_shinra_json(self.label_path, limit)

    def load_wikidata(self, limit):
        wikidata = self.load_shinra_json(self.wiki_path, limit)
        return self.merge_wikidata(wikidata)

    def merge_wikidata(self, wikidata):
        wikidatas = []
        for i in range(0, len(wikidata), 2):
            d = dict(**wikidata[i], **wikidata[i + 1])
            wikidatas.append(d)
        return wikidatas

    def load_shinra_json(self, path: str, limit=False):
        data = []
        with open(path, "r") as f, \
                tqdm.tqdm(desc=os.path.basename(path)) as t:
            for line in f:
                if line == '\n':
                    continue
                file = json.loads(line)
                data.append(file)
                if limit and len(data) == self.limit:
                    break
                t.update()
        return data


def label_preprocess(labels):
    df = pd.DataFrame()
    label_list = []
    for label in tqdm.tqdm(labels, desc = "extracting page id and page name"):
        label_dict = {'pageid': label['pageid'], 'title': label['title']}
        enes = label['ENEs']
        label_ene_dict = {}
        for ene in enes:
            label_ene_dict = dict(**label_dict, **ene)
            label_list.append(label_ene_dict)
    df = pd.DataFrame(label_list)
    return df


def wikidata_preprocess(wikidata):
    df = pd.DataFrame()
    wiki_dict_list = []
    for wiki in tqdm.tqdm(wikidata, desc = "extracting wiki data"):
        # print(wiki.keys())
        wikidict = {"type": wiki['index']['_type'],
                    "id": wiki['index']['_id'],
                    "text": wiki['text']}
        wiki_dict_list.append(wikidict)
    wiki_df = pd.DataFrame(wiki_dict_list)
    df = pd.concat([df, wiki_df])
    return df
# 実装のテスト　


def main():
    print("hello")
    base_path = "./trial_en/trial_en/en/"
    label_name = "en_ENEW_LIST.json"
    wiki_name = "en-trial-wiki-20190121-cirrussearch-content.json"
    files = FileUtils(base_path, label_name, wiki_name)
    labels = files.load_labeldata(False)
    wikidata = files.load_wikidata(True)
    label_df = label_preprocess(labels)
    wiki_df = wikidata_preprocess(wikidata)
    label_df.to_csv("./csv/labels.csv", index=False)
    wiki_df.to_csv("./csv/wiki.csv", index=False)


if __name__ == "__main__":
    main()
