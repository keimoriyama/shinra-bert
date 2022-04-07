import json
from collections import OrderedDict
import tqdm
import os


class FileUtils:
    def __init__(self, label_path, wiki_path):
        self.label_path = label_path
        self.wiki_path = wiki_path

    def load_labeldata(self):
        return self.load_shinra_json(self.label_path)

    def load_shinra_json(self, path):
        data = []
        with open(self.label_path, "r") as f, \
                tqdm.tqdm(desc=os.path.basename(path)) as t:
            for d in map(json.loads, f):
                data.append(d)
                t.update()
        return data


def main():
    print("hello")
    path = "./trial_en/trial_en/en/en-trial-wiki-20190121-cirrussearch-content.json"
    data = []
    with open(path, "r") as f, \
            tqdm.tqdm(desc=os.path.basename(path)) as t:
        for d in map(json.loads, f):
            data.append(d)
            t.update()
    print(data[0])


if __name__ == "__main__":
    main()
