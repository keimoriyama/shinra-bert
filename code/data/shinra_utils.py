import json
from collections import OrderedDict
import tqdm
import os


class FileUtils:
    def __init__(self, base_path: str, label_name: str, wiki_name: str, limit=1000):
        self.base_path = base_path
        self.label_path = self.base_path + label_name
        self.wiki_path = self.base_path + wiki_name
        self.limit = limit

    def load_labeldata(self):
        return self.load_shinra_json(self.label_path)

    def load_wikidata(self):
        wikidata = self.load_shinra_json(self.wiki_path, True)
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


class ShinraData():
    def __init__(self) -> None:
        pass


def main():
    print("hello")
    base_path = "./trial_en/trial_en/en/"
    label_name = "en_ENEW_LIST.json"
    wiki_name = "en-trial-wiki-20190121-cirrussearch-content.json"
    files = FileUtils(base_path, label_name, wiki_name)
    labels = files.load_labeldata()
    wikidata = files.load_wikidata()
    # print(type(labels[0]), type(wikidata[0]))
    print(wikidata[0])


if __name__ == "__main__":
    main()
