import json
from collections import OrderedDict
import tqdm
import os


def main():
    print("hello")
    path = "./trial_en/trial_en/en/en_ENEW_LIST.json"
    data = []
    with open(path, "r") as f, \
            tqdm.tqdm(desc=os.path.basename(path)) as t:
        for d in map(json.loads, f):
            data.append(d)
            t.update()
    print(data)


if __name__ == "__main__":
    main()
