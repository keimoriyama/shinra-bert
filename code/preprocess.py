import os 
import argparse

import multiprocessing as multiprocessing
from data.shinra_utils import ShinraData, ShinraSystemData



def load_arg():
    parser = argparse.ArgumentParser()
    return parser.parse_args()

def main():
    args = load_arg()

if __name__ == "__main__":
    main()