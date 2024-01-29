import os
import numpy as np
import yaml
import argparse
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn


def main(config):

    print(config["value_one"])
    print(config["value_two"])


if __name__ == '__main__':

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-c", "--config", default="./configs/default.yaml", help="path to default.yaml file")
    args = argParser.parse_args()

    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)

    main(config)
