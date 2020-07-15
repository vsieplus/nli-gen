# training script for the inference RNN

import infer_rnn

import argparse
import pathlib
import os
import sys

import pandas as pd
import torch
import torch.nn.functional as F
import torchtext


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help="output directory")
    parser.add_argument("--num_epochs", type=int, default=5)

    args = parser.parse_args()

    torch.manual_seed(321)



if __name__ == "__main__":
    main()