import sys
import shutil
import torch
from subprocess import run
from os import path


def train_model(run_params='venue_classifier'):
    try:
        print("Torch Version:", torch.__version__)
        print("GPU Count:", torch.cuda.device_count())
        command = f"allennlp train ./experiments/{run_params}.json -s ./tmp/{run_params} --include-package library"
        run(command.split(' '), check=True)
    finally:
        print("\nTraining Done! - param_file:", run_params)


if __name__ == "__main__":
    argv = sys.argv
    param_file = 'venue_classifier'
    training = True
    for k, v in enumerate(argv):
        if v == '-f':  # num of cores
            param_file = argv[k + 1]
        if v == '-p':
            training = False

    if training:
        train_model()
