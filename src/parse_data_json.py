import argparse
import json
import glob
import numpy as np

from pathlib import Path
from tqdm import tqdm

def parse_json(data_dir):
    sequence = []
    labels = []
    pbar = tqdm(glob.glob(f'{data_dir}/p*.json'))
    for filename in pbar:
        pbar.set_description(filename)
        with open(filename, 'r') as f:
            data = json.load(f)

        for segment in data:
            sequence.append(segment['ppg_raw'])
            labels.append(segment['targets'])
    return np.vstack(sequence), np.vstack(labels)

def save_data(filename, array):
    np.save(filename, array)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")

    args = parser.parse_args()
    data_dir = Path(args.path)

    for dataset in ['train', 'test', 'val']:
        filepath = Path(data_dir, f"{dataset}_set")
        print(f"Loading data for {dataset} dataset")
        X, y = parse_json(filepath)

        save_data(Path(data_dir, f"vitals_X_{dataset}.npy"), X)
        save_data(Path(data_dir, f"vitals_y_{dataset}.npy"), y)

    print("PARSING DONE")



