import h5py
import glob
import argparse
import numpy as np

from tqdm import tqdm
from pathlib import Path

def get_data_from_ref(f, feature_name, i):
    ref = f['Subj_Wins'][feature_name][0][i]
    name = h5py.h5r.get_name(ref, f.id)
    return f[name][0]

def load_data(filenames):
    sequence = []
    labels = []
    for filename in tqdm(filenames):
        f = h5py.File(filename, 'r')
        for i in range(len(f['Subj_Wins']['SegSBP'][0])):
            if isinstance(f['Subj_Wins']['SegSBP'][0][i], np.generic):
                input_data = f['Subj_Wins']['PPG_Raw'][i]
                sbp_data = f['Subj_Wins']['SegSBP'][i]
                dbp_data = f['Subj_Wins']['SegDBP'][i]

            elif isinstance(f['Subj_Wins']['SegSBP'][0][i], h5py.h5r.Reference):
                input_data = get_data_from_ref(f, 'PPG_Raw', i)

                sbp_data = get_data_from_ref(f, 'SegSBP', i)
                dbp_data = get_data_from_ref(f, 'SegDBP', i)


            sequence.append(input_data)
            labels.append((sbp_data, dbp_data))
        f.close()
    return np.vstack(sequence), np.array(labels).reshape(-1,2)

def train_test_split_np(array):
    np.random.seed(42)
    np.random.shuffle(array)
    split = int(len(array) * 0.85)
    train, test = array[:split], array[split:]
    return train, test

def get_filenames(path):
    return glob.glob(f"{path}/p*.mat")

def save_data(filename, array):
    np.save(filename, array)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("target_name")

    args = parser.parse_args()
    data_dir = Path(args.path)
    target_name = args.target_name

    filenames = get_filenames(data_dir)
    train_files, test_files = train_test_split_np(filenames)

    print("Loading data for TRAIN dataset")
    X_train, y_train = load_data(train_files)

    print("Loading data for EVAL dataset")
    X_test, y_test = load_data(test_files)

    save_data(Path(data_dir, f"{target_name}_x_train.npy"), X_train)
    save_data(Path(data_dir, f"{target_name}_y_train.npy"), y_train)
    save_data(Path(data_dir, f"{target_name}_x_test.npy"), X_test)
    save_data(Path(data_dir, f"{target_name}_y_test.npy"), y_train)

    print("PARSING DONE")




