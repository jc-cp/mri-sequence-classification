import argparse
import os
import random
import time
import pandas as pd

import matplotlib.pyplot as plt
import numpy
import torch
import torch.utils.data as data

from MedicalDataset_long import MedicalDataset
from models import select_net
from time_util import time_format
import os
import csv
from tqdm import tqdm
from config import BASE_DIR_NO_OPS, OUTPUT_DIR, GET_CSV, MODELS_DIR
from statistics import mode

# TODO:
# check storage of prediciton


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        dest="test_data_path",
        type=str,
        # required=True,
        help="Path containing data to be tested.",
        default=OUTPUT_DIR,
    )
    parser.add_argument(
        "-m",
        dest="model_file",
        type=str,
        # required=True,
        help="Name of the trained model file.",
        default="main_sl10.pth",
    )
    parser.add_argument(
        "-sl",
        dest="slices",
        type=int,
        default=10,
        help="Number of central slices considered by the trained model.",
    )
    parser.add_argument(
        "-3d",
        dest="tridim",
        action="store_true",
        help="Use if the trained model used tridimensional convolution.",
    )
    parser.add_argument(
        "--no-other",
        dest="no_other",
        action="store_true",
        help='If specified, "Other" class is not considered.',
    )
    parser.add_argument(
        "--net",
        dest="net",
        type=str,
        default="resnet18",
        help="Network architecture to be used.",
    )
    return parser.parse_args()


def fix_random_seeds():
    torch.backends.cudnn.deterministic = True
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    numpy.random.seed(1)


def get_files_and_write_to_csv(base_dir, output_dir):
    # check if output directory exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # loop through patient_id directories
    for patient in os.listdir(base_dir):
        patient_path = os.path.join(base_dir, patient)

        # check if it is a directory
        if os.path.isdir(patient_path):
            file_paths = []

            # loop through scan_id directories
            for scan in os.listdir(patient_path):
                scan_path = os.path.join(patient_path, scan)

                # check if it is a directory
                if os.path.isdir(scan_path):
                    # loop through files in files directory
                    for file in os.listdir(scan_path):
                        file_path = os.path.join(scan_path, file)

                        # check if it is a file, not a directory
                        if os.path.isfile(file_path):
                            file_paths.append(file_path)

            # if there are any files, write them to a CSV
            if file_paths:
                with open(
                    os.path.join(output_dir, f"{patient}_file_paths.csv"),
                    "w",
                    newline="",
                ) as f:
                    fieldnames = ["Path", "Prognosis"]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for path in file_paths:
                        writer.writerow({"Path": path, "Prognosis": ""})


def perform_prognosis_on_csvs(
    test_data_path,
    model_file,
    n_slices,
    tridim,
    consider_other_class,
    architecture,
    models_dir,
):
    # Make sure the models_dir is a directory
    assert os.path.isdir(models_dir), f"{models_dir} is not a directory."
    assert os.path.isdir(test_data_path), f"{test_data_path} is not a directory."

    for csv_file in tqdm(os.listdir(test_data_path), desc="Processing CSVs"):
        if csv_file.endswith(".csv"):
            csv_path = os.path.join(test_data_path, csv_file)
            data_frame = pd.read_csv(csv_path)

            fix_random_seeds()

            test_set = MedicalDataset(
                csv_path,
                min_slices=n_slices,
                consider_other_class=consider_other_class,
            )

            skipped_data = test_set.skipped_data

            for file in skipped_data:
                data_frame.loc[data_frame["Path"] == file, "Prognosis"] = "other"

            test_loader = data.DataLoader(test_set, num_workers=8, pin_memory=True)

            n_test_files = test_set.__len__()
            classes = ["FLAIR", "T1", "T1c", "T2", "OTHER"]

            net = select_net(architecture, n_slices, tridim, consider_other_class)

            if torch.cuda.is_available():
                net = net.cuda()

            start_time = time.time()

            # test
            net.load_state_dict(torch.load(os.path.join(models_dir, model_file)))
            net.eval()
            with torch.no_grad():
                for i, (pixel_data, path) in tqdm(
                    enumerate(test_loader),
                    total=len(test_loader),
                    desc=f"Predicting from {csv_file}",
                ):
                    if tridim:
                        pixel_data = pixel_data.view(-1, 1, 10, 200, 200)

                    outputs = net(pixel_data.cuda())
                    _, predicted = torch.max(outputs.data, 1)

                    predicted_class = classes[predicted.cpu().numpy()[0]]
                    print(f"For file at {path[0]}, model predicts: {predicted_class}")

                    # find the corresponding row in data_frame and set its 'Prognosis' value
                    # Assuming 'path' is a batch with a single file path string
                    current_file_path = path[0]
                    data_frame.loc[
                        data_frame["Path"] == current_file_path, "Prognosis"
                    ] = predicted_class

                # save the data_frame with the new 'Prognosis' values to the csv file
                data_frame.to_csv(csv_path, index=False)

            # time
            print()
            end_time = time.time()
            elapsed_time = time_format(end_time - start_time)
            print(
                "Testing elapsed time for file {0}: {1}".format(csv_file, elapsed_time)
            )


if __name__ == "__main__":
    if GET_CSV:
        get_files_and_write_to_csv(BASE_DIR_NO_OPS, OUTPUT_DIR)
    else:
        args = parse_args()
        perform_prognosis_on_csvs(
            args.test_data_path,
            args.model_file,
            args.slices,
            args.tridim,
            not args.no_other,
            args.net,
            MODELS_DIR,
        )
