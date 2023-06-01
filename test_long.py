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

# TODO:
# change the test_data_path
# check default for 3D
# check column of csv -- currently: 'data-frame'
# check storage of prediciton


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        dest="test_data_path",
        type=str,
        # required=True,
        help="Path containing data to be tested.",
        default="data_csv/generated/own_tests_brats2020/test.csv",
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


if __name__ == "__main__":
    args = parse_args()
    test_data_path = args.test_data_path
    data_frame = pd.read_csv(test_data_path)
    model_file = args.model_file
    n_slices = args.slices
    tridim = args.tridim
    consider_other_class = not args.no_other
    architecture = args.net
    models_dir = "/mnt/93E8-0534/JuanCarlos/mri-classifcation-pretrained-models/Models"

    assert architecture in ["resnet18", "alexnet", "vgg", "squeezenet", "mobilenet"]

    fix_random_seeds()

    test_set = MedicalDataset(
        test_data_path,
        min_slices=n_slices,
        consider_other_class=consider_other_class,
        test=True,
        predict=True,
    )
    test_loader = data.DataLoader(test_set, num_workers=8, pin_memory=True)
    # test_loader = data.DataLoader(test_set, pin_memory = True)

    n_test_files = test_set.__len__()
    classes = ["FLAIR", "T1", "T1c", "T2", "OTHER"]  # train_set.classes

    net = select_net(architecture, n_slices, tridim, consider_other_class)

    if torch.cuda.is_available():
        net = net.cuda()

    start_time = time.time()

    # test
    net.load_state_dict(torch.load(os.path.join(models_dir, model_file)))
    net.eval()
    predicted_classes = []
    with torch.no_grad():
        for i, (pixel_data, path) in enumerate(test_loader):
            if tridim:
                pixel_data = pixel_data.view(-1, 1, 10, 200, 200)

            outputs = net(pixel_data.cuda())
            _, predicted = torch.max(outputs.data, 1)

            predicted_class = classes[predicted.cpu().numpy()[0]]
            predicted_classes.append(predicted_class)

            print(f"For file at {path[0]}, model predicts: {predicted_classes}")
            # add a new column 'prediction' to the corresponding row in the data_frame
            data_frame.loc[
                data_frame["image_path"] == path[0], "prediction"
            ] = predicted_class

    # time
    print()
    end_time = time.time()
    elapsed_time = time_format(end_time - start_time)
    print("Testing elapsed time:", elapsed_time)

    os.makedirs(os.path.join("results", "test"), exist_ok=True)

    # save the updated data_frame as csv
    data_frame.to_csv(
        os.path.join(
            "results",
            "test",
            test_data_path.replace(os.sep, "_").replace(".", "_")
            + "--"
            + model_file.replace(".pth", ".txt"),
        ),
        index=False,
    )
