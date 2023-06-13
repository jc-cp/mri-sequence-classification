import pandas as pd
import os
import shutil
from parser_long_cfg import DIR_DGM, DIR_MIXED, DIR_NO_OPS, DIR_RADART


def extract_ids_from_path(path):
    # Extract the patient and scan ids from the file path
    ids = path.split("/")[-3:-1]
    return ids


# Specify the names of the directories here
main_directories = [
    # DIR_RADART,
    DIR_NO_OPS,
    # DIR_MIXED,
    # DIR_DGM,
]  # edit this list to process single directories

sub_directories = ["FLAIR", "T1", "T1c", "T2", "OTHER", "NO PREDICTION"]

# Strings to be ignored
ignore_strings = [
    "tse",
    "turbo_spin_echo",
    "spine",
    "ctl",
    "lumbar",
    "tlsp",
    "tsp",
    "lsp",
    "csp",
    "t1_flair",
]


for main_dir in main_directories:
    for sub_dir in sub_directories:
        os.makedirs(os.path.join(main_dir, sub_dir), exist_ok=True)

    csv_files = [f for f in os.listdir(main_dir) if f.endswith(".csv")]

    for file_name in csv_files:
        print(f"\tProcessing file: {file_name}")

        df = pd.read_csv(os.path.join(main_dir, file_name))

        # Handle 'NO PREDICTION' special case
        no_prediction_df = df[df["Prediction"].str.startswith("NO PREDICTION")]
        if not no_prediction_df.empty:
            no_prediction_df.to_csv(
                os.path.join(main_dir, "NO PREDICTION", file_name), index=False
            )
            # Copy files to 'NO PREDICTION' sub-folder
            for _, row in no_prediction_df.iterrows():
                assert os.path.exists(row["Path"])
                if not any(
                    ignore_string in row["Path"] for ignore_string in ignore_strings
                ):
                    patient_id, scan_id = extract_ids_from_path(row["Path"])
                    dest_path = os.path.join(
                        main_dir,
                        "NO PREDICTION",
                        f'{patient_id}_{scan_id}_{os.path.basename(row["Path"])}',
                    )
                    if not os.path.exists(dest_path):
                        shutil.copy(row["Path"], dest_path)

        for prediction in sub_directories:
            if (
                prediction != "NO PREDICTION"
            ):  # Skip 'NO PREDICTION' as it has been handled
                df_filtered = df[df["Prediction"] == prediction]

                if not df_filtered.empty:
                    df_filtered.to_csv(
                        os.path.join(main_dir, prediction, file_name), index=False
                    )
                    for _, row in df_filtered.iterrows():
                        assert os.path.exists(row["Path"])
                        if not any(
                            ignore_string in row["Path"]
                            for ignore_string in ignore_strings
                        ):
                            patient_id, scan_id = extract_ids_from_path(row["Path"])
                            dest_path = os.path.join(
                                main_dir,
                                prediction,
                                f'{patient_id}_{scan_id}_{os.path.basename(row["Path"])}',
                            )

                            if not os.path.exists(dest_path):
                                shutil.copy(row["Path"], dest_path)
