import pandas as pd
import os
import shutil
from parser_long_cfg import DIR_DGM, DIR_MIXED, DIR_NO_OPS, DIR_RADART

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
]

# Iterate through each main directory
for main_dir in main_directories:
    # Create sub-directories within main directory
    for sub_dir in sub_directories:
        os.makedirs(os.path.join(main_dir, sub_dir), exist_ok=True)

    # List all csv files in the directory
    csv_files = [f for f in os.listdir(main_dir) if f.endswith(".csv")]
    # Iterate through each .csv file in the main directory
    for file_name in csv_files:
        print(f"\tProcessing file: {file_name}")
        # Load the .csv file as a pandas DataFrame
        df = pd.read_csv(os.path.join(main_dir, file_name))

        # Handle 'NO PREDICTION' special case
        no_prediction_df = df[df["Prognosis"].str.startswith("NO PREDICTION")]
        if not no_prediction_df.empty:
            no_prediction_df.to_csv(
                os.path.join(main_dir, "NO PREDICTION", file_name), index=False
            )
            # Copy files to 'NO PREDICTION' sub-folder
            for _, row in no_prediction_df.iterrows():
                if not any(
                    ignore_string in row["path"] for ignore_string in ignore_strings
                ):
                    shutil.copy(row["path"], os.path.join(main_dir, "NO PREDICTION"))

        # Iterate through each type of prediction and create corresponding .csv files in subfolders
        for prediction in sub_directories:
            if (
                prediction != "NO PREDICTION"
            ):  # Skip 'NO PREDICTION' as it has been handled
                # Filter the DataFrame for the current type of prediction
                df_filtered = df[df["Prognosis"] == prediction]

                # If there are any rows of this prediction type, save them in a new .csv file
                if not df_filtered.empty:
                    df_filtered.to_csv(
                        os.path.join(main_dir, prediction, file_name), index=False
                    )
                    for _, row in df_filtered.iterrows():
                        if not any(
                            ignore_string in row["path"]
                            for ignore_string in ignore_strings
                        ):
                            shutil.copy(row["path"], os.path.join(main_dir, prediction))
