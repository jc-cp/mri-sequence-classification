import os


def get_all_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)


# specify your directory
directory = "/mnt/93E8-0534/JuanCarlos/BraTS2020/MICCAI_BraTS2020_ValidationData"

file_list = list(get_all_files(directory))

# print out the first few files to verify
for file in enumerate(file_list[:20], start=1):
    print(str(file) + "\n")
