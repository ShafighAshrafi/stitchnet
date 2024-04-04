import cv2
import numpy as np
import pandas as pd
from os import makedirs
from random import seed
from random import random
from shutil import rmtree

# seed random number generator
seed(1)
# define ratio of pictures to use for validation
val_ratio = 0.2
# copy training dataset images into subdirectories
imageSize  = 224

dataset_home = 'src/dataset/main_dataset/'
csv_path   = "src/dataset/main_dataset/data_information.csv"
file_path  = "src/dataset/main_dataset/NEH_UT_2021RetinalOCTDataset/"

def read_worstcase_images() -> tuple:
    rmtree(dataset_home + 'train/' + 'normal/', ignore_errors=True)
    rmtree(dataset_home + 'train/' + 'drusen/', ignore_errors=True)
    rmtree(dataset_home + 'train/' + 'cnv/', ignore_errors=True)
    rmtree(dataset_home + 'test/' + 'normal/', ignore_errors=True)
    rmtree(dataset_home + 'test/' + 'drusen/', ignore_errors=True)
    rmtree(dataset_home + 'test/' + 'cnv/', ignore_errors=True)
    makedirs(dataset_home + 'train/' + 'normal/', exist_ok=True)
    makedirs(dataset_home + 'train/' + 'drusen/', exist_ok=True)
    makedirs(dataset_home + 'train/' + 'cnv/', exist_ok=True)
    makedirs(dataset_home + 'test/' + 'normal/', exist_ok=True)
    makedirs(dataset_home + 'test/' + 'drusen/', exist_ok=True)
    makedirs(dataset_home + 'test/' + 'cnv/', exist_ok=True)
    df = pd.read_csv(csv_path)
    for patient_class in np.unique(df['Class']):
        df_classwise = df[df['Class'] == patient_class]
        count = 0
        for i in range(len(df_classwise)):
            if df_classwise.iloc[i]['Class'] == df_classwise.iloc[i]['Label']:
                file_extension = df_classwise.iloc[i]['Directory'].split(".")[-1]
                img = cv2.imread(file_path + df_classwise.iloc[i]['Directory'])
                img = cv2.resize(img, (imageSize, imageSize))
                # img = np.asarray(img, dtype=np.float32)
                # img = img / 255
                # img = np.moveaxis(img, 2, 0)
                destination_dir = 'train/'
                if random() < val_ratio:
                    destination_dir = 'test/'
                count += 1
                if df_classwise.iloc[i]['Class'].lower() == "normal":
                    destination = dataset_home + destination_dir + 'normal/' + str(count) + "." + file_extension
                elif df_classwise.iloc[i]['Class'].lower() == "drusen":
                    destination = dataset_home + destination_dir + 'drusen/'  + str(count) + "." + file_extension
                elif df_classwise.iloc[i]['Class'].lower() == "cnv":
                    destination = dataset_home + destination_dir + 'cnv/'  + str(count) + "." + file_extension
                cv2.imwrite(destination, img)