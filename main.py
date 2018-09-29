import sys
import os


def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    print(dirs)


if __name__ == '__main__':
    prepare_training_data('samples/thumbnails_features_deduped_sample')
