import os

# This file is used to count number of phonemes in train and test data.

def count(root_folder):

  sub_folders =  os.listdir(root_folder)

  for folder in sub_folders:
      if folder != '.DS_Store':
        print(str(len(os.listdir(root_folder+folder)) - 1))

#count(os.getcwd() + '/timit_data/train/nosab/phonemes/flattened_mfcc/')
count(os.getcwd() + '/timit_data/test/nosab/phonemes/flattened_mfcc/')