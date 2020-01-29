# This file corrects the error - file does not start with RIFF id occuring on TIMIT data set
# @ refer - https://fieldarchives.wordpress.com/2014/02/18/converting-the-wav-files/


import subprocess
import glob
import os

# Lists all the wav files in train
wav_files_list = glob.glob(os.getcwd()+'/timit_data/train/*/*/*.wav')
# Lists all the wav files in test
wav_files_list.extend(glob.glob(os.getcwd()+'/timit_data/test/*/*/*.wav'))


wav_prime = []
# Create temporary names for the wav files to be converted. They will be renamed later on.
for f in wav_files_list:
    fileName, fileExtension = os.path.splitext(f)
    fileName += 'm' # m for modified
    wav_prime.append(fileName + fileExtension)

# Command strings
cmd = "sox {0} -t wav {1}"
mv_cmd = "mv {0} {1}"

# Convert the wav_files first. Remove it. Rename the new file created by sox to its original name
for i, f in enumerate(wav_files_list):
    subprocess.call(cmd.format(f, wav_prime[i]), shell=True)
    os.remove(f)
    subprocess.call(mv_cmd.format(wav_prime[i], f), shell=True)
