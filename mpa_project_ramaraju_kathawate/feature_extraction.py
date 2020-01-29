# Using python_speech_features and librosa libraries for mfcc and derivatives
# Author: Naveenkumar Ramaraju
#

from scipy.io import wavfile
import numpy as np
from python_speech_features import mfcc
import glob
import os
import time
import shutil
import librosa


phonemes_list = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx',
                     'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix',
                     'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's',
                     'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']


def extract_features(directory_name):

    for phoneme in phonemes_list:
        if os.path.exists(directory_name+'extracted_features/'+phoneme):
            shutil.rmtree(directory_name+'extracted_features/'+phoneme)
        os.mkdir(directory_name+'extracted_features/'+phoneme)

    folders = os.listdir(directory_name)

    for folder in folders:
        folder_name, file_extension = os.path.splitext(folder)
        folder_name = folder_name.split('/')[-1]
        if folder_name in phonemes_list:
            print(folder_name)
            wav_files_list = glob.glob(directory_name+ folder_name+'/*.wav')
            #print(directory_name+ folder_name)
            #print(wav_files_list)
            for utterance in wav_files_list:
                #file_name, file_extension = os.path.splitext(directory_name + phoneme + '/' + utterance)
                phoneme = utterance.split('/')[-2]
                file_name = utterance.split('/')[-1].replace('.wav','')

                sr, audio = wavfile.read(utterance)
                try:
                    mfcc_ = mfcc(audio,samplerate=16000,winlen=0.025,winstep=0.01,numcep=13,
                                 nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True)
                    d1 = librosa.feature.delta(mfcc_)

                    d2 = librosa.feature.delta(mfcc_, order=2)
                    np.save(directory_name+'extracted_features/'+phoneme+'/'+file_name,np.concatenate([np.concatenate([mfcc_,d1],axis=0), d2], axis=0))
                except Exception as e:
                    raise e
                    print(' Error on ' + str(file_name))

def compute_averages(directory_name):

    csum = np.zeros([1,13], float)
    d1_sum = np.zeros([1,13], float)
    d2_sum = np.zeros([1,13], float)

    phonemes = os.listdir(directory_name + 'extracted_features/')
    no_of_frames = 0
    for phoneme_folder in phonemes:
        if str(phoneme_folder) in phonemes_list:
            np_files = os.listdir(directory_name + 'extracted_features/' + phoneme_folder)

            for file in np_files:
                no_of_frames += 1
                current_mfcc = np.load(directory_name + 'extracted_features/' + phoneme_folder + '/' + file)
                # print(len(np.split(current_mfcc, 3)))
                # print(np.split(current_mfcc, 3))
                mfcc, d1, d2 = np.split(current_mfcc, 3)
                mfcc_sum = np.sum(mfcc, axis=0)
                cd1_sum = np.sum(d1, axis=0)
                cd2_sum = np.sum(d2, axis=0)

                csum = np.sum([csum,mfcc_sum], axis = 0)
                d1_sum = np.sum([d1_sum, cd1_sum], axis = 0)
                d2_sum = np.sum([d2_sum, cd2_sum], axis = 0)

    print(no_of_frames)

    mean = csum/no_of_frames
    d1_mean = d1_sum/no_of_frames
    d2_mean = d2_sum/no_of_frames

    return mean, d1_mean, d2_mean

def compute_sd(directory_name, mean, d1_mean, d2_mean):
    # here sum is used for sum of squared difference
    csum = np.zeros([1,13], float)
    d1_sum = np.zeros([1,13], float)
    d2_sum = np.zeros([1,13], float)

    phonemes = os.listdir(directory_name + 'extracted_features/')
    no_of_frames = 0
    for phoneme_folder in phonemes:
        if str(phoneme_folder) in phonemes_list:
            np_files = os.listdir(directory_name + 'extracted_features/' + phoneme_folder)

            for file in np_files:
                no_of_frames += 1
                current_mfcc = np.load(directory_name + 'extracted_features/' + phoneme_folder + '/' + file)
                mfcc, d1, d2 = np.split(current_mfcc, 3)

                # computing squared difference
                sq_dif = np.power((mfcc - mean), 2)
                d1_sq_dif = np.power((d1 - d1_mean), 2)
                d2_sq_dif = np.power((d2 - d2_mean), 2)

                # summing up for the frame for each cepstrum
                mfcc_sum = np.sum(sq_dif, axis=0)
                cd1_sum = np.sum(d1_sq_dif, axis=0)
                cd2_sum = np.sum(d2_sq_dif, axis=0)

                #
                csum = np.sum([csum, mfcc_sum], axis=0)
                d1_sum = np.sum([d1_sum, cd1_sum], axis=0)
                d2_sum = np.sum([d2_sum, cd2_sum], axis=0)

    sd = np.sqrt(csum/no_of_frames)
    d1_sd = np.sqrt(d1_sum/no_of_frames)
    d2_sd = np.sqrt(d2_sum/no_of_frames)

    return sd, d1_sd, d2_sd

def unit_normalize_extracted_features(directory_name, mean, d1_mean, d2_mean, sd, d1_sd, d2_sd):


    phonemes = os.listdir(directory_name + 'extracted_features/')
    if os.path.exists(directory_name+'normalized_mfcc/'):
        shutil.rmtree(directory_name+'normalized_mfcc/')
    os.mkdir(directory_name+'normalized_mfcc/')

    for phoneme in phonemes_list:
        if os.path.exists(directory_name+'normalized_mfcc/'+phoneme):
            shutil.rmtree(directory_name+'normalized_mfcc/'+phoneme)
        os.mkdir(directory_name+'normalized_mfcc/'+phoneme)

    for phoneme_folder in phonemes:
        if str(phoneme_folder) in phonemes_list:
            np_files = os.listdir(directory_name + 'extracted_features/'+ phoneme_folder)
            for file in np_files:
                phoneme = file.split('_')[-2]
                file_name = file.replace('.npy', '')
                current_mfcc = np.load(directory_name + 'extracted_features/' + phoneme_folder + '/' + file)
                mfcc, d1, d2 = np.split(current_mfcc, 3)

                nmfcc = (mfcc - mean)/sd
                nd1 = (d1 - d1_mean)/d1_sd
                nd2 = (d2 - d2_mean) / d2_sd
                np.save(directory_name + 'normalized_mfcc/' + phoneme + '/' + file_name,
                        np.concatenate([np.concatenate([nmfcc, nd1], axis=0), nd2], axis=0))


# This method flattens the array to one diemensional array in the order of normalized mfcc, 1st derivative, then 2nd derivative for each cepstrum
def flatten(directory_name):
    if os.path.exists(directory_name+'flattened_mfcc/'):
        shutil.rmtree(directory_name+'flattened_mfcc/')
    os.mkdir(directory_name+'flattened_mfcc/')

    for phoneme in phonemes_list:
        if os.path.exists(directory_name+'flattened_mfcc/'+phoneme):
            shutil.rmtree(directory_name+'flattened_mfcc/'+phoneme)
        os.mkdir(directory_name+'flattened_mfcc/'+phoneme)

    phonemes = os.listdir(directory_name + 'normalized_mfcc/')
    for phoneme_folder in phonemes:
        if str(phoneme_folder) in phonemes_list:
            np_files = os.listdir(directory_name + 'normalized_mfcc/'+ phoneme_folder)
            for file in np_files:
                feature_vector = []
                phoneme = file.split('_')[-2]
                file_name = file.replace('.npy', '')
                phone_mfcc_frames = np.load(directory_name + 'normalized_mfcc/' + phoneme + '/' + file_name+'.npy')
                number_of_windows = len(phone_mfcc_frames)/3
                mfcc, d1, d2 = np.split(phone_mfcc_frames, 3)

                for wind_index in range(int(number_of_windows)):

                    for cep_coeff_index in range(13):
                        feature_vector.append(mfcc[wind_index][cep_coeff_index])
                        feature_vector.append(d1[wind_index][cep_coeff_index])
                        feature_vector.append(d2[wind_index][cep_coeff_index])
                np.save( directory_name + 'flattened_mfcc/' + phoneme + '/' + file_name, np.asarray(feature_vector,float))







wrk_dir = os.getcwd()
data_sets = [wrk_dir+'/timit_data/train/nosa/phonemes/', wrk_dir+'/timit_data/test/nosa/phonemes/']

mean, d1_mean, d2_mean = None, None, None
sd, d1_sd, d2_sd = None, None, None
for item in data_sets:
    print(item)
    print(str(time.ctime()))
    extract_features(item) #- #commented out temp as extraction is done

    # computing mean and sd only for train data and using the same to normalize test data.
    if item == wrk_dir+'/timit_data/train/nosa/phonemes/':
         mean, d1_mean, d2_mean = compute_averages(item)
         sd, d1_sd, d2_sd = compute_sd(item, mean, d1_mean, d2_mean)
    unit_normalize_extracted_features(item, mean, d1_mean, d2_mean, sd, d1_sd, d2_sd)

    flatten(item)

print(mean, d1_mean, d2_mean, sd, d1_sd, d2_sd)
#print np.load('/Users/naveenkumar2703/PycharmProjects/UniversalSpeechRecognition/timit_data/train/nosa/phonemes/extracted_features/aa/dr1_fcjf0_si1657_aa_0.npy')


# values of mean, d1_mean, d2_mean, sd, d1_sd, d2_sd
# [[  91.13916113  -76.93350378  -78.19828892  -46.38542021 -106.78014944
#    -65.37575041  -50.64767956  -44.89397624   -1.64586778  -30.70583561
#     -9.87635851  -23.15012079  -23.12218467]] [[-28.51664252 -27.73673651 -23.36637725 -16.24932375  -3.98184407
#     7.89472157   9.18461355   8.3665702    9.28399218   4.78079946
#     2.33367392   0.40966152  -1.49363385]] [[-5.55173165 -5.39040519 -4.50075893 -2.94973428 -0.71511854  2.37763232
#    4.49106025  5.50279831  5.81264143  4.84032496  3.21490159  1.32423383
#   -0.49649982]] [[ 204.71338742  179.26650839  179.53701865  112.08947743  243.16188326
#    152.74045228  120.69031066  108.33061498   38.4597235    77.80022957
#     38.52565523   60.32462383   58.80479616]] [[ 64.13987248  62.40869319  52.63948571  36.80347495  10.10073778
#    18.4629577   21.29773551  19.2986981   21.33134272  11.88718215
#     7.49556729   5.3759007    6.57722845]] [[ 12.47231809  12.11338026  10.12249321   6.66233725   1.81534451
#     5.43032574  10.13217944  12.39946847  13.09682706  10.93029892
#     7.32326635   3.22966388   1.68489764]]