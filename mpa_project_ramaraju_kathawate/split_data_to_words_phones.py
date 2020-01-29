# This file splits audio sententences to words and phones
# Stores the words and phones into respective directory
# @ author Naveen

from scipy.io import wavfile
import os
import glob
import shutil
import time



def validate_text_audio_length_match(audio_length, text_file_location):
    valid_file = False
    try:
        with open(str(text_file_location), 'r', encoding='utf-8') as file:
            length_from_file = int(file.readline().split(' ')[1])
            valid_file = (length_from_file == audio_length)
            if not valid_file:
                print('Mismatch in file: ' + str(text_file_location))
                print('Expected: ' + str(audio_length)+', Actual: '+ str(length_from_file))
    except:
        print('Unable to open file - '+str(text_file_location))


    return True


# This file reads the input file and returns components in it with start and end location
# File is expected to have form - 'start_duration<space>end_duration<space>component
def get_parts_and_duration(file_location):
    component_dict = {}
    try:
        with open(str(file_location), 'r', encoding='utf-8') as file:
            for line in file:
                line_data = line.rstrip('\n').split(' ')
                if len(line_data) == 3:
                    start_duration = int(line_data[0])
                    end_duration = int(line_data[1])
                    pos = str(line_data[2]) # part of speech

                    if pos in component_dict.keys():
                        component_dict[pos].append([start_duration,end_duration])
                    else:
                        component_dict[pos] = [[start_duration,end_duration]]


    except:
        print('Unable to process file - '+str(file_location))
    return component_dict


def split_audio_to_words_and_phones(directory_name):
    # deleting and creating directory for phonemes and words
    phoneme_directory = directory_name + 'nosa/phonemes/'
    words_directory = directory_name + 'nosa/words/'
    if os.path.exists(phoneme_directory):
        shutil.rmtree(phoneme_directory)
    os.mkdir(phoneme_directory)
    if os.path.exists(words_directory):
        shutil.rmtree(words_directory)
    os.mkdir(words_directory)


    # getting list of wav files
    wav_files_list = glob.glob(directory_name+'/*/*/*.wav')

    dot_wav_suffix = '.wav'
    dot_txt_suffix = '.txt'
    dot_phn_suffix = '.phn'
    dot_wrd_suffix = '.wrd'
    # creating a list of prefixes - without .wav
    audio_file_prefixes = []
    for file in wav_files_list:
        file_name, file_extension = os.path.splitext(file)
        if not (str(file_name).endswith('sa1') or str(file_name).endswith('sa2')):
            audio_file_prefixes.append(str(file_name))

    for sentence in audio_file_prefixes:
        dr_speaker_sent = sentence.split(directory_name)[1] # dr stands for dialect region
        dr_speaker_sent = dr_speaker_sent.replace('/','_')
        text_file_name = sentence + dot_txt_suffix
        word_file_name = sentence + dot_wrd_suffix
        phone_file_name = sentence + dot_phn_suffix
        audio_file_name = sentence + dot_wav_suffix
        rate, audio = wavfile.read(audio_file_name)

        if validate_text_audio_length_match(len(audio), text_file_name):
            #print('Processing file - ' + str(dr_speaker_sent))

            # splitting by words
            words_duration = get_parts_and_duration(word_file_name)

            for word in words_duration:
                current_word_directory = words_directory + word + '/'
                if not os.path.exists(current_word_directory):
                    os.mkdir(current_word_directory)

                current_word_file_name_pref = current_word_directory + dr_speaker_sent + '_' + word +'_'

                for index in range(len(words_duration[word])):
                    clipped_audio = audio[words_duration[word][index][0]:words_duration[word][index][1]]
                    wavfile.write(current_word_file_name_pref + str(index) + dot_wav_suffix, rate, clipped_audio)

            # Now splitting by phones
            phones_duration = get_parts_and_duration(phone_file_name)

            for phoneme in phones_duration:
                current_phoneme_directory = phoneme_directory + phoneme + '/'
                if not os.path.exists(current_phoneme_directory):
                    os.mkdir(current_phoneme_directory)

                current_phoneme_file_name_pref = current_phoneme_directory + dr_speaker_sent + '_' + phoneme + '_'

                for index in range(len(phones_duration[phoneme])):
                    start_time = phones_duration[phoneme][index][0]
                    end_time = phones_duration[phoneme][index][1]
                    mid_point = int((start_time + end_time)/2)
                    # clip_start = 0
                    # clip_end = 150 * 16
                    #
                    # if mid_point > (75 * 16) and (end_time - mid_point - (75 * 16)) > 0:
                    #     clip_start = mid_point - (75 * 16)
                    #     clip_end = mid_point + (75 * 16)
                    # else:
                    #     clip_end = end_time
                    #     clip_start = end_time - (150 * 16)


                    clipped_audio = audio[start_time:end_time]
                    wavfile.write(current_phoneme_file_name_pref + str(index) + dot_wav_suffix, rate, clipped_audio)



wrk_dir = os.getcwd()

data_sets = [wrk_dir+'/timit_data/train/', wrk_dir+'/timit_data/test/']

for item in data_sets:
    print(item)
    print(str(time.ctime()))
    split_audio_to_words_and_phones(item)


print(str(time.ctime()))

