# Authors: Naveenkumar Ramaraju, Roshan Kathawate
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import os
from random import shuffle
import random


phonemes_list = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx',
                     'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix',
                     'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's',
                     'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']
phonemes_list_1 = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx',
                     'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix',
                     'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's',
                     'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']

reduction_class = {'aa':['aa', 'ao'],'ah':['ah', 'ax', 'ax-h'],'er':['er', 'axr'],
                 'hh':['hh', 'hv'],'ih':['ih', 'ix'],'l':['l', 'el'],'m':['m', 'em'],
                 'n':['n', ' en', 'nx'], 'ng':['ng', 'eng'], 'sh':['sh', 'zh'],
                 'uw':['uw', 'ux'], 'pcl':['pcl', 'tcl', 'kcl', 'bcl', 'dcl', 'gcl', 'h#', 'pau', 'epi'], # using 'pcl' as key while suggestion is 'sil' to reuse phoneme list.
                 'q': []}

# To reduce 61 class to 39 classes
reduction_map = {'aa':'aa', 'ao':'aa',
                 'ah':'ah', 'ax':'ah', 'ax-h':'ah',
                 'er':'er', 'axr':'er',
                 'hh':'hh', 'hv':'hh',
                 'ih':'ih', 'ix':'ih',
                 'l':'l', 'el':'l',
                 'm':'m', 'em':'m',
                 'n':'n', 'en':'n', 'nx':'n',
                 'ng':'ng', 'eng':'ng',
                 'sh':'sh', 'zh':'sh',
                 'uw':'uw', 'ux':'uw',
                 'pcl':'pcl', 'tcl':'pcl', 'kcl':'pcl', 'bcl':'pcl', 'dcl':'pcl', 'gcl':'pcl', 'h#':'pcl', 'pau':'pcl', 'epi':'pcl', 'q':'q'}

reduced_phones = ['y', 's', 'w', 'g', 'th', 'dh', 'eh', 'b', 'en', 'f', 'ow', 'uh', 'v', 'z', 'p', 't', 'ey', 'q', 'jh', 'oy', 'ch', 'd', 'ay', 'r', 'k', 'ae', 'dx', 'iy', 'aw']


phones_to_reduced = ['aa', 'ao', 'ah', 'ax', 'ax-h','er', 'axr', 'hh', 'hv', 'ih', 'ix',
                     'l', 'el', 'm', 'em', 'n', ' en', 'nx', 'ng', 'eng','sh', 'zh',
                     'uw', 'ux','pcl', 'tcl', 'kcl', 'bcl', 'dcl', 'gcl', 'h#', 'pau', 'epi'] # ignoring 'q' as it has only one mapping

wrk_dir = os.getcwd()
train_folder = wrk_dir+'/timit_data/train/nosa/phonemes/flattened_mfcc/'
test_folder = wrk_dir+'/timit_data/test/nosa/phonemes/flattened_mfcc/'

train_phone_file_index = {}
cv_phone_file_index = {}
test_phone_file_index = {}
phone_index_dict = {}




def constructOneHotEncodedLabels(phoneme):
    output = np.zeros([61], float) # 61 for 61 class labels
    output[phone_index_dict[phoneme]] = 1.
    return output

def fix_window_size(number_of_windows_in_phoneme, phoneme_vector):
    number_of_windows_in_phoneme = int(number_of_windows_in_phoneme)
    return_array = None
    if number_of_windows_in_phoneme == 11:
        return_array = np.array(phoneme_vector, float)
    elif number_of_windows_in_phoneme > 11:
        extra_frames = number_of_windows_in_phoneme - 11
        if extra_frames % 2 == 0:  # truncating same number of frames at the end
            trunc_phoneme_data = phoneme_vector[
                                 int(extra_frames / 2) * 39:(len(phoneme_vector) - int(extra_frames / 2) * 39)]
            return_array = np.array(trunc_phoneme_data, float)
        else:
            if bool(random.getrandbits(1)):  # truncated more in front
                if extra_frames == 1:
                    trunc_phoneme_data = phoneme_vector[int(extra_frames) * 39:]
                    return_array = np.array(trunc_phoneme_data, float)

                else:
                    trunc_phoneme_data = phoneme_vector[(int(extra_frames / 2) + 1) * 39:(
                    len(phoneme_vector) - int(extra_frames / 2) * 39)]
                    return_array = np.array(trunc_phoneme_data, float)

            else:  # truncate more in back
                if extra_frames == 1:
                    trunc_phoneme_data = phoneme_vector[:len(phoneme_vector) - int(extra_frames) * 39]
                    return_array = np.array(trunc_phoneme_data, float)

                else:
                    trunc_phoneme_data = phoneme_vector[int(extra_frames / 2) * 39:(
                    len(phoneme_vector) - (int(extra_frames / 2) + 1) * 39)]
                    return_array = np.array(trunc_phoneme_data, float)
    else:
        # currently centering and filling missing values with zeros - TODO try random appending from same speaker
        shortage_features = (11 - number_of_windows_in_phoneme) * 39
        extended_vector = []

        for index in range(int(shortage_features / 2)):
            extended_vector.append(0.)
        extended_vector.extend(phoneme_vector)
        for index in range(int(int(shortage_features / 2) + number_of_windows_in_phoneme * 39), 429):
            extended_vector.append(0.)
        return_array = np.array(extended_vector,float)
    return_array = np.reshape(return_array, (39, 11))
    return return_array

def map_classification_to_reduced_class(class_labels):
    reduced_labels = []
    for label_index in class_labels:
        if phonemes_list[label_index] in reduction_map.keys():
            equivalent_phone = reduction_map[phonemes_list[label_index]]
            reduced_labels.append(phonemes_list.index(equivalent_phone))
        else:
            reduced_labels.append(label_index)

    #reduced_labels = np.fromiter(reduced_labels, float).reshape(len(reduced_labels),1)
    return reduced_labels


def create_test_data():
    td = []
    tdl = []
    tdln = []
    reduced_tdl = []

    for phoneme in test_phone_file_index:
        files = test_phone_file_index[phoneme]
        for file in files:
            tdln.append(phonemes_list.index(phoneme))
            #tdl.append(phonemes_list.index(phoneme))
            tdl.append(constructOneHotEncodedLabels(phoneme))
            phoneme_vector = np.load(test_folder + phoneme + '/' + file)
            number_of_windows_in_phoneme = len(phoneme_vector) / 39  # 13 each for mfcc, d1, d2
            # handling varying number of frames in phonemes
            td.append(fix_window_size(number_of_windows_in_phoneme, phoneme_vector))
            if phoneme in reduction_map.keys():
                equivalent_phone = reduction_map[phoneme]
                reduced_tdl.append(phonemes_list.index(equivalent_phone))
            else:
                reduced_tdl.append(phonemes_list.index(phoneme))

    td = np.array(td)
    tdl = np.array(tdl)
    tdln = np.array(tdln)
    #reduced_tdl = np.fromiter(reduced_tdl,float)
    #reduced_tdl = reduced_tdl.reshape((len(reduced_tdl), 1))
    return td, tdl, reduced_tdl, tdln


def create_cv_data():
    cv = []
    cvl = []

    for phoneme in cv_phone_file_index:
        files = cv_phone_file_index[phoneme]
        for file in files:
            #cvl.append(constructOneHotEncodedLabels(phoneme))
            cvl.append(phonemes_list.index(phoneme))
            phoneme_vector = np.load(train_folder + phoneme + '/' + file)
            number_of_windows_in_phoneme = len(phoneme_vector) / 39  # 13 each for mfcc, d1, d2
            # handling varying number of frames in phonemes
            cv.append(fix_window_size(number_of_windows_in_phoneme, phoneme_vector))

    cv = np.array(cv)
    cvl = np.array(cvl)
    return cv, cvl

def initialize():
    for phone_index in range(len(phonemes_list)):
        phoneme = phonemes_list[phone_index]
        phone_index_dict[phoneme] = phone_index
        train_folders = os.listdir(train_folder + phoneme)
        shuffle(train_folders)  # shuffling to avoid speaker/dr bias
        # splitting 0.85 as train and remain as cross validation
        train_phone_file_index[phoneme] = train_folders[:int(len(train_folders) * .85)]
        cv_phone_file_index[phoneme] = train_folders[int(len(train_folders) * .85):]
        test_phone_file_index[phoneme] = os.listdir(test_folder + phoneme)
    #create_cv_data()

# This method gets phoneme data in batch
def get_batch_data(directory, batch, batch_size):
    phoneme_data = []
    phoneme_label = []
    shuffle(phonemes_list_1) # shuffling the phones to change order in different calls
    for phoneme in phonemes_list_1:
        number_of_windows_in_phoneme = None
        phoneme_vector = None
        batch_start = (batch) * batch_size
        batch_end = None

        if (batch+1)* batch_size < len(train_phone_file_index[phoneme]):
            batch_end = (batch+1)* batch_size

        elif batch* batch_size < len(train_phone_file_index[phoneme]):
            batch_end = len(train_phone_file_index[phoneme])
        else:
            continue

        for ex_index in range(batch_start, batch_end):#((batch) * batch_size, len(train_phone_file_index[phoneme])):
            phoneme_label.append(constructOneHotEncodedLabels(phoneme))
            phoneme_vector = np.load(directory + phoneme +'/'+train_phone_file_index[phoneme][ex_index])
            number_of_windows_in_phoneme = len(phoneme_vector)/39  # 13 each for mfcc, d1, d2

            # handling varying number of frames in phonemes
            temp = fix_window_size(number_of_windows_in_phoneme, phoneme_vector)
            phoneme_data.append(fix_window_size(number_of_windows_in_phoneme, phoneme_vector))

    # for item in phoneme_data:
    #     print(item.shape)
    #print(phoneme_data)
    return np.array(phoneme_data, float), np.array(phoneme_label, float)


# Parameters
learning_rate = 0.001
training_iters = 100
batch_size = 128
display_step = 10

# Network Parameters
n_input = 39  # MNIST data input (img shape: 28*28)
n_steps = 11  # timesteps
n_hidden = 140  # hidden layer num of features
n_classes = 61  # 61 phonemes

# tf Graph input
x = tf.placeholder("float", [None, n_input, n_steps])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_steps])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_input, x)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

initialize()
print('Done Initialization')
cv,cvl = create_cv_data()
print('Created cv data')

# Initializing the variables
init = tf.initialize_all_variables()
reduced_predicted_phonemes = tf.placeholder(tf.float32)
reduced_expected_phonemes = tf.placeholder(tf.float32)
reduced_correct_prediction = tf.equal(reduced_expected_phonemes, reduced_predicted_phonemes)
reduced_label_accuracy = tf.reduce_mean(tf.cast(reduced_correct_prediction, tf.float32))

print('Commencing training')
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    cv_accuracy = 0
    epoch = 0
    # Keep training until reach max iterations
    while cv_accuracy < .5 and epoch < 50:
        print('Starting Epoch - ' + str(epoch))
        for batch_no in range(1000):
            features, labels = get_batch_data(train_folder, batch_no, 2)
            if len(labels) > 0:
            # Run optimization op (backprop)
                sess.run(optimizer, feed_dict={x: features, y: labels})

        if epoch % 1 == 0:
            predictions = sess.run(tf.argmax(pred, 1), feed_dict={x: cv})
            cv_accuracy = reduced_label_accuracy.eval(feed_dict={reduced_expected_phonemes: cvl,
                                                                     reduced_predicted_phonemes: predictions})
            print("step %d, cross validation accuracy %g" % (epoch, cv_accuracy))
        epoch += 1

    td, tdl, reduced_tdl, ntdl = create_test_data()
    # td_accuracy = sess.run(accuracy, feed_dict={input_features: td, expected_phonemes: tdl})
    predictions = sess.run(tf.argmax(pred, 1), feed_dict={x: td})
    td1_accuracy = reduced_label_accuracy.eval(feed_dict={reduced_expected_phonemes: ntdl,
                                                          reduced_predicted_phonemes: predictions})
    # print(predictions.tolist()[:300])
    # print(ntdl[:300])
    # print("Accuracy1 on test data is %g" % (td_accuracy))
    print("Accuracy on test data is %g" % (td1_accuracy))

    classifications = sess.run(tf.argmax(pred, 1), feed_dict={x: td})
    reduced_predicted_labels = map_classification_to_reduced_class(classifications)
    # print(ntdl)
    # print(classifications.tolist())
    # print(reduced_tdl)
    # print(reduced_predicted_labels)
    reduced_td_accuracy = reduced_label_accuracy.eval(
        feed_dict={reduced_expected_phonemes: reduced_tdl, reduced_predicted_phonemes: reduced_predicted_labels})
    print("Accuracy on test data with 39 labels is %g" % (reduced_td_accuracy))
