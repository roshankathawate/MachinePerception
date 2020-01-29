*****************************************************************
* Machine perception and Audition - Project - Fall 2016         *
* Submitted by - Roshan Kathawate, Naveenkumar Ramaraju         *
* Professor - Donald Williamson                                 *
* Comparison of Different Algorithms for Phoneme Recognition    *
* Programming Language - Python 3.5                             *
* Package Dependencies - numpy, scipy.io,python_speech_features,*
*                         shutil, librosa and tensorflow        *
* Executed environment - Mac Pro, i7, 8GB RAM                   *
*****************************************************************

Data: TIMIT Data set was used for this project. Place the timit data with the name timit_data with train and test folders as provided by the timit inside the folder timit_data.

Execute the below instructions in the same order to run the code. All the files that ends with '.py' can be executed using any python interpreter after having the dependencies listed above in the runtime of the interpreter.
1) Inside /timit_data/train/ create a folder name 'nosa'. /timit_data/train/ has all the sentences separated by dialect region.
2) Inside /timit_data/test/ create a folder name 'nosa'. /timit_data/test/ has all the test sentences separated by dialect region.

'nosa' - folders indicate that the sentences that are of type 'sa' is not included while splitting words and phonemes.

3) audio_convert.py - This file formats the audio file and removes RIFF issue faced when reading the file using scipy. This would take about two minutes

4) split_data_to_words_phones.py - This file splits sentences into words and phonemes and stores them into individual wav files. This would take a minute.

5) Now manually create folder - extracted_features in side /timit_data/train/nosa/phonemes
6) Now manually create folder for test - extracted_features in side /timit_data/test/nosa/phonemes

'extracted_features' - stores the computed mfcc in .npy format for faster accesible to the data during train and test.

7) feature_extraction.py - This file reads audio files as '.wav' format and converts them to mfcc, with first order and second order derivative. And it will unit normalize the and store it as flattened '.npy' array. This would take about 15 minutes.

8) phone_classification.py - This file trains and predicts the phoneme using DNN. To change the number of layers, edit the code with 'TODO' comment.
9) lstm_phone_classification.py - This file trains and predicts the phoneme using RNN-LSTM.

References:
1) python_speech_features - https://pypi.python.org/pypi/python_speech_features
2) librosa - https://github.com/librosa/librosa
3) scipy - https://www.scipy.org/
4) Tensorflow - https://www.tensorflow.org/
5) Tensorflow - https://www.tensorflow.org/versions/r0.12/tutorials/mnist/beginners/index.html
6) https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
7) https://tspace.library.utoronto.ca/bitstream/1807/44123/1/Mohamed_Abdel-rahman_201406_PhD_thesis.pdf
8) https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/CNN_ASLPTrans2-14.pdf
9) http://cdn.intechopen.com/pdfs/15948/InTech-Phoneme_recognition_on_the_timit_database.pdf
10) https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf