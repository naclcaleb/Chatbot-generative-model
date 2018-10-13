import numpy as np
import scipy.io.wavfile as wvfle
import os

def process_data(dir, eos, pad, l):
    #Define file arrays
    train_data_files = []
    target_data_files = []

    #Populate file arrays
    for sound_file_name in os.listdir(dir):
        #Check if file is a feature or a label
        if "_train" in sound_file_name:
            #It's a feature
            train_data_files.append(sound_file_name)
        else:
            #It's a target
            target_data_files.append(sound_file_name)

    #Create return arrays
    train_data = []
    target_data = []

    #Populate return arrays
    for i in train_data_files:
        numpy_audio_array = wvfle.read(dir + "/" + i)[1]
        train_data.append(numpy_audio_array)
    for i in target_data_files:
        numpy_audio_array = wvfle.read(dir + "/" + i)[1]
        target_data.append(numpy_audio_array)



    #Pad each item in training array, and add an EOS token
    for i in range(len(train_data)):
        train_data[i] = train_data[i].tolist()
        train_pad_num = l - len(train_data[i])
        print(train_pad_num)
        for j in range(1,train_pad_num-1):
            train_data[i].append(pad)

        train_data[i].append(eos)

    #Convert back to numpy arrays
    for i in range(len(train_data)):
        train_data[i] = np.array(train_data[i])
    return train_data, target_data
