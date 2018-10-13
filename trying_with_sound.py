#Import dependencies
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keas.layers import Input, LSTM, Dense
from load_sound_data import process_data

#Easy-access variables
sound_file_dir = "./data/sound/"

#Very high input sizes for sound files
max_sequence_length = 1000000
eos_token = 0.001
pad_token = 0


#Fetch and preprocess data
train_data, target_data = process_data(sound_file_dir, eos_token, pad_token, max_sequence_length)


#Load encoder and decoder networks
encoder_inputs = Input(shape=(None, max_sequence_length))
encoder = LSTM(latent_dim, return_state=True)

#Train them on trainign data

#Test them on target data
