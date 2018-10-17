#Import dependencies
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keas.layers import Input, LSTM, Dense
from load_sound_data import process_data
import numpy as np

#Easy-access variables
sound_file_dir = "./data/sound/"
epochs = 10

#Very high input sizes for sound files
max_sequence_length = 1000000
eos_token = 0.001
pad_token = 0


#Fetch and preprocess data
train_data, target_data = process_data(sound_file_dir, eos_token, pad_token, max_sequence_length)

timestepped_target = target_data
for i in range(len(timestepped_target)):
    timestepped_target[i] = np.delete(timestepped_target[i], 0)
    timestepped_target[i] = np.append(timestepped_target[i], eos_token)

#Load encoder and decoder networks
encoder_inputs = Input(shape=(max_sequence_length, 1))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(max_sequence_length, 1))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(max_sequence_length, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

#Train them on training data
model.fit([train_data, target_data], timestepped_target,
          batch_size=len(train_data)/epochs,
          epochs=epochs,
          validation_split=0.2)



#Test them on target data
