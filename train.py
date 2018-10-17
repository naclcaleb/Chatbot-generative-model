import tensorflow
from tensorflow import keras
import numpy as np

#Easy-acces variables
max_string_length = 20

#hyperparameters
batch_size = 64
epochs = 100

#Fetch data
train_data = []
target_data = []
with open("data/train_dialogue.txt") as file:
    txt = file.read()
    txts = txt.split("\n")
    for i in range(0, len(txts)-1):
        #Every other line is target data
        if i%2==0:
            train_data.append("\t " + txts[i] + " \n")
        else:
            target_data.append("\t " + txts[i] + " \n")

#Preprocess the data

print(target_data)

#Tokenize and measure vocabulary
train_tokenizer = keras.preprocessing.text.Tokenizer()
train_tokenizer.fit_on_texts(train_data)

train_vocab = train_tokenizer.word_index
train_seqs = train_tokenizer.texts_to_sequences(train_data)

target_tokenizer = keras.preprocessing.text.Tokenizer()
target_tokenizer.fit_on_texts(target_data)
target_vocab = target_tokenizer.word_index
target_seqs = target_tokenizer.texts_to_sequences(target_data)

#Pad the sequences
keras.preprocessing.sequence.pad_sequences(
train_seqs, maxlen=max_string_length, dtype="int32", padding="pre", truncating="pre", value=0.0
)
keras.preprocessing.sequence.pad_sequences(
target_seqs, maxlen=max_string_length, dtype="int32", padding="pre", truncating="pre", value=0.0
)


train_seqs = np.array(train_seqs)
target_seqs = np.array(target_seqs)

#Create the encoder model
encoder_inputs = keras.layers.Input(shape=(None,))
x = keras.layers.Embedding(len(train_seqs), 256)(encoder_inputs)
x, state_h, state_c = keras.layers.LSTM(256, return_state=True)(x)
encoder_states = [state_h, state_c]

#Create the decoder model
decoder_inputs = keras.layers.Input(shape=(None,))
x = keras.layers.Embedding(len(target_seqs), 256)(decoder_inputs)
x = keras.layers.LSTM(256, return_sequences=True)(x, initial_state=encoder_states)
decoder_outputs = keras.layers.Dense(len(target_seqs), activation="softmax")(x)


#Combine them into one model
model = keras.models.Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')


#Train the model
model.fit([train_seqs, target_seqs[:len(target_seqs)-1]],target_seqs[1:len(target_seqs)], batch_size = batch_size, epochs=epochs, validation_split=0.2 )
