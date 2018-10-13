#Import dependencies
import tensorflow as tf

#Fetch and preprocess data

#Define a tokenizer function
def tokenize(string):
    tokenized_list = []
    tmp_indx = 0
    for i in range(len(string)):
        if string[i] in "?.,!;":
            tokenized_list.append(string[tmp_indx:i])
            tokenized_list.append(string[i])
            tmp_indx = i+1
        elif string[i] == " ":
            tokenized_list.append(string[tmp_indx:i])
            tmp_indx = i+1
            
    #A quick and dirty way out :/
    tokenized_list = [x for x in tokenized_list if x!=""]
            
            
    return tokenized_list


raw_file_data = ""
with open("training_dialogue.txt") as file:
    raw_file_data = file.read()
raw_file_data = raw_file_data.split("\n")
    
#Train data as list of values like so: [query, target_response]
train_data = []

for i in range(0,len(raw_file_data)):
    if i%2!=0:
        #Perform the most basic tokenization algorithm
        query = tokenize(raw_file_data[i-1])
        target_response = tokenize(raw_file_data[i])
        train_data.append([query, target_response])

#Now that I have a list of tokens in the form of strings, I need to map these to numbers somehow


#Load encoder and decoder networks



#Define hyperparameters


#Train them on the data



#Test them on other data
