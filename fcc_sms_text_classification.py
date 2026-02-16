# import libraries 
try: 
    # %tensorflow_version only exists in Colab. 
    !pip install tf-nightly 
    except Exception: 
    pass 
import tensorflow as tf 
import pandas as pd 
from tensorflow import keras 
!pip install tensorflow-datasets 
import tensorflow_datasets as tfds 
import numpy as np 
import matplotlib.pyplot as plt 
print(tf.__version__) 

# get data files 
!wget https://cdn.freecodecamp.org/project-data/sms/train-data.tsv
!wget https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

# Load datasets
train_df = pd.read_csv(train_file_path, sep='\t', header=None, names=['label', 'message'])
test_df = pd.read_csv(test_file_path, sep='\t', header=None, names=['label', 'message'])

# Convert labels to binary (ham=0, spam=1)
train_df['label'] = train_df['label'].map({'ham': 0, 'spam': 1})
test_df['label'] = test_df['label'].map({'ham': 0, 'spam': 1})

train_messages = train_df['message'].values
train_labels = train_df['label'].values

test_messages = test_df['message'].values
test_labels = test_df['label'].values

vocab_size = 10000
embedding_dim = 32
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"

tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=vocab_size,
    oov_token=oov_tok
)

tokenizer.fit_on_texts(train_messages)

train_sequences = tokenizer.texts_to_sequences(train_messages)
train_padded = tf.keras.preprocessing.sequence.pad_sequences(
    train_sequences,
    maxlen=max_length,
    padding=padding_type,
    truncating=trunc_type
)

test_sequences = tokenizer.texts_to_sequences(test_messages)
test_padded = tf.keras.preprocessing.sequence.pad_sequences(
    test_sequences,
    maxlen=max_length,
    padding=padding_type,
    truncating=trunc_type
)

model = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    keras.layers.Bidirectional(keras.layers.LSTM(32)),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    train_padded,
    train_labels,
    epochs=10,
    validation_data=(test_padded, test_labels),
    verbose=2
)

# function to predict messages based on model 
# (should return list containing prediction and label, ex. [0.008318834938108921, 'ham']) 
def predict_message(pred_text):

    sequence = tokenizer.texts_to_sequences([pred_text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        sequence,
        maxlen=max_length,
        padding=padding_type,
        truncating=trunc_type
    )

    prob = model.predict(padded)[0][0]

    label = "spam" if prob >= 0.5 else "ham"

    return [float(prob), label]

# Run this cell to test your function and model. Do not modify contents. 
def test_predictions(): 
    test_messages = ["how are you doing today", 
                     "sale today! to stop texts call 98912460324", 
                     "i dont want to go. can we try it a different day? available sat", 
                     "our new mobile video service is live. just install on your phone to start watching.", 
                     "you have won £1000 cash! call to claim your prize.", 
                     "i'll bring it tomorrow. don't forget the milk.", 
                     "wow, is your arm alright. that happened to me one time too" ] 
    test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"] 
    passed = True 
    for msg, ans in zip(test_messages, test_answers): 
        prediction = predict_message(msg) 
        if prediction[1] != ans: 
            passed = False 
            if passed: 
                print("You passed the challenge. Great job!") 
            else: 
                print("You haven't passed yet. Keep trying.") 
        test_predictions() 
        