import json
import numpy as np
import random
import tensorflow as tf
import pickle

# Load processed files
with open("words.pkl", "rb") as f:
    words = pickle.load(f)
with open("classes.pkl", "rb") as f:
    classes = pickle.load(f)
with open("intents.json", "r") as f:
    intents = json.load(f)

# Tokenizer function
def simple_tokenizer(text):
    return text.lower().replace("?", "").replace("!", "").replace(",", "").replace(".", "").split()

# Build training data
training = []
output_empty = [0] * len(classes)

for intent in intents['intents']:
    for pattern in intent['text']:
        bag = []
        tokenized = simple_tokenizer(pattern)
        for w in words:
            bag.append(1 if w in tokenized else 0)

        output_row = output_empty[:]
        output_row[classes.index(intent['intent'])] = 1

        training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Build and train the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(train_y[0]), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save the model
model.save("model.h5")
print("✅ model.h5 has been created successfully!")
