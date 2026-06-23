import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.utils import to_categorical

texts = [
    "The match was exciting and the team played well",
    "He scored the winning goal in the last minute",
    "New smartphone models are released every year",
    "The latest software update improves performance",
    "A balanced diet helps maintain good health",
    "Regular exercise is important for wellness"
]
labels = [0, 0, 1, 1, 2, 2]

max_words = 100
tokenizer = Tokenizer(num_words=max_words, lower=True)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

max_len = 10
X = pad_sequences(sequences, maxlen=max_len, padding='post')
X = np.asarray(X, dtype=np.int32)

y = to_categorical(labels, num_classes=3)

inputs = Input(shape=(max_len,))
x = Embedding(input_dim=max_words, output_dim=16)(inputs)
x = Bidirectional(LSTM(20, dropout=0.2, recurrent_dropout=0.2))(x)
outputs = Dense(3, activation='softmax')(x)
model = Model(inputs, outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(X, y, epochs=20, batch_size=2, verbose=1)

# Predict a new sentence
new_text = ["I love learning about new technology"]
new_seq = tokenizer.texts_to_sequences(new_text)
new_padded = pad_sequences(new_seq, maxlen=max_len, padding='post')
pred = model.predict(new_padded)
class_idx = pred.argmax(axis=-1)[0]

category_names = ['sports', 'technology', 'health']
print(f"Input: {new_text[0]}")
print(f"Predicted category: {category_names[class_idx]}")
print(f"Scores: {pred[0].round(4)}")
