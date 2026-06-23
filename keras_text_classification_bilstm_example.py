import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense

texts = [
    "I love this product",
    "This is the best purchase I made",
    "I am very happy with the quality",
    "The product is terrible",
    "I do not like this at all",
    "This was a bad choice"
]
labels = [1, 1, 1, 0, 0, 0]

max_words = 50
tokenizer = Tokenizer(num_words=max_words, lower=True)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

max_len = 6
X = pad_sequences(sequences, maxlen=max_len, padding='post')
X = np.asarray(X, dtype=np.int32)

y = np.asarray(labels, dtype=np.float32)

model = Sequential([
    Embedding(input_dim=max_words, output_dim=16),
    Bidirectional(LSTM(16, dropout=0.2, recurrent_dropout=0.2)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X, y, epochs=15, batch_size=2, verbose=1)

new_text = ["I do not like the service"]
new_seq = tokenizer.texts_to_sequences(new_text)
new_padded = pad_sequences(new_seq, maxlen=max_len, padding='post')
pred = model.predict(new_padded)
print(f"Input: {new_text[0]}")
print(f"Prediction score: {pred[0][0]:.4f}")
print("Label: positive" if pred[0][0] > 0.5 else "Label: negative")
