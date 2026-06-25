import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

TRAINING_SIZE = 50000
DIGITS = 3
REVERSE = True
MAXLEN = DIGITS + 1 + DIGITS
EPOCHS = 15
BATCH_SIZE = 128

chars = "0123456789+ "
rng = np.random.default_rng(42)


class CharacterTable:
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = {c: i for i, c in enumerate(self.chars)}
        self.indices_char = {i: c for i, c in enumerate(self.chars)}

    def encode(self, text, length):
        encoded = np.zeros((length,), dtype=np.int32)
        for i, ch in enumerate(text):
            encoded[i] = self.char_indices[ch]
        return encoded

    def decode(self, indices):
        if hasattr(indices, "ndim") and indices.ndim > 1:
            indices = indices.argmax(axis=-1)
        return "".join(self.indices_char[int(i)] for i in indices)


ctable = CharacterTable(chars)

print("Generating data...")
questions = []
expected = []
seen = set()

while len(questions) < TRAINING_SIZE:
    def make_number():
        length = int(rng.integers(1, DIGITS + 1))
        return int("".join(str(int(rng.integers(0, 10))) for _ in range(length)))

    a, b = make_number(), make_number()
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)

    query = f"{a}+{b}"
    query = query + " " * (MAXLEN - len(query))
    answer = str(a + b)
    answer = answer + " " * (DIGITS + 1 - len(answer))

    if REVERSE:
        query = query[::-1]

    questions.append(query)
    expected.append(answer)

print("Vectorizing data...")
x = np.array([ctable.encode(q, MAXLEN) for q in questions], dtype=np.int32)
y = np.array([ctable.encode(a, DIGITS + 1) for a in expected], dtype=np.int32)

split_at = len(x) - len(x) // 10
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]

print("Building the Seq2Seq model...")
input_layer = layers.Input(shape=(MAXLEN,))
embedded = layers.Embedding(len(chars), 64)(input_layer)
encoded = layers.LSTM(128)(embedded)
decoded = layers.RepeatVector(DIGITS + 1)(encoded)
decoded = layers.LSTM(128, return_sequences=True)(decoded)
output = layers.TimeDistributed(layers.Dense(len(chars), activation="softmax"))(decoded)

model = keras.Model(input_layer, output)
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

print(f"Training the model for {EPOCHS} epochs...")
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_val, y_val))

print("\nTesting on validation examples:")
for i in range(10):
    ind = int(rng.integers(0, len(x_val)))
    rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]

    preds = model.predict(rowx, verbose=0)[0].argmax(axis=-1)

    q = ctable.decode(rowx[0])
    correct = ctable.decode(rowy[0]).strip()
    guess = ctable.decode(preds).strip()

    q_display = q[::-1] if REVERSE else q
    print(f"Q: {q_display.strip()} \t Target: {correct} \t Guess: {guess} \t Correct? {'Yes' if correct == guess else 'No'}")
