import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

TRAINING_SIZE = 50000
DIGITS = 3
REVERSE = True
MAXLEN = DIGITS + 1 + DIGITS 

chars = "0123456789+ "

class CharacterTable:
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return "".join(self.indices_char[x] for x in x)

ctable = CharacterTable(chars)

print("Generating data...")
questions = []
expected = []
seen = set()

while len(questions) < TRAINING_SIZE:
    f = lambda: int("".join(np.random.choice(list("0123456789")) for _ in range(np.random.randint(1, DIGITS + 1))))
    a, b = f(), f()
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    
    q = "{}+{}".format(a, b)
    query = q + " " * (MAXLEN - len(q))
    
    ans = str(a + b)
    ans += " " * (DIGITS + 1 - len(ans))
    
    if REVERSE:
        query = query[::-1]
        
    questions.append(query)
    expected.append(ans)

print("Vectorizing data...")
x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.float32)
y = np.zeros((len(questions), DIGITS + 1, len(chars)), dtype=np.float32)

for i, sentence in enumerate(questions):
    x[i] = ctable.encode(sentence, MAXLEN)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, DIGITS + 1)

split_at = len(x) - len(x) // 10
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]

print("Building the Seq2Seq model...")
model = keras.Sequential()

model.add(layers.LSTM(128, input_shape=(MAXLEN, len(chars))))
model.add(layers.RepeatVector(DIGITS + 1))
model.add(layers.LSTM(128, return_sequences=True))
model.add(layers.Dense(len(chars), activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

print("Training the model for 5 epochs...")
model.fit(x_train, y_train, batch_size=128, epochs=5, validation_data=(x_val, y_val))

print("\nTesting on validation examples:")
for i in range(10):
    ind = np.random.randint(0, len(x_val))
    rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
    
    preds = np.argmax(model.predict(rowx, verbose=0), axis=-1)
    
    q = ctable.decode(rowx[0])
    correct = ctable.decode(rowy[0])
    guess = ctable.decode(preds[0], calc_argmax=False)
    
    q_display = q[::-1] if REVERSE else q
    print(f"Q: {q_display.strip()} \t Target: {correct.strip()} \t Guess: {guess.strip()} \t Correct? {'Yes' if correct == guess else 'No'}")
