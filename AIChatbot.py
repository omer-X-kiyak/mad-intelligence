from keras.layers import Dropout
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from tensorflow import keras
from tensorflow.keras import layers




# Veri(kelime) kümesi
words = ["napiyorsun", "1", "merhaba", "nasılsın", "iyiyim", "sen", "kimsin", "ben", "adın", "ne", "nedir", "saat", "kaç", "bugün", "hava", "nasıl", "olacak"]
tags = ["napiyorsun", "1", "selam", "iyi", "iyi", "bot", "bot", "kullanıcı", "ad", "bot", "bot", "zaman", "zaman", "zaman", "hava", "hava", "hava"]
responses = ["iyiyim sen napiyorsun bea", "1","Merhaba!", "İyiyim, teşekkürler.", "Harika!", "Ben bir chatbot'um.", "Ben bir chatbot'um.", "Ben kimseyim, sen kendini tanıtır mısın?", "Benim adım Chatbot.", "Ben bir chatbot'um.", "Benim adım Chatbot.", "Şu anda saat %H:%M.", "Şu anda saat %H:%M.", "Bugün %d.%m.%Y.", "Hava şu anda güzel.", "Hava şu anda güzel.", "Bugün hava %s olacak."]

# Tokenization
lemmatizer = WordNetLemmatizer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def lemmatize(token):
    return lemmatizer.lemmatize(token.lower())

def preprocess(sentence):
    tokens = tokenize(sentence)
    lemmas = [lemmatize(token) for token in tokens]
    return " ".join(lemmas)


def create_model(vocab_size, num_classes):
    model = keras.Sequential(
        [
            layers.Dense(256, input_shape=(vocab_size,), activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model

processed_words = [preprocess(word) for word in words]


vocab = sorted(set(" ".join(processed_words).split()))


word_to_index = {word: i for i, word in enumerate(vocab)}
tag_to_index = {tag: i for i, tag in enumerate(set(tags))}


X = np.zeros((len(words), len(vocab)), dtype=np.float32)
for i, word in enumerate(processed_words):
    for token in word.split():
        X[i, word_to_index[token]] = 1


y = np.zeros((len(tags), len(tag_to_index)), dtype=np.float32)
for i, tag in enumerate(tags):
    y[i, tag_to_index[tags[i]]] = 1


model = Sequential()
model.add(Dense(512, input_shape=(len(X[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y[0]), activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X, y, epochs=500, batch_size=10, verbose=1)


model.save('chatbot_model.h5')

def output(user_input):
    
    processed_input = preprocess(user_input)

    
    X_input = np.zeros((1, len(vocab)), dtype=np.float32)
    for token in processed_input.split():
        if token in word_to_index:
            X_input[0, word_to_index[token]] = 1

    
    results = model.predict([X_input])[0]
    tag_index = np.argmax(results)
    tag = list(tag_to_index.keys())[list(tag_to_index.values()).index(tag_index)]
    response = responses[tag_index]

    
    print("Chatbot: " + response)


while True:
  user_input = input("Sen: ")
  if user_input.lower() == 'exit':
    print("Chatbot: Görüşmek üzere!")
    break
  output(user_input)
