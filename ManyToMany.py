import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, TimeDistributed
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# =====================================
# Configuration
# =====================================

MODEL_FILE = "ner_model.keras"

WORD_TOKENIZER = "word_tokenizer.pkl"
TAG_TOKENIZER = "tag_tokenizer.pkl"

MAX_LEN = 30

EMBEDDING_DIM = 128

RNN_UNITS = 128

# =====================================
# Load Dataset
# =====================================

def load_dataset():

    print("Loading Dataset...")

    df = pd.read_csv("ner_dataset.csv", encoding="latin1",nrows=50000)

    df = df.ffill()

    return df
def preprocess(df):

    print("Preprocessing...")

    grouped = df.groupby("Sentence #")

    sentences = grouped["Word"].apply(list).values

    tags = grouped["Tag"].apply(list).values

    words = sorted(list(set(df["Word"].values)))

    tags_list = sorted(list(set(df["Tag"].values)))

    word2idx = {
        w:i+2 for i,w in enumerate(words)
    }

    word2idx["PAD"] = 0

    word2idx["UNK"] = 1

    tag2idx = {
        t:i+1 for i,t in enumerate(tags_list)
    }

    tag2idx["PAD"] = 0

    X = []

    y = []

    for sent,tag in zip(sentences,tags):

        X.append(
            [word2idx.get(w,1) for w in sent]
        )

        y.append(
            [tag2idx[t] for t in tag]
        )

    X = pad_sequences(
        X,
        maxlen=MAX_LEN,
        padding="post",
        value=0
    )

    y = pad_sequences(
        y,
        maxlen=MAX_LEN,
        padding="post",
        value=0
    )

    y = to_categorical(
        y,
        num_classes=len(tag2idx)
    )

    with open(WORD_TOKENIZER,"wb") as f:
        pickle.dump(word2idx,f)

    with open(TAG_TOKENIZER,"wb") as f:
        pickle.dump(tag2idx,f)

    return X,y,len(word2idx),len(tag2idx)
# =====================================
# Train Model
# =====================================

def train_model():

    df = load_dataset()

    X, y, vocab_size, tag_size = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = Sequential()

    model.add(
        Embedding(
            input_dim=vocab_size,
            output_dim=EMBEDDING_DIM,
            input_length=MAX_LEN
        )
    )

    model.add(
        SimpleRNN(
            RNN_UNITS,
            return_sequences=True
        )
    )

    model.add(
        TimeDistributed(
            Dense(
                tag_size,
                activation="softmax"
            )
        )
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=5,
        batch_size=32,
        verbose=1
    )

    model.save(MODEL_FILE)

    print("Model Saved Successfully")
    # =====================================
# Prediction Function
# =====================================

def predict_sentence(sentence):

    model = load_model(MODEL_FILE)

    with open(WORD_TOKENIZER, "rb") as f:
        word2idx = pickle.load(f)

    with open(TAG_TOKENIZER, "rb") as f:
        tag2idx = pickle.load(f)

    idx2tag = {v: k for k, v in tag2idx.items()}

    words = sentence.strip().split()

    sequence = [word2idx.get(word, 1) for word in words]

    sequence = pad_sequences(
        [sequence],
        maxlen=MAX_LEN,
        padding="post"
    )

    prediction = model.predict(sequence, verbose=0)

    prediction = np.argmax(prediction[0], axis=1)

    result = []

    for word, tag in zip(words, prediction):

        result.append(
            (word, idx2tag.get(tag, "O"))
        )

    return result


# =====================================
# Train if Model Doesn't Exist
# =====================================

if not os.path.exists(MODEL_FILE):

    train_model()


# =====================================
# Streamlit UI
# =====================================

st.set_page_config(
    page_title="NER using Simple RNN",
    page_icon="🧠"
)

st.title("🧠 Named Entity Recognition")

st.write("### Many-to-Many RNN Example")

sentence = st.text_input(
    "Enter a sentence"
)

if st.button("Predict"):

    if sentence.strip() == "":

        st.warning("Please enter a sentence.")

    else:

        result = predict_sentence(sentence)

        st.subheader("Predicted Tags")

        for word, tag in result:

            st.write(f"{word}  ➜  {tag}")