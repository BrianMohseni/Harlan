import tensorflow as tf
import numpy as np
import os
from nlp_utils import create_tf_dataset, Tokenizer, Chatbot
from keras_nlp.layers import TransformerDecoder
import keras
from model_utils import build_transformer_decoder_model

vocab_size = 5000
maxlen = 512
maxlen += 1
batch_size = 16

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=vocab_size,
    pad_to_max_tokens=True,
    output_sequence_length=maxlen,
    standardize=None,
)

test_vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=vocab_size,
    #pad_to_max_tokens=True,
    #output_sequence_length=maxlen,
    standardize=None,
)


#vectorizer.adapt(text)
#test_vectorizer.adapt(text)

def adapt_vectorizers(vectorizer, test_vectorizer):
    num_files = os.walk("datasets")

    idx = 0
    data = ""
    with open(f"harlan_0.txt", "r") as f:
        data += f.read() + " "    #for _ in range(50000):
        #with open(f"validation_dataset/data/prompt_{idx}.txt", "r") as f:
            #data += f.read() + " "

    #    idx += 1
    vectorizer.adapt(data)
    test_vectorizer.adapt(data)

    return data


dataset = adapt_vectorizers(vectorizer, test_vectorizer)
vocab_size = len(test_vectorizer.get_vocabulary())

tokenizer = Tokenizer(vectorizer, maxlen)

dataset = create_tf_dataset(dataset, tokenizer, maxlen, test_vectorizer, batch_size=batch_size)

model = build_transformer_decoder_model(vocab_size, maxlen, 128, 4, 4, 64)
model = tf.keras.models.load_model("harlan-model.keras")
model.compile(loss='SparseCategoricalCrossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4), metrics=["accuracy"])

#model.summary()

model.fit(dataset.repeat(), steps_per_epoch=4000, epochs=1)
#model.save("harlan-model.keras")


chatbot = Chatbot(model, maxlen, test_vectorizer, tokenizer)


chatbot.main_loop()
