import tensorflow as tf
import numpy as np
import os


class Tokenizer:
    def __init__(self, vectorizer, maxlen):
        self.vectorizer = vectorizer
        self.vocab_length = len(vectorizer.get_vocabulary())

    def dictionary(self):
        return self.vectorizer.get_vocabulary()

    def encode(self, text):

        return self.vectorizer(text)

    def get_word_from_token(self, token):
        if token > self.vocab_length:
            return None

        return self.dictionary()[token]

    def extract_between_markers(self, memory):
        last_model_index = memory.rfind('<MODEL> <BOS>')
        eos_index = memory.rfind('<EOS>')


        start = last_model_index + len('<MODEL>')
        return memory[start:eos_index].strip()


    def decode(self, vectors):
        words = ""
        for vector in vectors:
            word = self.get_word_from_token(vector)
            if word != "":
                words += word + " "

        words = self.extract_between_markers(words)
        return words

    def decode_test(self, vectors):
        words = ""

        for vector in vectors:
            word = self.get_word_from_token(vector)
            if word != "" and word != "<BOS>" and word != "<EOS>" and word != "[UNK]" and word is not None:
                if word == "<MODEL>":
                    words = ""
                    words += "\nmodel: "
                else:
                    words += word + " "

        return words


    def decode_processed(self, vectors):
        words = ""
        for vector in vectors:
            word = self.get_word_from_token(vector)
            if word != "" and word != "<BOS>" and word != "<EOS>" and word != "[UNK]" and word != f"<USER>" and word != "<MODEL>":
                words += word + " "

        return words

    def get_random_pos(self, text, maxlen):
        text = self.vectorizer(text)
        zeros = np.zeros(shape=(maxlen,))
        padded = np.concatenate([zeros, text])
        start_at = np.random.randint(0, maxlen)
        end_at = start_at+maxlen-1

        input_sequence = padded[start_at:end_at]
        output_token = padded[end_at]
        return input_sequence, output_token


class Chatbot:
    def __init__(self, model, maxlen, test_vectorizer, tokenizer):
        self.model = model
        self.maxlen = maxlen
        self.memory = tf.zeros((maxlen - 1,), dtype=tf.int32)
        self.vectorizer = test_vectorizer
        self.tokenizer = tokenizer
        self.c_limit = 0

    def generate_token(self):
        context = tf.expand_dims(self.memory, axis=0)

        pred_token = tf.argmax(self.model.predict(context, verbose=0), axis=1)
        pred_token = tf.cast(pred_token, dtype=tf.int32)

        if self.tokenizer.get_word_from_token(int(pred_token)) != "<EOS>" and self.c_limit < self.maxlen:
            self.memory = tf.concat([self.memory, pred_token], axis=0)
            self.memory = self.memory[1:]
            self.c_limit += 1
            return True

        return False

    def user_input(self):
        user_input = input("User: ")
        if user_input == "<RESET_CHAT>":
            self.reset_memory()
            user_input = input("User: ")
        elif user_input == "<<STOP>>":
            self.halt_chat()
        user_input = " <USER> <BOS> " + user_input + " <EOS>"
        user_input = tf.cast(self.vectorizer(user_input), dtype=tf.int32)
        len_input = user_input.shape[0]
        self.memory = tf.concat([self.memory, user_input], axis=0)
        self.memory = self.memory[len_input:]
        #print(self.tokenizer.decode(self.memory))

    def generate_response(self):
        model_start = " <MODEL> <BOS> "
        model_start = tf.cast(self.vectorizer(model_start), dtype=tf.int32)
        len_input = model_start.shape[0]
        self.memory = tf.concat([self.memory, model_start], axis=0)
        self.memory = self.memory[len_input:]
        loop = True

        while loop:
            loop = self.generate_token()
        model_end = " <EOS>"
        model_end = tf.cast(self.vectorizer(model_end), dtype=tf.int32)
        len_input = model_end.shape[0]
        self.memory = tf.concat([self.memory, model_end], axis=0)
        self.memory = self.memory[len_input:]

    def print_response(self):
        print(self.tokenizer.decode_test(self.memory))
        #print(self.tokenizer.decode_test(self.memory))

    def main_loop(self):
        while True:
            self.user_input()
            self.generate_response()
            self.print_response()

    def reset_memory(self):
        self.memory = tf.zeros((self.maxlen - 1,), dtype=tf.int32)
        print("Model memory has been reset")

    def halt_chat(self):
        exit()


def pull_random_file(tokenizer):
    num_files = os.walk("datasets")
    idx = np.random.randint(0, 1000000)
    with open(f"train_dataset/data/prompt_{idx}.txt") as f:
        data = f.read()

    return data
    

def create_tf_dataset(text, tokenizer, maxlen, test_vectorizer, batch_size=32):
    #text = test_vectorizer(text)

    def generator():
        while True:
            #text = pull_random_file(tokenizer)
            tokenizer.vectorizer.adapt(text)

            x, y = tokenizer.get_random_pos(text, maxlen)
            yield x, y

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(maxlen - 1,), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset
