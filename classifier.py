from keras.models import model_from_json
import numpy as np
from nltk.tokenize import TreebankWordTokenizer
from gensim.models.keyedvectors import KeyedVectors

class ReviewClassifier:

    def __init__(self):
        EMBEDDING_FILE = 'GoogleNews-vectors-negative300.bin.gz'
        self.word_vectors = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
        self.model = self.create_model()
        self.maxlen = 300
        self.embedding_dims = 300

    def create_model(self):
        model_json = "cnn_model.json"
        weights = 'cnn_weights.h5'
        with open(model_json, "r") as json_file:
            json_string = json_file.read()
        model = model_from_json(json_string)
        model.load_weights(weights)
        return model


    def classify_text(self, text):
        vectorized_data = self.tokenize_and_vectorize(text)
        padded_truncated_data = self.pad_trunc(vectorized_data, self.maxlen)
        reshaped_data = self.reshape_data(padded_truncated_data)
        classification = self.classify_data(reshaped_data)
        return classification

    def tokenize_and_vectorize(self, dataset):
        tokenizer = TreebankWordTokenizer()
        vectorized_data = []
        for sample in dataset:
            tokens = tokenizer.tokenize(sample)
            sample_vecs = []
            for token in tokens:
                try:
                    sample_vecs.append(self.word_vectors[token])
                except KeyError:
                    pass  # No matching token in the Google w2v vocab
            vectorized_data.append(sample_vecs)
        return vectorized_data

    def pad_trunc(self, data, maxlen):
        """
        For a given dataset pad with zero vectors or truncate to maxlen
        """
        new_data = []

        # Create a vector of 0s the length of our word vectors
        zero_vector = []
        for _ in range(len(data[0][0])):
            zero_vector.append(0.0)

        for sample in data:
            if len(sample) > maxlen:
                temp = sample[:maxlen]
            elif len(sample) < maxlen:
                temp = sample
                # Append the appropriate number 0 vectors to the list
                additional_elems = maxlen - len(sample)
                for _ in range(additional_elems):
                    temp.append(zero_vector)
            else:
                temp = sample
            new_data.append(temp)
        return new_data

    def reshape_data(self, data):
        return np.reshape(data, (len(data), self.maxlen, self.embedding_dims))

    def classify_data(self, data):
        predications = self.model.predict_classes(data)
        return predications[0][0]