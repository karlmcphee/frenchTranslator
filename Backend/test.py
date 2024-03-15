import tensorflow as tf 
from keras.utils import CustomObjectScope
from tensorflow.keras.layers import Embedding, MultiHeadAttention, LayerNormalization, Dense, Add
import tensorflow.keras.layers as layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

LATENT_DIM = 2048
EMBEDDING_DIM = 100

class PositionalEmbedding(tf.keras.layers.Layer):

    def __init__(self, sequence_length, vocab_size, embedding_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.base_embeddings= Embedding(vocab_size, embedding_dim)
        self.position_embeddings = Embedding(
            sequence_length, embedding_dim)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.sequence_length= sequence_length

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        print("hi ", tf.shape(inputs)[-1])
        positions = tf.range(start = 0, limit=length, delta=1)
        tokens = self.base_embeddings(inputs)
        positions = self.position_embeddings(positions)
        return tokens + positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim
        })
        return config

class Encoder(layers.Layer):
    def __init__(self, embedding_dim, num_heads, dense_dim, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dense_dim = dense_dim
        self.attention = MultiHeadAttention(num_heads = num_heads, key_dim = embedding_dim)
        self.normalization = LayerNormalization()
        self.normalization2 = LayerNormalization()
        self.dense1 = Dense(LATENT_DIM, activation="relu")
        self.dense2 = Dense(EMBEDDING_DIM)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        x = self.attention(query = inputs, value = inputs, key = inputs)
        x2 = self.normalization(x + inputs)
        x3 = self.dense1(x2)
        x4 = self.dense2(x3)
        return self.normalization2(x2 + x4)

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update(
            {
                "embed_dim": self.embedding_dim,
                "dense_dim": self.dense_dim,
                "num_heads": self.num_heads,
            }
        )
        return config

class Decoder(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, latent_dim, num_heads, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.embedding = Embedding(embedding_dim, latent_dim)
        self.attention1 = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
        self.attention2 = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.layernorm3 = LayerNormalization()
        self.dense1 = Dense(latent_dim, activation='relu')
        self.dense2 = Dense(embedding_dim)
        self.add = Add()
        self.supports_masking = True
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads


    def call(self, inputs, encoder_outputs, mask=None): 
        attention1 = self.attention1(query=inputs, key=inputs, value=inputs, use_causal_mask=True)
        x = self.layernorm1(self.add([inputs, attention1]))
        x2 = self.attention2(query=x, key=encoder_outputs, value=encoder_outputs)
        x3 = self.layernorm2(self.add([x2, x]))
        x4 = self.dense1(x3)
        x5 = self.dense2(x4)
        return(self.layernorm3(self.add([x2, x5])))

    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update(
            {
                "embed_dim": self.embedding_dim,
                "latent_dim": self.latent_dim,
                "num_heads": self.num_heads,
            }
        )
        return config

model = tf.keras.models.load_model('fr_model1.h5',  custom_objects={'MyLayers>PositionalEmbedding': PositionalEmbedding,                         
                                                                    'MyLayers>Encoder': Encoder,
                                                                    'MyLayers>Decoder': Decoder })
print(model.summary())

def translate(input_sentence, tokenizer1, tokenizer2, transformer):
    word2idx_outputs = tokenizer2.word_index
    idx2word_trans = {v: k for k, v in word2idx_outputs.items()}
    word = ""
    sentence = tokenizer1.texts_to_sequences([input_sentence])
    sentence = pad_sequences(sentence, maxlen=20, padding='post')
    decoded_sentence = "[start]"
    for i in range(20):
        target = tokenizer2.texts_to_sequences([decoded_sentence])
        target = pad_sequences(target, maxlen=20, padding='post')[:, :-1]
        print(target)
        predictions = transformer([sentence, target])
        # ops.argmax(predictions[0, i, :]) is not a concrete value for jax here
        idx = np.argmax(predictions[0, i, :])
        if idx > 0:
            word = idx2word_trans[idx]
            decoded_sentence+= " " + word
        else:
            decoded_sentence += " test"

        if word == "[end]" and i > 1:
            break
    return decoded_sentence

filehandler_eng = open('tokenizer_eng2.pickle', 'rb')
eng_tokenizer = pickle.load(filehandler_eng)
filehandler_fr = open('tokenizer_fra2.pickle', 'rb')
fr_tokenizer = pickle.load(filehandler_fr)
phrase = "Where are we going"
translation = translate(phrase, eng_tokenizer, fr_tokenizer, model)
print(translation)
print('hi')

