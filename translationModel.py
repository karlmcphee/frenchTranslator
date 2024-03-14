import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Embedding, MultiHeadAttention, LayerNormalization, Dropout, Add, TextVectorization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import sys
import random
import tensorflow.keras.layers as layers
import tensorflow.keras as keras
from sklearn.utils  import shuffle
import re

BATCH_SIZE = 64
EPOCHS =40
NUM_SAMPLES = 1000000
LATENT_DIM = 2048
EMBEDDING_DIM = 100
MAX_NUM_WORDS=15000
MAX_INPUTS = 20

input_texts = []
target_texts = []

t = 0
s = set()
max_size = 0
for line in open('fra.txt', encoding='utf8'):
    t += 1
    if '\t' not in line:
        continue
    if t > NUM_SAMPLES:
        break
    input_text, translation, *rest = line.rstrip().split('\t')
    input_text = re.sub(r'[^\w\s]', '', input_text)
    if input_text in s:
        continue
    s.add(input_text)
    translation = re.sub(r'[^\w\s]', '', translation)
    input_texts.append(input_text)
    target_texts.append('[start] ' + translation + ' [end]')


def make_dataset(eng_texts, fra_texts):
    eng_texts = list(eng_texts)
    fra_texts = list(fra_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, fra_texts))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(format_dataset)
    return dataset.shuffle(2048).prefetch(16).cache()

def format_dataset(eng_texts, fra_texts):
    new_set = (
        {
        "encoder_inputs": eng_texts,
        "decoder_inputs": fra_texts[:, 0:-1],
        },
        fra_texts[:, 1:],
    )
    return new_set

tokenizer_inputs = Tokenizer(num_words = MAX_NUM_WORDS)
tokenizer_inputs.fit_on_texts(input_texts)
eng_texts1 = tokenizer_inputs.texts_to_sequences(input_texts)
words1 = tokenizer_inputs.word_index

tokenizer_outputs = Tokenizer(num_words = MAX_NUM_WORDS, filters='')
tokenizer_outputs.fit_on_texts(target_texts)


output_texts = tokenizer_outputs.texts_to_sequences(target_texts)
MAX_OUTPUTS = max(len(s) for s in output_texts)
MAX_INPUTS = max(len(s) for s in eng_texts1)
MAX_INPUTS = max(MAX_INPUTS, MAX_OUTPUTS)
MAX_OUTPUTS = MAX_INPUTS
#MAX_OUTPUTS = 20
fra_texts = pad_sequences(output_texts, maxlen=MAX_OUTPUTS, padding='post')
eng_texts = pad_sequences(eng_texts1, maxlen=MAX_INPUTS, padding='post')

print("hi", MAX_INPUTS, MAX_OUTPUTS)

pairs = []
for i in range(0, len(eng_texts)):
    pairs.append((eng_texts[i], fra_texts[i]))

random.shuffle(pairs)
eng_texts, fra_texts = zip(*pairs)
val_eng = eng_texts[int(.7*len(eng_texts)):int(.9*len(eng_texts))]
val_fra = fra_texts[int(.7*len(fra_texts)):int(.9*len(fra_texts))]
eng_texts1=eng_texts[0:int(.7*len(eng_texts))]
fra_texts1=fra_texts[0:int(.7*len(fra_texts))]

test_eng = eng_texts[int(.9*len(eng_texts)):]
test_fra = fra_texts[int(.9*len(eng_texts)):]

val_pairs = target_texts[int(len(target_texts)*8//1):]
train_ds = make_dataset(eng_texts1, fra_texts1)
val_ds = make_dataset(val_eng, val_fra)

tf.keras.utils.get_custom_objects().clear()

@tf.keras.saving.register_keras_serializable(package="MyLayers")

class Encoder(layers.Layer):
    def __init__(self, embedding_dim, num_heads, dense_dim, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dense_dim = dense_dim
        self.attention = MultiHeadAttention(num_heads = num_heads, key_dim = embedding_dim)
        self.normalization = LayerNormalization()
        self.normalization2 = LayerNormalization()
        self.dropout = Dropout(.1)
        self.dense1 = Dense(LATENT_DIM, activation="relu")
        self.dense2 = Dense(EMBEDDING_DIM)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        x = self.attention(query = inputs, value = inputs, key = inputs)
        x2 = self.normalization(x + inputs)
        x3 = self.dropout(x2 + x)
        x3 = self.dense1(x2)
        x4 = self.dense2(x3)
        return self.normalization2(x2 + x4)

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "dense_dim": self.dense_dim,
                "num_heads": self.num_heads,
            }
        )
        return config

@tf.keras.saving.register_keras_serializable(package="MyLayers", name="PositionalEmbedding")
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

@tf.keras.saving.register_keras_serializable(package="MyLayers")
class Decoder(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, latent_dim, num_heads, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.embedding = Embedding(embedding_dim, latent_dim)
        self.attention1 = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
        self.attention2 = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.layernorm3 = LayerNormalization()
        self.dropout1 = Dropout(.1)
        self.dropout2 = Dropout(.1)
        self.dense1 = Dense(latent_dim, activation='relu')
        self.dense2 = Dense(embedding_dim)
        self.add = Add()
        self.supports_masking = True
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads


    def call(self, inputs, encoder_outputs, mask=None): 
        attention1 = self.attention1(query=inputs, key=inputs, value=inputs, use_causal_mask=True)
        x = self.dropout1(attention1)
        x2 = self.layernorm1(self.add([inputs, x]))
        x3 = self.attention2(query=x2, key=encoder_outputs, value=encoder_outputs)
        x4 = self.dropout2(x3)
        x5 = self.layernorm2(self.add([x2, x4]))
        x6 = self.dense1(x5)
        x7 = self.dense2(x6)
        
        return(self.layernorm3(self.add([x5, x7])))

    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "latent_dim": self.latent_dim,
                "num_heads": self.num_heads,
            }
        )
        return config

num_heads = 8
max_inputs = MAX_INPUTS
input = Input(shape=(None,), dtype="int64", name="encoder_inputs")
#print("seq length ", max_inputs, "max words ", MAX_NUM_WORDS, "input shape ", tf.shape(input)[-1])

x = PositionalEmbedding(MAX_INPUTS, MAX_NUM_WORDS, EMBEDDING_DIM)(input)
encoder_outputs = Encoder(EMBEDDING_DIM, num_heads, LATENT_DIM)(x)

encoder = Model(input, encoder_outputs)
decoder_input1 = Input(shape=(None,), name="decoder_inputs")
decoder_input2 = Input(shape=(None, EMBEDDING_DIM))
x = PositionalEmbedding(MAX_INPUTS, MAX_NUM_WORDS, EMBEDDING_DIM)(decoder_input1)
x = Decoder(EMBEDDING_DIM, LATENT_DIM, num_heads)(x, decoder_input2)
x = Dropout(.3)(x)
decoder_output = layers.Dense(MAX_NUM_WORDS, activation="softmax")(x)
decoder = Model([decoder_input1, decoder_input2], decoder_output)
decoder_outputs = decoder([decoder_input1, encoder_outputs])
transformer = Model(
    [input, decoder_input1], decoder_outputs
)

transformer.compile(
    tf.keras.optimizers.Adam(lr=.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

r = transformer.fit(train_ds, batch_size=16, epochs=30, validation_data=val_ds)
#r = model.fit(
#  [encoder_inputs, decoder_inputs], decoder_targets_one_hot,
#  batch_size=BATCH_SIZE,
#  epochs=EPOCHS,
#  validation_split=0.2,
#)