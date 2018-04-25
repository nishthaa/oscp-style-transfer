from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential, Model, load_model
from keras.preprocessing.sequence import pad_sequences

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers
import argparse
import cPickle as pickle
import keras
import keras.optimizers as optimizers
import numpy as np
import os
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
keras.backend.set_session(sess)

parser = argparse.ArgumentParser(
    description='Script for training a binary classifier over embeddings',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    '--tweets_file',
    help='Specify the ground truth file',
    default=os.path.join(os.pardir, 'data', 'twitter_celeb_gender.csv'),
    type=str
)
parser.add_argument(
    '--embedding_size',
    help='Specify size of embeddings',
    default=100,
    type=int
)
parser.add_argument(
    '--maxlen',
    help='Specify the maximum sequence length',
    default=30,
    type=int
)
parser.add_argument(
    '--batch_size',
    help='Specify the batch_size',
    default=16,
    type=int
)
parser.add_argument(
    '--use_attention',
    help='Use hierarchical attention instead of CNN',
    default=False,
    action='store_true'
)
parser.add_argument(
    '--train',
    help='Train the classifier',
    default=False,
    action='store_true'
)
parser.add_argument(
    '--test',
    help='Test the trained classifier',
    default=False,
    action='store_true'
)


class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.init((input_shape[-1],))
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))
        ai = K.exp(eij)
        weights = ai / K.sum(ai, axis=1).dimshuffle(0, 'x')
        weighted_input = x * weights.dimshuffle(0, 1, 'x')
        return weighted_input.sum(axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])


def get_embedding_matrix(tokenizer, embedding_size):
    embeddings_index = {}
    with open('glove.twitter.27B.100d.txt') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    relevant_words = sorted(
        tokenizer.word_counts.keys(),
        key=lambda x: tokenizer.word_counts[x],
        reverse=True
    )[:tokenizer.num_words - 1]
    embedding_matrix = np.zeros((tokenizer.num_words, embedding_size))
    for word in relevant_words:
        index = tokenizer.word_index[word]
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[index] = embedding_vector
    return embedding_matrix


def load_content(filename):
    return np.load(filename)


def get_model(
    embedding_size,
    tokenizer_save_filename,
    nb_classes,
    input_length=20,
    use_attention=False
):
    from keras.layers import Embedding, Dense, Dropout
    with open(tokenizer_save_filename, 'rb') as f:
        tokenizer = pickle.load(f)
    embedding_matrix = get_embedding_matrix(
        tokenizer,
        embedding_size
    )
    vocab_size = tokenizer.num_words

    if (not use_attention):
        from keras.layers import Conv1D, MaxPool1D, Flatten
        model = Sequential()
        model.add(
            Embedding(
                vocab_size,
                embedding_size,
                input_length=input_length,
                weights=[embedding_matrix],
                trainable=True
            )
        )
        model.add(Conv1D(128, 3, activation='relu'))
        model.add(MaxPool1D(pool_size=2))
        model.add(Conv1D(128, 3, activation='relu'))
        model.add(MaxPool1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(nb_classes, activation='softmax'))
    else:
        from keras.layers import Bidirectional, LSTM, TimeDistributed, Input
        sentence_input = Input(shape=(input_length,), dtype='int32')
        embedded_sequences = Embedding(
            vocab_size,
            embedding_size,
            input_length=input_length,
            weights=[embedding_matrix],
            trainable=True
        )(sentence_input)
        l_lstm = Bidirectional(LSTM(100))(embedded_sequences)
        sentEncoder = Model(sentence_input, l_lstm)

        review_input = Input(shape=(input_length, 1, ), dtype='int32')
        review_encoder = TimeDistributed(sentEncoder)(review_input)
        l_lstm_sent = Bidirectional(LSTM(100))(review_encoder)
        preds = Dense(nb_classes, activation='softmax')(l_lstm_sent)
        model = Model(review_input, preds)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.Adam(lr=0.0005),
        metrics=['accuracy', 'mse']
    )
    return model


def generator(
    tweets,
    labels,
    batch_size,
    max_len,
    tokenizer_save_filename,
    use_attention
):
    with open(tokenizer_save_filename, 'rb') as f:
        tokenizer = pickle.load(f)
    batch_num = 0
    while (True):
        batch_num %= (len(tweets) // batch_size)
        batch_start = batch_num * batch_size
        batch_end = min(len(tweets), (batch_num + 1) * batch_size)
        tweet_batch = tweets[batch_start:batch_end]
        labels_batch = labels[batch_start:batch_end]
        integer_encoded_tweets = tokenizer.texts_to_sequences(tweet_batch)
        integer_encoded_tweets = pad_sequences(
            integer_encoded_tweets,
            maxlen=max_len
        )
        batch_num += 1
        if (use_attention):
            integer_encoded_tweets = np.expand_dims(
                integer_encoded_tweets,
                axis=-1
            )
        yield integer_encoded_tweets, labels_batch


if __name__ == '__main__':
    args = parser.parse_args()
    batch_size = args.batch_size
    maxlen = args.maxlen
    use_attention = args.use_attention
    tweets_GT_file = args.tweets_file
    checkpoint = 'GenderPredictor.h5'
    tokenizer_save_filename = 'tokenizer.pickle'
    if ('celeb' in tweets_GT_file):
        tokenizer_save_filename = 'tokenizer_celeb.pickle'
        checkpoint = 'GenderPredictor_celeb.h5'
    if (use_attention):
        checkpoint = 'attention_{}'.format(checkpoint)
    csv = tweets_GT_file.split(os.sep)[-1]
    csv = csv.split('.')[0]
    train_savefile = '{}_train'.format(csv)
    test_savefile = '{}_test'.format(csv)
    train_tweets = load_content('{}_tweets.npy'.format(train_savefile))
    test_tweets = load_content('{}_tweets.npy'.format(test_savefile))
    train_labels = load_content('{}_labels.npy'.format(train_savefile))
    test_labels = load_content('{}_labels.npy'.format(test_savefile))
    nb_classes = len(np.unique(train_labels))

    train_labels = np.array(map(
        lambda x: keras.utils.to_categorical(x, nb_classes),
        train_labels
    ))
    test_labels = np.array(map(
        lambda x: keras.utils.to_categorical(x, nb_classes),
        test_labels
    ))

    try:
        model = load_model(checkpoint)
    except IOError:
        model = get_model(
            args.embedding_size,
            tokenizer_save_filename,
            nb_classes,
            maxlen,
            use_attention
        )
    train_data_size = int(0.8 * len(train_tweets))
    train_gen_tweets, train_gen_labels = map(
        lambda x: x,
        [train_tweets[:train_data_size], train_labels[:train_data_size]]
    )
    test_gen_tweets, test_gen_labels = map(
        lambda x: x,
        [test_tweets, test_labels]
    )
    validation_gen_tweets, validation_gen_labels = map(
        lambda x: x,
        [train_tweets[train_data_size:], train_labels[train_data_size:]]
    )
    del train_tweets, train_labels
    del test_tweets, test_labels
    train_gen = generator(
        train_gen_tweets,
        train_gen_labels,
        batch_size,
        max_len=maxlen,
        tokenizer_save_filename=tokenizer_save_filename,
        use_attention=use_attention
    )
    test_gen = generator(
        test_gen_tweets,
        test_gen_labels,
        batch_size,
        max_len=maxlen,
        tokenizer_save_filename=tokenizer_save_filename,
        use_attention=use_attention
    )
    validation_gen = generator(
        validation_gen_tweets,
        validation_gen_labels,
        batch_size,
        max_len=maxlen,
        tokenizer_save_filename=tokenizer_save_filename,
        use_attention=use_attention
    )
    print 'Number of 0s, 1s and 2s in train data: {}, {} and {}'.format(
        len(filter(lambda x: np.argmax(x) == 0, train_gen_labels)),
        len(filter(lambda x: np.argmax(x) == 1, train_gen_labels)),
        len(filter(lambda x: np.argmax(x) == 2, train_gen_labels))
    )
    print 'Number of 0s, 1s and 2s in test data: {}, {} and {}'.format(
        len(filter(lambda x: np.argmax(x) == 0, test_gen_labels)),
        len(filter(lambda x: np.argmax(x) == 1, test_gen_labels)),
        len(filter(lambda x: np.argmax(x) == 2, test_gen_labels))
    )
    print 'Number of 0s, 1s and 2s in validation data: {}, {} and {}'.format(
        len(filter(lambda x: np.argmax(x) == 0, validation_gen_labels)),
        len(filter(lambda x: np.argmax(x) == 1, validation_gen_labels)),
        len(filter(lambda x: np.argmax(x) == 2, validation_gen_labels))
    )
    save_callback = ModelCheckpoint(
        filepath=checkpoint,
        verbose=1,
        save_best_only=True
    )
    lr_callback = ReduceLROnPlateau(
        verbose=1,
        patience=1,
        min_lr=0.001,
        factor=0.2
    )
    if (args.train):
        model.fit_generator(
            train_gen,
            steps_per_epoch=(len(train_gen_tweets) // batch_size),
            epochs=10,
            validation_data=validation_gen,
            validation_steps=(len(validation_gen_tweets) // batch_size),
            callbacks=[save_callback, lr_callback],
            class_weight={
                0: len(filter(lambda x: np.argmax(x) != 0, train_gen_labels)),
                1: len(filter(lambda x: np.argmax(x) != 1, train_gen_labels)),
                2: len(filter(lambda x: np.argmax(x) != 2, train_gen_labels))
            }
        )
    if (args.test):
        print model.evaluate_generator(
            test_gen,
            steps=(len(test_gen_tweets) // batch_size)
        )
        test_gen = generator(
            test_gen_tweets,
            test_gen_labels,
            batch_size,
            max_len=maxlen,
            tokenizer_save_filename=tokenizer_save_filename,
            use_attention=use_attention
        )
        import pdb
        # pdb.set_trace()
        y_predicted = []
        y_actual = []
        for i in xrange((len(test_gen_tweets) // batch_size)):
            x, y = test_gen.next()
            y_ = model.predict_on_batch(x)
            y_actual.extend(y.tolist())
            y_predicted.extend(y_.tolist())
        acc = 0.
        count = 0
        threshold = 1.5e-1
        for i in xrange(len(y_predicted)):
            y_ = sorted(y_predicted[i], reverse=True)
            if (y_[0] - y_[1] >= threshold):
                count += 1
                if (np.argmax(y_predicted[i]) == np.argmax(y_actual[i])):
                    acc += 1
        print 'Filtered test accuracy: {}%'.format(acc / count * 100)
