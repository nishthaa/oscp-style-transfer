from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Embedding, Dense, LSTM, Conv1D, MaxPool1D,\
    BatchNormalization
from keras.models import Sequential, load_model
from keras.preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical
import argparse
import cPickle as pickle
import keras
import numpy as np
import os
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
keras.backend.set_session(sess)

parser = argparse.ArgumentParser(
    description='Script for training a binary classifier over embeddings'
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
    default=300,
    type=int
)
parser.add_argument(
    '--batch_size',
    help='Specify the batch_size',
    default=16,
    type=int
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


def load_content(filename):
    return np.load(filename)


def get_model(
    embedding_size,
    tweet,
    tokenizer_save_filename='tokenizer.pickle'
):
    model = Sequential()
    # with open(tokenizer_save_filename, 'rb') as f:
    #     tokenizer = pickle.load(f)
    # vocab_size = tokenizer.num_words
    vocab_size = 50
    # embedded_tweet = tokenizer.texts_to_sequences([tweet])[0]
    model.add(
        Embedding(
            vocab_size,
            embedding_size,
            input_length=50
        )
    )
    model.add(Conv1D(
        filters=32,
        kernel_size=3,
        padding='same',
        activation='relu'
    ))
    model.add(MaxPool1D(pool_size=2))
    model.add(BatchNormalization())
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    # model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy', 'mse']
    )
    return model


def generator(
    tweets,
    labels,
    batch_size=16,
    max_len=50,
    tokenizer_save_filename='tokenizer.pickle'
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
        # labels_batch = np.array(map(
        #     lambda x: to_categorical(x, nb_classes),
        #     labels_batch
        # ))
        integer_encoded_tweets = tokenizer.texts_to_sequences(tweet_batch)
        integer_encoded_tweets = pad_sequences(
            integer_encoded_tweets,
            maxlen=max_len
        )
        # embedded_tweets_batch = tokenizer.texts_to_matrix(tweet_batch)
        batch_num += 1
        # yield embedded_tweets_batch, labels_batch
        yield integer_encoded_tweets, labels_batch


if __name__ == '__main__':
    args = parser.parse_args()
    batch_size = args.batch_size
    tweets_GT_file = args.tweets_file
    csv = tweets_GT_file.split(os.sep)[-1]
    csv = csv.split('.')[0]
    train_savefile = '{}_train'.format(csv)
    test_savefile = '{}_test'.format(csv)
    train_tweets = load_content('{}_tweets.npy'.format(train_savefile))
    test_tweets = load_content('{}_tweets.npy'.format(test_savefile))
    train_labels = load_content('{}_labels.npy'.format(train_savefile))
    test_labels = load_content('{}_labels.npy'.format(test_savefile))
    # nb_classes = len(np.unique(test_labels))

    # Training on 10000 as proof of concept
    try:
        model = load_model('GenderPredictor.h5')
    except IOError:
        model = get_model(
            args.embedding_size,
            train_tweets[0]
            # ,nb_classes
        )
    # [:10000]
    # [:2000]
    # [10000:12000]
    train_gen_tweets, train_gen_labels = map(
        lambda x: x,
        [train_tweets, train_labels]
    )
    test_gen_tweets, test_gen_labels = map(
        lambda x: x,
        [test_tweets, test_labels]
    )
    validation_gen_tweets, validation_gen_labels = map(
        lambda x: x,
        [train_tweets, train_labels]
    )
    train_gen = generator(
        train_gen_tweets,
        train_gen_labels,
        batch_size
    )
    test_gen = generator(
        test_gen_tweets,
        test_gen_labels,
        batch_size
    )
    validation_gen = generator(
        validation_gen_tweets,
        validation_gen_labels,
        batch_size
    )
    del train_tweets, train_labels
    del test_tweets, test_labels
    print 'Number of 0s and 1s in train data: {} and {}'.format(
        len(filter(lambda x: x == 0, train_gen_labels)),
        len(filter(lambda x: x == 1, train_gen_labels))
    )
    print 'Number of 0s and 1s in test data: {} and {}'.format(
        len(filter(lambda x: x == 0, test_gen_labels)),
        len(filter(lambda x: x == 1, test_gen_labels))
    )
    print 'Number of 0s and 1s in validation data: {} and {}'.format(
        len(filter(lambda x: x == 0, validation_gen_labels)),
        len(filter(lambda x: x == 1, validation_gen_labels))
    )
    save_callback = ModelCheckpoint(
        filepath='GenderPredictor.h5',
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
                0: len(filter(lambda x: x == 1, train_gen_labels)),
                1: len(filter(lambda x: x == 0, train_gen_labels))
            }
        )
    if (args.test):
        print model.evaluate_generator(
            test_gen,
            steps=(len(test_gen_tweets) // batch_size)
        )
