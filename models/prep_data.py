import argparse
from keras.preprocessing.text import Tokenizer
import numpy as np
import os
import pdb
import preprocessor

parser = argparse.ArgumentParser(
    description='Script for generating word one-hot vectors for model',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    '--save_tokenizer',
    help='Save the tokenizer generated from all tweets',
    default=False,
    action='store_true'
)
parser.add_argument(
    '--tweets_file',
    help='Specify the ground truth file',
    default=os.path.join(os.pardir, 'data', 'twitter_celeb_gender.csv'),
    type=str
)


def read_tweets(filename):
    from unidecode import unidecode
    labels, tweets = [[] for _ in xrange(2)]
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            label, tweet = line.split(',', 1)
            try:
                labels.append(int(label))
            except ValueError:
                continue
            tweet = preprocessor.tokenize(tweet)
            tweet = unidecode(tweet)
            tweets.append(tweet)
    pdb.set_trace()
    return labels, tweets


def fit_tokenizer(
    tweets,
    save_tokenizer,
    tokenizer_save_filename
):
    import cPickle as pickle
    # pdb.set_trace()
    if (not os.path.exists(tokenizer_save_filename)):
        tokenizer = Tokenizer(num_words=50000)
        print 'Fitting tokenizer on tweets'
        tokenizer.fit_on_texts(tweets)
        print 'Finished fitting tokenizer'
        print 'Vocab size: {}'.format(len(tokenizer.word_counts))
        if (save_tokenizer):
            with open(tokenizer_save_filename, 'wb') as f:
                pickle.dump(
                    tokenizer,
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL
                )


if __name__ == '__main__':
    args = parser.parse_args()
    tweets_GT_file = args.tweets_file
    tokenizer_save_filename = 'tokenizer.pickle'
    if ('celeb' in tweets_GT_file):
        tokenizer_save_filename = 'tokenizer_celeb.pickle'
    labels, tweets = read_tweets(tweets_GT_file)
    fit_tokenizer(tweets, args.save_tokenizer, tokenizer_save_filename)
    rnd = np.random.get_state()
    np.random.shuffle(tweets)
    np.random.set_state(rnd)
    np.random.shuffle(labels)

    csv = tweets_GT_file.split(os.sep)[-1]
    csv = csv.split('.')[0]
    train_savefile = '{}_train'.format(csv)
    test_savefile = '{}_test'.format(csv)
    if (not os.path.exists('{}_tweets'.format(train_savefile))):
        train_tweets, test_tweets = [[] for _ in xrange(2)]
        train_labels, test_labels = [[] for _ in xrange(2)]
        tweets_with_0, tweets_with_1, tweets_with_2 =\
            filter(lambda x: x == 0, labels),\
            filter(lambda x: x == 1, labels),\
            filter(lambda x: x == 2, labels)
        train_0_split_size, train_1_split_size, train_2_split_size =\
            int(0.8 * len(tweets_with_0)), int(0.8 * len(tweets_with_1)),\
            int(0.8 * len(tweets_with_2))
        train_0_counter, train_1_counter, train_2_counter = [
            0 for _ in xrange(3)
        ]
        # pdb.set_trace()
        for i, tweet in enumerate(tweets):
            if (labels[i] == 0):
                if (train_0_counter < train_0_split_size):
                    train_0_counter += 1
                    train_tweets.append(tweet)
                    train_labels.append(labels[i])
                else:
                    test_tweets.append(tweet)
                    test_labels.append(labels[i])
            elif (labels[i] == 1):
                if (train_1_counter < train_1_split_size):
                    train_1_counter += 1
                    train_tweets.append(tweet)
                    train_labels.append(labels[i])
                else:
                    test_tweets.append(tweet)
                    test_labels.append(labels[i])
            elif (labels[i] == 2):
                if (train_2_counter < train_2_split_size):
                    train_2_counter += 1
                    train_tweets.append(tweet)
                    train_labels.append(labels[i])
                else:
                    test_tweets.append(tweet)
                    test_labels.append(labels[i])
        np.save('{}_tweets'.format(train_savefile), train_tweets)
        np.save('{}_tweets'.format(test_savefile), test_tweets)
        np.save('{}_labels'.format(train_savefile), train_labels)
        np.save('{}_labels'.format(test_savefile), test_labels)
