import numpy as np
import tweepy
from tweepy.error import TweepError
from tweepy import OAuthHandler


def read(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        if (',') in lines[0]:
            lines = map(lambda x: x.split(','), lines[1:])
        else:
            lines = map(lambda x: x.split(' - ')[::-1], lines)
        content = map(lambda x: (x[0].strip(), x[-1].strip()), lines)
    return content


if __name__ == '__main__':
    # Ideally, following should be hidden.
    consumer_key = 'NaxHmBVQVAb8lf3qTd1DF1o2A'
    consumer_secret = 'OCqQkczH3ZM51YYUlKXkcyIHXfWaPn98cKO1ALSeuVem4LNGV4'
    access_token = '974376510583967746-uKEQ44qF5yr8PU1K2vcystnzouTxmH3'
    access_token_secret = 'HewoZx4L15Ud9AugrHRUrIVhqgA4Eo9dvQ27cN4SuUfPG'

    # Allow tweepy to access profile
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    # Save all scraped tweets to one csv in the same form as twitter_gender.csv
    # 0 for male, 1 for female, followed by tweet
    save_csv = 'twitter_celeb_gender.csv'

    # for file in ('Top-1000-Celebrity-Twitter-Accounts.csv', 'bollywood.txt'):
    for file in ['news.txt']:
        content = read(file)
        scraped_profiles_file = '{}_scraped.npy'.format(file.split('.')[0])
        try:
            scraped_profiles = np.load(scraped_profiles_file).item()
        except IOError:
            scraped_profiles = 0
        for i, (celeb_handle, gender) in enumerate(content):
            print i, celeb_handle, gender
            if (i < scraped_profiles):
                print 'Already scraped'
                continue
            all_tweets = []
            oldest = None
            while (True):
                try:
                    new_tweets = api.user_timeline(
                        screen_name=celeb_handle,
                        count=200,
                        max_id=oldest,
                        tweet_mode='extended')
                except TweepError:
                    break
                if (len(new_tweets) == 0):
                    break
                all_tweets.extend(new_tweets)
                oldest = all_tweets[-1].id - 1
            tweets_to_save = map(
                lambda x: x.full_text.encode('utf-8').replace('\n', ''),
                all_tweets)
            print 'Writing {} tweets to {}'\
                .format(len(tweets_to_save), save_csv)
            with open(save_csv, 'a') as f:
                if (gender == 'F'):
                    gender = 1
                elif (gender == 'M'):
                    gender = 0
                elif (gender == 'N'):
                    gender = 2
                for tweet in tweets_to_save:
                    f.write('{},{}\n'.format(gender, tweet))
            scraped_profiles += 1
            np.save(scraped_profiles_file, scraped_profiles)
