import argparse
import subprocess as sp
import os
"""
Script for downloading the data sets from Kaggle

Usage:

    python download_data.py

Run

    python download_data.py --help

for help on the usage of command line arguments

Note: Kaggle requires a user to accept the rules of a competition
before they can download any data. So, downloading the data sets
from commandline requires Kaggle user specific data. Make sure
that DATA_DIRECTORY contains a "cookies.txt" from
"https:www.kaggle.com". You can get a copy of your cookies using
an extension like this:

    "https://chrome.google.com/webstore/detail/cookiestxt/njabckikapfpffapmjgojcnbfjonfjfg?hl=en"

"""

COOKIES = 'cookies.txt'
TRAIN = 'train.csv'
TEST = 'test.csv'


def download(url, directory, cookie):
    curl_bin = [
        'curl',
        '-L',
        '--cookie', cookie,
        '-o', directory,
        url
    ]
    sp.run(curl_bin)


def get_dataset(source, directory):
    train_path = os.path.join(directory, TRAIN)
    if os.path.exists(train_path):
        print('%s already exists' % train_path)
    else:
        url = os.path.join(source, TRAIN)
        print('downloading %s ...' % url)
        download(url, train_path, os.path.join(directory, COOKIES))
        print('done')

    test_path = os.path.join(directory, TEST)
    if os.path.exists(test_path):
        print('%s already exists' % test_path)
    else:
        url = os.path.join(source, TEST)
        print('downloading %s ...' % url)
        download(url, test_path, os.path.join(directory, COOKIES))
        print('done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--data_directory',
                        default='./data/',
                        help='directory to download the data sets')
    parser.add_argument('-L', '--source_url',
                        default='https://www.kaggle.com/c/digit-recognizer/download/',
                        help='url to the data sets')
    args = parser.parse_args()
    get_dataset(args.source_url, args.data_directory)
