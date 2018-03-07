#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2018 AISpeech Inc. All Rights Reserved.
# Author: sharon.wu@aispeech.com (Qian.Wu)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import json

from gensim import corpora, models, similarities

def process_music_data(data_path, dictionary_path, corpus_path):
    texts = [[word.encode('utf-8') for word in json.loads(line)['comment_seg']]
             for line in open(data_path) if json.loads(line)['comment_seg'] is not ""]

    # create dictionary
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=2, no_above=0.5, keep_n=100000)
    dictionary.save(dictionary_path)
    print("dictionary", dictionary.token2id)

    # create corpus
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize(corpus_path, corpus)
    print("len_corpus", len(corpus))

def process_nytimes_data(data_path, dictionary_path, corpus_path):
    pass


def train_lda_model(dictionary_path, corpus_path, topic):
    dictionary = corpora.Dictionary.load(dictionary_path)
    corpus = corpora.MmCorpus(corpus_path)
    print('used files generated from string2vector')

    # create a tfidf model
    tfidf = models.TfidfModel(corpus=corpus)
    tfidf.save('/tmp/model.tfidf')
    tfidf_corpus = tfidf[corpus]
    corpora.MmCorpus.serialize('/tmp/tfidf_corpus.mm', tfidf_corpus)

    # create a lda model
    lda = models.LdaModel(corpus=tfidf_corpus, id2word=dictionary, num_topics=topic, iterations=100)
    lda_corpus = lda[tfidf_corpus]
    lda.save('/tmp/model.lda')
    corpora.MmCorpus.serialize('/tmp/lda_corpus.mm', lda_corpus)
    print('LDA Topics:')
    print(lda.print_topics(topic))


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description="Simple LDA using gensim")

    parser.add_argument('--data_path', '-da',
                        help="The path of dataset",
                        default='./all.txt',
                        required=False)
    parser.add_argument('--dictionary_path', '-di',
                        help="The path of dictionary",
                        default='/tmp/music_dict.dict',
                        required=False)
    parser.add_argument('--corpus_path', '-c',
                        help="The path of corpus",
                        default='/tmp/music_corpus.mm',
                        required=False)
    parser.add_argument('--topic', '-t',
                        help="The path of topic",
                        default=200,
                        required=False)
    parser.add_argument('--verbose', '-v',
                        help="Be verbose -- debug logging level",
                        required=False,
                        action='store_true')
    args = parser.parse_args()

    # Logging
    logLevel = logging.INFO
    if args.verbose:
        logLevel = logging.DEBUG
    logging.basicConfig(level=logLevel)
    logging.info('Initializing...')

    # if os.path.isfile(args.dictionary_path) is False or os.path.isfile(args.corpus_path) is False:
    process_music_data(args.data_path, args.dictionary_path, args.corpus_path)
    train_lda_model(args.dictionary_path, args.corpus_path, args.topic)
