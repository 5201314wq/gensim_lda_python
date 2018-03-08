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
    # texts = [[word.encode('utf-8') for word in json.loads(line)['comment_seg']]
    #          for line in open(data_path)]
    texts = []
    for line in open(data_path):
        word_list = []
        for word in json.loads(line)['comment_seg']:
            word = word.encode('utf-8')
            if len(word) == 3 or word == "":
                continue
            word_list.append(word)
        texts.append(word_list)

    print("texts", json.dumps(texts, encoding="UTF-8", ensure_ascii=False))
    print("texts")
    raw_input()
    # create dictionary
    dictionary = corpora.Dictionary(texts)
    # dictionary.filter_extremes(no_below=2, no_above=0.5, keep_n=1000000)
    # dictionary.filter_extremes(no_below=3, no_above=0.4, keep_n=10000000)
    dictionary.save(dictionary_path)
    print("dictionary", json.dumps(dictionary.token2id, encoding="UTF-8", ensure_ascii=False))
    print("dictionary")
    raw_input()
    # create corpus
    corpus = []
    for text in texts:
        if text == []:
            continue
        text = dictionary.doc2bow(text)
        corpus.append(text)
    # corpus = [dictionary.doc2bow(text) for text in texts]
    print("corpus", json.dumps(corpus, encoding="UTF-8", ensure_ascii=False))
    print("corpus")
    raw_input()
    corpora.MmCorpus.serialize(corpus_path, corpus)
    print("len_corpus", len(corpus))
    raw_input()

# load vocab from vocab_file
def load_vocab_file(vocab_path):
    vocab_list = []
    for word in open(vocab_path):
        vocab_list.append(word.rstrip())

def process_nytimes_data(data_path, vocab_path, dictionary_path, corpus_path):
    texts = []
    # load vocab file
    vocab = load_vocab_file(vocab_path)

    for line in open(data_path).readline()[3:]:
        print("line",line)
        raw_input()
        # specify word_id to word
        line = line.rstrip()

        # make file into the format of [[],[]]


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
    lda = models.LdaModel(corpus=tfidf_corpus, id2word=dictionary, num_topics=topic, iterations=200)
    lda_corpus = lda[tfidf_corpus]
    lda.save('/tmp/model.lda')
    corpora.MmCorpus.serialize('/tmp/lda_corpus.mm', lda_corpus)
    print('LDA Topics:')
    print(lda.print_topics(topic))


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description="Simple LDA using gensim")

    parser.add_argument('--data_choose', '-da_c',
                        help="The chose of dataset's process[nytimes or music]",
                        default='music',
                        required=False)
    parser.add_argument('--data_path', '-da',
                        help="The path of dataset",
                        default='./all.txt',
                        required=False)
    parser.add_argument('--vocab_path', '-vo',
                        help="The path of vocab set,only used in nytimes processing",
                        default='./vocab.nytimes.txt',
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
                        default=300,
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
    if args.data_choose == 'nytimes':
        process_nytimes_data(args.data_path, args.vocab_path, args.dictionary_path, args.corpus_path)
    elif args.data_choose == 'music':
        process_music_data(args.data_path, args.dictionary_path, args.corpus_path)
    else:
        print("you should choose a existed dataset")
        exit(0)
    train_lda_model(args.dictionary_path, args.corpus_path, args.topic)
