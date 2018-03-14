#!/usr/bin/python
#coding:utf-8
#
# Copyright 2018 AISpeech Inc. All Rights Reserved.
# Author: sharon.wu@aispeech.com (Qian.Wu)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pyLDAvis
import argparse
import logging
import os
import json
import sys

from gensim import corpora, models, similarities

def load_lda_model(lda_path='/tmp/model.lda'):
    lda  = models.LdaModel.load(lda_path)
    for topic in lda.show_topics(formatted=False, num_topics=lda.num_topics, num_words=100):
        print(topic[0])
        raw_input()
        for item in topic[1]:
            print(" ",json.dumps(item[0], encoding="UTF-8", ensure_ascii=False))
        raw_input()


if __name__ == '__main__':
    load_lda_model()