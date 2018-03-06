from six import iteritems
from gensim import corpora
from pprint import pprint  # pretty-printer

class MyCorpus(object):
    def __iter__(self):
        for line in open('mycorpus.txt'):
            yield dictionary.doc2bow(line.lower().split())

# remove common words and tokenize
stoplist = set('for a of the and to in'.split())

dictionary = corpora.Dictionary(line.lower().split() for line in open('mycorpus.txt'))
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
            if stopword in dictionary.token2id]

once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
dictionary.filter_tokens(stop_ids + once_ids)  # remove stop words and words that appear only once
dictionary.compactify()   # remove gaps in id sequence after words that were removed

dictionary.save('/tmp/deerwester.dict')
print(dictionary)
print(dictionary.token2id)

new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)

corpus_memory_friendly = MyCorpus()
print(corpus_memory_friendly)

for vector in corpus_memory_friendly: # load one vector into memory at a time
    print(vector)

corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus_memory_friendly)

