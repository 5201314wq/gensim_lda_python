from gensim import corpora

# create a toy corpus of 2 documents, as a plain python list
corpus = [[(1, 0.5)], []]
# serialise to disk
corpora.MmCorpus.serialize('/tmp/corpus.mm', corpus)

# read from disk
corpus = corpora.MmCorpus('/tmp/corpus.mm')
print(corpus)
print(list(corpus))
# or
for doc in corpus:
    print(doc)


# other way to serialise to disk
# corpora.SvmLightCorpus.serialize('/tmp/corpus.svmlight', corpus)
# corpora.BleiCorpus.serialize('/tmp/corpus.lda-c', corpus)
# corpora.LowCorpus.serialize('/tmp/corpus.low', corpus)
