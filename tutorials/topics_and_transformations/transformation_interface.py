import os
from gensim import corpora, models, similarities

## Creating a transformation
# first you should run the script memory_friendly.py
if (os.path.exists("/tmp/deerwester.dict")):
    dictionary = corpora.Dictionary.load('/tmp/deerwester.dict')
    corpus = corpora.MmCorpus('/tmp/deerwester.mm')
    print("Used files generated from first tutorial")
else:
    print("Please run first tutorial/corpora_and_vector_spaces/memory_friendly\
          to generate data set")


tfidf = models.TfidfModel(corpus) # step 1 --initialize a model
## transforming vectors
doc_bow = [(0, 1), (1, 1)]
print(tfidf[doc_bow]) # step 2 -- use the model to transform vectors
corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
    print("tfidf", doc)

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2) # initialize an LSI transformation
corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
lsi.print_topics(2)
for doc in corpus_lsi: # both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
    print("lsimodel", doc)
lsi.save('/tmp/model.lsi') # same for tfidf, lda, ...
lsi = models.LsiModel.load('/tmp/model.lsi')

lda = models.LdaModel(corpus, id2word=dictionary, num_topics=1)
corpus_lda = lda[corpus_tfidf]
lda.print_topic(2)
lda.save('/tmp/model.lda')
corpora.MmCorpus.serialize('/tmp/lda_corpus.mm', corpus_lda)
for doc in corpus_lda:
    print("ldamodel", doc)


