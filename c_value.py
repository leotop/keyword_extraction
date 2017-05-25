import nltk, string
import gensim
from itertools import takewhile, tee, izip, chain, groupby, combinations
from collections import Counter, defaultdict
from functools import partial
from operator import itemgetter
import math
import networkx, nltk
import collections, math, nltk, re
from nltk import word_tokenize, sent_tokenize, pos_tag_sents, chunk
from nltk.tokenize import RegexpTokenizer

punct = set(string.punctuation)
stop_words = set(nltk.corpus.stopwords.words('english'))
grammar = r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'
chunker = chunk.regexp.RegexpParser(grammar)
good_tags = set(['JJ', 'JJR', 'JJS', 'NN', 'NNP', 'NNS', 'NNPS'])

import re
import spacy

def filtered_chunks(doc, pattern):
  for chunk in doc.noun_chunks:
    signature = ''.join(['<%s>' % w.tag_ for w in chunk])
    if pattern.match(signature) is not None:
      yield chunk

def chunker_for_spacy():
	nlp = spacy.load('en')
	doc = nlp(u'Great work!')
	pattern = re.compile(r'(<JJ>)*(<NN>|<NNS>|<NNP>)+')

	print(list(filtered_chunks(doc, pattern)))

#condition_1 = r"""(\d*-NN\D?\s)+(\d*-NN\D?\s?)"""
#
#condition_2 = re.compile(r"""(\d*-JJ\D?\s|\d*-NN\D?\s)+(\d*-NN\D?\s?)""")
#
#condition_3 = re.compile(r"""((\d*-JJ\D?\s|\d*-NN\D?\s)+(\d*-NN\D?\s?))|
#                             (((\d*-JJ\D?\s|\d*-NN\D?\s)*(\d*-IN\s)?)
#                             (\d*-JJ\D?\s|\d*-NN\D?\s)*(\d*-NN\D?\s?))""")
#chunker = chunk.regexp.RegexpParser(condition_1)

def condition_matches(sentence, condition):
    tags = " ".join(["{0}-{1}".format(t.index, t.pos)
                     for t in sentence])
    for match in re.finditer(condition, tags):
        g = match.group().strip().split()
        s_index, s_pos = g[0].split("-")
        e_index, e_pos = g[-1].split("-")
        yield int(s_index), int(e_index)+1

def sentens_to_candidate(sentences):
    words = [[word for word in word_tokenize(sent)] for sent in sent_tokenize(sentences)]
    tagged_sents = pos_tag_sents(words)
    all_chunks = list(
        chain.from_iterable(chunk.tree2conlltags(chunker.parse(tagged_sent)) for tagged_sent in tagged_sents))
    # join constituent chunk words into a single chunked phrase
    return [' '.join(word for word, pos, chunk_tag in group).lower()
                  for key, group in groupby(all_chunks, lambda (word, pos, chunk_tag): chunk_tag != 'O')
                  if key]


class CValue(object):
    def __init__(self, minum_freq=2):
        self.minum_freq = minum_freq
        self.freq = defaultdict(int)
        self.prefixs = defaultdict(set)
        self.cvalue = {}

    def fit(self, docs):
        term_list = chain(*docs)
        for words in term_list:
            self.freq[words] += 1
            for word in words.split():
                self.prefixs[word].add(words)

        self.freq = {k: v for k, v in self.freq.iteritems() if v >= self.minum_freq}
        
        words_list = set(chain(*docs))
        for words in words_list:
            word_list = set(words.split())
            t_b = filter(lambda r: words in r, (reduce(lambda x, y: x.intersection(y), map(lambda r: self.prefixs[r], word_list))-set([words])))
            base = math.log(len(words.split()))
            if len(t_b) == 0:
                self.cvalue[words] = base*self.freq[words]
            else:
                self.cvalue[words] = base*(self.freq[words] - 1/len(t_b)*sum(map(lambda r: self.freq[r], t_b)))
        return self

    def transform(self, docs):
        res = []
        for term_list in docs:
            valid_terms = filter(self.cvalue.get, term_list)
            weight_terms = map(lambda r: (r, self.cvalue[r]), valid_terms)
            res.append(sorted(weight_terms, key=itemgetter(1), reverse=True))
        return res


def cvalue(list_of_sentences):
    docs = map(lambda r: sentens_to_candidate(r), list_of_sentences)
    cv = CValue(1)
    cv.fit(docs)
    print docs
    for i in cv.transform(docs):
        print i


if __name__ == "__main__":
    text_one = """Connect E Commerce and FreshBooks with CarryTheOne\nAs your business takes off, you need to concentrate on the things that matter and not get caught up in endless admin.\nGet your store & accounts system working together seamlessly with the E Commerce integrator for FreshBooks accounts.\nDeveloped by CarryTheOne.\nWhat is FreshBooks?\nNorth America's leading online accountancy & bookkeeping software, with over 2 million small business users.\n(Free Trial available)\nHow Does It Work?\nIt automatically imports invoices from your E Commerce store directly into your FreshBooks accounts software in real time, creating clients/customers where required and payments automatically (if desired).\nAlso compatible with E Commerce POS (E Commerce's Point-of-Sale system) so that you may automate accounting for your physical store as well as your online store!\nWhy Does Your Business Need It?\n\nSaves Time and Money\nIncreases Accuracy\nProvides Instant Financial Reporting\nCentralizes Invoicing\nSimplifies Tax returns\n\nHow Much Does It Cost?\n\nFree 30 day trial. Then $31.99 USD per month.\nFree Set-up\nFree support and upgrades.\n"""

    text_two = """FEATURES:\n\nCreate product catalogs in minutes using products in custom collections/smart collections/Vendors/Product types.\nCreate a front cover page and a back cover page for the catalog using text,custom fonts and images (upload your own)\nAdd social media links , company info on the back cover\nChoose one of three available layouts (Portrait/Landscape/Tabular)\nChoose a color schema/template from many freely available\nIf you want you can build your own template using tons of customization options ; save it and re-use them for next catalogs\nPreview your custom template before using with real products, this gives you a real quick way to create custom templates and modify them.\nSave up-to 12 catalogs\nOnce click PDF creation using the saved catalogs\n\n\nNot Sure?? Try it before you buy it! All plans come with 1-2 days trial!"""
    cvalue([text_one, text_two])
