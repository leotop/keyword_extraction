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

def cvalue(list_of_sentences):
    candidates = list(chain(*map(lambda r: sentens_to_candidate(r), list_of_sentences)))
    freq = defaultdict(int)
    prefixs = defaultdict(set)
    for words in candidates:
        freq[words] += 1
        for word in words.split():
            prefixs[word].add(words)
    for words in candidates:
        t_b = filter(lambda r: words in r, reduce(lambda x, y: x.intersection(y), map(lambda r: prefixs[r], words.split()))-set([words]))
        base = math.log(len(words.split()))
        if len(t_b) == 0:
            print [words], base*freq[words]
        else:
            print [words], base*(freq[words] - 1/len(t_b)*sum(map(lambda r: freq[r], t_b)))
    return candidates



if __name__ == "__main__":
    text_one = """Connect Shopify and FreshBooks with CarryTheOne\nAs your business takes off, you need to concentrate on the things that matter and not get caught up in endless admin.\nGet your store & accounts system working together seamlessly with the Shopify integrator for FreshBooks accounts.\nDeveloped by CarryTheOne.\nWhat is FreshBooks?\nNorth America's leading online accountancy & bookkeeping software, with over 2 million small business users.\n(Free Trial available)\nHow Does It Work?\nIt automatically imports invoices from your Shopify store directly into your FreshBooks accounts software in real time, creating clients/customers where required and payments automatically (if desired).\nAlso compatible with Shopify POS (Shopify's Point-of-Sale system) so that you may automate accounting for your physical store as well as your online store!\nWhy Does Your Business Need It?\n\nSaves Time and Money\nIncreases Accuracy\nProvides Instant Financial Reporting\nCentralizes Invoicing\nSimplifies Tax returns\n\nHow Much Does It Cost?\n\nFree 30 day trial. Then $31.99 USD per month.\nFree Set-up\nFree support and upgrades.\n"""

    text_two = """FEATURES:\n\nCreate product catalogs in minutes using products in custom collections/smart collections/Vendors/Product types.\nCreate a front cover page and a back cover page for the catalog using text,custom fonts and images (upload your own)\nAdd social media links , company info on the back cover\nChoose one of three available layouts (Portrait/Landscape/Tabular)\nChoose a color schema/template from many freely available\nIf you want you can build your own template using tons of customization options ; save it and re-use them for next catalogs\nPreview your custom template before using with real products, this gives you a real quick way to create custom templates and modify them.\nSave up-to 12 catalogs\nOnce click PDF creation using the saved catalogs\n\n\nNot Sure?? Try it before you buy it! All plans come with 1-2 days trial!"""
    cvalue([text_one, text_two])
