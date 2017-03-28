import nltk, string
import gensim
from itertools import takewhile, tee, izip, chain, groupby, combinations
from collections import Counter, defaultdict
from functools import partial
from operator import itemgetter
import networkx, nltk
import collections, math, nltk, re
from nltk import word_tokenize, sent_tokenize, pos_tag_sents, chunk
from nltk.tokenize import RegexpTokenizer

punct = set(string.punctuation)
stop_words = set(nltk.corpus.stopwords.words('english'))
grammar = r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'
chunker = chunk.regexp.RegexpParser(grammar)
good_tags = set(['JJ', 'JJR', 'JJS', 'NN', 'NNP', 'NNS', 'NNPS'])


def convert_text_into_sentences(text):
    return [[word for word in word_tokenize(sent)] for sent in sent_tokenize(text)]


def pos_tag_sentences(sentences):
    return pos_tag_sents(sentences)


def convert_pos_tagged_sentences_into_chunks(tagged_sents):
    all_chunks = list(
        chain.from_iterable(chunk.tree2conlltags(chunker.parse(tagged_sent)) for tagged_sent in tagged_sents))
    # join constituent chunk words into a single chunked phrase
    candidates = [' '.join(word for word, pos, chunk_tag in group).lower()
                  for key, group in groupby(all_chunks, lambda (word, pos, chunk_tag): chunk_tag != 'O')
                  if key]

    return [candidate for candidate in candidates
            if candidate.lower() not in stop_words
            and not all(char in punct for char in candidate)]


def convert_pos_tagged_sentences_into_words(tagged_sents):
    tagged_words = list(chain(*tagged_sents))
    candidates = [word.lower() for word, tag in tagged_words
                  if tag in good_tags
                  and word.lower() not in stop_words
                  and not all(char in punct for char in word)]

    return candidates


def merge_keyword_into_keyphrases(word_ranks, words):
    keywords = set(word_ranks.keys())
    keyphrases = {}
    j = 0
    for i, word in enumerate(words):
        if i < j:
            continue
        if word in keywords:
            kp_words = list(takewhile(lambda x: x in keywords, words[i:i + 10]))
            avg_rank = sum(word_ranks[w] for w in kp_words) / float(len(kp_words))
            keyphrases[' '.join(kp_words)] = avg_rank
            # counter as hackish way to ensure merged keyphrases are non-overlapping
            j = i + len(kp_words)

    return sorted(keyphrases.iteritems(), key=lambda x: x[1], reverse=True)


def convert_text_into_docs(texts, candidates='chunks'):
    word_in_sentences = map(convert_text_into_sentences, texts)

    tagged_sents = map(pos_tag_sentences, word_in_sentences)
    if candidates == 'chunks':
        boc_texts = map(convert_pos_tagged_sentences_into_chunks, tagged_sents)
    elif candidates == 'words':
        boc_texts = map(convert_pos_tagged_sentences_into_words, tagged_sents)
    return boc_texts

def convert_text_into_tfidf(texts, candidates='chunks'):
    boc_texts = convert_text_into_docs(texts, candidates)

    # make gensim dictionary and corpus
    dictionary = gensim.corpora.Dictionary(boc_texts)
    corpus = [dictionary.doc2bow(boc_text) for boc_text in boc_texts]
    # transform corpus with tf*idf model
    tfidf = gensim.models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    return boc_texts, corpus_tfidf, dictionary


def score_keyphrases_by_tfidf(texts, candidates='chunks', top_k=10):
    boc_texts, corpus_tfidf, dictionary = convert_text_into_tfidf(texts, candidates)
    res = []
    for n, doc in enumerate(corpus_tfidf):
        word_ranks = dict(map(lambda (x, y): (dictionary[x], y), doc[:top_k]))
        res.append(merge_keyword_into_keyphrases(word_ranks, boc_texts[n]))
    return res


def pairwise(iterable, window_size=2):
    iters = tee(iterable, window_size)
    for n, i in enumerate(iters):
        for _ in xrange(n):
            next(i, None)

    return izip(*iters)

def graph_rank(candidates, n_keywords=10, personalization=None):
    graph = networkx.Graph()
    graph.add_nodes_from(set(candidates))
    # iterate over word-pairs, add unweighted edges into graph
    for _ in pairwise(candidates):
        for w1, w2 in combinations(_, 2):
            if w2:
                graph.add_edge(*sorted([w1, w2]))
    # score nodes using default pagerank algorithm, sort by score, keep top n_keywords
    ranks = networkx.pagerank(graph, personalization=personalization)
    if 0 < n_keywords < 1:
        n_keywords = int(round(len(candidates) * n_keywords))
    word_ranks = {word_rank[0]: word_rank[1]
                  for word_rank in sorted(ranks.iteritems(), key=lambda x: x[1], reverse=True)[:n_keywords]}
    return word_ranks

def _score_keyphrases_by_textrank(text, n_keywords=10):
    # tokenize for all words, and extract *candidate* words
    candidates = convert_text_into_docs([text], 'words')[0]
    # build graph, each node is a unique candidate
    word_ranks = graph_rank(candidates)
    return merge_keyword_into_keyphrases(word_ranks, candidates)

def score_keyphrases_by_textrank(texts, n_keywords=10):
    return map(partial(_score_keyphrases_by_textrank, n_keywords=n_keywords), texts)


def split_sentences(text):
    pattern = re.compile('\s(%s)(?!\w)|(\\n)' % ('|'.join(stop_words)))
    word_tokenizer = RegexpTokenizer(pattern, gaps=True)
    sent_in_pharses = [word_tokenizer.tokenize(sent) for sent in sent_tokenize(text)]
    for pharses in sent_in_pharses:
        for pharse in pharses:
            words = filter(lambda r: r not in punct, word_tokenize(pharse))
            if len(words) > 0:
                yield words


def _score_keypharses_by_rake(text):
    pharses = [pharses for pharses in split_sentences(text)]
    freq_score = Counter(chain(*pharses))
    degress_score = Counter()
    for pharse in pharses:
        for word in pharse:
            degress_score.update({word:len(pharse)-1})
    word_score = {}
    for word in freq_score:
        word_score[word] = (degress_score[word]+freq_score[word])/(freq_score[word]*1.0)
    pharse_score = defaultdict(int)
    for pharse in pharses:
        pharse_score[' '.join(pharse)] += sum(map(lambda r: word_score[r], pharse))
    scored_phrase = sorted(pharse_score.items(), key=itemgetter(1), reverse=True)[:10]
    return scored_phrase

def score_keypharses_by_rake(texts):
    return map(_score_keypharses_by_rake, texts)

def score_keyphrases_by_topical_page_rank(texts, candidates='chunks', num_topics=10):
    doc_texts, corpus_tfidf, dictionary = convert_text_into_tfidf(texts, candidates)
    lda = gensim.models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics)
    corpus_lda = lda[corpus_tfidf]
    res = []
    for x, doc in enumerate(doc_texts):
        rank = Counter()
        topics = corpus_lda[x]
        for t_id, weight in topics:
            personalization = dict((dictionary[i], j) for (i, j) in lda.get_topic_terms(t_id, lda.num_terms) if dictionary[i] in doc)
            word_rank = graph_rank(doc, personalization=personalization)
            rank.update(dict((i, j*weight) for i, j in word_rank.items()))
        res.append(merge_keyword_into_keyphrases(rank, doc))
    return res


def main():
    text_one = """Connect Shopify and FreshBooks with CarryTheOne\nAs your business takes off, you need to concentrate on the things that matter and not get caught up in endless admin.\nGet your store & accounts system working together seamlessly with the Shopify integrator for FreshBooks accounts.\nDeveloped by CarryTheOne.\nWhat is FreshBooks?\nNorth America's leading online accountancy & bookkeeping software, with over 2 million small business users.\n(Free Trial available)\nHow Does It Work?\nIt automatically imports invoices from your Shopify store directly into your FreshBooks accounts software in real time, creating clients/customers where required and payments automatically (if desired).\nAlso compatible with Shopify POS (Shopify's Point-of-Sale system) so that you may automate accounting for your physical store as well as your online store!\nWhy Does Your Business Need It?\n\nSaves Time and Money\nIncreases Accuracy\nProvides Instant Financial Reporting\nCentralizes Invoicing\nSimplifies Tax returns\n\nHow Much Does It Cost?\n\nFree 30 day trial. Then $31.99 USD per month.\nFree Set-up\nFree support and upgrades.\n"""

    text_two = """FEATURES:\n\nCreate product catalogs in minutes using products in custom collections/smart collections/Vendors/Product types.\nCreate a front cover page and a back cover page for the catalog using text,custom fonts and images (upload your own)\nAdd social media links , company info on the back cover\nChoose one of three available layouts (Portrait/Landscape/Tabular)\nChoose a color schema/template from many freely available\nIf you want you can build your own template using tons of customization options ; save it and re-use them for next catalogs\nPreview your custom template before using with real products, this gives you a real quick way to create custom templates and modify them.\nSave up-to 12 catalogs\nOnce click PDF creation using the saved catalogs\n\n\nNot Sure?? Try it before you buy it! All plans come with 1-2 days trial!"""

    keypharses = score_keyphrases_by_tfidf([text_one], 'words')

    for doc_keyphases in keypharses:
        print doc_keyphases

    keypharses = score_keyphrases_by_tfidf([text_one])

    for doc_keyphases in keypharses:
        print doc_keyphases

    for doc_keyphases in score_keyphrases_by_textrank([text_one]):
        print doc_keyphases

    for doc_keyphases in score_keypharses_by_rake([text_one]):
        print doc_keyphases

    for doc_keyphases in score_keyphrases_by_topical_page_rank([text_one]):
        print doc_keyphases


if __name__ == "__main__":
    main()
