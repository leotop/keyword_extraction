import nltk, string
import gensim
from itertools import takewhile, tee, izip, chain, groupby
import networkx, nltk
import collections, math, nltk, re
from nltk import word_tokenize, sent_tokenize, pos_tag_sents, chunk

punct = set(string.punctuation)
stop_words = set(nltk.corpus.stopwords.words('english'))
grammar=r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'
chunker = chunk.regexp.RegexpParser(grammar)
good_tags=set(['JJ','JJR','JJS','NN','NNP','NNS','NNPS'])

def seg_word_in_sentence(text):
    return [[word for word in word_tokenize(sent)] for sent in sent_tokenize(text)]

def tag_word_in_sentence(word_in_sentence):
    return [pos_tag_sents(words) for words in word_in_sentence]

def text_to_tagged_sents(text):
    word_in_sentence = seg_word_in_sentence(text)
    tagged_sents = tag_word_in_sentence(word_in_sentence)
    return tagged_sent

def extract_candidate_chunks(tagged_sents):
    all_chunks = list(chain.from_iterable(chunk.tree2conlltags(chunker.parse(tagged_sent)) for tagged_sent in tagged_sents))
    # join constituent chunk words into a single chunked phrase
    candidates = [' '.join(word 
                        for word, pos, chunk in group).lower() 
                            for key, group in 
                                    groupby(all_chunks, lambda (word,pos,chunk): chunk != 'O') 
                                    if key]

    return [cand for cand in candidates 
                    if cand.lower() not in stop_words 
                        and not all(char in punct for char in cand)]


def extract_candidate_words(tagged_sents):
    tagged_words = chain.from_iterable(tagged_sents)

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
            kp_words = list(takewhile(lambda x: x in keywords, words[i:i+10]))
            avg_pagerank = sum(word_ranks[w] for w in kp_words) / float(len(kp_words))
            keyphrases[' '.join(kp_words)] = avg_pagerank
            # counter as hackish way to ensure merged keyphrases are non-overlapping
            j = i + len(kp_words)

    return sorted(keyphrases.iteritems(), key=lambda x: x[1], reverse=True)


def score_keyphrases_by_tfidf(texts, candidates='chunks'):
    word_in_sentences = map(seg_word_in_sentence, texts)
    tagged_sents = map(tag_word_in_sentence, word_in_sentences)

    if candidates == 'chunks':
        boc_texts = map(extract_candidate_chunks, tagged_sents)
    elif candidates == 'words':
        boc_texts = map(extract_candidate_words, tagged_sents)

    # make gensim dictionary and corpus
    dictionary = gensim.corpora.Dictionary(boc_texts)
    corpus = [dictionary.doc2bow(boc_text) for boc_text in boc_texts]
    # transform corpus with tf*idf model
    tfidf = gensim.models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    for n, doc in enumerate(corpus_tfidf):
        print merge_keyword_into_keyphrases(doc, chain.from_iterable(*word_in_sentences[n]))



def pairwise(iterable, window_size=2):
    iters = tee(iterable, window_size)
    for n, i in enumerate(iters):
        for _ in xrange(n):
            next(i, None)

    return izip(*iters)

def score_keyphrases_by_textrank(text, n_keywords=0.05):

    # tokenize for all words, and extract *candidate* words
    words = [word.lower()
                for sent in sent_tokenize(text)
                    for word in word_tokenize(sent)]
    candidates = extract_candidate_words(text)
    # build graph, each node is a unique candidate
    graph = networkx.Graph()
    graph.add_nodes_from(set(candidates))
    # iterate over word-pairs, add unweighted edges into graph
    for w1, w2 in pairwise(candidates):
        if w2:
            graph.add_edge(*sorted([w1, w2]))
    # score nodes using default pagerank algorithm, sort by score, keep top n_keywords
    ranks = networkx.pagerank(graph)
    if 0 < n_keywords < 1:
        n_keywords = int(round(len(candidates) * n_keywords))
    word_ranks = {word_rank[0]: word_rank[1]
                for word_rank in sorted(ranks.iteritems(), key=lambda x: x[1], reverse=True)[:n_keywords]}

    return merge_keyword_into_keyphrases(word_ranks, words)



def get_stop_words_pattern(stop_word_list):
    stop_word_regex_list = []
    for word in stop_word_list:
        word_regex = r'\b' + word + r'(?![\w-])'  # added look ahead for hyphen
        stop_word_regex_list.append(word_regex)
    stop_word_pattern = re.compile('|'.join(stop_word_regex_list), re.IGNORECASE)
    return stop_words_pattern


def score_keypharses_by_rake(text):
    sents = [sent for sent in sent_tokenize(text)]
    

def score_keyphrases_by_topical_page_rank(texts, candidates='chunks', num_topics=10):
    corpus_tfidf, dictionary = score_keyphrases_by_tfidf(texts, candidates)
    lda = gensim.models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics)
    corpus_lda = lda[corpus_tfidf]

    for t_id in xrange(lda.num_topics):
        personlaiztion = lda.get_topic_terms(t_id)

    return corpus_lda, lda 

def main():
    text_one = """Connect Shopify and FreshBooks with CarryTheOne\nAs your business takes off, you need to concentrate on the things that matter and not get caught up in endless admin.\nGet your store & accounts system working together seamlessly with the Shopify integrator for FreshBooks accounts.\nDeveloped by CarryTheOne.\nWhat is FreshBooks?\nNorth America's leading online accountancy & bookkeeping software, with over 2 million small business users.\n(Free Trial available)\nHow Does It Work?\nIt automatically imports invoices from your Shopify store directly into your FreshBooks accounts software in real time, creating clients/customers where required and payments automatically (if desired).\nAlso compatible with Shopify POS (Shopify's Point-of-Sale system) so that you may automate accounting for your physical store as well as your online store!\nWhy Does Your Business Need It?\n\nSaves Time and Money\nIncreases Accuracy\nProvides Instant Financial Reporting\nCentralizes Invoicing\nSimplifies Tax returns\n\nHow Much Does It Cost?\n\nFree 30 day trial. Then $31.99 USD per month.\nFree Set-up\nFree support and upgrades.\n"""
    
    text_two = """FEATURES:\n\nCreate product catalogs in minutes using products in custom collections/smart collections/Vendors/Product types.\nCreate a front cover page and a back cover page for the catalog using text,custom fonts and images (upload your own)\nAdd social media links , company info on the back cover\nChoose one of three available layouts (Portrait/Landscape/Tabular)\nChoose a color schema/template from many freely available\nIf you want you can build your own template using tons of customization options ; save it and re-use them for next catalogs\nPreview your custom template before using with real products, this gives you a real quick way to create custom templates and modify them.\nSave up-to 12 catalogs\nOnce click PDF creation using the saved catalogs\n\n\nNot Sure?? Try it before you buy it! All plans come with 1-2 days trial!"""

    corpus_tfidf, dictionary =  score_keyphrases_by_tfidf([text_one, text_two], 'words')
    
    for n, doc in enumerate(corpus_tfidf):
        print n, '----'
        for word, weight in doc[:10]:
            print dictionary[word], weight

    corpus_tfidf, dictionary =  score_keyphrases_by_tfidf([text_one, text_two])
    
    for n, doc in enumerate(corpus_tfidf):
        print n, '----'
        for word, weight in doc[:10]:
            print dictionary[word], weight

    corpus_lda, lda =  score_keyphrases_by_topical_page_rank([text_one, text_two], num_topics=5)
    import ipdb; ipdb.set_trace()
    for n, doc in enumerate(corpus_lda):
        print n, '----'
        for topic, weight in doc[:10]:
            print lda.print_topic(topic), weight

if __name__ == "__main__":
    main()
