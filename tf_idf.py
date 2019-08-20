import string, math
import numpy as np
from documents import documents

# TF-IDF(term): TF(term in a document) * IDF(term)
# TF(term): # of times the term appears in document / total # of terms in document
# IDF(term): log(total # of documents / # of documents with term in it)

def parse(doc):
    return doc.translate(str.maketrans("", "", string.punctuation))

def create_vocab_map(parsed_documents):
    vocabulary = list(set("".join(parsed_documents).split()))
    return { word: index for index, word in enumerate(vocabulary) }

def create_index_to_vocab_map(vocab_map):
    return { value: key for key, value in vocab_map.items() }

def create_word_freq_array(document, vocab_map):
    vocabulary = vocab_map.keys()
    word_freqs = np.zeros(len(vocabulary))
    doc_total_word_count = len(document.split())

    for word in set(document.split()):
        if word in vocabulary:
            word_freqs[vocab_map[word]] += calculate_term_freq(word, document, doc_total_word_count)
    return word_freqs    

def calculate_term_freq(word, document, doc_total_word_count):
    word_count = document.split().count(word)
    return word_count / doc_total_word_count

def create_idf_array(vocab_map, documents):
    document_count = len(documents)
    vocabulary = vocab_map.keys()
    word_freqs = np.zeros(len(vocabulary))

    for word in vocabulary:
        docs_with_word = (len(list(filter(lambda x: word in x, documents))))
        if not docs_with_word == 0:
            word_freqs[vocab_map[word]] = math.log(document_count / docs_with_word)

    return word_freqs

def calculate_tf_idf(vocab_map, term_frequencies, inverse_document_frequencies):
    vocabulary = vocab_map.keys()
    tf_idf = np.zeros(len(vocabulary))

    for word in vocabulary:
        index = vocab_map[word]
        tf_idf_value = term_frequencies[index] * inverse_document_frequencies[index]
        tf_idf[index] = tf_idf_value

    return tf_idf

# parsed_documents = [parse(doc) for doc in documents]
# vocab_map = create_vocab_map(parsed_documents)
# index_to_vocab_map = create_index_to_vocab_map(vocab_map)
# idf = create_idf_array(vocab_map, parsed_documents)

# all_docs_td_idfs = np.array([calculate_tf_idf(vocab_map, create_word_freq_array(parse(document), vocab_map), idf) for document in parsed_documents])


def top_10_words(doc):
    tf_idf = calculate_tf_idf(vocab_map, create_word_freq_array(parse(doc), vocab_map), idf)
    indexes = np.argpartition(tf_idf, -10)[-10:]
    indexes = indexes[np.argsort(tf_idf[indexes])][::-1]

    top_10_tf_idf_values = [tf_idf[i] for i in indexes]
    top_10_tf_idf_words = [index_to_vocab_map[i] for i in indexes]

    tf_idf_scores_and_words = np.array(list(zip(top_10_tf_idf_values, top_10_tf_idf_words)))

    return tf_idf_scores_and_words


print(top_10_words(parsed_documents[0]))