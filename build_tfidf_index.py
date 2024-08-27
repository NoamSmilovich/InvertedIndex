import re, os, shutil
from math import log
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# nltk.download('punkt') # might have to run this line of code normally (meaning not as part of a batch job) to download necessary files for nltk to work
stemmer = PorterStemmer()

conf = SparkConf()
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

# input files paths
docs_data_path = os.path.join(os.getcwd(), "docs.dat")

# output files paths
preprocessed_df_output_path = os.path.join(os.getcwd(), "preprocessed_df")
inverted_index_output_path = os.path.join(os.getcwd(), "inverted_index")
inverted_index_with_tfidf_output_path = os.path.join(os.getcwd(), "inverted_index_with_tfidf")
docs_df_output_path = os.path.join(os.getcwd(), "docs_df_output_path")

# removal of pre-existing output files
output_paths = [preprocessed_df_output_path, inverted_index_output_path, inverted_index_with_tfidf_output_path, docs_df_output_path]
for p in output_paths:
    if os.path.exists(p):
        shutil.rmtree(p)

# processing input docs into dataframe
df = spark.read.csv('file://' + docs_data_path, sep="\t", header=True)

def preprocess_text(text):
    clean_text = re.sub(r'\W+', ' ', text).lower()
    words = word_tokenize(clean_text)
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

preprocessed_df = df.filter(df["Doc ID"] != "Doc ID").rdd.map(lambda row: (int(row[0]), preprocess_text(row[1])))
preprocessed_df.coalesce(1).saveAsTextFile('file://' + preprocessed_df_output_path)

docs_df = spark.createDataFrame(preprocessed_df, ["doc ID", "doc Text"]) # holds { doc_id_1: "doc_text_1", doc_id_2: "doc_text_2", ...}
docs_df = Tokenizer(inputCol="doc Text", outputCol="words").transform(docs_df) # tokenization
docs_df = StopWordsRemover(inputCol="words", outputCol="filtered").transform(docs_df) # remove stopwords
docs_df.rdd.coalesce(1).saveAsTextFile('file://' + docs_df_output_path) # holds { doc_id_1: "doc_text_1", doc_id_2: "doc_text_2", ...} after tokenization and removal of stopwords

# building the inverted index
def create_inverted_index(df):
    inverted_index = {}
    for row in df.collect():
        doc_id = row["doc ID"]
        for word in row["filtered"]: 
            if word not in inverted_index:
                inverted_index[word] = {doc_id: 1}
            else:
                if doc_id in inverted_index[word]:
                    inverted_index[word][doc_id] += 1
                else:
                    inverted_index[word][doc_id] = 1
    return inverted_index

inverted_index = create_inverted_index(docs_df)
inverted_index_rdd = sc.parallelize(inverted_index.items(), 2)
inverted_index_rdd.coalesce(1).saveAsTextFile('file://' + inverted_index_output_path) # holds { "word_1": {doc_id_a: count(word_1), doc_id_b: count(word_1), ...}, "word_2": {"doc_id_c", ...}, ...} 
# print(inverted_index)
print("finished inverted_index_rdd calculation")


doc_frequencies = {}
for word, doc_counts in inverted_index.items():
    doc_frequencies[word] = len(doc_counts)
# print(doc_frequencies) # holds { "word_1": total_appearences_of_word_1, "word_2": total_appearences_of_word_1, ...}
print("finished doc_frequencies calculation")

total_doc_count = docs_df.count()
print(f"total number of documents: {total_doc_count}")

# building max term frequency dictionary, to be able to normalize tf of each word by the max in each document
def build_max_tf_by_doc_dict(inverted_index):
    max_tf_by_doc = {}
    for word, doc_counts in inverted_index.items():
        for doc_id, count in doc_counts.items():
            if doc_id not in max_tf_by_doc:
                max_tf_by_doc[doc_id] = count
            else:
                max_tf_by_doc[doc_id] = max(max_tf_by_doc[doc_id], count)
    return max_tf_by_doc
max_term_freqs = build_max_tf_by_doc_dict(inverted_index)
# print(f"max_term_freqs: {max_term_freqs}")

# building tf-idf inverted index
inverted_index_with_tfidf = {}
for word, doc_counts in inverted_index.items(): # word, {doc_id_a: count(word_1), doc_id_b: count(word_1), ...}
    for doc_id in doc_counts:  # iterate over document id
        term_frequency = doc_counts[doc_id] # count of word in specific document
        normalized_tf = term_frequency / max_term_freqs[doc_id]
        # calculate TF-IDF using the formula: TF * log(N / DF)
        tf_idf = normalized_tf * log(total_doc_count / doc_frequencies[word])
        if word not in inverted_index_with_tfidf:
            inverted_index_with_tfidf[word] = {doc_id: tf_idf}
        else:
            inverted_index_with_tfidf[word][doc_id] = tf_idf

print("finished tf_idf calculation")
inverted_index_with_tfidf_rdd = sc.parallelize(inverted_index_with_tfidf.items(), 2)
inverted_index_with_tfidf_rdd.coalesce(1).saveAsTextFile('file://' + inverted_index_with_tfidf_output_path) # holds {word_1: {doc_a: tf_idf, doc_b: tf_idf, ...}, word_2: {...}, ...}

sc.stop()