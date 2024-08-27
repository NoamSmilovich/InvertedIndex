import os, shutil, re
from math import sqrt, log
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

# input docs path
docs_data_path = os.path.join(os.getcwd(), "docs.dat")

# input queries paths
train_queries_path = os.path.join(os.getcwd(), "train.queries")
test_queries_path = os.path.join(os.getcwd(), "test.queries")

# tfidf index builder paths
preprocessed_df_output_path = os.path.join(os.getcwd(), "preprocessed_df", "part-00000")
inverted_index_output_path = os.path.join(os.getcwd(), "inverted_index", "part-00000")
inverted_index_with_tfidf_output_path = os.path.join(os.getcwd(), "inverted_index_with_tfidf", "part-00000")
docs_df_output_path = os.path.join(os.getcwd(), "docs_df_output_path", "part-00000")

loaded_rdd = sc.textFile(inverted_index_with_tfidf_output_path)
inverted_index_with_tfidf = loaded_rdd.map(eval).collectAsMap()

#output file paths
relevant_train_docs_path = os.path.join(os.getcwd(), "relevant_train")
relevant_test_docs_path = os.path.join(os.getcwd(), "relevant_test")

# removal of pre-existing output files
output_paths = [os.path.join(os.getcwd(), "relevant"), relevant_train_docs_path, relevant_test_docs_path]
for p in output_paths:
    if os.path.exists(p):
        shutil.rmtree(p)
        
# processing input docs into dataframe
docs_df = spark.read.csv('file://' + docs_data_path, sep="\t", header=True) # strictly used to get the document count

def preprocess_text(text):
    clean_text = re.sub(r'\W+', ' ', text).lower()
    words = word_tokenize(clean_text)
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

def tfidf_from_docs_for_cosine_similarity(inverted_index_with_tfidf, query): 
    # function outputs a dictionary with the primary index being the doc_id, additionaly only words from the query are indexed here
    tfidf = {}
    for word in query:
        if word in inverted_index_with_tfidf:
            for doc_id in inverted_index_with_tfidf[word]:
                if doc_id not in tfidf:
                    tfidf[doc_id] = {}    
                tfidf[doc_id][word] = inverted_index_with_tfidf[word][doc_id]
    return tfidf

def process_single_query(query_terms):
    # transform to query term counts
    query_term_counts = {term: query_terms.count(term) for term in set(query_terms)}
    
    # get relevant tf-idf for cosine similarity computation
    docs_tfidf_vectors = tfidf_from_docs_for_cosine_similarity(inverted_index_with_tfidf, query_term_counts)
    # print(docs_tfidf_vectors)
    
    # preliminary calculations to get ready to compute tf-idf for the query
    max_tf = max(query_term_counts.values())
    total_doc_count = docs_df.count()
    
    # tf-idf computation for the query
    query_vector = {}
    for word in query_term_counts:
        normalized_tf = query_term_counts[word] / max_tf
        tf_idf = normalized_tf * log(total_doc_count / doc_frequencies[word]) if word in doc_frequencies else 0
        query_vector[word] = tf_idf
        
    # computation of cosine similarity
    cosine_similarities = {}
    for doc_id in docs_tfidf_vectors:
        product = 0
        for word in query_vector:
            if word in docs_tfidf_vectors[doc_id]:
                product += query_vector[word] * docs_tfidf_vectors[doc_id][word]
        
        def root_of_squares(vector):
            return sqrt(sum([i ** 2 for i in vector]))
        cosine_similarities[doc_id] = product / (root_of_squares(query_vector.values()) * root_of_squares(docs_tfidf_vectors[doc_id].values()))
    
    # # rank documents by cosine similarity
    ranked = sorted(cosine_similarities.items(), key=lambda item: item[1], reverse=True)
 
    # print(ranked)
    return [r[0] for r in ranked][0:5] # take the top 5 results


def process_queries(df):
    relevant = []
    for query in df.collect():
        query_id = query["Query ID"]
        cur_relevant_docs = process_single_query(query['filtered'])
        for d in cur_relevant_docs:
            relevant.append((query_id, d))
    return relevant

doc_frequencies = {}
for word, doc_counts in inverted_index_with_tfidf.items():
    doc_frequencies[word] = len(doc_counts)
    
paths = [(train_queries_path, relevant_train_docs_path), (test_queries_path, relevant_test_docs_path)] # tuples of paths (input, output)

for input_query_path, output_path in paths:
    df = spark.read.csv('file://' + input_query_path, sep="\t", header=True)
    preprocessed_df = df.filter(df["Query ID"] != "Query ID").rdd.map(lambda row: (int(row[0]), preprocess_text(row[1])))
    query_df = spark.createDataFrame(preprocessed_df, ["Query ID", "Query text"])
    query_df = Tokenizer(inputCol="Query text", outputCol="words").transform(query_df) # tokenization
    query_df = StopWordsRemover(inputCol="words", outputCol="filtered").transform(query_df) # remove stopwords

    relevant_list = process_queries(query_df)
    header = sc.parallelize(["Training Query ID\tDocument ID"])
    relevant_list_rdd = sc.parallelize(relevant_list, 2)
    relevant_list_rdd_with_header = header.union(relevant_list_rdd.map(lambda x: f"{x[0]}\t{x[1]}"))
    relevant_list_rdd_with_header.coalesce(1).saveAsTextFile('file://' + output_path)


# Stop the Spark Context
sc.stop()