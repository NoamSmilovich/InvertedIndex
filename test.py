import os, csv

# ground truth
train_rel_path = os.path.join(os.getcwd(), "train.rel")

# my results
relevant_result_path = os.path.join(os.getcwd(), "relevant_train", "part-00000")

def process_file_to_dictionary(filepath):
    data = []
    with open(filepath, mode='r') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)
        for row in reader:
            data.append(row)
    data = [[int(query_id), int(doc_id)] for query_id, doc_id in data]
    # print(data)

    dictionary_data = {}
    for query, document in data:
        if query not in dictionary_data:
            dictionary_data[query] = []
        dictionary_data[query].append(document)
        
    # print(dictionary_data)
    return dictionary_data

query_results = process_file_to_dictionary(relevant_result_path)
ground_truth = process_file_to_dictionary(train_rel_path)

def precision_at_k(relevant_docs, retrieved_docs, k=5):
    retrieved_k = retrieved_docs[:k]
    relevant_and_retrieved = set(relevant_docs).intersection(retrieved_k)
    return len(relevant_and_retrieved) / k

def recall_at_k(relevant_docs, retrieved_docs, k=5):
    retrieved_k = retrieved_docs[:k]
    relevant_and_retrieved = set(relevant_docs).intersection(retrieved_k)
    return len(relevant_and_retrieved) / len(relevant_docs)

# evaluate metrics for each query
precision_scores = []
recall_scores = []
lines = []

for query_id in ground_truth.keys():
    relevant_docs = ground_truth[query_id]
    retrieved_docs = query_results[query_id]
    
    k = 5
    precision = precision_at_k(relevant_docs, retrieved_docs, k)
    recall = recall_at_k(relevant_docs, retrieved_docs, k)
    
    precision_scores.append(precision)
    recall_scores.append(recall)
    
    lines.append(f"Query ID: {query_id}, Precision@5: {precision}, Recall@5: {recall}")
    print(lines[-1])

# calculate average precision and recall
avg_precision = sum(precision_scores) / len(precision_scores)
avg_recall = sum(recall_scores) / len(recall_scores)

lines.append(f"Average Precision@5: {avg_precision}")
print(lines[-1])
lines.append(f"Average Recall@5: {avg_recall}")
print(lines[-1])

with open("output.txt", "w") as file:
    for line in lines:
        file.write(line + "\n") 