import annoy 
import time
import numpy as np

vector_count = 1000000
num_of_queries = 10000
vector_dim = 128
result_size = 100
k = 10

#For approximate recall:
#70% - 8 trees
#80% - 11 trees
#90% - 17 trees
#95% - 23 trees
#99% - 42 trees
num_of_forests = 8

file = open('C:\\Users\\mikul\\OneDrive\\VSB\\ING\\3SM\\AVD\\cviceni\\cv7\\sift1M\\sift1M.bin', 'r')
vector_data = np.reshape(np.fromfile(file, dtype=np.float32), [vector_count, vector_dim])
vector_labels = np.arange(vector_count)

file = open('C:\\Users\\mikul\\OneDrive\\VSB\\ING\\3SM\\AVD\\cviceni\\cv7\\sift1M\\siftQ1M.bin', 'r')
query_vectors = np.reshape(np.fromfile(file, dtype=np.float32), [num_of_queries, vector_dim])

file = open('C:\\Users\\mikul\\OneDrive\\VSB\\ING\\3SM\\AVD\\cviceni\\cv7\\sift1M\\knnQA1M.bin', 'r')
distance_results = np.reshape(np.fromfile(file, dtype=np.int32), [num_of_queries, result_size])

start = time.time()

t = annoy.AnnoyIndex(vector_dim, "euclidean")

for i in range(len(vector_labels)):
    t.add_item(i, vector_data[i])

end = time.time()
print("Index inserting time:", end-start, "s")    

start = time.time()

t.build(num_of_forests, -1)
#t.save('C:\\Users\\mikul\\test.ann')

end = time.time()
print("Index building time:", end-start, "s")

labels = []

start = time.time()

for i in range(len(query_vectors)):
    labels.append(t.get_nns_by_vector(query_vectors[i], k))

end = time.time()
print("Query time:", ((end - start) / num_of_queries) * 1000, "ms")

fn_count = 0
tp_count = 0
for i in range(len(labels)):
    for j in range(k):
        for h in range(result_size):
            if labels[i][j] == distance_results[i][h]:
                tp_count += 1
            else:
                fn_count += 1

print("Recall: ", (tp_count / (tp_count + fn_count)) * result_size) 