from txtai.embeddings import Embeddings
import numpy as np
import sys

embeddings = Embeddings({'method': 'transformers', 'path': 'sentence-transformers/roberta-base-nli-stsb-mean-tokens'})

input_file = sys.argv[1]
mode = sys.argv[2]

index_name = 'index'

with open(input_file, 'r') as infile:
    sections = infile.readlines()

# Create an index for the list of sections
doc_dict = {}
index_text = []

for uid, text in enumerate(sections):
    doc_dict[uid] = text.split('\t')
    session_id, raw_text = doc_dict[uid][:2]
    if len(raw_text) > 250:
        index_text.append((uid, raw_text, None))

if mode == 'index':
    print("--indexing-- %d documents" % (len(index_text)))
    embeddings.index(index_text)
    embeddings.save(index_name)
elif mode == 'search':
    print("--searching-- %d documents" % (len(index_text)))
    embeddings.load(index_name)
    for query in ("What is possible today", "My philosophy has always been don't solve the human", "story about Larry", "biological memory", "short-term memory", "memory blocks", "nothing to do with us"):
    # Extract uid of first result
    # search result format: (uid, score)
        print(query)
        for i in range(0, 3):
            uid = embeddings.search(query, 3)[i][0]
            print("%-20s %s" % (query, doc_dict[uid]))
