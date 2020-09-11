
from txtai.embeddings import Embeddings 
import numpy as np

embeddings = Embeddings({'method': 'transformers', 'path': 'sentence-transformers/roberta-base-nli-stsb-mean-tokens'})

# Create an index for the list of sections
embeddings.index([(uid, text, None) for uid, text in enumerate(sections)])

print("%-20s %s" % ("Query", "Best Match"))
print("-" * 50)

# Run an embeddings search for each query
for query in ("feel good story", "climate change", "health", "war", "wildlife", "asia",
              "north america", "dishonest junk"):
    # Extract uid of first result
    # search result format: (uid, score)
    uid = embeddings.search(query, 1)[0][0]

    # Print section
    print("%-20s %s" % (query, sections[uid]))

