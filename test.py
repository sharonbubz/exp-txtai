
from txtai.embeddings import Embeddings 
import numpy as np

embeddings = Embeddings({'method': 'transformers', 'path': 'sentence-transformers/roberta-base-nli-stsb-mean-tokens'})

sections = ["US tops 5 million confirmed virus cases",
            "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
            "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
            "The National Park Service warns against sacrificing slower friends in a bear attack",
            "Maine man wins $1M from $25 lottery ticket",
            "Make huge profits without work, earn up to $100,000 a day"]

print("%-20s %s" % ("Query", "Best Match"))
print("-" * 50)

for query in ("feel good story", "climate change", "health", "war", "wildlife", "asia",
              "north america", "dishonest junk"):
    # Get index of best section that best matches query
    uid = np.argmax(embeddings.similarity(query, sections))

    print("%-20s %s" % (query, sections[uid]))