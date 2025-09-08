import spacy
import networkx as nx
import json

nlp = spacy.load("en_core_web_sm")

# Load crawled docs (replace with your actual data source)
CRAWLED_DOCS = "crawled_finance_docs.json"
KG_FILE = "finance_knowledge_graph.gml"

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ["ORG", "PERSON", "MONEY", "GPE"]]

def build_knowledge_graph(docs):
    G = nx.Graph()
    for doc in docs:
        url = doc["url"]
        ents = extract_entities(doc["content"])
        for ent, label in ents:
            G.add_node(ent, label=label)
            G.add_edge(url, ent, type="mentions")
    nx.write_gml(G, KG_FILE)
    print(f"Knowledge graph saved to {KG_FILE}")

if __name__ == "__main__":
    with open(CRAWLED_DOCS) as f:
        docs = json.load(f)
    build_knowledge_graph(docs)
