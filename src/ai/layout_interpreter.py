import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LayoutLMv2Model, LayoutLMv2Config
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
import numpy as np

class LayoutInterpreter(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=128, num_layers=2):
        super(LayoutInterpreter, self).__init__()
        self.config = LayoutLMv2Config.from_pretrained("microsoft/layoutlmv2-base-uncased")
        self.layoutlm = LayoutLMv2Model(self.config)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, token_type_ids, bbox):
        outputs = self.layoutlm(input_ids=input_ids, 
                                attention_mask=attention_mask, 
                                token_type_ids=token_type_ids, 
                                bbox=bbox)
        sequence_output = outputs.last_hidden_state
        lstm_out, _ = self.lstm(sequence_output)
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_output = self.dropout(attn_output)
        output = self.fc(attn_output.mean(dim=1))
        return F.normalize(output, p=2, dim=1)

def process_layout_predictions(model, tokenizer, image, text):
    encoding = tokenizer(text, image=image, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoding)
    return output

def cluster_embeddings(embeddings, min_cluster_size=2, min_samples=1):
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean')
    clusters = clusterer.fit_predict(embeddings.numpy())
    return clusters

sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

def interpret_embedding(embedding):
    # Convert embedding to text using Sentence Transformers
    sentences = ["folder", "file", "config", "source", "test", "docs", "data"]
    sentence_embeddings = sentence_model.encode(sentences)
    similarities = np.dot(sentence_embeddings, embedding)
    most_similar = sentences[np.argmax(similarities)]
    return f"{most_similar}_{hash(tuple(embedding.numpy()))}"

def build_directory_structure(interpreted_clusters):
    structure = {}
    for name in interpreted_clusters:
        parts = name.split('/')
        current = structure
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = None
    return structure
