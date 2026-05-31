"""
ClinicalShield Module 3: Semantic Intent Verification for RAG

Verifies that retrieved documents' semantic intent aligns with
the clinical query intent. Flags documents with divergent intent
(e.g., injection instructions disguised as clinical content).
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel


class IntentVerifier:
    """
    Semantic Intent Verification for RAG-retrieved documents.
    Uses PubMedBERT embeddings to compute query-document alignment.
    """

    def __init__(self, model_name='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                 similarity_threshold=0.3, device='cuda'):
        self.device = device
        self.threshold = similarity_threshold
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

        # Clinical intent keywords for additional rule-based checking
        self.clinical_intents = [
            'diagnose', 'treatment', 'dosage', 'medication', 'prognosis',
            'symptom', 'adverse', 'contraindication', 'interaction',
            'guideline', 'protocol', 'assessment', 'monitoring',
            'laboratory', 'imaging', 'referral', 'follow-up',
        ]
        self.injection_markers = [
            'ignore', 'override', 'forget', 'disregard', 'bypass',
            'instead', 'new instruction', 'system prompt', 'you are now',
            'do not follow', 'pretend', 'act as', 'jailbreak',
            'DAN', 'developer mode', 'output the',
        ]

    def _get_embedding(self, text, max_length=256):
        """Get [CLS] embedding for a text."""
        encoding = self.tokenizer(
            text, max_length=max_length, truncation=True,
            padding='max_length', return_tensors='pt'
        )
        with torch.no_grad():
            outputs = self.model(
                input_ids=encoding['input_ids'].to(self.device),
                attention_mask=encoding['attention_mask'].to(self.device),
            )
        # Mean pooling over non-padding tokens
        mask = encoding['attention_mask'].unsqueeze(-1).to(self.device)
        hidden = outputs.last_hidden_state
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1)
        return (summed / counts).squeeze(0).cpu().numpy()

    def _cosine_similarity(self, a, b):
        """Compute cosine similarity between two vectors."""
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return dot / max(norm, 1e-8)

    def _check_injection_markers(self, text):
        """Rule-based check for known injection patterns."""
        text_lower = text.lower()
        found = [m for m in self.injection_markers if m in text_lower]
        return len(found) > 0, found

    def verify_document(self, query, document):
        """
        Verify if a retrieved document's intent aligns with the query.

        Args:
            query: The clinical query (e.g., "What is the dosage of metformin?")
            document: The retrieved document text

        Returns:
            dict with:
              - 'aligned': bool — True if document intent matches query intent
              - 'similarity': float — cosine similarity score
              - 'has_injection_markers': bool — True if known injection patterns found
              - 'injection_markers_found': list — which markers were found
              - 'risk_level': str — 'low', 'medium', 'high'
        """
        # Semantic similarity check
        query_emb = self._get_embedding(query)
        doc_emb   = self._get_embedding(document)
        similarity = self._cosine_similarity(query_emb, doc_emb)

        # Injection marker check
        has_markers, markers_found = self._check_injection_markers(document)

        # Determine risk level
        if has_markers and similarity < self.threshold:
            risk = 'high'
            aligned = False
        elif has_markers or similarity < self.threshold:
            risk = 'medium'
            aligned = False
        else:
            risk = 'low'
            aligned = True

        return {
            'aligned': aligned,
            'similarity': float(similarity),
            'has_injection_markers': has_markers,
            'injection_markers_found': markers_found,
            'risk_level': risk,
        }

    def verify_batch(self, query, documents):
        """Verify multiple retrieved documents against a query."""
        return [self.verify_document(query, doc) for doc in documents]
