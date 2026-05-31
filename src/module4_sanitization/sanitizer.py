"""
ClinicalShield Module 4: Utility-Preserving Clinical Sanitization

Removes adversarial content from clinical text while preserving
clinical entities (drugs, dosages, conditions, lab values).
Introduces CEPS (Clinical Entity Preservation Score) for validation.
"""

import re
import spacy
from collections import Counter


class ClinicalSanitizer:
    """
    Removes adversarial instructions while preserving clinical content.
    Uses POS tagging to identify imperative commands and NER to
    verify clinical entity preservation.
    """

    def __init__(self, risk_threshold=0.5, utility_threshold=0.95):
        self.risk_threshold = risk_threshold
        self.utility_threshold = utility_threshold
        self.nlp = spacy.load('en_core_web_sm')

        # Imperative/instruction patterns
        self.imperative_patterns = [
            r'(?i)\b(ignore|disregard|forget|bypass|override|skip)\b.*?[.!]',
            r'(?i)\b(do not follow|don\'t follow|stop following)\b.*?[.!]',
            r'(?i)\b(you are now|you must|you should now|act as|pretend)\b.*?[.!]',
            r'(?i)\b(output|print|display|reveal|show)\s+(the|all|your)\b.*?[.!]',
            r'(?i)\b(new instruction|system prompt|developer mode|DAN mode)\b.*?[.!]',
            r'(?i)\b(instead of|rather than)\s+follow.*?[.!]',
            r'(?i)\bEditor\'s Note:.*?(?=\n|$)',
        ]

        # Clinical entity patterns (regex-based for speed)
        self.clinical_patterns = {
            'drug': r'(?i)\b(metformin|insulin|aspirin|lisinopril|atorvastatin|omeprazole|'
                    r'amoxicillin|ibuprofen|acetaminophen|warfarin|heparin|prednisone|'
                    r'levothyroxine|amlodipine|losartan|gabapentin|fluoxetine|sertraline|'
                    r'ciprofloxacin|metoprolol|hydrochlorothiazide|clopidogrel|pantoprazole|'
                    r'thalidomide|doxycycline|azithromycin|ceftriaxone)\b',
            'dosage': r'\b\d+\s*(?:mg|mcg|g|ml|mL|IU|units?)(?:/(?:day|dose|kg|hr|h))?\b',
            'lab_value': r'\b\d+\.?\d*\s*(?:mg/dL|mmol/L|mEq/L|g/dL|%|ng/mL|pg/mL|U/L|IU/L)\b',
            'vital_sign': r'(?i)\b(?:BP|HR|RR|SpO2|O2 sat|temp)\s*:?\s*\d+[/.]?\d*\b',
            'condition': r'(?i)\b(diabetes|hypertension|epilepsy|seizure|cancer|infection|'
                        r'pneumonia|asthma|COPD|anemia|hepatitis|cirrhosis|stroke|'
                        r'myocardial infarction|heart failure|renal failure)\b',
        }

    def extract_clinical_entities(self, text):
        """Extract all clinical entities from text."""
        entities = {}
        for entity_type, pattern in self.clinical_patterns.items():
            matches = re.findall(pattern, str(text))
            if matches:
                entities[entity_type] = list(set(
                    m if isinstance(m, str) else m[0] for m in matches
                ))
        return entities

    def identify_adversarial_segments(self, text, attention_weights=None, tokens=None):
        """
        Identify adversarial segments in the text using:
        1. Imperative pattern matching
        2. POS tagging for imperative verbs
        3. Attention weights from Module 2 (if available)
        """
        adversarial_spans = []

        # 1. Pattern-based detection
        for pattern in self.imperative_patterns:
            for match in re.finditer(pattern, text):
                adversarial_spans.append({
                    'start': match.start(),
                    'end': match.end(),
                    'text': match.group(),
                    'method': 'pattern',
                })

        # 2. POS-based detection: find imperative verb sentences
        doc = self.nlp(text)
        for sent in doc.sents:
            sent_text = sent.text.strip()
            # Check if sentence starts with imperative verb
            if sent[0].pos_ == 'VERB' and sent[0].tag_ == 'VB':
                # Verify it's not a clinical instruction (e.g., "Administer 500mg")
                clinical_verbs = {'administer', 'monitor', 'assess', 'evaluate',
                                  'prescribe', 'discontinue', 'titrate', 'measure',
                                  'record', 'observe', 'examine', 'diagnose'}
                if sent[0].text.lower() not in clinical_verbs:
                    adversarial_spans.append({
                        'start': sent.start_char,
                        'end': sent.end_char,
                        'text': sent_text,
                        'method': 'pos_imperative',
                    })

        # 3. Attention-based (if Module 2 provides attention weights)
        if attention_weights is not None and tokens is not None:
            # Find tokens with anomalously high attention
            mean_attn = attention_weights.mean()
            std_attn = attention_weights.std()
            threshold = mean_attn + 2 * std_attn

            suspicious_tokens = []
            for i, (token, weight) in enumerate(zip(tokens, attention_weights)):
                if weight > threshold and token not in ['[CLS]', '[SEP]', '[PAD]']:
                    suspicious_tokens.append({
                        'token': token,
                        'index': i,
                        'weight': float(weight),
                    })

        return adversarial_spans

    def sanitize(self, text, attention_weights=None, tokens=None):
        """
        Remove adversarial segments while preserving clinical content.

        Returns:
            dict with:
              - 'original': original text
              - 'sanitized': cleaned text
              - 'removed_segments': list of removed adversarial segments
              - 'entities_before': clinical entities in original
              - 'entities_after': clinical entities after sanitization
              - 'ceps': Clinical Entity Preservation Score
              - 'passed_threshold': bool — True if CEPS >= utility_threshold
        """
        # Extract entities before sanitization
        entities_before = self.extract_clinical_entities(text)

        # Identify adversarial segments
        adv_spans = self.identify_adversarial_segments(text, attention_weights, tokens)

        # Sort by start position (reverse) to remove from end first
        adv_spans.sort(key=lambda x: x['start'], reverse=True)

        # Remove adversarial segments
        sanitized = text
        removed = []
        for span in adv_spans:
            segment = sanitized[span['start']:span['end']]
            # Double-check: don't remove if it contains critical clinical entities
            segment_entities = self.extract_clinical_entities(segment)
            if not any(segment_entities.values()):
                sanitized = sanitized[:span['start']] + sanitized[span['end']:]
                removed.append(span)

        # Clean up whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        sanitized = re.sub(r'\s*\.\s*\.', '.', sanitized)

        # Extract entities after sanitization
        entities_after = self.extract_clinical_entities(sanitized)

        # Compute CEPS
        ceps = self.compute_ceps(entities_before, entities_after)

        return {
            'original': text,
            'sanitized': sanitized,
            'removed_segments': removed,
            'entities_before': entities_before,
            'entities_after': entities_after,
            'ceps': ceps,
            'passed_threshold': ceps >= self.utility_threshold,
        }

    def compute_ceps(self, entities_before, entities_after):
        """
        Clinical Entity Preservation Score (CEPS)

        CEPS = (number of preserved entities) / (total entities before sanitization)

        Entity types: drugs, dosages, lab values, vital signs, conditions
        """
        total_before = 0
        preserved = 0

        for entity_type in self.clinical_patterns.keys():
            before_set = set(entities_before.get(entity_type, []))
            after_set = set(entities_after.get(entity_type, []))
            total_before += len(before_set)
            preserved += len(before_set.intersection(after_set))

        if total_before == 0:
            return 1.0  # No entities to preserve = perfect score

        return preserved / total_before
