"""
ClinicalShield: Full End-to-End Pipeline

Module 1 (Encoding) → Module 2 (Detection) → Module 3 (Intent) → Module 4 (Sanitization)
"""
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn


class ClinicalShieldDetector(nn.Module):
    def __init__(self, model_name, num_labels=2, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    def forward(self, input_ids, attention_mask, return_attention=False):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(outputs.last_hidden_state[:, 0, :])
        logits = self.classifier(cls)
        if return_attention:
            attn = outputs.attentions[-1].mean(dim=1)[:, 0, :]
            return logits, attn
        return logits


class ClinicalShieldPipeline:
    def __init__(self, model_dir='models/clinicalshield_detector',
                 weights_path='models/best_model.pt',
                 device='cuda', detection_threshold=0.5):
        self.device = device
        self.threshold = detection_threshold

        # Module 1: Encoding Detector
        from src.module1_encoding.detector import EncodingDetector
        self.encoder = EncodingDetector(confidence_threshold=0.7)

        # Module 2: PubMedBERT Detector
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.detector = ClinicalShieldDetector(model_dir)
        self.detector.load_state_dict(torch.load(weights_path, map_location=device))
        self.detector.to(device)
        self.detector.eval()

        # Module 4: Sanitizer
        from src.module4_sanitization.sanitizer import ClinicalSanitizer
        self.sanitizer = ClinicalSanitizer()

    def process(self, text):
        """
        Full pipeline processing.
        Returns dict with detection result, sanitized text, and all module outputs.
        """
        result = {
            'original_text': text,
            'is_attack': False,
            'attack_probability': 0.0,
            'encoding_detected': False,
            'sanitized_text': text,
            'ceps': 1.0,
        }

        # Step 1: Encoding detection + decoding
        enc_result = self.encoder.detect_and_decode(str(text))
        result['encoding_detected'] = enc_result['has_encoding']
        processed_text = enc_result['decoded_text']

        # Step 2: PubMedBERT classification
        encoding = self.tokenizer(
            processed_text, max_length=512, truncation=True,
            padding='max_length', return_tensors='pt'
        )
        with torch.no_grad():
            ids = encoding['input_ids'].to(self.device)
            masks = encoding['attention_mask'].to(self.device)
            logits, attn = self.detector(ids, masks, return_attention=True)
            probs = torch.softmax(logits, dim=1)

        attack_prob = probs[0, 1].item()
        result['attack_probability'] = attack_prob
        result['is_attack'] = attack_prob > self.threshold

        # Step 3: If attack detected, sanitize
        if result['is_attack']:
            tokens = self.tokenizer.convert_ids_to_tokens(ids[0].cpu())
            san_result = self.sanitizer.sanitize(
                processed_text,
                attention_weights=attn[0].cpu().numpy(),
                tokens=tokens
            )
            result['sanitized_text'] = san_result['sanitized']
            result['ceps'] = san_result['ceps']
            result['removed_segments'] = san_result['removed_segments']

        # Elevated risk if encoding was detected
        if result['encoding_detected'] and not result['is_attack']:
            result['risk_note'] = 'Encoding detected but classifier passed — review recommended'

        return result
