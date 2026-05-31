"""
ClinicalShield Module 2: PubMedBERT-based IPI Detector
with attention extraction for token-level localization.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class ClinicalShieldDetector(nn.Module):
    def __init__(self, model_name, num_labels=2, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, return_attention=False):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        if return_attention:
            last_attn = outputs.attentions[-1]
            avg_attn = last_attn.mean(dim=1)
            cls_attn = avg_attn[:, 0, :]
            return logits, cls_attn
        return logits


def load_detector(model_dir, device='cuda'):
    """Load a trained ClinicalShield detector from disk."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = ClinicalShieldDetector(
        model_name=model_dir, num_labels=2, dropout=0.1
    )
    state_dict = torch.load(
        model_dir.rstrip('/') + '/../clinicalshield_full.pt',
        map_location=device
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, tokenizer


def predict(model, tokenizer, text, device='cuda', threshold=0.5):
    """Predict if a text is adversarial."""
    encoding = tokenizer(
        text, max_length=512, truncation=True,
        padding='max_length', return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    masks     = encoding['attention_mask'].to(device)

    with torch.no_grad():
        logits, cls_attn = model(input_ids, masks, return_attention=True)
        probs = torch.softmax(logits, dim=1)

    attack_prob = probs[0, 1].item()
    return {
        'is_attack': attack_prob > threshold,
        'attack_probability': attack_prob,
        'attention_weights': cls_attn[0].cpu().numpy(),
        'tokens': tokenizer.convert_ids_to_tokens(input_ids[0].cpu()),
    }
