"""
ClinicalShield Module 1: Encoding-Aware Ingestion Layer (v2 — fixed)

Detects and decodes obfuscated payloads in clinical text.
Supports: Base64, Hex, Unicode, Leetspeak, URL encoding, Nested.
"""

import base64
import re
import math
from urllib.parse import unquote
from collections import Counter


class EncodingDetector:

    def __init__(self, confidence_threshold=0.85):
        self.confidence_threshold = confidence_threshold
        self.leet_reverse = {
            '4':'a','@':'a','3':'e','1':'i','!':'i',
            '0':'o','5':'s','$':'s','7':'t','|':'l','9':'g','8':'b'
        }
        # Base64: 20+ valid chars with optional padding
        self.b64_pattern = re.compile(r'[A-Za-z0-9+/]{20,}={0,2}')

    def _entropy(self, text):
        if not text:
            return 0.0
        freq = Counter(text)
        n = len(text)
        return -sum((c/n) * math.log2(c/n) for c in freq.values())

    # ── Base64 ─────────────────────────────────────────────
    def detect_base64(self, text):
        matches = self.b64_pattern.findall(text)
        results = []
        for match in matches:
            try:
                decoded = base64.b64decode(match).decode('utf-8', errors='ignore')
                alpha = sum(c.isalpha() or c.isspace() for c in decoded) / max(len(decoded),1)
                if alpha > 0.4 and len(decoded) > 5:
                    results.append({
                        'encoding':'base64','encoded':match,
                        'decoded':decoded,'confidence':min(alpha+0.2,1.0)
                    })
            except Exception:
                continue
        return results

    # ── Hex ─────────────────────────────────────────────────
    def detect_hex(self, text):
        results = []
        # Match both \\xNN and \xNN patterns (escaped and literal)
        patterns = [
            re.compile(r'(?:\\\\x[0-9a-fA-F]{2}){3,}'),  # \\xNN (escaped backslash)
            re.compile(r'(?:\\x[0-9a-fA-F]{2}){3,}'),     # \xNN (single backslash)
        ]
        for pat in patterns:
            for match in pat.finditer(text):
                try:
                    raw = match.group()
                    # Normalize to get just the hex digits
                    clean = re.sub(r'\\\\x|\\x', '', raw)
                    decoded = bytes.fromhex(clean).decode('utf-8', errors='ignore')
                    if len(decoded) > 3:
                        results.append({
                            'encoding':'hex','encoded':raw[:100],
                            'decoded':decoded,'confidence':0.95
                        })
                except Exception:
                    continue
        return results

    # ── Unicode ─────────────────────────────────────────────
    def detect_unicode(self, text):
        results = []
        patterns = [
            re.compile(r'(?:\\\\u[0-9a-fA-F]{4}){3,}'),  # \\uNNNN
            re.compile(r'(?:\\u[0-9a-fA-F]{4}){3,}'),     # \uNNNN
        ]
        for pat in patterns:
            for match in pat.finditer(text):
                try:
                    raw = match.group()
                    # Try to decode unicode escapes
                    normalized = raw.replace('\\\\u', '\\u')
                    decoded = normalized.encode().decode('unicode_escape')
                    if len(decoded) > 3:
                        results.append({
                            'encoding':'unicode','encoded':raw[:100],
                            'decoded':decoded,'confidence':0.95
                        })
                except Exception:
                    continue
        return results

    # ── URL Encoding ────────────────────────────────────────
    def detect_url_encoding(self, text):
        results = []
        # Match any stretch containing %XX patterns (at least 3)
        pat = re.compile(r'(?:[A-Za-z0-9]*%[0-9a-fA-F]{2}){3,}[A-Za-z0-9]*')
        for match in pat.finditer(text):
            try:
                raw = match.group()
                decoded = unquote(raw)
                if decoded != raw and len(decoded) > 5:
                    results.append({
                        'encoding':'url','encoded':raw[:100],
                        'decoded':decoded,'confidence':0.90
                    })
            except Exception:
                continue
        return results

    # ── Leetspeak ───────────────────────────────────────────
    def detect_leetspeak(self, text):
        leet_chars = set(self.leet_reverse.keys())
        # Count leet substitution characters in the text
        total_alpha_like = sum(1 for c in text if c.isalpha() or c in leet_chars)
        if total_alpha_like == 0:
            return []

        leet_count = sum(1 for c in text if c in leet_chars)
        leet_ratio = leet_count / total_alpha_like

        # If more than 10% of alpha-like characters are leet substitutions
        if leet_ratio > 0.10 and leet_count >= 3:
            decoded = ''.join(self.leet_reverse.get(c, c) for c in text)
            return [{
                'encoding':'leetspeak','encoded':text[:200],
                'decoded':decoded[:200],
                'confidence': min(leet_ratio + 0.5, 1.0)
            }]
        return []

    # ── Main Detect ─────────────────────────────────────────
    def detect(self, text):
        if not isinstance(text, str) or len(text) < 10:
            return {
                'has_encoding':False,'detections':[],
                'decoded_text':text,'original_text':text
            }

        all_det = []
        all_det.extend(self.detect_base64(text))
        all_det.extend(self.detect_hex(text))
        all_det.extend(self.detect_unicode(text))
        all_det.extend(self.detect_url_encoding(text))
        all_det.extend(self.detect_leetspeak(text))

        confident = [d for d in all_det if d['confidence'] >= self.confidence_threshold]

        decoded_text = text
        for det in confident:
            if det['encoding'] != 'leetspeak':
                decoded_text = decoded_text.replace(det['encoded'], det['decoded'])

        return {
            'has_encoding': len(confident) > 0,
            'detections': confident,
            'decoded_text': decoded_text,
            'original_text': text
        }

    def detect_and_decode(self, text, max_depth=3):
        current = text
        all_det = []
        for _ in range(max_depth):
            result = self.detect(current)
            if not result['has_encoding']:
                break
            all_det.extend(result['detections'])
            current = result['decoded_text']
        return {
            'has_encoding': len(all_det) > 0,
            'detections': all_det,
            'decoded_text': current,
            'original_text': text,
            'encoding_depth': len(set(d['encoding'] for d in all_det))
        }
