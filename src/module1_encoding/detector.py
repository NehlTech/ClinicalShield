"""
ClinicalShield Module 1: Encoding-Aware Ingestion Layer (v3 — fixed Base64 FP)
"""
import base64, re, math
from urllib.parse import unquote
from collections import Counter

class EncodingDetector:
    def __init__(self, confidence_threshold=0.85):
        self.confidence_threshold = confidence_threshold
        self.leet_reverse = {
            '4':'a','@':'a','3':'e','1':'i','!':'i',
            '0':'o','5':'s','$':'s','7':'t','|':'l','9':'g','8':'b'
        }
        self.b64_pattern = re.compile(r'[A-Za-z0-9+/]{20,}={0,2}')

    def _is_real_base64(self, s):
        """Check if string has real Base64 characteristics vs being a long word."""
        has_upper = any(c.isupper() for c in s)
        has_lower = any(c.islower() for c in s)
        has_digit = any(c.isdigit() for c in s)
        has_special = any(c in '+/=' for c in s)

        # Real Base64 has mixed case + usually digits or special chars
        # Single long words are typically all lowercase or title case
        if not has_digit and not has_special:
            # No digits or special chars — check case pattern
            upper_count = sum(1 for c in s if c.isupper())
            if upper_count <= 1:
                return False  # Likely a regular word (all lowercase or Title case)

        # Check character entropy — Base64 has higher entropy than words
        freq = Counter(s)
        n = len(s)
        entropy = -sum((c/n) * math.log2(c/n) for c in freq.values())
        if entropy < 3.5:
            return False  # Low entropy = repetitive = likely a word

        return True

    def detect_base64(self, text):
        matches = self.b64_pattern.findall(text)
        results = []
        for match in matches:
            if not self._is_real_base64(match):
                continue
            try:
                decoded = base64.b64decode(match).decode('utf-8', errors='ignore')
                alpha = sum(c.isalpha() or c.isspace() for c in decoded) / max(len(decoded),1)
                if alpha > 0.4 and len(decoded) > 5:
                    results.append({'encoding':'base64','encoded':match,
                                    'decoded':decoded,'confidence':min(alpha+0.2,1.0)})
            except Exception:
                continue
        return results

    def detect_hex(self, text):
        results = []
        patterns = [re.compile(r'(?:\\\\x[0-9a-fA-F]{2}){3,}'),
                    re.compile(r'(?:\\x[0-9a-fA-F]{2}){3,}')]
        for pat in patterns:
            for match in pat.finditer(text):
                try:
                    raw = match.group()
                    clean = re.sub(r'\\\\x|\\x', '', raw)
                    decoded = bytes.fromhex(clean).decode('utf-8', errors='ignore')
                    if len(decoded) > 3:
                        results.append({'encoding':'hex','encoded':raw[:100],
                                        'decoded':decoded,'confidence':0.95})
                except: continue
        return results

    def detect_unicode(self, text):
        results = []
        patterns = [re.compile(r'(?:\\\\u[0-9a-fA-F]{4}){3,}'),
                    re.compile(r'(?:\\u[0-9a-fA-F]{4}){3,}')]
        for pat in patterns:
            for match in pat.finditer(text):
                try:
                    raw = match.group()
                    normalized = raw.replace('\\\\u', '\\u')
                    decoded = normalized.encode().decode('unicode_escape')
                    if len(decoded) > 3:
                        results.append({'encoding':'unicode','encoded':raw[:100],
                                        'decoded':decoded,'confidence':0.95})
                except: continue
        return results

    def detect_url_encoding(self, text):
        results = []
        pat = re.compile(r'(?:[A-Za-z0-9]*%[0-9a-fA-F]{2}){3,}[A-Za-z0-9]*')
        for match in pat.finditer(text):
            try:
                raw = match.group()
                decoded = unquote(raw)
                if decoded != raw and len(decoded) > 5:
                    results.append({'encoding':'url','encoded':raw[:100],
                                    'decoded':decoded,'confidence':0.90})
            except: continue
        return results

    def detect_leetspeak(self, text):
        leet_chars = set(self.leet_reverse.keys())
        total = sum(1 for c in text if c.isalpha() or c in leet_chars)
        if total == 0: return []
        leet_count = sum(1 for c in text if c in leet_chars)
        ratio = leet_count / total
        if ratio > 0.10 and leet_count >= 3:
            decoded = ''.join(self.leet_reverse.get(c, c) for c in text)
            return [{'encoding':'leetspeak','encoded':text[:200],
                     'decoded':decoded[:200],'confidence':min(ratio+0.5,1.0)}]
        return []

    def detect(self, text):
        if not isinstance(text, str) or len(text) < 10:
            return {'has_encoding':False,'detections':[],'decoded_text':text,'original_text':text}
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
        return {'has_encoding':len(confident)>0,'detections':confident,
                'decoded_text':decoded_text,'original_text':text}

    def detect_and_decode(self, text, max_depth=3):
        current, all_det = text, []
        for _ in range(max_depth):
            result = self.detect(current)
            if not result['has_encoding']: break
            all_det.extend(result['detections'])
            current = result['decoded_text']
        return {'has_encoding':len(all_det)>0,'detections':all_det,
                'decoded_text':current,'original_text':text,
                'encoding_depth':len(set(d['encoding'] for d in all_det))}
