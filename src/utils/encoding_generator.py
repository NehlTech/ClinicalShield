"""
Encoding Attack Generator — ClinicalShield Module 1 Support
Builds obfuscated adversarial samples for robustness testing.
"""
import base64, random
from urllib.parse import quote

LEET = {'a':['4','@'],'e':['3'],'i':['1','!'],'o':['0'],
        's':['5','$'],'t':['7'],'l':['1','|'],'g':['9'],'b':['8']}

def to_base64(t):  return base64.b64encode(t.encode()).decode()
def to_hex(t):     return ''.join(f'\\x{ord(c):02x}' for c in t)
def to_unicode(t): return ''.join(f'\\u{ord(c):04x}' for c in t)
def to_leet(t):    return ''.join(random.choice(LEET[c]) if c in LEET and random.random()>0.3 else c for c in t.lower())
def to_url(t):     return quote(t)

ENCODERS = {'base64':to_base64,'hex':to_hex,'unicode':to_unicode,'leetspeak':to_leet,'url':to_url}

def embed_payload(payload, clinical):
    parts = clinical.split('. ')
    mid = max(1, len(parts)//2)
    parts.insert(mid, payload)
    return '. '.join(parts)

def build_encoded_dataset(attacks, clinical_texts, n_per_type=500, seed=42):
    random.seed(seed)
    samples = []
    for name, fn in ENCODERS.items():
        for _ in range(n_per_type):
            try:
                encoded = fn(random.choice(attacks))
                context = random.choice(clinical_texts)
                samples.append({'text':embed_payload(encoded,context),'label':1,'encoding_type':name,'source':f'encoded_{name}'})
            except: continue
        for _ in range(n_per_type//5):
            try:
                text = random.choice(clinical_texts)
                samples.append({'text':text+' '+fn(text[:80]),'label':0,'encoding_type':name,'source':f'benign_encoded_{name}'})
            except: continue
    return samples
