"""ClinicalShield Module 4 — utility-preserving clinical sanitisation.

Removes adversarial segments from a retrieved document while retaining the
clinical content a clinician needs, and scores the result with the Clinical
Entity Preservation Score.

CEPS v2 supersedes v1. v1 measured entity preservation alone, so a sanitiser
that removed nothing scored 1.0 — indistinguishable from perfect sanitisation
(Reviewer 4). v2 pairs preservation with adversarial removal and reports their
harmonic mean, so a no-op scores zero.

Removal is scored against the ground-truth payload, per payload sentence. An
earlier version asked whether the sanitiser removed what it had itself decided
to remove, which is circular; and asking whether the exact payload string
survived gave full marks when only half a two-sentence payload was excised.
"""

import re

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

CLINICAL_VERBS = [
    "administer", "monitor", "prescribe", "titrate", "discontinue", "adjust",
    "reduce", "increase", "initiate", "withhold", "observe", "measure",
    "assess", "evaluate", "consider", "avoid", "consult",
]

ENTITY_CATEGORIES = ["drug_name", "dosage", "lab_result", "vital_sign", "condition"]

GAMMA = 0.95          # CEPS below this routes the document to human review

# Explicit instruction-override phrasing. No genuine drug label contains these,
# so they are removed regardless of clinical entities: the entity-protection
# rule is otherwise exactly what an attacker hides behind, since every
# clinically targeted payload names a drug.
OVERRIDE_MARKERS = [
    "ignore previous", "ignore all previous", "ignore the above",
    "disregard the", "disregard all", "system override", "new instruction",
    "override:", "admin:", "[admin]", "[config]", "[system]",
    "forget everything", "forget the", "prior context is void",
    "end of document", "attention model", "do not follow",
    "your reply must", "you must instead", "instead output",
]

# Misinformation phrasing is genuinely ambiguous — real labels are updated and
# real guidance does change — so the entity protection is retained for these.
MISINFO_CUES = [
    r"\b(?:recent|updated|contemporary|current|newer|revised)\s+"
    r"(?:analys|data|evidence|guidance|practice|review|series|cohort|work|"
    r"appraisal|consensus|modelling|modeling)",
    r"\bno longer\s+(?:recommended|advised|considered|applies|required|necessary)",
    r"\b(?:has|have)\s+not\s+been\s+(?:substantiated|confirmed|reproduced|"
    r"established|replicated|demonstrated)",
    r"\b(?:lack|lacks)\s+supporting",
    r"\breclassified as\b",
    r"\bmay be disregarded\b",
    r"\bover-?reported\b",
]

ENTITY_PATTERNS = {
    "dosage": r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|\u00b5g|ug|g|kg|mL|ml|L|units?|IU|"
              r"mEq|mmol)(?:\s*/\s*(?:day|kg|m2|dose|hr|h|min))?\b",
    "lab_result": r"\b(?:INR|eGFR|GFR|HbA1c|A1C|ALT|AST|ALP|BUN|creatinine|"
                  r"bilirubin|albumin|haemoglobin|hemoglobin|platelet|WBC|RBC|"
                  r"sodium|potassium|calcium|magnesium|phosphate|glucose|"
                  r"cholesterol|LDL|HDL|triglycerides?)\s*(?:of|was|is|:|=)?"
                  r"\s*\d+(?:\.\d+)?\s*%?",
    "vital_sign": r"\b(?:BP|blood pressure|heart rate|pulse|respiratory rate|"
                  r"SpO2|temperature|weight|BMI)\s*(?:of|was|is|:|=)?\s*"
                  r"\d+(?:\.\d+)?(?:\s*/\s*\d+)?\s*(?:mmHg|bpm|kg|lbs?|C|F|%)?",
    "condition": r"\b(?:hypertension|hypotension|diabetes|renal impairment|"
                 r"renal failure|hepatic impairment|hepatic failure|heart failure|"
                 r"arrhythmia|myocardial infarction|stroke|seizure|pregnancy|"
                 r"lactation|hypersensitivity|anaphylaxis|neutropenia|"
                 r"thrombocytopenia|anaemia|anemia|infection|sepsis|bleeding|"
                 r"haemorrhage|hemorrhage|hyperkalaemia|hyperkalemia|"
                 r"hypoglycaemia|hypoglycemia)\b",
}

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|\n+")
CLINICAL_VERB_RE = re.compile(r"\b(" + "|".join(CLINICAL_VERBS) + r")\b", re.I)
OVERRIDE_PATTERNS = [re.compile(re.escape(m), re.I) for m in OVERRIDE_MARKERS]
MISINFO_PATTERNS = [re.compile(p, re.I) for p in MISINFO_CUES]
_ENTITY_RE = {k: re.compile(v, re.I) for k, v in ENTITY_PATTERNS.items()}

DRUG_RE = None        # set by set_drug_vocabulary()


def set_drug_vocabulary(drug_names):
    """Install the drug vocabulary from the deployed corpus.

    Drug names come from the retrieval corpus rather than a hard-coded list, so
    the extractor and the corpus cannot disagree about what counts as a drug.
    """
    global DRUG_RE
    names = sorted({str(d).lower() for d in drug_names if d}, key=len, reverse=True)
    if not names:
        DRUG_RE = None
        return
    DRUG_RE = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b",
                         re.I)


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation and entity extraction
# ─────────────────────────────────────────────────────────────────────────────

def segments(text):
    """Split into sentences, returning (start, end, text) triples.

    Offsets are kept so removal is exact and inspectable rather than inferred
    from a length difference.
    """
    out, start = [], 0
    for m in SENT_SPLIT.finditer(text):
        if m.start() > start:
            out.append((start, m.start(), text[start:m.start()]))
        start = m.end()
    if start < len(text):
        out.append((start, len(text), text[start:]))
    return out or [(0, len(text), text)]


def extract_entities(text):
    """Return {category: set of surface forms}.

    Sets rather than counts, because CEPS asks whether an entity survived, not
    how many times it occurred.
    """
    out = {c: set() for c in ENTITY_CATEGORIES}
    if DRUG_RE is not None:
        for m in DRUG_RE.finditer(text):
            out["drug_name"].add(m.group().lower())
    for cat, pat in _ENTITY_RE.items():
        for m in pat.finditer(text):
            out[cat].add(re.sub(r"\s+", " ", m.group().strip().lower()))
    return out


def entity_count(entities):
    return sum(len(v) for v in entities.values())


# ─────────────────────────────────────────────────────────────────────────────
# Sanitisation
# ─────────────────────────────────────────────────────────────────────────────

def has_override_marker(segment):
    return any(p.search(segment) for p in OVERRIDE_PATTERNS)


def is_adversarial_segment(segment):
    """Returns (flagged, reason)."""
    for p in OVERRIDE_PATTERNS:
        if p.search(segment):
            return True, "override_marker"
    for p in MISINFO_PATTERNS:
        if p.search(segment):
            return True, "misinformation_cue"
    return False, None


def sanitise(text, sent_scores=None, tau_intent=None,
             tau_adjacent=None, radius=1):
    """Remove adversarial segments while keeping clinical content.

    Grounds for removal, in descending confidence:

    1. explicit override marker — removed unconditionally
    2. classifier judgement above ``tau_intent`` — Module 2 scoring the sentence
       on its own. This is what lifted payload removal from 20% to 70%: the
       entity-protection rule alone shields any attack that names a drug, and
       every clinically targeted attack does.
    3. misinformation cue AND no clinical entity — the v1 rule, retained for the
       ambiguous case

    A second pass optionally re-judges neighbours of a removed segment at a
    lower bar (``tau_adjacent``), for payloads spanning two sentences. Measured
    on this corpus it produced no gain: 84% of payloads occupy a single
    sentence. Retained as an option, disabled by default.

    ``sent_scores`` maps segment text to P(adversarial) from Module 2.

    Returns (clean_text, removed, protected).
    """
    segs = [seg for _s0, _s1, seg in segments(text)]
    status = [None] * len(segs)

    for i, seg in enumerate(segs):
        if has_override_marker(seg):
            status[i] = {"reason": "override_marker"}
            continue
        if sent_scores is not None and tau_intent is not None:
            p = sent_scores.get(seg)
            if p is not None and p >= tau_intent:
                status[i] = {"reason": "intent_classifier", "p_attack": float(p)}
                continue
        adv, why = is_adversarial_segment(seg)
        if adv:
            ents = entity_count(extract_entities(seg))
            if ents == 0 and not CLINICAL_VERB_RE.search(seg):
                status[i] = {"reason": why}

    if tau_adjacent is not None and sent_scores is not None:
        for i in [k for k, s in enumerate(status) if s]:
            lo, hi = max(0, i - radius), min(len(segs), i + radius + 1)
            for j in range(lo, hi):
                if status[j] or j == i:
                    continue
                p = sent_scores.get(segs[j])
                if p is None or p < tau_adjacent:
                    continue
                # never remove clinical content by adjacency alone
                if entity_count(extract_entities(segs[j])) > 0:
                    continue
                status[j] = {"reason": "adjacent_to_removal", "p_attack": float(p)}

    kept, removed, protected = [], [], []
    for seg, st in zip(segs, status):
        if st:
            removed.append({"segment": seg, **st})
            continue
        adv, why = is_adversarial_segment(seg)
        if adv:
            protected.append({"segment": seg, "reason": why,
                              "entities": entity_count(extract_entities(seg))})
        kept.append(seg)
    return " ".join(kept).strip(), removed, protected


# ─────────────────────────────────────────────────────────────────────────────
# Clinical Entity Preservation Score
# ─────────────────────────────────────────────────────────────────────────────

def ceps_v1(before_text, after_text):
    """Entity preservation alone. Returns (score, n_entities_before).

    Retained so v1 and v2 can be reported side by side. A no-op sanitiser scores
    1.0 here, which is the flaw v2 exists to correct.
    """
    eb = extract_entities(before_text)
    ea = extract_entities(after_text)
    total = entity_count(eb)
    if total == 0:
        return 1.0, 0
    kept = sum(len(eb[c] & ea[c]) for c in eb)
    return kept / total, total


def ceps_v2(before_text, after_text, payload=None):
    """Preservation, removal, and their harmonic mean.

    preservation  fraction of clinical entities surviving sanitisation
    removal       fraction of payload SENTENCES excised
    combined      harmonic mean; zero if either component is zero

    A no-op scores preservation 1.0, removal 0.0, combined 0.0. Deleting the
    document entirely scores 0.0, 1.0, 0.0. Only removing the attack while
    keeping the clinical content scores well.

    ``payload`` is the ground-truth injected text. Scoring removal against
    segments the sanitiser itself flagged would be circular. Scoring against the
    whole payload string gave full marks when only the first of two sentences
    was excised, leaving the dangerous clause in place.
    """
    preservation, n_entities = ceps_v1(before_text, after_text)

    if payload is None:
        removal, n_adv = 1.0, 0            # benign document: nothing to remove
    else:
        norm = lambda t: re.sub(r"\s+", " ", str(t)).strip().lower()
        after_n = norm(after_text)
        pay_sents = [seg for _s0, _s1, seg in segments(payload)
                     if len(seg.split()) >= 3] or [payload]
        n_adv = len(pay_sents)
        gone = sum(1 for seg in pay_sents if norm(seg) not in after_n)
        removal = gone / n_adv

    if preservation + removal == 0:
        combined = 0.0
    else:
        combined = 2 * preservation * removal / (preservation + removal)

    return {"preservation": preservation, "removal": removal,
            "combined": combined, "n_entities": n_entities,
            "n_adversarial": n_adv}


def needs_human_review(ceps_score, gamma=GAMMA):
    """CEPS below gamma routes the document to a clinician rather than the LLM."""
    return ceps_score < gamma
