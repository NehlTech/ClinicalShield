"""ClinicalShield Module 5 — clinical fact verification against openFDA.

Extracts drug-property claims from a retrieved document and checks them against
the FDA's structured labelling. Where a claim contradicts the label, the
document is flagged regardless of what the upstream detectors concluded.

The boundary is stated rather than discovered: openFDA covers DRUG PROPERTIES,
not whether a PROCEDURE is appropriate. On MPIB, which asserts intervention
appropriateness rather than drug facts, this module extracted no checkable claim
from any sample and detected 0 of 64. That is a property of the knowledge source,
not of the extraction, and it is why the paper treats clinical misinformation of
that kind as an open problem rather than a solved one.
"""

import re
import time
import urllib.parse
import urllib.request
import json

OPENFDA_ENDPOINT = "https://api.fda.gov/drug/label.json"

# Claim shapes the module can check. Each maps to the label section holding the
# authoritative answer.
CLAIM_PATTERNS = {
    "pregnancy_safety": (
        r"\b(?P<drug>[a-z][a-z\- ]{2,30}?)\s+(?:is|are)\s+"
        r"(?:considered\s+)?(?P<polarity>safe|unsafe|contraindicated|appropriate)"
        r"\s+(?:for\s+use\s+)?(?:in|during)\s+pregnan",
        "use_in_specific_populations"),
    "contraindication": (
        r"\b(?:no|without)\s+contraindications?\s+(?:for|to|with)\s+"
        r"(?P<drug>[a-z][a-z\- ]{2,30})",
        "contraindications"),
    "interaction": (
        r"\b(?P<drug>[a-z][a-z\- ]{2,30}?)\s+(?:has|have)\s+no\s+"
        r"(?:known\s+)?(?:clinically\s+)?(?:significant\s+)?"
        r"(?:drug\s+)?interactions?",
        "drug_interactions"),
    "dose_ceiling": (
        r"\b(?P<drug>[a-z][a-z\- ]{2,30}?)\s+may\s+be\s+(?:given|administered|"
        r"dosed)\s+(?:at\s+)?(?P<mult>two|three|four|five|ten|\d+)[\s-]*"
        r"(?:times|fold|x)",
        "dosage_and_administration"),
    "allergy_dismissal": (
        r"\b(?:hypersensitivity|allergy|allergic)\s+(?:reactions?\s+)?(?:to\s+)?"
        r"(?P<drug>[a-z][a-z\- ]{2,30}?)\s+(?:is|are)\s+"
        r"(?:clinically\s+)?(?:insignificant|not\s+clinically|over-?reported)",
        "warnings_and_cautions"),
}
_CLAIM_RE = {k: (re.compile(p, re.I), sec) for k, (p, sec) in CLAIM_PATTERNS.items()}

# Terms whose presence in the authoritative section contradicts the claim.
CONTRADICTION_TERMS = {
    "pregnancy_safety": ["contraindicated", "should not be used", "fetal harm",
                         "teratogenic", "avoid", "discontinue"],
    "contraindication": ["contraindicated", "should not be used", "do not use"],
    "interaction": ["interaction", "coadministration", "concomitant",
                    "increases exposure", "reduces clearance"],
    "dose_ceiling": ["maximum", "do not exceed", "should not exceed"],
    "allergy_dismissal": ["anaphylaxis", "hypersensitivity", "serious",
                          "discontinue immediately"],
}


class LabelCache:
    """openFDA lookups, cached. The API is public, so a deployed system does not
    need its own copy of the labelling."""

    def __init__(self, api_key=None, timeout=30):
        self.api_key = api_key
        self.timeout = timeout
        self._cache = {}

    def fetch(self, drug, retries=3):
        drug = drug.strip().lower()
        if drug in self._cache:
            return self._cache[drug]
        params = {"search": 'openfda.generic_name:"%s"' % drug, "limit": 1}
        if self.api_key:
            params["api_key"] = self.api_key
        url = OPENFDA_ENDPOINT + "?" + urllib.parse.urlencode(params)
        label = None
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(url, timeout=self.timeout) as r:
                    res = json.loads(r.read().decode()).get("results", [])
                    label = res[0] if res else None
                break
            except Exception:
                if attempt == retries - 1:
                    label = None
                else:
                    time.sleep(2 ** attempt)
        self._cache[drug] = label
        return label

    def section(self, drug, field):
        label = self.fetch(drug)
        if not label:
            return None
        val = label.get(field)
        if not val:
            return None
        return " ".join(val) if isinstance(val, list) else str(val)


def extract_claims(text):
    """Return the checkable drug-property claims in the text.

    Pattern-based, and therefore limited to phrasings anticipated in advance —
    the reason this module extracted nothing from MPIB, whose payloads assert
    intervention appropriateness rather than drug facts.
    """
    out = []
    for kind, (pat, section) in _CLAIM_RE.items():
        for m in pat.finditer(text):
            d = m.groupdict()
            out.append({"kind": kind, "drug": (d.get("drug") or "").strip().lower(),
                        "section": section, "span": list(m.span()),
                        "text": m.group()})
    return out


def verify(text, cache=None, api_key=None):
    """Check every extracted claim against the FDA label.

    Returns (flagged, findings). ``flagged`` is True where any claim contradicts
    its authoritative section. Claims whose drug or section is unavailable are
    reported as ``unverifiable`` rather than as passing: silently treating an
    unchecked claim as safe is the failure mode this module exists to avoid.
    """
    cache = cache or LabelCache(api_key=api_key)
    findings = []
    flagged = False
    for claim in extract_claims(text):
        sec = cache.section(claim["drug"], claim["section"]) if claim["drug"] else None
        if not sec:
            findings.append({**claim, "verdict": "unverifiable",
                             "reason": "no label section available"})
            continue
        low = sec.lower()
        hits = [t for t in CONTRADICTION_TERMS.get(claim["kind"], []) if t in low]
        if hits:
            flagged = True
            findings.append({**claim, "verdict": "contradicted",
                             "evidence_terms": hits[:4]})
        else:
            findings.append({**claim, "verdict": "consistent"})
    return flagged, findings
