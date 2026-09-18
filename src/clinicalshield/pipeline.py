"""ClinicalShield — the assembled pipeline.

Wires the modules in the order the paper describes, with the operating points
selected on validation data in the accompanying notebooks.

    ingest -> detect -> verify intent -> sanitise -> fact-check -> policy

The policy step is the part v1 left implicit. Sanitisation removes roughly 70%
of payloads, so passing every sanitised document onward fails open. Three
policies are provided with their measured residual risk; ``verify_or_block`` is
the default because it fails closed, which is the appropriate posture for a
system informing prescribing.
"""

from . import module1, module4, module5

POLICIES = ("sanitise_and_pass", "verify_or_block", "detect_and_block")


class ClinicalShield:
    def __init__(self, detector, drug_vocabulary=None, tau_detect=0.50,
                 tau_intent=0.90, tau_verify=0.50, policy="verify_or_block",
                 fact_check=False, openfda_key=None):
        if policy not in POLICIES:
            raise ValueError("policy must be one of %s" % (POLICIES,))
        self.detector = detector
        self.tau_detect = tau_detect
        self.tau_intent = tau_intent
        self.tau_verify = tau_verify
        self.policy = policy
        self.fact_check = fact_check
        self._cache = module5.LabelCache(api_key=openfda_key) if fact_check else None
        if drug_vocabulary:
            module4.set_drug_vocabulary(drug_vocabulary)

    def process(self, document):
        """Returns the decision for one retrieved document.

        ``release`` is the text to forward, or None where the document is
        withheld.
        """
        ing = module1.module1(document)
        decoded = ing["decoded"]

        score = float(self.detector.scores([decoded])[0])
        detected = score >= self.tau_detect

        sent = self.detector.sentence_scores(decoded, module4.segments)
        clean, removed, protected = module4.sanitise(
            decoded, sent, tau_intent=self.tau_intent)

        ceps = module4.ceps_v1(decoded, clean)[0]

        fact_flagged, findings = (False, [])
        if self.fact_check:
            fact_flagged, findings = module5.verify(clean, cache=self._cache)

        residual = float(self.detector.scores([clean or " "])[0])
        still_flagged = residual >= self.tau_verify

        if self.policy == "detect_and_block":
            release = None if (detected or fact_flagged) else decoded
        elif self.policy == "verify_or_block":
            withhold = (detected and still_flagged) or fact_flagged
            release = None if withhold else clean
        else:
            release = clean

        return {
            "release": release,
            "withheld": release is None,
            "ingestion": {"encodings": ing["formats"],
                          "canonicalisation_drift": ing.get("canon_drift")},
            "detection": {"score": score, "detected": detected},
            "sanitisation": {"removed": len(removed), "protected": len(protected),
                             "reasons": [r["reason"] for r in removed]},
            "ceps": ceps,
            "needs_human_review": module4.needs_human_review(ceps),
            "verification": {"residual_score": residual,
                             "still_flagged": still_flagged},
            "fact_check": {"flagged": fact_flagged, "findings": findings},
            "policy": self.policy,
        }
