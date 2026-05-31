"""
ClinicalShield Module 1: Pipeline Preprocessor

This is the entry point for the ClinicalShield pipeline.
It takes raw clinical text, runs encoding detection, and outputs
both the original and decoded text for downstream classification.
"""

from src.module1_encoding.detector import EncodingDetector


class ClinicalIngestionLayer:
    """
    Universal Ingestion Layer (UIL) for ClinicalShield.
    Preprocesses clinical text before it enters the detection pipeline.
    """

    def __init__(self, confidence_threshold=0.85, max_decode_depth=3):
        self.detector = EncodingDetector(confidence_threshold=confidence_threshold)
        self.max_depth = max_decode_depth

    def process(self, text):
        """
        Process a single clinical input through the ingestion layer.

        Args:
            text: Raw clinical text input

        Returns:
            dict with:
              - 'original': original text
              - 'processed': decoded text (or original if no encoding found)
              - 'encoding_detected': bool
              - 'encoding_types': list of detected encoding types
              - 'risk_elevated': bool — True if encoding was detected (signals Module 2 to apply stricter thresholds)
        """
        result = self.detector.detect_and_decode(str(text), max_depth=self.max_depth)

        encoding_types = list(set(d['encoding'] for d in result['detections']))

        return {
            'original': text,
            'processed': result['decoded_text'],
            'encoding_detected': result['has_encoding'],
            'encoding_types': encoding_types,
            'risk_elevated': result['has_encoding'],
        }

    def process_batch(self, texts):
        """Process a batch of clinical texts."""
        return [self.process(t) for t in texts]
