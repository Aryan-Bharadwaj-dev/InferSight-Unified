"""
Text Steganography Detector
Detects steganography in text files
"""

import os
import re
import numpy as np

class TextStegoDetector:
    def __init__(self):
        self.detection_methods = [
            self._detect_whitespace_steganography,
            self._detect_character_anomalies,
            self._analyze_word_patterns,
            self._detect_unusual_encoding
        ]

    def detect(self, text_path):
        """Run all detection methods on a text file"""
        detections = []

        try:
            with open(text_path, 'r', encoding='utf-8', errors='ignore') as text_file:
                text = text_file.read()

                # Run each detection method
                for method in self.detection_methods:
                    try:
                        result = method(text)
                        if result:
                            detections.append(result)
                    except Exception as e:
                        print(f"Error in detection method {method.__name__}: {e}")

        except Exception as e:
            print(f"Error loading text {text_path}: {e}")

        return detections

    def _detect_whitespace_steganography(self, text):
        """Detect hidden messages using whitespace characters"""
        try:
            spaces = text.count(' ')
            tabs = text.count('\t')
            newlines = text.count('\n')

            # Check for unusual patterns in whitespace distribution
            if abs(spaces - tabs) + abs(spaces - newlines) > 100:
                return {
                    'method': 'Whitespace Steganography',
                    'confidence': 80,
                    'description': 'Unusual whitespace distribution detected'
                }

        except Exception:
            pass

        return None

    def _detect_character_anomalies(self, text):
        """Detect anomalies in character distribution"""
        try:
            letters = [c for c in text if c.isalpha()]
            unique_chars = len(set(letters))
            char_entropy = self._calculate_entropy(letters)

            # Low entropy might indicate hidden data
            if unique_chars < 10 and char_entropy < 3:
                confidence = 70
                return {
                    'method': 'Character Anomalies',
                    'confidence': confidence,
                    'description': f'Low character diversity: unique_chars={unique_chars}, entropy={char_entropy:.2f}'
                }

        except Exception:
            pass

        return None

    def _analyze_word_patterns(self, text):
        """Analyze word patterns for anomalies"""
        try:
            words = re.findall(r"\b\w+\b", text)
            word_counts = dict()

            for word in words:
                if word not in word_counts:
                    word_counts[word] = 0
                word_counts[word] += 1

            most_common = max(word_counts.items(), key=lambda x: x[1])
            if most_common[1] > 100:  # Overuse of a single word can be suspicious
                confidence = 65
                return {
                    'method': 'Word Pattern Analysis',
                    'confidence': confidence,
                    'description': 'Overuse of a single word: "{}" appears {} times'.format(most_common[0], most_common[1])
                }

        except Exception:
            pass

        return None

    def _detect_unusual_encoding(self, text):
        """Detect unusual text encoding patterns"""
        try:
            byte_data = text.encode('utf-8', errors='ignore')

            # Check for non-standard encoding markers
            if b'\xfe\xff' in byte_data or b'\xff\xfe' in byte_data:
                return {
                    'method': 'Unusual Encoding',
                    'confidence': 75,
                    'description': 'Potentially suspicious encoding markers detected'
                }

        except Exception:
            pass

        return None

    def _calculate_entropy(self, data):
        """Calculate the entropy of the given data"""
        unique, counts = np.unique(list(data), return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -sum(probabilities * np.log2(probabilities))
        return entropy

