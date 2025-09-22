"""
Audio Steganography Detector
Detects steganography in audio files using various techniques
"""

import numpy as np
import wave
import os

class AudioStegoDetector:
    def __init__(self):
        self.detection_methods = [
            self._detect_lsb_steganography,
            self._detect_spectral_anomalies,
            self._analyze_noise_patterns,
            self._statistical_analysis
        ]

    def detect(self, audio_path):
        """Run all detection methods on an audio file"""
        detections = []

        try:
            with wave.open(audio_path, 'rb') as wav_file:
                params = wav_file.getparams()
                frames = wav_file.readframes(params.nframes)
                data = np.frombuffer(frames, dtype=np.int16)

                # Run each detection method
                for method in self.detection_methods:
                    try:
                        result = method(data, params)
                        if result:
                            detections.append(result)
                    except Exception as e:
                        print(f"Error in detection method {method.__name__}: {e}")

        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")

        return detections

    def _detect_lsb_steganography(self, data, params):
        """Detect least significant bit (LSB) steganography"""
        try:
            lsb = data & 1

            # Calculate the entropy of the LSBs
            def calculate_entropy(data):
                unique, counts = np.unique(data, return_counts=True)
                probabilities = counts / counts.sum()
                entropy = -sum(probabilities * np.log2(probabilities))
                return entropy

            entropy = calculate_entropy(lsb)

            # High entropy could indicate hidden data
            if entropy > 0.9:
                confidence = min(100, (entropy - 0.9) * 1000)
                return {
                    'method': 'LSB Steganography',
                    'confidence': confidence,
                    'description': f'High entropy in LSBs: {entropy:.3f}'
                }

        except Exception:
            pass

        return None

    def _detect_spectral_anomalies(self, data, params):
        """Detect anomalies in the spectral domain"""
        try:
            # Perform FFT
            fft_data = np.fft.fft(data)
            magnitude = np.abs(fft_data)

            # Check for unusual frequency magnitudes
            if np.max(magnitude) > 1e6:
                confidence = 75
                return {
                    'method': 'Spectral Anomaly',
                    'confidence': confidence,
                    'description': 'Unusual frequency magnitudes detected'
                }

        except Exception:
            pass

        return None

    def _analyze_noise_patterns(self, data, params):
        """Analyze noise patterns for irregularities"""
        try:
            noise = data - np.mean(data)
            variance = np.var(noise)

            # High variance could indicate unnatural noise patterns
            if variance > 1e4:
                confidence = min(80, (variance - 1e4) / 1000)
                return {
                    'method': 'Noise Pattern Analysis',
                    'confidence': confidence,
                    'description': f'High variance in noise: {variance:.2f}'
                }

        except Exception:
            pass

        return None

    def _statistical_analysis(self, data, params):
        """Perform basic statistical analysis"""
        try:
            mean = np.mean(data)
            std_dev = np.std(data)

            # Unusual mean or standard deviation might indicate steganography
            if abs(mean) > 1000 or std_dev > 10000:
                confidence = 65
                return {
                    'method': 'Statistical Analysis',
                    'confidence': confidence,
                    'description': f'Unusual mean/std deviation: mean={mean:.2f}, std={std_dev:.2f}'
                }

        except Exception:
            pass

        return None

