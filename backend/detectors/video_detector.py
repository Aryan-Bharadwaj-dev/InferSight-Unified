"""
Video Steganography Detector
Detects steganography in video files
"""

import cv2
import numpy as np
import os
from pathlib import Path

class VideoStegoDetector:
    def __init__(self):
        self.detection_methods = [
            self._detect_frame_anomalies,
            self._detect_motion_vector_anomalies,
            self._detect_compression_artifacts,
            self._analyze_file_structure
        ]
    
    def detect(self, video_path):
        """Run all detection methods on a video file"""
        detections = []
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                print(f"Error opening video file: {video_path}")
                return detections
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            video_info = {
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'path': video_path
            }
            
            # Run each detection method
            for method in self.detection_methods:
                try:
                    result = method(cap, video_info)
                    if result:
                        detections.append(result)
                except Exception as e:
                    print(f"Error in detection method {method.__name__}: {e}")
            
            cap.release()
            
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            
        return detections
    
    def _detect_frame_anomalies(self, cap, video_info):
        """Detect anomalies in individual frames"""
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            frame_differences = []
            prev_frame = None
            samples_checked = 0
            max_samples = min(100, video_info['frame_count'] // 10)  # Sample every 10th frame, max 100
            
            for i in range(0, video_info['frame_count'], 10):
                if samples_checked >= max_samples:
                    break
                    
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Convert to grayscale for analysis
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # Calculate frame difference
                    diff = cv2.absdiff(gray_frame, prev_frame)
                    frame_differences.append(np.mean(diff))
                
                prev_frame = gray_frame
                samples_checked += 1
            
            if frame_differences:
                avg_diff = np.mean(frame_differences)
                std_diff = np.std(frame_differences)
                
                # Look for unusual frame differences that might indicate steganography
                if std_diff > 20 and avg_diff > 30:
                    confidence = min(70, (std_diff - 20) * 2)
                    return {
                        'method': 'Frame Anomaly Detection',
                        'confidence': confidence,
                        'description': f'Unusual frame differences detected (avg: {avg_diff:.2f}, std: {std_diff:.2f})',
                        'details': {
                            'avg_difference': avg_diff,
                            'std_difference': std_diff,
                            'samples_analyzed': len(frame_differences)
                        }
                    }
        except Exception:
            pass
        return None
    
    def _detect_motion_vector_anomalies(self, cap, video_info):
        """Detect anomalies in motion vectors (simplified version)"""
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Sample a few frames to check for motion consistency
            motion_magnitudes = []
            prev_frame = None
            
            for i in range(0, min(50, video_info['frame_count']), 5):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # Calculate optical flow (Lucas-Kanade method)
                    flow = cv2.calcOpticalFlowPyrLK(
                        prev_frame, gray_frame, 
                        np.array([[100, 100]], dtype=np.float32), 
                        None, 
                        winSize=(15, 15),
                        maxLevel=2
                    )[0]
                    
                    if flow is not None and len(flow) > 0:
                        motion_mag = np.linalg.norm(flow[0])
                        motion_magnitudes.append(motion_mag)
                
                prev_frame = gray_frame
            
            if motion_magnitudes and len(motion_magnitudes) > 5:
                motion_variance = np.var(motion_magnitudes)
                
                # High variance in motion vectors might indicate steganography
                if motion_variance > 50:
                    confidence = min(60, (motion_variance - 50) / 2)
                    return {
                        'method': 'Motion Vector Anomaly',
                        'confidence': confidence,
                        'description': f'Unusual motion vector patterns detected (variance: {motion_variance:.2f})',
                        'details': {
                            'motion_variance': motion_variance,
                            'samples_analyzed': len(motion_magnitudes)
                        }
                    }
        except Exception:
            pass
        return None
    
    def _detect_compression_artifacts(self, cap, video_info):
        """Detect unusual compression artifacts"""
        try:
            # Sample a few frames to analyze compression quality
            cap.set(cv2.CAP_PROP_POS_FRAMES, video_info['frame_count'] // 2)
            ret, frame = cap.read()
            
            if not ret:
                return None
            
            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate Laplacian variance (measure of blurriness/compression)
            laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
            
            # Calculate JPEG-like compression artifacts using DCT
            # Divide image into 8x8 blocks and analyze DCT coefficients
            h, w = gray_frame.shape
            dct_variances = []
            
            for i in range(0, h-8, 8):
                for j in range(0, w-8, 8):
                    block = gray_frame[i:i+8, j:j+8].astype(np.float32)
                    dct_block = cv2.dct(block)
                    dct_variances.append(np.var(dct_block))
            
            if dct_variances:
                avg_dct_var = np.mean(dct_variances)
                
                # Unusual DCT variance patterns might indicate steganography
                if avg_dct_var > 1000 or avg_dct_var < 10:
                    confidence = 55
                    return {
                        'method': 'Compression Artifact Analysis',
                        'confidence': confidence,
                        'description': f'Unusual compression patterns detected (DCT var: {avg_dct_var:.2f})',
                        'details': {
                            'laplacian_variance': laplacian_var,
                            'dct_variance': avg_dct_var
                        }
                    }
        except Exception:
            pass
        return None
    
    def _analyze_file_structure(self, cap, video_info):
        """Analyze video file structure for anomalies"""
        try:
            file_path = Path(video_info['path'])
            file_size = file_path.stat().st_size
            
            # Estimate expected file size based on video properties
            expected_bitrate = video_info['width'] * video_info['height'] * video_info['fps'] * 0.1  # rough estimate
            expected_size = expected_bitrate * (video_info['frame_count'] / video_info['fps']) / 8
            
            # Check if file size is significantly larger than expected
            if file_size > expected_size * 3:
                size_ratio = file_size / expected_size
                confidence = min(75, (size_ratio - 3) * 15)
                return {
                    'method': 'File Structure Analysis',
                    'confidence': confidence,
                    'description': f'File size {file_size} bytes is {size_ratio:.1f}x larger than expected',
                    'details': {
                        'actual_size': file_size,
                        'expected_size': int(expected_size),
                        'size_ratio': size_ratio
                    }
                }
            
            # Check for unusual metadata or extra data at the end of file
            with open(file_path, 'rb') as f:
                f.seek(-1024, 2)  # Seek to last 1KB
                tail_data = f.read()
                
                # Look for suspicious patterns in the tail
                if b'PK' in tail_data or b'Rar!' in tail_data:
                    return {
                        'method': 'File Structure Analysis',
                        'confidence': 85,
                        'description': 'Suspicious data patterns found at end of video file',
                        'details': {'suspicious_tail': True}
                    }
                    
        except Exception:
            pass
        return None
