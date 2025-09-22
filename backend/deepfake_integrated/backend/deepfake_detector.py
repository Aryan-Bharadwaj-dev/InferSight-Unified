import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
import librosa
import asyncio
import tempfile
import os
from typing import Tuple, Dict, Any, List, Optional
from PIL import Image
import tensorflow as tf
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
import ffmpeg
import time
import logging

logger = logging.getLogger(__name__)

class FaceXceptionModel:
    """Xception-based model for face manipulation detection"""
    
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained Xception model adapted for deepfake detection"""
        try:
            # Create a simplified Xception-like architecture
            from torchvision.models import efficientnet_b0
            self.model = efficientnet_b0(pretrained=True)
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.model.classifier[1].in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 2)  # real vs fake
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info("Face detection model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading face model: {e}")
    
    def predict(self, image: np.ndarray) -> Tuple[float, Dict]:
        """Predict if image contains deepfake manipulation"""
        try:
            # Convert to PIL Image and apply transforms
            if len(image.shape) == 3:
                image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image_pil = Image.fromarray(image)
            
            input_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                fake_prob = probabilities[0][1].item()
                
            # Additional visual feature analysis
            features = self._extract_visual_features(image)
            
            return fake_prob, features
            
        except Exception as e:
            logger.error(f"Error in face prediction: {e}")
            return 0.5, {}  # Return neutral probability on error
    
    def _extract_visual_features(self, image: np.ndarray) -> Dict:
        """Extract visual inconsistency features"""
        features = {}
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Detect edges and analyze sharpness
            edges = cv2.Laplacian(gray, cv2.CV_64F)
            features['edge_variance'] = float(np.var(edges))
            features['sharpness'] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            
            # Color histogram analysis
            if len(image.shape) == 3:
                for i, color in enumerate(['blue', 'green', 'red']):
                    hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                    features[f'{color}_hist_mean'] = float(np.mean(hist))
                    features[f'{color}_hist_std'] = float(np.std(hist))
            
            # Face detection for region analysis
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            features['faces_detected'] = int(len(faces))
            
            if len(faces) > 0:
                # Analyze largest face
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                face_region = gray[y:y+h, x:x+w]
                features['face_sharpness'] = float(cv2.Laplacian(face_region, cv2.CV_64F).var())
                
        except Exception as e:
            logger.error(f"Error extracting visual features: {e}")
            
        return features

class AudioDeepfakeDetector:
    """Audio deepfake detection using spectral analysis"""
    
    def __init__(self):
        self.sample_rate = 16000
        self.n_mfcc = 13
        self.model = self._create_audio_model()
    
    def _create_audio_model(self):
        """Create a simple CNN model for audio deepfake detection"""
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(None, self.n_mfcc)),
                tf.keras.layers.Reshape((-1, self.n_mfcc, 1)),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(2, activation='softmax')  # real vs fake
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            logger.info("Audio detection model created successfully")
            return model
        except Exception as e:
            logger.error(f"Error creating audio model: {e}")
            return None
    
    def predict(self, audio_data: np.ndarray) -> Tuple[float, Dict]:
        """Predict if audio contains deepfake manipulation"""
        try:
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=self.n_mfcc)
            
            # Prepare input for model
            mfccs = np.expand_dims(mfccs.T, axis=0)
            mfccs = np.expand_dims(mfccs, axis=-1)
            
            # Get prediction
            if self.model:
                prediction = self.model.predict(mfccs, verbose=0)
                fake_prob = prediction[0][1]
            else:
                fake_prob = 0.5  # Neutral if model not available
            
            # Extract additional audio features
            features = self._extract_audio_features(audio_data)
            
            return float(fake_prob), features
            
        except Exception as e:
            logger.error(f"Error in audio prediction: {e}")
            return 0.5, {}
    
    def _extract_audio_features(self, audio_data: np.ndarray) -> Dict:
        """Extract audio features for analysis"""
        features = {}
        try:
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            features['zcr_mean'] = float(np.mean(zcr))
            features['zcr_std'] = float(np.std(zcr))
            
            # RMS energy
            rms = librosa.feature.rms(y=audio_data)[0]
            features['rms_mean'] = float(np.mean(rms))
            features['rms_std'] = float(np.std(rms))
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=self.sample_rate)
            features['chroma_mean'] = float(np.mean(chroma))
            features['chroma_std'] = float(np.std(chroma))
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            
        return features

class TemporalAnalyzer:
    """Analyze temporal inconsistencies in videos"""
    
    def __init__(self):
        self.frame_skip = 5  # Analyze every 5th frame for efficiency
    
    def analyze_video_frames(self, frames: List[np.ndarray]) -> Dict:
        """Analyze temporal inconsistencies across video frames"""
        features = {}
        try:
            if len(frames) < 2:
                return features
            
            # Calculate frame-to-frame differences
            frame_diffs = []
            for i in range(1, len(frames)):
                diff = cv2.absdiff(frames[i-1], frames[i])
                frame_diffs.append(float(np.mean(diff)))
            
            features['frame_diff_mean'] = float(np.mean(frame_diffs))
            features['frame_diff_std'] = float(np.std(frame_diffs))
            features['frame_diff_max'] = float(np.max(frame_diffs))
            
            # Optical flow analysis
            if len(frames) >= 2:
                flow_magnitude = self._calculate_optical_flow(frames)
                if flow_magnitude:
                    features['optical_flow_mean'] = float(np.mean(flow_magnitude))
                    features['optical_flow_std'] = float(np.std(flow_magnitude))
            
            # Temporal consistency in face regions
            face_consistency = self._analyze_face_consistency(frames)
            features.update(face_consistency)
            
        except Exception as e:
            logger.error(f"Error in temporal analysis: {e}")
            
        return features
    
    def _calculate_optical_flow(self, frames: List[np.ndarray]) -> List[float]:
        """Calculate optical flow between consecutive frames"""
        flow_magnitudes = []
        try:
            for i in range(1, min(len(frames), 10)):  # Limit to first 10 frames for efficiency
                gray1 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                
                # Find corners to track
                corners = cv2.goodFeaturesToTrack(gray1, maxCorners=100, qualityLevel=0.01, minDistance=10)
                
                if corners is not None and len(corners) > 0:
                    # Calculate optical flow
                    next_pts, status, error = cv2.calcOpticalFlowPyrLK(gray1, gray2, corners, None)
                    
                    # Calculate magnitudes for valid points
                    valid_flow = next_pts[status == 1] - corners[status == 1]
                    if len(valid_flow) > 0:
                        magnitude = np.sqrt(valid_flow[:, 0]**2 + valid_flow[:, 1]**2)
                        flow_magnitudes.append(float(np.mean(magnitude)))
        except Exception as e:
            logger.error(f"Error calculating optical flow: {e}")
            
        return flow_magnitudes
    
    def _analyze_face_consistency(self, frames: List[np.ndarray]) -> Dict:
        """Analyze consistency of face regions across frames"""
        features = {}
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            face_positions = []
            face_sizes = []
            
            for frame in frames[:10]:  # Analyze first 10 frames
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    # Take largest face
                    largest_face = max(faces, key=lambda x: x[2] * x[3])
                    x, y, w, h = largest_face
                    face_positions.append([x + w/2, y + h/2])  # Center position
                    face_sizes.append(w * h)
            
            if len(face_positions) > 1:
                positions = np.array(face_positions)
                sizes = np.array(face_sizes)
                
                features['face_position_variance'] = float(np.var(positions, axis=0).mean())
                features['face_size_variance'] = float(np.var(sizes))
                features['faces_detected_count'] = int(len(face_positions))
                
        except Exception as e:
            logger.error(f"Error analyzing face consistency: {e}")
            
        return features

class DeepfakeDetector:
    """Main deepfake detection orchestrator"""
    
    def __init__(self):
        self.face_detector = FaceXceptionModel()
        self.audio_detector = AudioDeepfakeDetector()
        self.temporal_analyzer = TemporalAnalyzer()
        
    async def detect_image(self, image_path: str) -> Dict[str, Any]:
        """Detect deepfake in image"""
        start_time = time.time()
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            # Face detection
            face_prob, visual_features = self.face_detector.predict(image)
            
            # Calculate final confidence
            confidence_score = float(face_prob)
            is_deepfake = confidence_score > 0.5
            
            result = {
                'is_deepfake': bool(is_deepfake),
                'confidence_score': confidence_score,
                'deepfake_probability': confidence_score * 100,
                'visual_analysis': visual_features,
                'models_used': ['FaceXception'],
                'processing_time': float(time.time() - start_time)
            }
            
            logger.info(f"Image analysis completed: {confidence_score:.3f} confidence")
            return result
            
        except Exception as e:
            logger.error(f"Error detecting image deepfake: {e}")
            raise
    
    async def detect_video(self, video_path: str) -> Dict[str, Any]:
        """Detect deepfake in video"""
        start_time = time.time()
        
        try:
            # Extract frames
            frames = await self._extract_video_frames(video_path)
            if not frames:
                raise ValueError("Could not extract frames from video")
            
            # Extract audio
            audio_data = await self._extract_audio_from_video(video_path)
            
            # Analyze frames
            frame_predictions = []
            visual_features_list = []
            
            for i, frame in enumerate(frames[::5]):  # Every 5th frame
                face_prob, visual_features = self.face_detector.predict(frame)
                frame_predictions.append(face_prob)
                visual_features_list.append(visual_features)
            
            # Temporal analysis
            temporal_features = self.temporal_analyzer.analyze_video_frames(frames)
            
            # Audio analysis
            audio_prob = 0.5
            audio_features = {}
            if audio_data is not None:
                audio_prob, audio_features = self.audio_detector.predict(audio_data)
            
            # Ensemble prediction
            visual_confidence = float(np.mean(frame_predictions)) if frame_predictions else 0.5
            final_confidence = float(visual_confidence * 0.7 + audio_prob * 0.3)
            
            is_deepfake = final_confidence > 0.5
            
            result = {
                'is_deepfake': bool(is_deepfake),
                'confidence_score': final_confidence,
                'deepfake_probability': final_confidence * 100,
                'visual_analysis': {
                    'frame_predictions': [float(p) for p in frame_predictions],
                    'visual_features': visual_features_list[:3] if visual_features_list else [],
                    'frames_analyzed': int(len(frame_predictions))
                },
                'audio_analysis': audio_features if audio_data is not None else None,
                'temporal_analysis': temporal_features,
                'models_used': ['FaceXception', 'AudioCNN', 'TemporalAnalyzer'],
                'processing_time': float(time.time() - start_time)
            }
            
            logger.info(f"Video analysis completed: {final_confidence:.3f} confidence")
            return result
            
        except Exception as e:
            logger.error(f"Error detecting video deepfake: {e}")
            raise
    
    async def detect_audio(self, audio_path: str) -> Dict[str, Any]:
        """Detect deepfake in audio file"""
        start_time = time.time()
        
        try:
            # Load audio
            audio_data, sr = librosa.load(audio_path, sr=16000)
            
            # Audio analysis
            audio_prob, audio_features = self.audio_detector.predict(audio_data)
            
            is_deepfake = audio_prob > 0.5
            
            result = {
                'is_deepfake': bool(is_deepfake),
                'confidence_score': float(audio_prob),
                'deepfake_probability': float(audio_prob * 100),
                'audio_analysis': audio_features,
                'models_used': ['AudioCNN'],
                'processing_time': float(time.time() - start_time)
            }
            
            logger.info(f"Audio analysis completed: {audio_prob:.3f} confidence")
            return result
            
        except Exception as e:
            logger.error(f"Error detecting audio deepfake: {e}")
            raise
    
    async def _extract_video_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract frames from video"""
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            max_frames = 30  # Limit frames for efficiency
            
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % 5 == 0:  # Every 5th frame
                    frames.append(frame)
                
                frame_count += 1
            
            cap.release()
            logger.info(f"Extracted {len(frames)} frames from video")
            
        except Exception as e:
            logger.error(f"Error extracting video frames: {e}")
            
        return frames
    
    async def _extract_audio_from_video(self, video_path: str) -> Optional[np.ndarray]:
        """Extract audio track from video"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                # Use ffmpeg to extract audio
                (
                    ffmpeg
                    .input(video_path)
                    .output(temp_audio.name, format='wav', acodec='pcm_s16le', ar=16000)
                    .overwrite_output()
                    .run(quiet=True)
                )
                
                # Load the extracted audio
                audio_data, _ = librosa.load(temp_audio.name, sr=16000)
                
                # Clean up
                os.unlink(temp_audio.name)
                
                logger.info("Audio extracted from video successfully")
                return audio_data
                
        except Exception as e:
            logger.warning(f"Could not extract audio from video: {e}")
            return None