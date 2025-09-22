#!/usr/bin/env python3
"""
Steganography Detection Tool
Detects hidden data in images, videos, audio files, and text files
"""

import os
import sys
import argparse
from pathlib import Path

# Import detection modules
from detectors.image_detector import ImageStegoDetector
from detectors.audio_detector import AudioStegoDetector
from detectors.video_detector import VideoStegoDetector
from detectors.text_detector import TextStegoDetector
from utils.logger import setup_logger
from utils.report_generator import ReportGenerator

class SteganographyDetector:
    def __init__(self):
        self.logger = setup_logger()
        self.image_detector = ImageStegoDetector()
        self.audio_detector = AudioStegoDetector()
        self.video_detector = VideoStegoDetector()
        self.text_detector = TextStegoDetector()
        self.report_generator = ReportGenerator()
        
    def detect_file(self, file_path, output_dir=None):
        """Detect steganography in a single file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            return None
            
        self.logger.info(f"Analyzing file: {file_path}")
        
        # Determine file type and use appropriate detector
        file_ext = file_path.suffix.lower()
        
        results = {
            'file_path': str(file_path),
            'file_type': self._get_file_type(file_ext),
            'file_size': file_path.stat().st_size,
            'detections': []
        }
        
        try:
            if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
                detections = self.image_detector.detect(file_path)
            elif file_ext in ['.wav', '.mp3', '.flac', '.ogg', '.aac']:
                detections = self.audio_detector.detect(file_path)
            elif file_ext in ['.mp4', '.avi', '.mkv', '.mov', '.wmv']:
                detections = self.video_detector.detect(file_path)
            elif file_ext in ['.txt', '.md', '.html', '.xml']:
                detections = self.text_detector.detect(file_path)
            else:
                self.logger.warning(f"Unsupported file type: {file_ext}")
                return results
                
            results['detections'] = detections
            
            # Generate report if output directory specified
            if output_dir:
                self.report_generator.generate_report(results, output_dir)
                
        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {str(e)}")
            results['error'] = str(e)
            
        return results
    
    def detect_directory(self, directory_path, output_dir=None, recursive=True):
        """Detect steganography in all supported files in a directory"""
        directory_path = Path(directory_path)
        
        if not directory_path.exists() or not directory_path.is_dir():
            self.logger.error(f"Directory not found: {directory_path}")
            return []
            
        self.logger.info(f"Scanning directory: {directory_path}")
        
        # Supported file extensions
        supported_extensions = {
            '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif',  # Images
            '.wav', '.mp3', '.flac', '.ogg', '.aac',          # Audio
            '.mp4', '.avi', '.mkv', '.mov', '.wmv',           # Video
            '.txt', '.md', '.html', '.xml'                    # Text
        }
        
        results = []
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                result = self.detect_file(file_path, output_dir)
                if result:
                    results.append(result)
                    
        return results
    
    def _get_file_type(self, file_ext):
        """Determine file type category"""
        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
            return 'image'
        elif file_ext in ['.wav', '.mp3', '.flac', '.ogg', '.aac']:
            return 'audio'
        elif file_ext in ['.mp4', '.avi', '.mkv', '.mov', '.wmv']:
            return 'video'
        elif file_ext in ['.txt', '.md', '.html', '.xml']:
            return 'text'
        else:
            return 'unknown'

def main():
    parser = argparse.ArgumentParser(description='Steganography Detection Tool')
    parser.add_argument('input', help='Input file or directory path')
    parser.add_argument('-o', '--output', help='Output directory for reports')
    parser.add_argument('-r', '--recursive', action='store_true', 
                       help='Recursively scan directories')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    detector = SteganographyDetector()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        result = detector.detect_file(input_path, args.output)
        if result and result.get('detections'):
            print(f"\n[ALERT] Potential steganography detected in: {input_path}")
            for detection in result['detections']:
                print(f"  - {detection['method']}: {detection['confidence']:.2f}% confidence")
                if detection.get('description'):
                    print(f"    {detection['description']}")
        else:
            print(f"\n[CLEAN] No steganography detected in: {input_path}")
            
    elif input_path.is_dir():
        results = detector.detect_directory(input_path, args.output, args.recursive)
        
        suspicious_files = [r for r in results if r.get('detections')]
        
        print(f"\nScanned {len(results)} files")
        print(f"Found {len(suspicious_files)} suspicious files")
        
        for result in suspicious_files:
            print(f"\n[ALERT] {result['file_path']}")
            for detection in result['detections']:
                print(f"  - {detection['method']}: {detection['confidence']:.2f}% confidence")
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        sys.exit(1)

if __name__ == "__main__":
    main()
