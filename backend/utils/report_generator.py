"""
Report generation utility for the steganography detector
Generates reports in JSON format
"""

import json
from pathlib import Path
from datetime import datetime

class ReportGenerator:
    def generate_report(self, detection_results, output_dir):
        """Generate a JSON report for the detection results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = output_dir / f'report_{timestamp}.json'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(detection_results, f, ensure_ascii=False, indent=4)
        
        return report_file

    def print_summary(self, detection_results):
        """Prints a summary of the detection results"""
        detected_files = [res for res in detection_results if res['detections']]
        
        print("Detection Summary:")
        print("-----------------")
        print(f"Total files analyzed: {len(detection_results)}")
        print(f"Files with potential steganography: {len(detected_files)}")
        
        for result in detected_files:
            print(f"\nFile: {result['file_path']}")
            for detection in result['detections']:
                print(f"  - Method: {detection['method']} - Confidence: {detection['confidence']:.2f}%")
