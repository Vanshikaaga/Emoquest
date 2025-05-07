import json
import matplotlib.pyplot as plt
from datetime import datetime
import os

class VoiceLogAnalyzer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = self.load_data()
        self.timestamps = []
        self.intensities = []
        self.confidences = []
        self.words = []
        self.rages = []
        self.avg_intensity = 0
        self.avg_confidence = 0
        self.INTENSITY_THRESHOLD = -6.2
        self.CONFIDENCE_THRESHOLD = 45
        self.RAGE_COLOR = 'red'

        self.process_data()
        self.compute_averages()
        self.detect_rage()  # calls report append inside

    def load_data(self):
        with open(self.filepath, 'r') as f:
            return json.load(f)

    def process_data(self):
        for entry in self.data:
            self.timestamps.append(datetime.fromtimestamp(entry['timestamp']))
            self.intensities.append(entry['intensity_dB'])
            self.confidences.append(entry['confidence'])
            self.words.append(entry['word'])

        for intensity, confidence in zip(self.intensities, self.confidences):
            rage_point = intensity > self.INTENSITY_THRESHOLD or confidence < self.CONFIDENCE_THRESHOLD
            self.rages.append(rage_point)

    def compute_averages(self):
        self.avg_intensity = sum(self.intensities) / len(self.intensities)
        self.avg_confidence = sum(self.confidences) / len(self.confidences)
        print(f"Average Intensity: {self.avg_intensity:.2f} dB")
        print(f"Average Confidence: {self.avg_confidence:.2f} %")

    def detect_rage(self):
        rage_detected = not (-6.4 <= self.avg_intensity <= -6.3 and 50 <= self.avg_confidence <= 65)
        print(" Final Output: RAGE DETECTED" if rage_detected else " Final Output: Rage NOT detected")

        # --- APPEND REPORT.JSON HERE ---
        report_data = {
            "analysis_time": datetime.now().isoformat(),
            "avg_intensity_dB": round(self.avg_intensity, 2),
            "avg_confidence_percent": round(self.avg_confidence, 2),
            "rage_detected": rage_detected,
            "total_entries": len(self.data),
            "intensity_threshold": self.INTENSITY_THRESHOLD,
            "confidence_threshold": self.CONFIDENCE_THRESHOLD
        }

        report_file = "report.json"

        if os.path.exists(report_file):
            with open(report_file, 'r') as f:
                existing_reports = json.load(f)
        else:
            existing_reports = []

        existing_reports.append(report_data)

        with open(report_file, 'w') as f:
            json.dump(existing_reports, f, indent=4)

        print(f"Report appended to {report_file}")

    def plot(self):
        plt.figure(figsize=(14, 6))

        plt.plot(self.timestamps, self.intensities, label="Intensity (dB)", color='blue', marker='o')
        plt.plot(self.timestamps, self.confidences, label="Confidence (%)", color='green', marker='x')

        plt.axhline(self.INTENSITY_THRESHOLD, color='blue', linestyle='--', linewidth=1,
                    label=f"Intensity Threshold ({self.INTENSITY_THRESHOLD} dB)")
        plt.axhline(self.CONFIDENCE_THRESHOLD, color='green', linestyle='--', linewidth=1,
                    label=f"Confidence Threshold ({self.CONFIDENCE_THRESHOLD}%)")

        for i, is_rage in enumerate(self.rages):
            if is_rage:
                plt.axvline(self.timestamps[i], color=self.RAGE_COLOR, linestyle='--', alpha=0.3)
                plt.text(self.timestamps[i], self.intensities[i] + 3, "RAGE", color=self.RAGE_COLOR, fontsize=9, rotation=90)

        for i, word in enumerate(self.words):
            plt.annotate(word, (self.timestamps[i], self.intensities[i]),
                         textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title("Voice Command Log: Intensity & Confidence vs Thresholds")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.gcf().autofmt_xdate()
        plt.show()

# Example usage:
# analyzer = VoiceLogAnalyzer('voice_log.json')
# analyzer.plot()
