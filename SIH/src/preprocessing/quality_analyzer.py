import cv2
import numpy as np

class QualityAnalyzer:
    def __init__(self, thresholds=None):
        default_thresholds = {
            "min_blur_score": 100.0,
            "min_brightness": 40.0,
            "max_brightness": 220.0,
            "min_contrast": 30.0,
            "max_similarity": 0.92,
        }
        if thresholds:
            default_thresholds.update(thresholds)
        self.thresholds = default_thresholds

    def compute_blur(self, gray):
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def compute_brightness(self, gray):
        return float(np.mean(gray))

    def compute_contrast(self, gray):
        return float(np.std(gray))

    def compute_similarity(self, gray1, gray2):
        if gray1 is None or gray2 is None:
            return 0.0
        
        target_size = (160, 120)
        g1 = cv2.resize(gray1, target_size).astype(np.float32)
        g2 = cv2.resize(gray2, target_size).astype(np.float32)
        
        mu1 = np.mean(g1)
        mu2 = np.mean(g2)
        var1 = np.var(g1)
        var2 = np.var(g2)
        cov = np.mean((g1 - mu1) * (g2 - mu2))
        
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * cov + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (var1 + var2 + c2))
        return float(np.clip(ssim, -1.0, 1.0))

    def analyze(self, frame_bgr, prev_selected_gray=None):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        blur = self.compute_blur(gray)
        brightness = self.compute_brightness(gray)
        contrast = self.compute_contrast(gray)
        similarity = self.compute_similarity(gray, prev_selected_gray) if prev_selected_gray is not None else 0.0

        scores = {
            "blur_score": round(blur, 2),
            "brightness_score": round(brightness, 2),
            "contrast_score": round(contrast, 2),
            "similarity_to_prev": round(similarity, 4),
        }

        rejection_reason = None

        if blur < self.thresholds["min_blur_score"]:
            rejection_reason = "BLURRY"
        elif brightness < self.thresholds["min_brightness"]:
            rejection_reason = "UNDEREXPOSED"
        elif brightness > self.thresholds["max_brightness"]:
            rejection_reason = "OVEREXPOSED"
        elif contrast < self.thresholds["min_contrast"]:
            rejection_reason = "LOW_CONTRAST"
        elif prev_selected_gray is not None and similarity > self.thresholds["max_similarity"]:
            rejection_reason = "NEAR_DUPLICATE"

        is_selected = rejection_reason is None

        return is_selected, rejection_reason, scores, gray
