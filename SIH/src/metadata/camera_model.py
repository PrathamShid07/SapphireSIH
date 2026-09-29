import math

class CameraIntrinsics:
    def __init__(self, width, height, custom_calib=None, sensor_width_mm=None, focal_length_mm=None):
        self.width = int(width)
        self.height = int(height)
        
        if custom_calib:
            self.fx = float(custom_calib.get("fx", 1.2 * max(self.width, self.height)))
            self.fy = float(custom_calib.get("fy", self.fx))
            self.cx = float(custom_calib.get("cx", self.width / 2.0))
            self.cy = float(custom_calib.get("cy", self.height / 2.0))
            self.k1 = float(custom_calib.get("k1", 0.0))
            self.k2 = float(custom_calib.get("k2", 0.0))
            self.p1 = float(custom_calib.get("p1", 0.0))
            self.p2 = float(custom_calib.get("p2", 0.0))
            self.calibration_type = custom_calib.get("calibration_type", "CALIBRATED")
            self.model = custom_calib.get("model", "OPENCV")
        elif sensor_width_mm and focal_length_mm and sensor_width_mm > 0:
            self.fx = float((focal_length_mm / sensor_width_mm) * self.width)
            self.fy = float(self.fx)
            self.cx = float(self.width / 2.0)
            self.cy = float(self.height / 2.0)
            self.k1 = 0.0
            self.k2 = 0.0
            self.p1 = 0.0
            self.p2 = 0.0
            self.calibration_type = "EXIF_CALCULATED"
            self.model = "PINHOLE"
        else:
            default_f = 1.2 * max(self.width, self.height)
            self.fx = float(default_f)
            self.fy = float(default_f)
            self.cx = float(self.width / 2.0)
            self.cy = float(self.height / 2.0)
            self.k1 = 0.0
            self.k2 = 0.0
            self.p1 = 0.0
            self.p2 = 0.0
            self.calibration_type = "ESTIMATED_DEFAULT"
            self.model = "PINHOLE"

    def to_dict(self):
        return {
            "width": self.width,
            "height": self.height,
            "fx": round(self.fx, 4),
            "fy": round(self.fy, 4),
            "cx": round(self.cx, 4),
            "cy": round(self.cy, 4),
            "distortion": {
                "k1": round(self.k1, 6),
                "k2": round(self.k2, 6),
                "p1": round(self.p1, 6),
                "p2": round(self.p2, 6),
            },
            "calibration_type": self.calibration_type,
            "model": self.model,
            "fov_deg_horizontal": round(2 * math.degrees(math.atan(self.width / (2 * self.fx))), 2),
            "fov_deg_vertical": round(2 * math.degrees(math.atan(self.height / (2 * self.fy))), 2),
        }
