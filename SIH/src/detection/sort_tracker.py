import numpy as np
from scipy.optimize import linear_sum_assignment


def bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h) if h > 0 else 1.0
    return np.array([x, y, s, r]).reshape((4, 1))


def x_to_bbox(x):
    w = np.sqrt(max(0.0, x[2, 0] * x[3, 0]))
    h = x[2, 0] / float(w) if w > 0 else 0.0
    x1 = x[0, 0] - w / 2.0
    y1 = x[1, 0] - h / 2.0
    x2 = x[0, 0] + w / 2.0
    y2 = x[1, 0] + h / 2.0
    return np.array([x1, y1, x2, y2])


def compute_iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    intersection = w * h
    area_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area_gt = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    union = area_test + area_gt - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox, score=1.0, class_id=None, class_name=None):
        self.F = np.eye(7)
        for i in range(3):
            self.F[i, i + 4] = 1.0

        self.H = np.zeros((4, 7))
        for i in range(4):
            self.H[i, i] = 1.0

        self.R = np.eye(4)
        self.R[0, 0] *= 1.0
        self.R[1, 1] *= 1.0
        self.R[2, 2] *= 10.0
        self.R[3, 3] *= 10.0

        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 1000.0

        self.Q = np.eye(7)
        self.Q[4:, 4:] *= 0.01

        self.x = np.zeros((7, 1))
        self.x[:4] = bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.score = float(score)
        self.class_id = class_id
        self.class_name = class_name or "object"

        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0
        self.history.append((float(center_x), float(center_y)))

    def update(self, bbox, score=None, class_id=None, class_name=None):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        if score is not None:
            self.score = float(score)
        if class_id is not None:
            self.class_id = class_id
        if class_name is not None:
            self.class_name = class_name

        z = bbox_to_z(bbox)
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))
        self.x = self.x + np.dot(K, y)
        I = np.eye(7)
        self.P = np.dot(I - np.dot(K, self.H), self.P)

        curr_box = self.get_state()
        center_x = (curr_box[0] + curr_box[2]) / 2.0
        center_y = (curr_box[1] + curr_box[3]) / 2.0
        self.history.append((float(center_x), float(center_y)))
        if len(self.history) > 50:
            self.history.pop(0)

    def predict(self):
        if (self.x[2, 0] + self.x[6, 0]) <= 0:
            self.x[6, 0] = 0.0
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.get_state()

    def get_state(self):
        return x_to_bbox(self.x)

    def get_velocity(self):
        return float(self.x[4, 0]), float(self.x[5, 0])


class SortTracker:
    def __init__(self, max_age=5, min_hits=2, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, detections):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            if np.any(np.isnan(pos)):
                to_del.append(t)
            else:
                trks[t, :] = pos

        for t in reversed(to_del):
            self.trackers.pop(t)
            trks = np.delete(trks, t, axis=0)

        num_dets = len(detections)
        num_trks = len(trks)

        matched_indices = []
        unmatched_dets = list(range(num_dets))
        unmatched_trks = list(range(num_trks))

        if num_dets > 0 and num_trks > 0:
            iou_matrix = np.zeros((num_dets, num_trks), dtype=np.float32)
            for d in range(num_dets):
                det_box = detections[d]["bbox"]
                for t in range(num_trks):
                    iou_matrix[d, t] = compute_iou(det_box, trks[t])

            row_ind, col_ind = linear_sum_assignment(-iou_matrix)

            matched_indices = []
            unmatched_dets = []
            unmatched_trks = []

            matched_d = set()
            matched_t = set()

            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    matched_indices.append((r, c))
                    matched_d.add(r)
                    matched_t.add(c)

            for d in range(num_dets):
                if d not in matched_d:
                    unmatched_dets.append(d)
            for t in range(num_trks):
                if t not in matched_t:
                    unmatched_trks.append(t)

        for d, t in matched_indices:
            det = detections[d]
            self.trackers[t].update(
                det["bbox"],
                score=det.get("score"),
                class_id=det.get("class_id"),
                class_name=det.get("class_name"),
            )

        for d in unmatched_dets:
            det = detections[d]
            trk = KalmanBoxTracker(
                det["bbox"],
                score=det.get("score", 1.0),
                class_id=det.get("class_id"),
                class_name=det.get("class_name"),
            )
            self.trackers.append(trk)

        active_tracks = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
                continue
            if (trk.time_since_update < 1) and (
                trk.hits >= self.min_hits or self.frame_count <= self.min_hits
            ):
                box = trk.get_state()
                vx, vy = trk.get_velocity()
                active_tracks.append(
                    {
                        "track_id": trk.id,
                        "bbox": [float(b) for b in box],
                        "score": trk.score,
                        "class_id": trk.class_id,
                        "class_name": trk.class_name,
                        "velocity": [vx, vy],
                        "history": list(trk.history),
                    }
                )

        return active_tracks
