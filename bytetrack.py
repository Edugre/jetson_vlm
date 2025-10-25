"""
Minimal ByteTrack implementation for multi-object tracking
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_batch(bboxes1, bboxes2):
    """
    Compute IoU between two sets of bounding boxes
    bboxes format: [x1, y1, x2, y2]
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)
    return o


class KalmanFilter:
    """Simple Kalman filter for bbox tracking"""
    def __init__(self):
        # State: [x, y, w, h, vx, vy, vw, vh]
        self.mean = np.zeros(8)
        self.covariance = np.eye(8) * 1000

        # Motion model
        self.motion_mat = np.eye(8)
        for i in range(4):
            self.motion_mat[i, i+4] = 1

        self.update_mat = np.eye(4, 8)
        self.std_weight_position = 1./20
        self.std_weight_velocity = 1./160

    def initiate(self, measurement):
        """Initialize filter with first measurement [x, y, w, h]"""
        self.mean[:4] = measurement
        self.mean[4:] = 0

        std = [
            2 * self.std_weight_position * measurement[2],
            2 * self.std_weight_position * measurement[3],
            2 * self.std_weight_position * measurement[2],
            2 * self.std_weight_position * measurement[3],
            10 * self.std_weight_velocity * measurement[2],
            10 * self.std_weight_velocity * measurement[3],
            10 * self.std_weight_velocity * measurement[2],
            10 * self.std_weight_velocity * measurement[3]
        ]
        self.covariance = np.diag(np.square(std))

    def predict(self):
        """Predict next state"""
        std_pos = [
            self.std_weight_position * self.mean[2],
            self.std_weight_position * self.mean[3],
            self.std_weight_position * self.mean[2],
            self.std_weight_position * self.mean[3]
        ]
        std_vel = [
            self.std_weight_velocity * self.mean[2],
            self.std_weight_velocity * self.mean[3],
            self.std_weight_velocity * self.mean[2],
            self.std_weight_velocity * self.mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        self.mean = np.dot(self.motion_mat, self.mean)
        self.covariance = np.linalg.multi_dot((
            self.motion_mat, self.covariance, self.motion_mat.T)) + motion_cov

    def update(self, measurement):
        """Update state with new measurement [x, y, w, h]"""
        projected_cov = np.linalg.multi_dot((
            self.update_mat, self.covariance, self.update_mat.T))

        std = [
            self.std_weight_position * self.mean[2],
            self.std_weight_position * self.mean[3],
            self.std_weight_position * self.mean[2],
            self.std_weight_position * self.mean[3]
        ]
        innovation_cov = projected_cov + np.diag(np.square(std))

        kalman_gain = np.linalg.multi_dot((
            self.covariance, self.update_mat.T, np.linalg.inv(innovation_cov)))
        innovation = measurement - np.dot(self.update_mat, self.mean)

        self.mean = self.mean + np.dot(kalman_gain, innovation)
        self.covariance = self.covariance - np.linalg.multi_dot((
            kalman_gain, self.update_mat, self.covariance))

    def get_state(self):
        """Return current bbox [x, y, w, h]"""
        return self.mean[:4].copy()


class Track:
    """Single track object"""
    _count = 0

    def __init__(self, bbox, score, class_id):
        self.track_id = Track._count
        Track._count += 1
        self.kf = KalmanFilter()
        self.kf.initiate(self._xyxy_to_xywh(bbox))
        self.score = score
        self.class_id = class_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

    @staticmethod
    def _xyxy_to_xywh(bbox):
        """Convert [x1, y1, x2, y2] to [cx, cy, w, h]"""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w/2
        cy = y1 + h/2
        return np.array([cx, cy, w, h])

    @staticmethod
    def _xywh_to_xyxy(bbox):
        """Convert [cx, cy, w, h] to [x1, y1, x2, y2]"""
        cx, cy, w, h = bbox
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        return np.array([x1, y1, x2, y2])

    def predict(self):
        """Predict next position"""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox, score):
        """Update with new detection"""
        self.kf.update(self._xyxy_to_xywh(bbox))
        self.score = score
        self.hits += 1
        self.time_since_update = 0

    def get_state(self):
        """Return current bbox in xyxy format"""
        return self._xywh_to_xyxy(self.kf.get_state())


class ByteTracker:
    """ByteTrack tracker"""
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.frame_id = 0

    def update(self, detections):
        """
        Update tracks with new detections
        detections: Nx6 array [x1, y1, x2, y2, score, class_id]
        returns: list of active tracks
        """
        self.frame_id += 1

        # Split detections by score
        if len(detections) > 0:
            high_det = detections[detections[:, 4] >= self.track_thresh]
            low_det = detections[detections[:, 4] < self.track_thresh]
        else:
            high_det = np.empty((0, 6))
            low_det = np.empty((0, 6))

        # Predict all tracks
        for track in self.tracked_tracks + self.lost_tracks:
            track.predict()

        # First association with high score detections
        unmatched_tracks, unmatched_dets = self._match(
            self.tracked_tracks, high_det, self.match_thresh)

        # Second association with low score detections
        unmatched_tracks2, _ = self._match(
            unmatched_tracks, low_det, 0.5)

        # Update matched tracks
        for track in self.tracked_tracks:
            if track.time_since_update == 0:
                continue

        # Move lost tracks
        for track in unmatched_tracks2:
            if track.time_since_update <= self.track_buffer:
                self.lost_tracks.append(track)
            else:
                self.removed_tracks.append(track)

        self.tracked_tracks = [t for t in self.tracked_tracks
                               if t.time_since_update == 0]

        # Try to match lost tracks with remaining detections
        matched_lost, unmatched_dets2 = self._match(
            self.lost_tracks, unmatched_dets, 0.5)

        # Recover matched lost tracks
        for track in matched_lost:
            if track in self.lost_tracks:
                self.lost_tracks.remove(track)
                self.tracked_tracks.append(track)

        # Initialize new tracks
        for det in unmatched_dets2:
            track = Track(det[:4], det[4], int(det[5]))
            self.tracked_tracks.append(track)

        # Remove old lost tracks
        self.lost_tracks = [t for t in self.lost_tracks
                           if t.time_since_update <= self.track_buffer]

        return self.tracked_tracks

    def _match(self, tracks, detections, thresh):
        """Match tracks to detections using Hungarian algorithm"""
        if len(tracks) == 0 or len(detections) == 0:
            return tracks, detections

        # Compute IoU matrix
        track_boxes = np.array([t.get_state() for t in tracks])
        det_boxes = detections[:, :4]
        iou_matrix = iou_batch(track_boxes, det_boxes)

        # Hungarian matching
        cost_matrix = 1 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_tracks = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= thresh:
                tracks[r].update(detections[c, :4], detections[c, 4])
                matched_tracks.append(tracks[r])
                unmatched_tracks.remove(r)
                unmatched_dets.remove(c)

        unmatched_tracks = [tracks[i] for i in unmatched_tracks]
        unmatched_dets = detections[unmatched_dets]

        return unmatched_tracks, unmatched_dets

    def get_tracked_objects(self):
        """Return list of tracked objects with (bbox, track_id, score, class_id)"""
        results = []
        for track in self.tracked_tracks:
            bbox = track.get_state()
            results.append({
                'bbox': bbox,
                'track_id': track.track_id,
                'score': track.score,
                'class_id': track.class_id
            })
        return results
