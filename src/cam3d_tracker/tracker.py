from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import bev_iou, yaw_cost
from .imm_ekf import IMMEKF, STATE_DIM
from .math_utils import clamp
from .models import Detection3D, TrackOutput


@dataclass
class TrackNode:
    track_id: int
    label: str
    filt: IMMEKF
    score_ema: float
    hits: int
    misses: int
    age_s: float
    time_since_update_s: float
    status: str


class Classical3DTracker:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.assoc_cfg = cfg["association"]
        self.tracker_cfg = cfg["tracker"]
        self.noise_cfg = cfg["noise"]
        self.imm_cfg = cfg["imm"]

        self.q_cv = np.diag(np.array(self.noise_cfg["process_cv_diag"], dtype=float) ** 2)
        self.q_ctrv = np.diag(np.array(self.noise_cfg["process_ctrv_diag"], dtype=float) ** 2)
        self.transition = np.array(self.imm_cfg["transition"], dtype=float)
        self.mode_prob_init = np.array(self.imm_cfg["mode_prob_init"], dtype=float)
        self.mode_prob_init = self.mode_prob_init / np.sum(self.mode_prob_init)

        self.tracks: dict[int, TrackNode] = {}
        self._next_id = 1
        self._last_timestamp_s: float | None = None

    def _max_age_for_label(self, label: str) -> float:
        table = self.tracker_cfg["max_age_s"]
        return float(table.get(label, table["default"]))

    def _meas_cov_for_label(self, label: str) -> np.ndarray:
        meas_map = self.noise_cfg["meas_by_class"]
        vals = np.array(meas_map.get(label, meas_map["default"]), dtype=float)
        return np.diag(vals**2)

    def _init_track(self, det: Detection3D) -> TrackNode:
        x0 = np.array(
            [det.x, det.y, det.z, 0.0, det.yaw, 0.0, max(det.l, 0.05), max(det.w, 0.05), max(det.h, 0.05)],
            dtype=float,
        )
        p0 = np.diag(np.array([6.0, 6.0, 3.0, 4.0, 0.8, 0.8, 1.0, 1.0, 1.0], dtype=float) ** 2)
        filt = IMMEKF(x0=x0, p0=p0, mode_prob_init=self.mode_prob_init, transition=self.transition)
        node = TrackNode(
            track_id=self._next_id,
            label=det.label,
            filt=filt,
            score_ema=det.score,
            hits=1,
            misses=0,
            age_s=0.0,
            time_since_update_s=0.0,
            status="tentative",
        )
        self._next_id += 1
        return node

    def _compute_dt(self, timestamp_s: float) -> float:
        if self._last_timestamp_s is None:
            return float(self.tracker_cfg["dt_fallback_s"])
        return max(1e-3, float(timestamp_s - self._last_timestamp_s))

    def _predict_all(self, dt: float) -> None:
        for trk in self.tracks.values():
            trk.filt.predict(dt=dt, q_cv=self.q_cv, q_ctrv=self.q_ctrv)
            trk.age_s += dt
            trk.time_since_update_s += dt
            trk.score_ema *= float(self.tracker_cfg["existence_decay"])

    def _cost_matrix(self, track_ids: list[int], detections: list[Detection3D]) -> np.ndarray:
        c = np.full((len(track_ids), len(detections)), fill_value=1e6, dtype=float)
        gate = float(self.assoc_cfg["maha_gate_threshold"])
        w = self.assoc_cfg["cost_weights"]

        for i, tid in enumerate(track_ids):
            trk = self.tracks[tid]
            r = self._meas_cov_for_label(trk.label)
            for j, det in enumerate(detections):
                if det.label != trk.label:
                    continue
                maha = trk.filt.innovation_mahalanobis(det.z_vec, r)
                if maha > gate:
                    continue
                iou_term = 1.0 - bev_iou(trk.filt.x, np.array([det.x, det.y, det.z, 0.0, det.yaw, 0.0, det.l, det.w, det.h]))
                yaw_term = yaw_cost(trk.filt.x[4], det.yaw)
                c[i, j] = (
                    float(w["maha"]) * (maha / gate)
                    + float(w["iou"]) * iou_term
                    + float(w["yaw"]) * yaw_term
                )
        return c

    def _second_stage_center_match(
        self,
        unmatched_tracks: list[int],
        unmatched_dets: list[int],
        detections: list[Detection3D],
    ) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        if not unmatched_tracks or not unmatched_dets:
            return out

        gate = float(self.assoc_cfg["second_stage_center_gate_m"])
        used_dets: set[int] = set()
        for tid in unmatched_tracks:
            trk = self.tracks[tid]
            best = None
            best_dist = 1e9
            for dj in unmatched_dets:
                if dj in used_dets:
                    continue
                det = detections[dj]
                if det.label != trk.label:
                    continue
                dist = float(np.linalg.norm(np.array([det.x - trk.filt.x[0], det.y - trk.filt.x[1]])))
                if dist < best_dist and dist <= gate:
                    best_dist = dist
                    best = dj
            if best is not None:
                out.append((tid, best))
                used_dets.add(best)
        return out

    def step(self, timestamp_s: float, detections: list[Detection3D]) -> list[TrackOutput]:
        dt = self._compute_dt(timestamp_s)
        self._predict_all(dt)

        track_ids = list(self.tracks.keys())
        matches: list[tuple[int, int]] = []
        unmatched_track_ids = track_ids.copy()
        unmatched_det_ids = list(range(len(detections)))

        if track_ids and detections:
            cost = self._cost_matrix(track_ids, detections)
            row_ind, col_ind = linear_sum_assignment(cost)

            gated_matches: list[tuple[int, int]] = []
            for r_i, c_i in zip(row_ind, col_ind):
                if cost[r_i, c_i] >= 1e5:
                    continue
                gated_matches.append((track_ids[r_i], c_i))

            matched_tids = {m[0] for m in gated_matches}
            matched_dids = {m[1] for m in gated_matches}
            unmatched_track_ids = [tid for tid in track_ids if tid not in matched_tids]
            unmatched_det_ids = [di for di in unmatched_det_ids if di not in matched_dids]
            matches.extend(gated_matches)

            second = self._second_stage_center_match(unmatched_track_ids, unmatched_det_ids, detections)
            if second:
                m2_tids = {m[0] for m in second}
                m2_dids = {m[1] for m in second}
                unmatched_track_ids = [tid for tid in unmatched_track_ids if tid not in m2_tids]
                unmatched_det_ids = [di for di in unmatched_det_ids if di not in m2_dids]
                matches.extend(second)

        for tid, det_idx in matches:
            trk = self.tracks[tid]
            det = detections[det_idx]
            r = self._meas_cov_for_label(trk.label)
            trk.filt.update(det.z_vec, r)
            trk.hits += 1
            trk.misses = 0
            trk.time_since_update_s = 0.0
            trk.score_ema = 0.6 * trk.score_ema + 0.4 * det.score

            min_hits = int(self.tracker_cfg["min_hits"].get(trk.label, self.tracker_cfg["min_hits"]["default"]))
            if trk.status == "tentative" and trk.hits >= min_hits and trk.score_ema >= float(self.tracker_cfg["confirm_score_threshold"]):
                trk.status = "confirmed"
            elif trk.status == "lost":
                trk.status = "confirmed"

        for tid in unmatched_track_ids:
            trk = self.tracks[tid]
            trk.misses += 1
            if trk.status == "confirmed":
                trk.status = "lost"

        for di in unmatched_det_ids:
            det = detections[di]
            if det.score < float(self.tracker_cfg["init_score_threshold"]):
                continue
            node = self._init_track(det)
            self.tracks[node.track_id] = node

        to_delete: list[int] = []
        for tid, trk in self.tracks.items():
            max_age = self._max_age_for_label(trk.label)
            if trk.time_since_update_s > max_age:
                to_delete.append(tid)
            if trk.status == "tentative" and trk.misses > 0:
                to_delete.append(tid)
            if trk.score_ema < 0.05:
                to_delete.append(tid)

        for tid in set(to_delete):
            self.tracks.pop(tid, None)

        self._last_timestamp_s = timestamp_s

        outputs: list[TrackOutput] = []
        for trk in self.tracks.values():
            if trk.status not in {"confirmed", "lost"}:
                continue
            out = TrackOutput(
                track_id=trk.track_id,
                label=trk.label,
                score=clamp(trk.score_ema, 0.0, 1.0),
                state=trk.filt.x.copy(),
                age_s=trk.age_s,
                hits=trk.hits,
                status=trk.status,
            )
            outputs.append(out)

        outputs.sort(key=lambda t: (t.track_id, t.label))
        return outputs
