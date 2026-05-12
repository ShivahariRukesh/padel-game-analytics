import json
import csv
import os

import cv2
import numpy as np


VELOCITY_SMOOTH_WINDOW   = 4    
MIN_VELOCITY_PX          = 8    
MIN_FRAMES_BETWEEN_HITS  = 18   
SHOT_LABEL_WINDOW        = 6    
FLASH_DURATION_FRAMES    = 14   



class ShotCounter:
    
    def __init__(self, fps: float = 24.0):
        self.fps = fps

        
        self.shot_log: list[dict] = []          
        self.forehand_count  = 0
        self.backhand_count  = 0
        self.per_player: dict[str, dict] = {}   

        
        self._frame_counts: dict[int, tuple] = {}  
        self._strike_index: dict[int, dict]  = {}  

    
    def analyze(
        self,
        ball_detections:      list[dict],
        player_detections:    list[dict],
        shot_type_detections: list[dict],
    ) -> list[dict]:
        """
        Runs the full analysis pipeline and populates shot_log.

        Parameters
        ----------
        ball_detections      : list of {1: [x1,y1,x2,y2]} per frame
        player_detections    : list of {track_id: [x1,y1,x2,y2], ...} per frame
        shot_type_detections : list of {shot_class: [x1,y1,x2,y2], ...} per frame

        Returns
        -------
        shot_log : list of dicts with keys frame / timestamp / shot_type / player_id
        """
        n = min(
            len(ball_detections),
            len(player_detections),
            len(shot_type_detections)
        )

        ball_detections = ball_detections[:n]
        player_detections = player_detections[:n]
        shot_type_detections = shot_type_detections[:n]
        
        
        raw_centres = [self._bbox_centre(list(bd.values())[0]) if bd else None
                       for bd in ball_detections]
        centres = self._interpolate(raw_centres)

        
        vx = self._smoothed_vx(centres)

        
        last_hit = -MIN_FRAMES_BETWEEN_HITS
        for i in range(VELOCITY_SMOOTH_WINDOW + 1, n - VELOCITY_SMOOTH_WINDOW):
            if i - last_hit < MIN_FRAMES_BETWEEN_HITS:
                continue

            v_prev = vx[i - VELOCITY_SMOOTH_WINDOW]
            v_curr = vx[i]
            if v_prev is None or v_curr is None:
                continue

            if (abs(v_prev) > MIN_VELOCITY_PX
                    and abs(v_curr) > MIN_VELOCITY_PX
                    and v_prev * v_curr < 0):                 

                shot_type = self._nearest_shot_label(shot_type_detections, i)
                if shot_type is None:
                    continue    

                player_id = self._nearest_player(centres[i], player_detections[i])
                timestamp = round(i / self.fps, 3)

                self._record(i, timestamp, shot_type, player_id)
                last_hit = i

        
        self._build_frame_counts(n)

        print(f"[ShotCounter] analysis complete — "
              f"{self.forehand_count} forehands, {self.backhand_count} backhands "
              f"({len(self.shot_log)} total hits detected)")

        return self.shot_log

    
    def draw_overlay_all(self, frames: list) -> list:
        """Stamp the running scoreboard (and strike flash) onto every frame."""
        output = []
        for i, frame in enumerate(frames):
            self._draw_scoreboard(frame, i)
            
            output.append(frame)
        return output

    def draw_overlay(self, frame, frame_num: int):
        """Single-frame version (useful if you loop yourself)."""
        self._draw_scoreboard(frame, frame_num)
        
        return frame

    
    def export_json(self, path: str = "output_data/shot_log.json") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "summary": {
                "total_shots":     len(self.shot_log),
                "forehand_total":  self.forehand_count,
                "backhand_total":  self.backhand_count,
                "per_player":      self.per_player,
            },
            "shots": self.shot_log,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[ShotCounter] JSON saved → {path}")

    def export_csv(self, path: str = "output_data/shot_log.csv") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["frame", "timestamp", "shot_type", "player_id"]
            )
            writer.writeheader()
            writer.writerows(self.shot_log)
        print(f"[ShotCounter] CSV saved  → {path}")

    
    @staticmethod
    def _bbox_centre(bbox) -> tuple:
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    @staticmethod
    def _interpolate(centres: list) -> list:
        """Linear interpolation over None gaps."""
        result = list(centres)
        n = len(result)
        valid = [(i, c) for i, c in enumerate(result) if c is not None]
        if not valid:
            return result
        for i in range(n):
            if result[i] is not None:
                continue
            prev = next(((vi, vc) for vi, vc in reversed(valid) if vi < i), None)
            nxt  = next(((vi, vc) for vi, vc in valid if vi > i), None)
            if prev and nxt:
                t = (i - prev[0]) / (nxt[0] - prev[0])
                result[i] = (
                    prev[1][0] + t * (nxt[1][0] - prev[1][0]),
                    prev[1][1] + t * (nxt[1][1] - prev[1][1]),
                )
        return result

    def _smoothed_vx(self, centres: list) -> list:
        """
        Rolling-window horizontal velocity:
        vx[i] = centre_x[i + W] - centre_x[i - W]
        where W = VELOCITY_SMOOTH_WINDOW.
        """
        w = VELOCITY_SMOOTH_WINDOW
        n = len(centres)
        vx = [None] * n
        for i in range(w, n - w):
            c_back = centres[i - w]
            c_fwd  = centres[i + w]
            if c_back is not None and c_fwd is not None:
                vx[i] = c_fwd[0] - c_back[0]
        return vx

    def _nearest_shot_label(self, detections: list[dict], frame: int) -> str | None:
        """
        Search a ±SHOT_LABEL_WINDOW frame band for 'forehand' or 'backhand'.
        Prioritise frames at or after the strike, then before.
        """
        lo = max(0, frame - SHOT_LABEL_WINDOW)
        hi = min(len(detections), frame + SHOT_LABEL_WINDOW + 1)

        for i in list(range(frame, hi)) + list(range(frame - 1, lo - 1, -1)):
            for label in detections[i]:
                if label.lower() in ("forehand", "backhand"):
                    return label.lower()
        return None

    def _nearest_player(self, ball_centre, player_dict: dict) -> str:
        """Return the track_id of the player closest to the ball."""
        if ball_centre is None or not player_dict:
            return "unknown"
        best_id, best_dist = "unknown", float("inf")
        for track_id, bbox in player_dict.items():
            if track_id == "racket":
                continue
            px = (bbox[0] + bbox[2]) / 2
            py = (bbox[1] + bbox[3]) / 2
            d  = np.hypot(ball_centre[0] - px, ball_centre[1] - py)
            if d < best_dist:
                best_dist = best_id = None  
                best_dist = d
                best_id   = str(track_id)
        return best_id or "unknown"

    
    def _record(self, frame: int, timestamp: float, shot_type: str, player_id: str) -> None:
        entry = {
            "frame":     frame,
            "timestamp": timestamp,
            "shot_type": shot_type,
            "player_id": player_id,
        }
        self.shot_log.append(entry)
        self._strike_index[frame] = entry

        if shot_type == "forehand":
            self.forehand_count += 1
        else:
            self.backhand_count += 1

        if player_id not in self.per_player:
            self.per_player[player_id] = {"forehand": 0, "backhand": 0}
        self.per_player[player_id][shot_type] = (
            self.per_player[player_id].get(shot_type, 0) + 1
        )

    def _build_frame_counts(self, n: int) -> None:
        """Pre-compute (forehand_count, backhand_count) up to each frame."""
        sorted_shots = sorted(self.shot_log, key=lambda x: x["frame"])
        idx = 0
        fh = bh = 0
        for f in range(n):
            while idx < len(sorted_shots) and sorted_shots[idx]["frame"] <= f:
                if sorted_shots[idx]["shot_type"] == "forehand":
                    fh += 1
                else:
                    bh += 1
                idx += 1
            self._frame_counts[f] = (fh, bh)

    
    def _draw_scoreboard(self, frame, frame_num: int) -> None:
        """Persistent top-left HUD showing running totals."""
        fh, bh = self._frame_counts.get(frame_num, (0, 0))

        x0, y0, w, h = 10, 10, 270, 115

        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        
        cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (180, 180, 180), 1)

        
        cv2.putText(frame, "SHOT TRACKER",
                    (x0 + 48, y0 + 27),
                    cv2.FONT_HERSHEY_COMPLEX, 0.55, (230, 230, 230), 1,
                    cv2.LINE_AA)

        
        cv2.putText(frame, f"FH  Forehand :  {fh}",
                    (x0 + 14, y0 + 62),
                    cv2.FONT_HERSHEY_COMPLEX, 0.72, (50, 230, 100), 2,
                    cv2.LINE_AA)

        
        cv2.putText(frame, f"BH  Backhand :  {bh}",
                    (x0 + 14, y0 + 100),
                    cv2.FONT_HERSHEY_COMPLEX, 0.72, (50, 200, 255), 2,
                    cv2.LINE_AA)

    def _draw_flash(self, frame, frame_num: int) -> None:
        """
        For FLASH_DURATION_FRAMES after a detected strike, display a
        centred banner announcing the shot type.
        """
        
        active_shot = None
        for f in range(max(0, frame_num - FLASH_DURATION_FRAMES), frame_num + 1):
            if f in self._strike_index:
                active_shot = self._strike_index[f]   

        if active_shot is None:
            return

        label = active_shot["shot_type"].upper()
        pid   = active_shot["player_id"]
        color = (50, 230, 100) if label == "FOREHAND" else (50, 200, 255)

        h_frame, w_frame = frame.shape[:2]
        text       = f"{label}!"
        player_txt = f"Player {pid}"

        
        frames_since = frame_num - active_shot["frame"]
        alpha = max(0.0, 1.0 - frames_since / FLASH_DURATION_FRAMES)

        overlay = frame.copy()

        
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1.6, 3)
        tx = (w_frame - tw) // 2
        ty = h_frame // 4

        cv2.putText(overlay, text, (tx, ty),
                    cv2.FONT_HERSHEY_COMPLEX, 1.6, color, 3, cv2.LINE_AA)

        
        (pw, _), _ = cv2.getTextSize(player_txt, cv2.FONT_HERSHEY_COMPLEX, 0.8, 2)
        cv2.putText(overlay, player_txt, ((w_frame - pw) // 2, ty + 42),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)