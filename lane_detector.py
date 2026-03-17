import cv2
import numpy as np
import warnings

class LaneDetector:
    def __init__(self, use_birds_eye=False):
        self.width = 640
        self.height = 480
        self.use_birds_eye = use_birds_eye

    def process(self, frame):
        '''
        Processes the frame using adaptive threshold + Hough Lines.
        Returns steering offset (-1.0 to 1.0) and annotated frame.
        '''
        h, w = frame.shape[:2]

        # ── Bird's Eye Transform Setup ──────────────────────────────────────
        top_y = int(h * 0.55)
        src = np.float32([
            [int(w * 0.05), h],
            [int(w * 0.95), h],
            [int(w * 0.65), top_y],
            [int(w * 0.35), top_y]
        ])
        dst_margin = 0.25
        dst = np.float32([
            [int(w * dst_margin), h],
            [int(w * (1 - dst_margin)), h],
            [int(w * (1 - dst_margin)), 0],
            [int(w * dst_margin), 0]
        ])

        # ── STEP 1: Grayscale ───────────────────────────────────────────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── STEP 2: Adaptive Threshold ──────────────────────────────────────
        # Automatically adjusts to lighting changes — no manual tuning needed
        # THRESH_BINARY_INV: black lines → WHITE pixels, white surface → BLACK pixels
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Uses local neighbourhood average
            cv2.THRESH_BINARY_INV,            # Invert: black line becomes white
            blockSize=31,                     # Neighbourhood size (must be odd number)
            C=10                              # Subtract from mean — increase to reduce noise
        )

        # ── STEP 3: Bird's Eye Warp ─────────────────────────────────────────
        if self.use_birds_eye:
            M = cv2.getPerspectiveTransform(src, dst)
            warped_edges = cv2.warpPerspective(edges, M, (w, h), flags=cv2.INTER_NEAREST)
            annotated_frame = cv2.warpPerspective(frame, M, (w, h), flags=cv2.INTER_LINEAR)
        else:
            warped_edges = edges.copy()
            annotated_frame = frame.copy()

        # ── STEP 4: Hough Lines ─────────────────────────────────────────────
        # Only look at bottom 60% of image — ignore sky/horizon area
        roi_y_start = int(h * 0.4)
        roi = warped_edges[roi_y_start:, :]

        lines = cv2.HoughLinesP(
            roi,
            rho=1,
            theta=np.pi / 180,
            threshold=30,       # Minimum votes — lower = more sensitive
            minLineLength=40,   # Ignore lines shorter than this (pixels)
            maxLineGap=60       # Bridge gaps up to this size (pixels)
        )

        left_lines = []
        right_lines = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                y1 += roi_y_start  # Add back the ROI offset
                y2 += roi_y_start

                if x2 - x1 == 0:
                    continue  # Skip vertical lines to avoid division by zero

                slope = (y2 - y1) / (x2 - x1)

                if abs(slope) < 0.3:
                    continue  # Skip near-horizontal lines (noise)

                # In image coordinates: negative slope = left lane, positive = right lane
                if slope < 0:
                    left_lines.append((x1, y1, x2, y2, slope))
                else:
                    right_lines.append((x1, y1, x2, y2, slope))

        # ── STEP 5: Average Lines ───────────────────────────────────────────
        def average_line(lines, h):
            '''Averages multiple Hough segments into one clean lane line.'''
            if not lines:
                return None
            slopes = [l[4] for l in lines]
            xs = [l[0] for l in lines] + [l[2] for l in lines]
            ys = [l[1] for l in lines] + [l[3] for l in lines]
            avg_slope = np.mean(slopes)
            avg_x = np.mean(xs)
            avg_y = np.mean(ys)
            if avg_slope == 0:
                return None
            y_bottom = h
            y_top = int(h * 0.4)
            x_bottom = int((y_bottom - avg_y) / avg_slope + avg_x)
            x_top = int((y_top - avg_y) / avg_slope + avg_x)
            return (x_bottom, y_bottom, x_top, y_top)

        left_avg = average_line(left_lines, h)
        right_avg = average_line(right_lines, h)

        # ── STEP 6: Steering Offset ─────────────────────────────────────────
        mid_x = w // 2
        steering_offset = None
        lane_center_x = None

        if left_avg and right_avg:
            lane_center_x = (left_avg[0] + right_avg[0]) // 2
        elif left_avg:
            lane_center_x = left_avg[0] + (w // 3)
        elif right_avg:
            lane_center_x = right_avg[0] - (w // 3)

        if lane_center_x is not None:
            pixel_offset = lane_center_x - mid_x
            steering_offset = pixel_offset / (w / 2)
            steering_offset = max(min(steering_offset, 1.0), -1.0)

        # ── STEP 7: Draw Annotations ─────────────────────────────────────────
        final_frame = frame.copy()

        if self.use_birds_eye:
            Minv = cv2.getPerspectiveTransform(dst, src)

        def draw_lane_line(frame, line, color, Minv=None):
            if line is None:
                return
            if Minv is not None:
                pts = np.array([[line[0], line[1]], [line[2], line[3]]], dtype=np.float32).reshape(-1, 1, 2)
                pts_unwarped = cv2.perspectiveTransform(pts, Minv)
                cv2.line(frame, tuple(pts_unwarped[0][0].astype(int)), tuple(pts_unwarped[1][0].astype(int)), color, 5)
            else:
                cv2.line(frame, (line[0], line[1]), (line[2], line[3]), color, 5)

        draw_lane_line(final_frame, left_avg, (0, 255, 0), Minv if self.use_birds_eye else None)
        draw_lane_line(final_frame, right_avg, (0, 255, 0), Minv if self.use_birds_eye else None)

        # Draw lane center (red) and car center (blue)
        if lane_center_x is not None:
            cv2.circle(final_frame, (lane_center_x, h - 20), 10, (0, 0, 255), -1)   # Red = lane center
            cv2.circle(final_frame, (mid_x, h - 20), 10, (255, 0, 0), -1)            # Blue = car center
            cv2.line(final_frame, (mid_x, h - 20), (lane_center_x, h - 20), (0, 255, 255), 2)

        # Draw trapezoid guide
        if self.use_birds_eye:
            cv2.polylines(final_frame, [src.astype(np.int32)], True, (0, 255, 255), 2)

        # Steering info text
        if steering_offset is not None:
            direction = "LEFT" if steering_offset < -0.1 else "RIGHT" if steering_offset > 0.1 else "STRAIGHT"
            cv2.putText(final_frame, f"Steering: {steering_offset:.2f} ({direction})", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(final_frame, "NO LANE DETECTED", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return steering_offset, final_frame
