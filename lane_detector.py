import cv2
import numpy as np

class LaneDetector:
    def __init__(self):
        # We assume a standard 640x480 resolution input from the cam
        self.width = 640
        self.height = 480
        pass
        
    def process(self, frame):
        '''
        Processes the frame, detects lanes, and returns a steering logic
        value (-1.0 to 1.0) and the annotated frame.
        '''
        h, w = frame.shape[:2]
        
        # 0. Perspective Transform (Bird's Eye View)
        # Define source points (trapezoid on original frame)
        src = np.float32([
            [int(w * 0.1), h],                 # Bottom-left
            [int(w * 0.9), h],                 # Bottom-right
            [int(w * 0.65), int(h * 0.6)],     # Top-right
            [int(w * 0.35), int(h * 0.6)]      # Top-left
        ])
        
        # Define destination points (rectangle on top-down view)
        dst = np.float32([
            [int(w * 0.2), h],                 # Bottom-left
            [int(w * 0.8), h],                 # Bottom-right
            [int(w * 0.8), 0],                 # Top-right
            [int(w * 0.2), 0]                  # Top-left
        ])
        
        # Get perspective transform matrix and warp the frame
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(frame, M, (w, h), flags=cv2.INTER_LINEAR)
        annotated_frame = warped.copy()
        
        # 1. Grayscale & Blur (on warped image)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 2. Canny Edge Detection
        edges = cv2.Canny(blur, 50, 150)
        
        # 3. Create Region of Interest (Simplified because we already warped and isolated the perspective)
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, np.array([[[0, h], [w, h], [w, 0], [0, 0]]]), 255)
        cropped_edges = cv2.bitwise_and(edges, mask)
        
        # 4. Hough Line Transform
        lines = cv2.HoughLinesP(
            cropped_edges, 1, np.pi/180, 
            threshold=50, maxLineGap=50, minLineLength=40
        )
        
        hough_frame = cv2.cvtColor(cropped_edges, cv2.COLOR_GRAY2BGR)
        
        steering_offset = 0.0 # -1.0 is full left, 1.0 is full right
        
        left_avg = None
        right_avg = None
        mid_x = w // 2
        y_eval_bottom = h # evaluate the difference at the very bottom
        y_eval_top = int(h * 0.5)
        target_x = mid_x

        def get_points(avg_line):
            slope, intercept = avg_line
            x1 = int((y_eval_bottom - intercept) / slope)
            x2 = int((y_eval_top - intercept) / slope)
            return ((x1, y_eval_bottom), (x2, y_eval_top))

        if lines is not None:
            left_lines = []
            right_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Draw raw hough line
                cv2.line(hough_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                
                # Avoid divide by zero
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                
                # Classify by slope
                if slope < -0.3: # Left lane
                    left_lines.append((slope, intercept))
                elif slope > 0.3: # Right lane
                    right_lines.append((slope, intercept))
            
            # Average the lines
            left_avg = np.average(left_lines, axis=0) if left_lines else None
            right_avg = np.average(right_lines, axis=0) if right_lines else None

            if left_avg is not None:
                pt1, pt2 = get_points(left_avg)
                cv2.line(annotated_frame, pt1, pt2, (255, 0, 0), 5) # Blue left lane
            if right_avg is not None:
                pt1, pt2 = get_points(right_avg)
                cv2.line(annotated_frame, pt1, pt2, (0, 0, 255), 5) # Red right lane

            center_of_lane = mid_x # Default

            # 5. Calculate steering offset
            if left_avg is not None and right_avg is not None:
                # Both lanes detected
                left_x = (y_eval_bottom - left_avg[1]) / left_avg[0]
                right_x = (y_eval_bottom - right_avg[1]) / right_avg[0]
                center_of_lane = (left_x + right_x) / 2
                
                target_left_x = (y_eval_top - left_avg[1]) / left_avg[0]
                target_right_x = (y_eval_top - right_avg[1]) / right_avg[0]
                target_x = (target_left_x + target_right_x) / 2
            elif left_avg is not None:
                # Only left lane detected, estimate by adding lane width
                left_x = (y_eval_bottom - left_avg[1]) / left_avg[0]
                center_of_lane = left_x + (w // 3) # Assumption of lane width
                
                target_left_x = (y_eval_top - left_avg[1]) / left_avg[0]
                target_x = target_left_x + (w // 3)
            elif right_avg is not None:
                # Only right lane detected, estimate
                right_x = (y_eval_bottom - right_avg[1]) / right_avg[0]
                center_of_lane = right_x - (w // 3)
                
                target_right_x = (y_eval_top - right_avg[1]) / right_avg[0]
                target_x = target_right_x - (w // 3)
            else:
                target_x = mid_x
                
            pixel_offset = center_of_lane - mid_x
            steering_offset = pixel_offset / (w / 2)
            
            # Draw the target center path (Green)
            cv2.line(annotated_frame, (mid_x, y_eval_bottom), (int(target_x), y_eval_top), (0, 255, 0), 4)

        # Cap the steering offset at [-1, 1]
        steering_offset = max(min(steering_offset, 1.0), -1.0)
        
        # Unwarp back to normal feed view by calculating inverse transform matrix
        Minv = cv2.getPerspectiveTransform(dst, src)
        
        # You can either show the top-down view or composite back into original frame.
        # We will composite the lane markers back onto a copy of the original feed.
        final_frame = frame.copy()
        
        # Create a blank image and draw the annotated lines on it, then unwarp it
        blank_annotated = np.zeros_like(annotated_frame)
        if left_avg is not None:
            pt1, pt2 = get_points(left_avg)
            cv2.line(blank_annotated, pt1, pt2, (255, 0, 0), 10)
        if right_avg is not None:
            pt1, pt2 = get_points(right_avg)
            cv2.line(blank_annotated, pt1, pt2, (0, 0, 255), 10)
        # Center target
        cv2.line(blank_annotated, (mid_x, y_eval_bottom), (int(target_x), y_eval_top), (0, 255, 0), 8)
        
        unwarped_annotations = cv2.warpPerspective(blank_annotated, Minv, (w, h), flags=cv2.INTER_LINEAR)
        
        # Combine the original frame with the unwarped lane overlays
        final_frame = cv2.addWeighted(final_frame, 1, unwarped_annotations, 0.5, 0)
        
        # Draw the source trapezoid as a visual guide (yellow)
        cv2.polylines(final_frame, [src.astype(np.int32)], True, (0, 255, 255), 2)
        
        # Add visual text
        cv2.putText(final_frame, f"Steering: {steering_offset:.2f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
        return steering_offset, final_frame, hough_frame
