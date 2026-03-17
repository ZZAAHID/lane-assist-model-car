import cv2
import numpy as np
import warnings

class LaneDetector:
    def __init__(self, use_birds_eye=True):
        # We assume a standard 640x480 resolution input from the cam
        self.width = 640
        self.height = 480
        self.use_birds_eye = use_birds_eye
        
    def process(self, frame):
        '''
        Processes the frame, detects lanes, and returns a steering logic
        value (-1.0 to 1.0) and the annotated frame.
        '''
        h, w = frame.shape[:2]
        
        # 0. Set up Bird's Eye View transformation parameters
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
        
        # 1. Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Morphological Line Extraction
        # --- CHANGE: Use larger kernel (71x71) for thicker black lines ---
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (71, 71))
        
        # For BLACK lines on WHITE surface:
        # blackhat isolates DARK features on BRIGHT backgrounds — this is our main signal
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # tophat isolates bright features — keep it but weight blackhat more
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        
        # --- CHANGE: Weight blackhat 2x more since lines are black on white ---
        combined_lines = cv2.addWeighted(blackhat, 2.0, tophat, 0.5, 0)
        
        # Blur to smooth
        blur = cv2.GaussianBlur(combined_lines, (9, 9), 0)
        
        # --- CHANGE: Raised threshold to 30 to reduce noise on bright white surface ---
        _, edges = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)
        
        # 3. Apply Perspective Transform (Bird's Eye View)
        if self.use_birds_eye:
            M = cv2.getPerspectiveTransform(src, dst)
            cropped_edges = cv2.warpPerspective(edges, M, (w, h), flags=cv2.INTER_NEAREST)
            annotated_frame = cv2.warpPerspective(frame, M, (w, h), flags=cv2.INTER_LINEAR)
        else:
            cropped_edges = edges.copy()
            annotated_frame = frame.copy()
        
        # 4. Discover Lane Pixels using Sliding Windows
        histogram = np.sum(cropped_edges[h//2:, :], axis=0)
        
        midpoint = int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        nwindows = 9
        window_height = int(h // nwindows)
        
        nonzero = cropped_edges.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        leftx_current = int(leftx_base)
        rightx_current = int(rightx_base)
        
        # --- CHANGE: Increased margin to 150 for wider black line tracks ---
        margin = 150
        # --- CHANGE: Lowered minpix to 20 so windows recenter more easily ---
        minpix = 20
        
        left_lane_inds = []
        right_lane_inds = []
        
        for window in range(nwindows):
            win_y_low = int(h - (window+1)*window_height)
            win_y_high = int(h - window*window_height)
            
            win_xleft_low = int(leftx_current - margin)
            win_xleft_high = int(leftx_current + margin)
            win_xright_low = int(rightx_current - margin)
            win_xright_high = int(rightx_current + margin)
            
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = int(np.mean(nonzerox[good_right_inds]))
                
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        left_fit = None
        right_fit = None
        
        # 5. Fit Polynomials
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', np.RankWarning)
            if len(leftx) > 10 and (np.max(lefty) - np.min(lefty)) > 10:
                left_fit = np.polyfit(lefty, leftx, 2)
            if len(rightx) > 10 and (np.max(righty) - np.min(righty)) > 10:
                right_fit = np.polyfit(righty, rightx, 2)
            
        ploty = np.linspace(0, h-1, h)
        
        left_fitx = None
        right_fitx = None
        center_fitx = None
        
        if left_fit is not None:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        if right_fit is not None:
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            
        # 6. Calculate Steering & Center
        mid_x = w // 2
        
        target_x = mid_x
        
        if left_fitx is not None and right_fitx is not None:
            center_fitx = (left_fitx + right_fitx) / 2
            target_x = center_fitx[-1]
        elif left_fitx is not None:
            target_x = left_fitx[-1] + (w // 3)
            center_fitx = left_fitx + (w // 3)
        elif right_fitx is not None:
            target_x = right_fitx[-1] - (w // 3)
            center_fitx = right_fitx - (w // 3)
            
        if center_fitx is None:
            steering_offset = None
        else:
            pixel_offset = target_x - mid_x
            steering_offset = pixel_offset / (w / 2)
            steering_offset = max(min(steering_offset, 1.0), -1.0)
        
        # 7. Draw the Path
        blank_annotated = np.zeros_like(annotated_frame)
        
        if center_fitx is not None:
            if left_fitx is not None and right_fitx is not None:
                pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
                pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
                pts = np.hstack((pts_left, pts_right))
                cv2.fillPoly(blank_annotated, np.int_([pts]), (0, 100, 0))
            
            if left_fitx is not None:
                pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))], np.int32)
                cv2.polylines(blank_annotated, pts_left, False, (0, 255, 0), 10)
            if right_fitx is not None:
                pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))], np.int32)
                cv2.polylines(blank_annotated, pts_right, False, (0, 255, 0), 10)
                
            pts_center = np.array([np.transpose(np.vstack([center_fitx, ploty]))], np.int32)
            cv2.polylines(blank_annotated, pts_center, False, (0, 0, 255), 5)
        
        if self.use_birds_eye:
            Minv = cv2.getPerspectiveTransform(dst, src)
            unwarped_annotations = cv2.warpPerspective(blank_annotated, Minv, (w, h), flags=cv2.INTER_LINEAR)
        else:
            unwarped_annotations = blank_annotated
        
        final_frame = cv2.addWeighted(frame, 1, unwarped_annotations, 0.5, 0)
        
        if self.use_birds_eye:
            cv2.polylines(final_frame, [src.astype(np.int32)], True, (0, 255, 255), 2)
            
        if steering_offset is not None:
            cv2.putText(final_frame, f"Steering: {steering_offset:.2f}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(final_frame, "Steering: None", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
        return steering_offset, final_frame
