import cv2
import numpy as np

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
        # Define source points (trapezoid on original frame)
        # We start near the bottom and go up to the horizon line.
        top_y = int(h * 0.55)
        src = np.float32([
            [int(w * 0.05), h],                # Bottom-left (inset to reduce edge distortion)
            [int(w * 0.95), h],                # Bottom-right
            [int(w * 0.65), top_y],            # Top-right
            [int(w * 0.35), top_y]             # Top-left
        ])
        
        # Define destination points (rectangle on top-down view)
        # Map them to a perfect rectangle in the middle of our warped frame
        dst_margin = 0.25 # 25% margin on both left and right (lane width = 50% of screen)
        dst = np.float32([
            [int(w * dst_margin), h],                 # Bottom-left
            [int(w * (1 - dst_margin)), h],           # Bottom-right
            [int(w * (1 - dst_margin)), 0],           # Top-right
            [int(w * dst_margin), 0]                  # Top-left
        ])
        
        # 1. Grayscale (Process original frame BEFORE warping)
        # This is CRITICAL because warping stretches features near the horizon massively.
        # Filtering before warping ensures standard thin lines are picked up cleanly everywhere.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Morphological Line Extraction
        # We use a 11x11 kernel, which is wide enough to cover standard model car lane lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        
        # Top-hat transforms isolate bright features on dark backgrounds
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        
        # Black-hat transforms isolate dark features on bright backgrounds
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # Combine both (so any line, whether black or white, is represented as bright pixels)
        combined_lines = cv2.add(tophat, blackhat)
        
        # Blur slightly to smooth the lines before thresholding
        blur = cv2.GaussianBlur(combined_lines, (5, 5), 0)
        
        # Threshold to get a clean binary image of strictly high-contrast thin features
        _, edges = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
        
        # 3. Apply Perspective Transform (Bird's Eye View)
        if self.use_birds_eye:
            M = cv2.getPerspectiveTransform(src, dst)
            # Use INTER_NEAREST for the binary edges to keep them crisp and prevent blurring
            cropped_edges = cv2.warpPerspective(edges, M, (w, h), flags=cv2.INTER_NEAREST)
            annotated_frame = cv2.warpPerspective(frame, M, (w, h), flags=cv2.INTER_LINEAR)
        else:
            cropped_edges = edges.copy()
            annotated_frame = frame.copy()
        
        # 4. Discover Lane Pixels using Sliding Windows
        # Take a histogram of the bottom half of the image
        histogram = np.sum(cropped_edges[h//2:, :], axis=0)
        
        # Find the peak of the left and right halves of the histogram
        midpoint = int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # Choose the number of sliding windows
        nwindows = 9
        window_height = int(h // nwindows)
        
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = cropped_edges.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Current positions to be updated for each window
        leftx_current = int(leftx_base)
        rightx_current = int(rightx_base)
        
        # Set the width of the windows +/- margin
        margin = 50
        # Set minimum number of pixels found to recenter window
        minpix = 50
        
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = int(h - (window+1)*window_height)
            win_y_high = int(h - window*window_height)
            
            win_xleft_low = int(leftx_current - margin)
            win_xleft_high = int(leftx_current + margin)
            win_xright_low = int(rightx_current - margin)
            win_xright_high = int(rightx_current + margin)
            
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = int(np.mean(nonzerox[good_right_inds]))
                
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        left_fit = None
        right_fit = None
        
        # 5. Fit Polynomials
        if len(leftx) > 0:
            left_fit = np.polyfit(lefty, leftx, 2)
        if len(rightx) > 0:
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
        y_eval = h # calculate steering from the bottom of the screen
        
        target_x = mid_x
        
        if left_fitx is not None and right_fitx is not None:
            center_fitx = (left_fitx + right_fitx) / 2
            target_x = center_fitx[-1]
        elif left_fitx is not None:
            # Hug left lane
            target_x = left_fitx[-1] + (w // 3)
            center_fitx = left_fitx + (w // 3)
        elif right_fitx is not None:
            target_x = right_fitx[-1] - (w // 3)
            center_fitx = right_fitx - (w // 3)
            
        if center_fitx is None:
            # Tell controller to STOP if no path exists
            steering_offset = None
        else:
            pixel_offset = target_x - mid_x
            steering_offset = pixel_offset / (w / 2)
            steering_offset = max(min(steering_offset, 1.0), -1.0)
        
        # 7. Draw the Path
        blank_annotated = np.zeros_like(annotated_frame)
        
        # Draw the lane region (Green Polygon)
        if center_fitx is not None:
            # Create points for left and right
            # If only one line is found, we approximate the other side
            if left_fitx is not None and right_fitx is not None:
                pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
                pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
                pts = np.hstack((pts_left, pts_right))
                cv2.fillPoly(blank_annotated, np.int_([pts]), (0, 100, 0)) # Faint green fill
            
            # Draw the solid lines
            if left_fitx is not None:
                pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))], np.int32)
                cv2.polylines(blank_annotated, pts_left, False, (0, 255, 0), 10)
            if right_fitx is not None:
                pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))], np.int32)
                cv2.polylines(blank_annotated, pts_right, False, (0, 255, 0), 10)
                
            # Draw center path line (Red)
            pts_center = np.array([np.transpose(np.vstack([center_fitx, ploty]))], np.int32)
            cv2.polylines(blank_annotated, pts_center, False, (0, 0, 255), 5)
        
        # Unwarp back to normal feed view if we used birds eye
        if self.use_birds_eye:
            Minv = cv2.getPerspectiveTransform(dst, src)
            unwarped_annotations = cv2.warpPerspective(blank_annotated, Minv, (w, h), flags=cv2.INTER_LINEAR)
        else:
            unwarped_annotations = blank_annotated
        
        # Combine the original frame with the unwarped lane overlays
        final_frame = cv2.addWeighted(frame, 1, unwarped_annotations, 0.5, 0)
        
        if self.use_birds_eye:
            # Draw the source trapezoid as a visual guide (yellow)
            cv2.polylines(final_frame, [src.astype(np.int32)], True, (0, 255, 255), 2)
            
        # Add visual text
        if steering_offset is not None:
            cv2.putText(final_frame, f"Steering: {steering_offset:.2f}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(final_frame, "Steering: None", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
        return steering_offset, final_frame
