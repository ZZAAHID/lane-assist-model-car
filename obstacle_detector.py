import cv2
import numpy as np

class ObstacleDetector:
    def __init__(self, model_path = 'MobileNetSSD_deploy.caffemodel', config_path = 'MobileNetSSD_deploy.prototxt'):
        '''
        Initializes obstacle detection.
        For robust detection, supply a pre-trained MobileNet SSD model:
        model_path = 'MobileNetSSD_deploy.caffemodel'
        config_path = 'MobileNetSSD_deploy.prototxt'
        
        If no model is supplied, it falls back to a simple contour size detection 
        in the direct path of the vehicle.
        '''
        self.net = None
        if model_path and config_path:
            try:
                self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)
                print("Loaded Neural Net for Obstacle Detection.")
            except Exception as e:
                print(f"Failed to load DNN model: {e}")
                
    def detect(self, frame):
        '''
        Returns True if an obstacle is immediately ahead, otherwise False.
        '''
        if self.net:
            # DNN Mode
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
            self.net.setInput(blob)
            detections = self.net.forward()
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.7:  # Adjusted confidence from 0.5 to 0.7 to reduce sensitivity
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # If box is large or near the bottom center, it's an immediate obstacle
                    box_width = endX - startX
                    box_height = endY - startY
                    if box_width > (w * 0.4) and endY > (h * 0.6): # Adjusted width > 0.4 and height near bottom > 0.6
                        return True
            return False

        else:
            # Fallback Pattern: simple color threshold / contour size in the central ROI
            # This is very naive and meant as a placeholder if you don't use AI sensors
            # or ultrasonic sensors (which are highly recommended).
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (21, 21), 0)
            
            # Simple thresholding
            _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)
            
            # Region of Interest directly in front of the car
            h, w = thresh.shape
            roi = thresh[int(h*0.5):h, int(w*0.3):int(w*0.7)] 
            
            # If a lot of dark objects are appearing as solid blobs in front
            obj_pixels = cv2.countNonZero(roi)
            if obj_pixels > (roi.size * 0.6): # Reduced sensitivity: from 40% obstructed to 60%
                return True
                
            return False
