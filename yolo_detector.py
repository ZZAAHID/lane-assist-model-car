class YoloDetector:
    def __init__(self, model_path='yolov8n.pt'):
        '''
        Initializes the YOLOv8 object detection model.
        This unified detector handles both obstacles (cars, people, etc.)
        and stop signs (class 11 in COCO).
        '''
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            print(f"Loaded YOLO model from {model_path}.")
            self.active = True
        except ImportError:
            print("ultralytics package not found. YOLO detection disabled. "
                  "Please install it using 'pip install ultralytics'.")
            self.active = False
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            self.active = False

    def detect(self, frame):
        '''
        Processes the frame through YOLO once and checks for:
        1. Stop Signs (COCO Class 11)
        2. Immediate Obstacles (Person=0, Car=2, Motorcycle=3, Bus=5, Dog=16, etc.)
        
        Returns a tuple: (sign_detected, obstacle_detected, annotated_frame)
        '''
        if not self.active:
            return None, False, frame

        # Run inference optimized for Raspberry Pi CPU:
        # - imgsz=160 forces it to downscale the image drastically before inference (huge speedup)
        # - half=False ensures we don't try to use FP16 (which is emulated slowly on Pi CPUs)
        # - device='cpu' ensures we don't waste time looking for a GPU
        results = self.model(frame, verbose=False, imgsz=160, half=False, device='cpu')
        result = results[0]
        
        h, w = frame.shape[:2]
        
        sign_detected = None
        obstacle_detected = False
        
        # We will annotate the frame with bounding boxes
        annotated_frame = result.plot()

        # Iterate through detected boxes
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            
            # Extract box coordinates (startX, startY, endX, endY)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            box_width = x2 - x1
            box_height = y2 - y1

            # 1. Stop Sign Detection
            if cls_id == 11 and conf > 0.2: # Class 11 is Stop Sign in COCO
                if box_width > 40 and box_height > 40: # Ignore tiny signs far away
                    sign_detected = "STOP"
            
            # 2. Obstacle Detection 
            # We want to detect persons, cars, and "random objects".
            # Exclude Stop Sign(11), TV(62), Cell Phone(67)
            ignored_classes = {11, 62, 67}
            if cls_id not in ignored_classes and conf > 0.3:
                # If the obstacle is large enough and near the lower portion of the frame
                if box_width > (w * 0.25) and y2 > (h * 0.5):
                    obstacle_detected = True

        return sign_detected, obstacle_detected, annotated_frame
