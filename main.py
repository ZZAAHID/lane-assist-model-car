import cv2
import time
import numpy as np
from motor import Car
from lane_detector import LaneDetector
from yolo_detector import YoloDetector

def main():
    print("Initialize Pi Car...")
    car = Car()
    
    # Initialize rpicam (via Picamera2 Python library)
    from picamera2 import Picamera2
    try:
        picam2 = Picamera2()
        # Configure video output
        config = picam2.create_video_configuration(main={"size": (640, 480)})
        picam2.configure(config)
        picam2.start()
    except Exception as e:
        print(f"Failed to initialize rpicam: {e}")
        return
    
    # Allow camera sensor to warm up
    time.sleep(2)

    lane_detector = LaneDetector()
    yolo_detector = YoloDetector()

    print("Starting Autonomous Loop. Press Ctrl+C to exit.")
    
    try:
        while True:
            try:
                # Capture frame as RGB array, then convert to BGR for OpenCV
                frame_rgb = picam2.capture_array()
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"Failed to grab frame from camera. Exiting. Error: {e}")
                break

            # Run YOLO unified detection
            sign, obstacle_detected, yolo_annotated_frame = yolo_detector.detect(frame)

            # 1. OBSTACLE DETECTION (High Priority)
            if obstacle_detected:
                car.stop()
                print("OBSTACLE DETECTED! Stopping.")
                
                # Show warning to prevent GUI from freezing
                cv2.putText(yolo_annotated_frame, "OBSTACLE WARNING", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.imshow('Lane Assist View', yolo_annotated_frame)
                
                hough_frame = np.zeros_like(yolo_annotated_frame)
                cv2.putText(hough_frame, "OBSTACLE WARNING", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.imshow('Hough Transform View', hough_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
                continue # Skip the rest of the loop until clear
                
            # 2. SIGN DETECTION (High Priority)
            if sign == "STOP":
                print("STOP SIGN DETECTED! Stopping for 3 seconds.")
                car.stop()
                
                # Wait for 3 seconds while keeping the camera feed alive
                start_time = time.time()
                while time.time() - start_time < 3.0:
                    try:
                        frame_rgb = picam2.capture_array()
                        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    except Exception:
                        pass
                    
                    annotated_frame = frame.copy()
                    cv2.putText(annotated_frame, "STOP SIGN - WAITING", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    cv2.imshow('Lane Assist View', annotated_frame)
                    
                    hough_frame = np.zeros_like(annotated_frame)
                    cv2.putText(hough_frame, "STOP SIGN - WAITING", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    cv2.imshow('Hough Transform View', hough_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Keep moving slightly forward to pass the sign for 1 second
                print("Proceeding forward...")
                car.move(0.5, 0.5)
                start_time = time.time()
                while time.time() - start_time < 1.0:
                    try:
                        frame_rgb = picam2.capture_array()
                        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    except Exception:
                        pass
                        
                    annotated_frame = frame.copy()
                    cv2.putText(annotated_frame, "PROCEEDING", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.imshow('Lane Assist View', annotated_frame)
                    
                    hough_frame = np.zeros_like(annotated_frame)
                    cv2.putText(hough_frame, "PROCEEDING", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.imshow('Hough Transform View', hough_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue

            # 3. LANE DETECTION & CONTROL
            steering_offset, annotated_frame, hough_frame = lane_detector.process(frame)
            
            # Simple Proportional Control (P-Controller)
            base_speed = 0.5 # Normal forward speed (50%)
            max_turn_reduction = 0.3 # Max speed reduction for turning
            
            # steering_offset is between -1.0 (left) and 1.0 (right)
            left_speed = base_speed + (steering_offset * max_turn_reduction)
            right_speed = base_speed - (steering_offset * max_turn_reduction)
            
            car.move(left_speed, right_speed)

            # Display output
            cv2.imshow('Lane Assist View', annotated_frame)
            cv2.imshow('Hough Transform View', hough_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user. Shutting down...")
    except Exception as e:
        print(f"\nCaught exception: {e}")
    finally:
        # Guarantee motors are stopped on exit
        car.stop()
        try:
            picam2.stop()
            picam2.close()
        except:
            pass
        cv2.destroyAllWindows()
        print("Shutdown complete.")

if __name__ == '__main__':
    main()
