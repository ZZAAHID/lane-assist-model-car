import cv2
import time
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

    # You can toggle the bird's eye view warp on or off here.
    # Set use_birds_eye=False to process the image from the camera's native perspective.
    lane_detector = LaneDetector(use_birds_eye=True)
    yolo_detector = YoloDetector()

    gui_enabled = True
    print("Starting Autonomous Loop. Press Ctrl+C to exit.")
    
    def active_delay(duration, message):
        nonlocal gui_enabled
        start_t = time.time()
        while time.time() - start_t < duration:
            try:
                fr_rgb = picam2.capture_array()
                fr = cv2.cvtColor(fr_rgb, cv2.COLOR_RGB2BGR)
                fr = cv2.flip(fr, -1) # Flip vertically and horizontally
                
                # We need to run YOLO during the delay so we can still display YOLO annotations
                _, _, _, annot = yolo_detector.detect(fr)
            except Exception:
                continue
            
            # Let's also draw the lanes quickly during the delay
            _, annot = lane_detector.process(annot)    
                
            cv2.putText(annot, message, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            if gui_enabled:
                try:
                    cv2.imshow('Autonomous Assist View', annot)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        return False
                except cv2.error:
                    print("\n[WARNING] OpenCV GUI not supported (headless mode). Disabling video output.")
                    gui_enabled = False
                    
        return True

    try:
        frame_counter = 0
        last_sign = None
        last_obstacle = False
        last_pedestrian = False
        last_yolo_frame = None

        while True:
            try:
                # Capture frame as RGB array, then convert to BGR for OpenCV
                frame_rgb = picam2.capture_array()
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                frame = cv2.flip(frame, -1) # Flip vertically and horizontally
            except Exception as e:
                print(f"Failed to grab frame from camera. Exiting. Error: {e}")
                break

            # Run YOLO unified detection only every 5 frames to save massive CPU cycles
            frame_counter += 1
            if frame_counter % 5 == 1 or last_yolo_frame is None:
                sign, obstacle_detected, pedestrian_detected, yolo_annotated_frame = yolo_detector.detect(frame)
                last_sign = sign
                last_obstacle = obstacle_detected
                last_pedestrian = pedestrian_detected
                last_yolo_frame = yolo_annotated_frame
            else:
                sign = last_sign
                obstacle_detected = last_obstacle
                pedestrian_detected = last_pedestrian
                yolo_annotated_frame = last_yolo_frame
                # Draw old boxes on the NEW frame to keep the feed looking smooth
                # (You could do a complex tracking algorithm here, but pasting old boxes is fine for 5 frames)
                pass

            # 0. PEDESTRIAN DETECTION (Highest Priority)
            if pedestrian_detected:
                print("PEDESTRIAN DETECTED! Stopping.")
                car.stop()
                if not active_delay(0.5, "PEDESTRIAN - STOPPED"): break
                continue

            # 1. OBSTACLE DETECTION (High Priority)
            if obstacle_detected:
                print("OBSTACLE DETECTED! Initiating overtake.")
                
                # 1. Swerve Right to change lane
                car.move(0.6, 0.2)
                if not active_delay(1.0, "OVERTAKING - SWERVE RIGHT"): break
                
                # 2. Straight to pass
                car.move(0.5, 0.5)
                if not active_delay(1.5, "OVERTAKING - PASSING"): break
                
                # 3. Swerve Left to re-enter lane
                car.move(0.2, 0.6)
                if not active_delay(1.0, "OVERTAKING - RETURN LEFT"): break
                
                continue
                
            # 2. SIGN DETECTION (High Priority)
            if sign == "STOP":
                print("STOP SIGN DETECTED! Stopping for 3 seconds.")
                car.stop()
                
                # Wait for 3 seconds while keeping the camera feed alive
                if not active_delay(3.0, "STOP SIGN - WAITING"): break

                # Keep moving slightly forward to pass the sign for 1 second
                print("Proceeding forward...")
                car.move(0.5, 0.5)
                if not active_delay(1.0, "PROCEEDING"): break
                continue

            # 3. LANE DETECTION & CONTROL
            # Pass the yolo_annotated_frame into the lane detector so it draws lane lines
            # on top of the already YOLO-marked frame.
            steering_offset, final_composite_frame = lane_detector.process(yolo_annotated_frame)
            
            # Simple Proportional Control (P-Controller)
            if steering_offset is None:
                # No lane detected, stop the car
                car.stop()
                cv2.putText(final_composite_frame, "NO LANE DETECTED - STOPPED", (20, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                # Lower base speed for better cornering control
                base_speed = 0.4 
                # Drastically increase steering authority so the inner wheel can stop or reverse
                steering_factor = 0.55 
                
                # steering_offset is between -1.0 (left) and 1.0 (right)
                left_speed = base_speed + (steering_offset * steering_factor)
                right_speed = base_speed - (steering_offset * steering_factor)
                
                car.move(left_speed, right_speed)

            # Display output
            if gui_enabled:
                try:
                    cv2.imshow('Autonomous Assist View', final_composite_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except cv2.error:
                    print("\n[WARNING] OpenCV GUI not supported (headless mode). Disabling video output.")
                    gui_enabled = False

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
        try:
            cv2.destroyAllWindows()
        except:
            pass
        print("Shutdown complete.")

if __name__ == '__main__':
    main()
