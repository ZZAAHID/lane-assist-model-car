import cv2
import time
from motor import Car
from lane_detector import LaneDetector
from yolo_detector import YoloDetector


class PIDController:
    '''
    PID Controller for smooth lane steering.

    P (Proportional): Reacts to current error  → main steering force
    I (Integral):     Reacts to accumulated error → corrects long-term drift
    D (Derivative):   Reacts to rate of change  → dampens oscillation/wobble

    Tune these values:
    - If car WOBBLES left-right: reduce Kp or increase Kd
    - If car is SLOW to correct: increase Kp
    - If car DRIFTS to one side over time: increase Ki
    - Start with Ki=0 and Kd=0, tune Kp first
    '''
    def __init__(self, Kp=0.30, Ki=0.001, Kd=0.08):
        self.Kp = Kp   # Proportional gain
        self.Ki = Ki   # Integral gain
        self.Kd = Kd   # Derivative gain

        self.prev_error = 0.0
        self.integral = 0.0
        self.integral_limit = 0.5  # Clamp integral to prevent windup

    def compute(self, error, dt):
        '''
        Compute PID output given current error and time delta.
        Returns a correction value between -1.0 and 1.0.
        '''
        if dt <= 0:
            dt = 0.033  # Default to ~30fps if time is weird

        # Proportional term
        P = self.Kp * error

        # Integral term (with anti-windup clamping)
        self.integral += error * dt
        self.integral = max(min(self.integral, self.integral_limit), -self.integral_limit)
        I = self.Ki * self.integral

        # Derivative term (rate of change of error)
        derivative = (error - self.prev_error) / dt
        D = self.Kd * derivative

        self.prev_error = error

        output = P + I + D
        return max(min(output, 1.0), -1.0)  # Clamp to [-1, 1]

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0


def main():
    print("Initialize Pi Car...")
    car = Car()

    from picamera2 import Picamera2
    try:
        picam2 = Picamera2()
        config = picam2.create_video_configuration(main={"size": (640, 480)})
        picam2.configure(config)
        picam2.start()
    except Exception as e:
        print(f"Failed to initialize rpicam: {e}")
        return

    time.sleep(2)

    lane_detector = LaneDetector(use_birds_eye=True)
    yolo_detector = YoloDetector()

    # ── PID Controller ──────────────────────────────────────────────────────
    pid = PIDController(
        Kp=0.30,   # Proportional — main steering strength (tune this first)
        Ki=0.001,  # Integral — corrects long-term drift (keep small)
        Kd=0.08    # Derivative — dampens wobbling (increase if car oscillates)
    )

    # ── Speed Settings ──────────────────────────────────────────────────────
    BASE_SPEED = 0.30       # Overall forward speed (0.0 to 1.0)
    MAX_STEERING = 0.30     # Max speed difference between wheels during a turn

    gui_enabled = True
    last_time = time.time()

    print("Starting Autonomous Loop. Press Ctrl+C to exit.")

    def active_delay(duration, message):
        nonlocal gui_enabled
        start_t = time.time()
        while time.time() - start_t < duration:
            try:
                fr_rgb = picam2.capture_array()
                fr = cv2.cvtColor(fr_rgb, cv2.COLOR_RGB2BGR)
                fr = cv2.flip(fr, -1)
                _, _, _, annot = yolo_detector.detect(fr)
            except Exception:
                continue
            _, annot = lane_detector.process(annot)
            cv2.putText(annot, message, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            if gui_enabled:
                try:
                    cv2.imshow('Autonomous Assist View', annot)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        return False
                except cv2.error:
                    print("\n[WARNING] OpenCV GUI not supported. Disabling video output.")
                    gui_enabled = False
        return True

    try:
        frame_counter = 0
        last_sign = None
        last_obstacle = False
        last_pedestrian = False
        last_yolo_frame = None

        while True:
            # ── Capture Frame ───────────────────────────────────────────────
            try:
                frame_rgb = picam2.capture_array()
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                frame = cv2.flip(frame, -1)
            except Exception as e:
                print(f"Failed to grab frame: {e}")
                break

            # ── Compute dt for PID ──────────────────────────────────────────
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            # ── YOLO Detection (every 5 frames) ────────────────────────────
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

            # ── 0. PEDESTRIAN (Highest Priority) ────────────────────────────
            if pedestrian_detected:
                print("PEDESTRIAN DETECTED! Stopping.")
                car.stop()
                pid.reset()
                if not active_delay(0.5, "PEDESTRIAN - STOPPED"): break
                continue

            # ── 1. OBSTACLE ─────────────────────────────────────────────────
            if obstacle_detected:
                print("OBSTACLE DETECTED! Initiating overtake.")
                pid.reset()
                car.move(0.6, 0.2)
                if not active_delay(1.0, "OVERTAKING - SWERVE RIGHT"): break
                car.move(0.5, 0.5)
                if not active_delay(1.5, "OVERTAKING - PASSING"): break
                car.move(0.2, 0.6)
                if not active_delay(1.0, "OVERTAKING - RETURN LEFT"): break
                continue

            # ── 2. STOP SIGN ─────────────────────────────────────────────────
            if sign == "STOP":
                print("STOP SIGN DETECTED! Stopping for 3 seconds.")
                car.stop()
                pid.reset()
                if not active_delay(3.0, "STOP SIGN - WAITING"): break
                print("Proceeding forward...")
                car.move(0.5, 0.5)
                if not active_delay(1.0, "PROCEEDING"): break
                continue

            # ── 3. LANE DETECTION + PID STEERING ────────────────────────────
            steering_offset, final_composite_frame = lane_detector.process(yolo_annotated_frame)

            if steering_offset is None:
                # No lane found — stop and reset PID
                car.stop()
                pid.reset()
                cv2.putText(final_composite_frame, "NO LANE - STOPPED", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                # ── PID computes smooth correction ──────────────────────────
                correction = pid.compute(steering_offset, dt)

                # Apply correction to wheel speeds
                left_speed  = BASE_SPEED + (correction * MAX_STEERING)
                right_speed = BASE_SPEED - (correction * MAX_STEERING)

                # Clamp speeds to valid range [0.0, 1.0]
                left_speed  = max(0.0, min(1.0, left_speed))
                right_speed = max(0.0, min(1.0, right_speed))

                car.move(left_speed, right_speed)

                # Show PID info on screen
                cv2.putText(final_composite_frame, f"PID: {correction:.2f}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # ── Display ──────────────────────────────────────────────────────
            if gui_enabled:
                try:
                    cv2.imshow('Autonomous Assist View', final_composite_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except cv2.error:
                    print("\n[WARNING] OpenCV GUI not supported. Disabling video output.")
                    gui_enabled = False

    except KeyboardInterrupt:
        print("\nInterrupted by user. Shutting down...")
    except Exception as e:
        print(f"\nCaught exception: {e}")
    finally:
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
