import time
from motor import Car

def test_motors():
    print("Initializing Motor Hardware Test...")
    try:
        car = Car()
    except Exception as e:
        print(f"Failed to initialize Car via motor.py: {e}")
        return

    print("\n--- Starting Individual Motor Test Sequence ---")
    
    try:
        # TEST 1: Left Motor Forward
        print("1. Left Motor: FORWARD (2 seconds)")
        car.move(left_speed=0.6, right_speed=0.0)
        time.sleep(2)
        car.stop()
        time.sleep(1)

        # TEST 2: Left Motor Backward
        print("2. Left Motor: BACKWARD (2 seconds)")
        car.move(left_speed=-0.6, right_speed=0.0)
        time.sleep(2)
        car.stop()
        time.sleep(1)

        # TEST 3: Right Motor Forward
        print("3. Right Motor: FORWARD (2 seconds)")
        car.move(left_speed=0.0, right_speed=0.6)
        time.sleep(2)
        car.stop()
        time.sleep(1)

        # TEST 4: Right Motor Backward
        print("4. Right Motor: BACKWARD (2 seconds)")
        car.move(left_speed=0.0, right_speed=-0.6)
        time.sleep(2)
        car.stop()
        time.sleep(1)

        # TEST 5: Both Motors Forward
        print("5. Both Motors: FORWARD (2 seconds)")
        car.move(left_speed=0.6, right_speed=0.6)
        time.sleep(2)
        car.stop()

        print("\nMotor Test Sequence Complete!")
        
    except KeyboardInterrupt:
        print("\nTest Interrupted by User.")
    finally:
        print("Shutting down motors.")
        car.stop()

if __name__ == "__main__":
    test_motors()
