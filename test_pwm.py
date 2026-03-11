import time
from motor import Car

def test_pwm(car):
    print("Starting PWM Speed Test...")
    
    # Step up the speed gradually forward
    print("\n--- Testing Forward Speeds ---")
    speeds = [0.2, 0.4, 0.6, 0.8, 1.0]
    for speed in speeds:
        print(f"Testing speed: {speed * 100:.0f}% PWM")
        car.move(speed, speed)
        time.sleep(2.0)  # run for 2 seconds
        
    car.stop()
    time.sleep(1.0)
    
    # Step up the speed gradually backward
    print("\n--- Testing Backward Speeds ---")
    for speed in speeds:
        print(f"Testing speed: {speed * 100:.0f}% PWM (Reverse)")
        car.move(-speed, -speed)
        time.sleep(2.0)

    car.stop()
    time.sleep(1.0)
    
    # Test motors independently to ensure left/right are wired correctly
    print("\n--- Testing Left Motor Only (50%) ---")
    car.move(0.5, 0.0)
    time.sleep(2.0)
    car.stop()
    time.sleep(1.0)
    
    print("\n--- Testing Right Motor Only (50%) ---")
    car.move(0.0, 0.5)
    time.sleep(2.0)
    
    print("\nPWM Test Complete.")

if __name__ == "__main__":
    my_car = None
    try:
        print("Initializing internal L298N motor driver...")
        my_car = Car()
        time.sleep(1.0) # wait for initialization
        test_pwm(my_car)
    except KeyboardInterrupt:
        print("\nTest interrupted by user. Processing stop command...")
    finally:
        if my_car:
            my_car.stop()
            print("Car safely stopped.")
