from gpiozero import Motor
from gpiozero.pins.lgpio import LGPIOFactory
from gpiozero import Device
# Crucial for Raspberry Pi 5!
# Forces gpiozero to use the lgpio library backend to talk to the RP1 chip.
Device.pin_factory = LGPIOFactory()

class Car:
    def __init__(self, left_pins=(24, 23, 25), right_pins=(22, 27, 17)):
        '''
        Initializes the car with an L298N Motor Driver.
        left_pins: (in1, in2, en_A)
        right_pins: (in3, in4, en_B)
        
        Note: on Raspberry Pi 5, we use gpiozero which utilizes the lgpio backend
        to successfully talk to the RP1 chip natively.
        '''
        # forward and backward are swapped to fix reversed direction
        self.left_motor = Motor(
            forward=right_pins[1],   # swapped
            backward=right_pins[0],  # swapped
            enable=right_pins[2],
            pwm=True
        )
        self.right_motor = Motor(
            forward=left_pins[1],    # swapped
            backward=left_pins[0],   # swapped
            enable=left_pins[2],
            pwm=True
        )

    def move(self, left_speed, right_speed):
        '''
        Drive the motors.
        Speeds should be between -1.0 (full backward) and 1.0 (full forward).
        Automatically compensates for the motor deadband (40% PWM minimum).
        '''
        left_speed = max(min(left_speed, 1.0), -1.0)
        right_speed = max(min(right_speed, 1.0), -1.0)

        min_pwm = 0.40

        def apply_deadband(speed):
            if abs(speed) < 0.05:
                return 0.0
            mapped_speed = min_pwm + (abs(speed) * (1.0 - min_pwm))
            return mapped_speed if speed > 0 else -mapped_speed

        actual_left = apply_deadband(left_speed)
        actual_right = apply_deadband(right_speed)

        # Control left motor
        if actual_left >= 0:
            self.left_motor.forward(actual_left)
        else:
            self.left_motor.backward(abs(actual_left))

        # Control right motor
        if actual_right >= 0:
            self.right_motor.forward(actual_right)
        else:
            self.right_motor.backward(abs(actual_right))

    def stop(self):
        '''Stop both motors.'''
        self.left_motor.stop()
        self.right_motor.stop()
