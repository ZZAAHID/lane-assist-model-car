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
        # We set pwm=True to allow speed control via the Enable pins
        self.left_motor = Motor(
            forward=left_pins[0], 
            backward=left_pins[1], 
            enable=left_pins[2], 
            pwm=True
        )
        self.right_motor = Motor(
            forward=right_pins[0], 
            backward=right_pins[1], 
            enable=right_pins[2], 
            pwm=True
        )

    def move(self, left_speed, right_speed):
        '''
        Drive the motors.
        Speeds should be between -1.0 (full backward) and 1.0 (full forward).
        '''
        # Constrain values to [-1.0, 1.0]
        left_speed = max(min(left_speed, 1.0), -1.0)
        right_speed = max(min(right_speed, 1.0), -1.0)

        # Control left motor
        if left_speed >= 0:
            self.left_motor.forward(left_speed)
        else:
            self.left_motor.backward(abs(left_speed))
            
        # Control right motor
        if right_speed >= 0:
            self.right_motor.forward(right_speed)
        else:
            self.right_motor.backward(abs(right_speed))

    def stop(self):
        '''Stop both motors.'''
        self.left_motor.stop()
        self.right_motor.stop()
