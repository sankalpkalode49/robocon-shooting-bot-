#include "CytronMotorDriver.h"

// Configure Cytron Motor Driver: PWM_DIR mode, PWM pin, DIR pin
CytronMD motor1(PWM_DIR, 5, 4);
CytronMD motor2(PWM_DIR, 9, 6);

char command = 0;

void setup() {
  // Initialize Serial Monitor at 9600 baud
  Serial.begin(9600); 
}

void loop() {
  // Check if data is available to read
  while (Serial.available() > 0) 
  {
    command = Serial.read();
    
    // 'r' = Rotate Right / Turn Right
    if(command == 'r')
    { 
        motor1.setSpeed(60); 
        motor2.setSpeed(-(60)); 
        Serial.write("LEFT"); // Debug response
    }
    // 'f' = Move Forward
    else if(command == 'f')
    {
        motor1.setSpeed(60); 
        motor2.setSpeed(60); 
        Serial.write("FORWARD");
    }
    // 'l' = Rotate Left / Turn Left
    else if(command == 'l')
    {
        motor1.setSpeed(-(60)); 
        motor2.setSpeed(60); 
        Serial.write("RIGHT");
    }
    // Stop if any other command or explicit stop is received
    else
    {
      motor1.setSpeed(0); 
      motor2.setSpeed(0);
    }
  }
}
