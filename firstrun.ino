#include "CytronMotorDriver.h"

CytronMD motor1(PWM_DIR, 5, 4);
CytronMD motor2(PWM_DIR, 9, 6);
char sleep_status = 0;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600); // Set monitor baudrate to 9600

}

void loop() {
  // put your main code here, to run repeatedly:
   while (Serial.available() > 0) 
  {
    sleep_status = Serial.read();
    if(sleep_status == 'r')
    { 
        motor1.setSpeed(60); 
        motor2.setSpeed(-(60)); 
        Serial.write("LEFT");
    }
    else if(sleep_status == 'f')
    {
       
      
        motor1.setSpeed(60); 
        motor2.setSpeed(60); 
        Serial.write("FORWARD");
    }
    else if(sleep_status == 'l')
    {
       
      
        motor1.setSpeed(-(60)); 
        motor2.setSpeed(60); 
        Serial.write("RIGHT");
    }
//    else if(sleep_status == 'o')
//    {
//        motor1.setSpeed(-(60)); 
//        motor2.setSpeed(60); 
//        delay(1000);
//        motor1.setSpeed(60); 
//        motor2.setSpeed(-(60));
//        
//      
//    }
    else
    {
      motor1.setSpeed(0); 
      motor2.setSpeed(0);
    }
  }

}
