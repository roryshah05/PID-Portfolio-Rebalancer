# Test file to ensure P, I and D are computed correctly

from pid_controller import PIDController
Kp = 0
Ki = 0
Kd = 1

t = 0
pid = PIDController(Kp, Ki, Kd, -100, 100)
while t <= 10:
    u = pid.step(10,1)
    print(u)
    t = t+1
