import numpy as np

# Inialising the PID Controller Class
class PIDController:
    def __init__(self, Kp, Ki, Kd, minintegral, maxintegral):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.minintegral = minintegral
        self.maxintegral = maxintegral
        self.integral = None # Accumulated integral sum
        self.prev_error = None # The error from the previous timestep

    def step(self, error, dt):
        if self.integral is None:
            self.integral = np.zeros_like(error)
            self.prev_error = np.zeros_like(error)

        u_P = self.Kp * error # Calculating the proportion term
        integral = self.integral + error * dt
        u_I = self.Ki * integral # Calculating the integral term
        u_D = self.Kd * ((error - self.prev_error) / dt) # Calculating the derivative term

        u = u_P + u_I + u_D

        self.prev_error = error # Updating the previous error term
        self.integral = np.clip(integral, self.minintegral, self.maxintegral) # Updating the accumulated integral term

        return(u)