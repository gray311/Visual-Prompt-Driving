import numpy as np
from cvxpy import *

# Define the MPC function
def mpc_fun(v_ego, lead_info, N, dt, Q, R, Q_h, tau, desired_speed, desired_headway, desired_yaw_rate, yaw_rate_weight):
    s0 = 4.2
    min_headway = 1
    slack_penalty = 1e5  # Penalty coefficient for soft constraints

    d = Variable((N+1, 1))  # Distance state variable
    v = Variable((N+1, 1))  # Speed state variable
    u = Variable((N, 1))    # Acceleration control variable
    delta = Variable((N, 1))  # Steering angle control variable

    a_sigma = Variable((N, 1), nonneg=True)  # Slack variable for acceleration constraint
    d_sigma = Variable((N, 1), nonneg=True)  # Slack variable for distance constraint
    delta_sigma = Variable((N, 1), nonneg=True)  # Slack variable for steering angle constraint

    constraints = [d[0] == 0, v[0] == v_ego, delta[0] == 0]  # Initial conditions
    for i in range(N):
        constraints += [
            d[i+1] == d[i] + dt * v[i],  # Distance update
            v[i+1] == v[i] + dt * (u[i] / tau),  # Acceleration model
            v[i] <= desired_speed + a_sigma[i],  # Speed upper bound
            v[i] >= 0,  # Speed lower bound
            u[i] <= 2.5 + a_sigma[i],  # Acceleration upper bound
            u[i] >= -3 - a_sigma[i],  # Deceleration lower bound
            delta[i] >= -np.pi/4,  # Lower bound for right turn steering angle (negative for right turn)
            delta[i] <= np.pi/4  # Upper bound for left turn steering angle (positive for left turn)
        ]

    # Cost function
    cost = sum([Q * (v[i] - desired_speed) ** 2 + R * u[i] ** 2 + yaw_rate_weight * (delta[i] - desired_yaw_rate) ** 2 +
                slack_penalty * (a_sigma[i] + delta_sigma[i]) for i in range(N)])

    if lead_info:
        d0, lead_v = lead_info
        for i in range(N):
            expected_distance = d0 + lead_v * i * dt - d[i] - desired_headway * v[i] - s0
            cost += Q_h * (expected_distance) ** 2
            constraints += [
                d0 + lead_v * i * dt - d[i] - min_headway * v[i] - s0 >= 0
            ]

    prob = Problem(Minimize(cost), constraints)
    prob.solve()

    if prob.status != 'optimal':
        raise ValueError("MPC optimization problem did not solve successfully.")

    return d.value, v.value, u.value, delta.value

if __name__ == "__main__":
    # Example setup for deceleration and left turn
    v_ego = 15.0  # Initial speed in m/s
    N = 8  # Prediction horizon steps
    dt = 0.5  # Time step in seconds
    Q = 1.5  # Slightly increased speed maintenance weight for quicker deceleration
    R = 1.2  # Slightly increased control effort weight for smoother deceleration and turning
    Q_h = 1.0  # Headway maintenance weight (unchanged)
    tau = 0.5  # Engine lag time constant
    desired_speed = 10.0  # Lower desired speed for deceleration
    desired_headway = 1.5  # Desired headway in seconds
    desired_yaw_rate = 0.2  # Positive value for left turn (X-axis negative direction)
    yaw_rate_weight = 1.0  # Yaw rate maintenance weight

    lead_info = None  # No leading vehicle information

    # Run MPC with modified parameters
    d_values, v_values, u_values, delta_values = mpc_fun(v_ego, lead_info, N, dt, Q, R, Q_h, tau, desired_speed, desired_headway, desired_yaw_rate, yaw_rate_weight)

    # Analyze the new trajectory
    start_x, start_y = 0.0, 0.0
    trajectory = [(start_x, start_y)]
    for i in range(N):
        delta_x = -v_values[i] * dt * np.sin(delta_values[i])  # Left turn, x decreases
        delta_y = v_values[i] * dt * np.cos(delta_values[i])  # Forward motion, y increases
        new_x = trajectory[-1][0] + delta_x
        new_y = trajectory[-1][1] + delta_y
        trajectory.append((new_x[0], new_y[0]))

    print(trajectory)

