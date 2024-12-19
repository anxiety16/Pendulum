import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation
import matplotlib.image as mpimg

def pendulum_equations(t, state, length, gravity):
    theta, omega = state
    dtheta_dt = omega
    domega_dt = - (gravity / length) * np.sin(theta)
    return [dtheta_dt, domega_dt]

def simulate_pendulum(length, initial_angle, initial_velocity, time_array):
    gravity = 9.81  # Acceleration due to gravity (m/s^2)
    initial_conditions = [initial_angle, initial_velocity]
    
    # Solve ODE using solve_ivp
    solution = solve_ivp(pendulum_equations, [time_array[0], time_array[-1]], 
                          initial_conditions, args=(length, gravity), t_eval=time_array)
    
    angles = solution.y[0]
    x_positions = length * np.sin(angles)
    y_positions = -length * np.cos(angles)
    
    return x_positions, y_positions

def setup_plot():
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 0.5])
    ax.set_title("Pendulum Motion")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    
    smiley_plot = ax.add_artist(plt.Circle((0, 0), 0.1))
    return fig, ax, smiley_plot  

def init_animation(line, smiley_plot):
    line.set_data([], [])
    smiley_plot.set_radius(0)
    return line, smiley_plot

def update_animation(frame, line, smiley_plot, x_positions, y_positions):
    line.set_data([0, x_positions[frame]], [0, y_positions[frame]])
    x, y = x_positions[frame], y_positions[frame]
    smiley_plot.set_radius(0.1)
    smiley_plot.set_center((x, y))
    return line, smiley_plot

def main():
    time_points = np.linspace(0, 10, 500)  # Time array (0 to 10 seconds)
    pendulum_length = 1.0
    initial_angle = np.pi / 4
    initial_velocity = 0

    x_positions, y_positions = simulate_pendulum(pendulum_length, initial_angle, initial_velocity, time_points)

    fig, ax, smiley_plot = setup_plot()  # Update to receive smiley_plot
    line, = ax.plot([], [], lw=3, color='brown')

    ani = animation.FuncAnimation(fig, update_animation, fargs=(line, smiley_plot, x_positions, y_positions),
                                  frames=len(time_points), init_func=lambda: init_animation(line, smiley_plot),
                                  interval=1000 * (time_points[1] - time_points[0]), blit=True, repeat=True)

    plt.show()

if __name__ == "__main__":
    main()
