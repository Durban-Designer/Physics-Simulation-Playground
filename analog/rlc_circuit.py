"""
Simulation of a simple RLC circuit, an electrical analog to a damped harmonic oscillator.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def simulate_rlc_circuit(R=1.0, L=1.0, C=1.0, initial_charge=1.0, initial_current=0.0, total_time=20.0):
    """
    Simulates the behavior of a series RLC circuit.

    Args:
        R (float): Resistance in ohms.
        L (float): Inductance in henries.
        C (float): Capacitance in farads.
        initial_charge (float): Initial charge on the capacitor.
        initial_current (float): Initial current in the circuit.
        total_time (float): Total time to simulate.

    Returns:
        A tuple containing the time array and the charge on the capacitor over time.
    """
    # The second-order ODE for an RLC circuit is L*d2q/dt2 + R*dq/dt + (1/C)*q = 0
    # We convert this to a system of two first-order ODEs:
    # dy1/dt = y2  (where y1=q, y2=dq/dt=I)
    # dy2/dt = -(R/L)*y2 - (1/(L*C))*y1

    def model(t, y):
        q, I = y
        dq_dt = I
        dI_dt = -(R / L) * I - (1 / (L * C)) * q
        return [dq_dt, dI_dt]

    y0 = [initial_charge, initial_current]
    t_span = [0, total_time]
    t_eval = np.linspace(0, total_time, 1000)

    sol = solve_ivp(model, t_span, y0, t_eval=t_eval)

    return sol.t, sol.y[0]

def plot_rlc_circuit(time, charge):
    """
    Plots the charge on the capacitor over time.

    Args:
        time (np.ndarray): The time array.
        charge (np.ndarray): The charge array.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(time, charge)
    plt.title("RLC Circuit Simulation")
    plt.xlabel("Time (s)")
    plt.ylabel("Charge on Capacitor (C)")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # Example: Underdamped oscillation
    time, charge = simulate_rlc_circuit(R=0.5, L=1.0, C=1.0)
    plot_rlc_circuit(time, charge)
