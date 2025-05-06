"""
Electron Trajectory Simulation in Magnetic Fields

Author: Koshish Aryal, 1002157343

Description:
This script simulates the trajectory of charged particles (electrons/positrons) 
in uniform magnetic fields using classical Lorentz force dynamics. It generates
a 2x2 grid of plots showing how different parameters affect the trajectories:
1. Magnetic field strength variation
2. Kinetic energy variation
3. Particle mass variation
4. Charge sign comparison (electron vs positron)

The simulation uses numerical integration of the equations of motion with a 
fixed time step. Results are plotted in centimeters for better readability.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Fundamental Physical Constants
e = 1.602176634e-19      # Electron charge (Coulomb)
me = 9.10938356e-31      # Electron mass (kg)

# Simulation Parameters
dt = 1e-11               # Time step (seconds) - small enough for accurate integration
steps = 5000             # Maximum number of simulation steps
Ek_default = 20          # Default kinetic energy (electron volts)
B_default = 2e-3         # Default magnetic field strength (Tesla)
cutoff_distance = 20     # Simulation cutoff distance (meters) to prevent runaway trajectories

def simulate_trajectory(v0, q, m, Bz, steps=steps, dt=dt):
    """
    Simulate charged particle trajectory in a uniform magnetic field.
    
    Parameters:
        v0 (float): Initial velocity in x-direction (m/s)
        q (float): Particle charge (Coulomb)
        m (float): Particle mass (kg)
        Bz (float): Magnetic field strength in z-direction (Tesla)
        steps (int): Maximum number of simulation steps
        dt (float): Time step for numerical integration (seconds)
    
    Returns:
        tuple: (x_traj, y_traj) trajectory arrays in centimeters
    """
    pos = np.array([0.0, 0.0])  # Initial position [x, y] in meters
    vel = np.array([v0, 0.0])   # Initial velocity [vx, vy] in m/s
    x_traj = []
    y_traj = []

    for _ in range(steps):
        # Calculate Lorentz force: F = q(v × B)
        # Only Bz component is non-zero (field is along z-axis)
        F = q * np.cross(vel, [0, 0, Bz])[:2]  # Force in xy-plane
        
        # Update velocity and position using Newton's laws
        a = F / m               # Acceleration
        vel += a * dt           # Velocity update
        pos += vel * dt         # Position update
        
        # Store trajectory (convert meters to centimeters)
        x_traj.append(pos[0] * 100)
        y_traj.append(pos[1] * 100)
        
        # Stop simulation if particle moves too far from origin
        if np.linalg.norm(pos) > cutoff_distance:
            break
    
    return x_traj, y_traj

def setup_axes(ax, title):
    """
    Configure matplotlib axes with consistent styling.
    
    Parameters:
        ax (matplotlib.axes.Axes): Axis object to configure
        title (str): Plot title
    
    Returns:
        matplotlib.axes.Axes: Configured axis object
    """
    ax.set_title(title, pad=15)
    ax.set_xlabel('x position (cm)', labelpad=10)
    ax.set_ylabel('y position (cm)', labelpad=10)
    ax.grid(True, linestyle='--', alpha=0.6)  # Add dashed grid lines
    ax.set_aspect('equal')                   # Ensure circular trajectories appear circular
    ax.xaxis.set_major_locator(MultipleLocator(5))  # Major ticks every 5 cm
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1))  # Legend outside plot
    return ax

def main():
    """Main function to run the simulation and generate plots."""
    # Create figure with 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Electron Trajectories in Magnetic Fields\nKoshish Aryal, 1002157343', 
                 fontsize=16, y=1.02)

    # Color scheme for consistent visualization
    colors = plt.cm.viridis(np.linspace(0, 0.8, 3))

    # 1. Magnetic Field Strength Variation
    print("Simulating magnetic field variations...")
    for Bz, color in zip([1e-3, 2e-3, 5e-3], colors):
        v0 = np.sqrt(2 * Ek_default * e / me)  # Calculate initial velocity from energy
        x, y = simulate_trajectory(v0, -e, me, Bz)  # Negative charge for electrons
        axs[0, 0].plot(x, y, color=color, linewidth=2,
                      label=f'B = {Bz*1e3:.1f} mT')
    setup_axes(axs[0, 0], 'Varying Magnetic Field Strength')

    # 2. Kinetic Energy Variation
    print("Simulating energy variations...")
    for Ek, color in zip([10, 20, 40], colors):
        v0 = np.sqrt(2 * Ek * e / me)
        x, y = simulate_trajectory(v0, -e, me, B_default)
        axs[0, 1].plot(x, y, color=color, linewidth=2,
                      label=f'E = {Ek} eV')
    setup_axes(axs[0, 1], 'Varying Kinetic Energy')

    # 3. Particle Mass Variation
    print("Simulating mass variations...")
    for m, color in zip([me, 2*me, 4*me], colors):
        v0 = np.sqrt(2 * Ek_default * e / m)
        x, y = simulate_trajectory(v0, -e, m, B_default)
        axs[1, 0].plot(x, y, color=color, linewidth=2,
                      label=f'm = {m/me:.1f}mₑ')
    setup_axes(axs[1, 0], 'Varying Particle Mass')

    # 4. Charge Sign Comparison (Electron vs Positron)
    print("Simulating charge sign comparison...")
    v0 = np.sqrt(2 * Ek_default * e / me)
    # Electron trajectory (negative charge)
    x1, y1 = simulate_trajectory(v0, -e, me, B_default)
    # Positron trajectory (positive charge)
    x2, y2 = simulate_trajectory(v0, e, me, B_default)
    axs[1, 1].plot(x1, y1, color='blue', linewidth=2, label='Electron (q=-e)')
    axs[1, 1].plot(x2, y2, color='red', linewidth=2, label='Positron (q=+e)')
    setup_axes(axs[1, 1], 'Charge Sign Comparison')

    # Save and display results
    plt.tight_layout()
    output_file = 'electron_trajectories_koshish.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Simulation complete. Results saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    main()