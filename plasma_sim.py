import sys
import os
import matplotlib

if not os.environ.get('DISPLAY') or '--headless' in sys.argv:
    matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from diagnostics import Diagnostics

class Plasma1D:
    def __init__(self, L=0.1, N_grid=100, N_particles=20000, dt=1e-11, frequency=13.56e6, voltage=100.0):
        """
        Initialize the 1D Electrostatic PIC simulation along the Z-axis.
        
        Parameters:
        L (float): System length (distance between electrodes) [m]
        N_grid (int): Number of grid points
        N_particles (int): Number of simulation particles
        dt (float): Time step [s]
        frequency (float): Frequency of the applied voltage [Hz]
        voltage (float): Amplitude of the applied voltage [V]
        """
        self.L = L
        self.Ng = N_grid
        self.Np = N_particles
        self.dt = dt
        self.dz = L / (N_grid - 1) # Note: N_grid points means N_grid-1 intervals
        self.frequency = frequency
        self.voltage_amp = voltage
        self.omega = 2 * np.pi * frequency
        self.current_time = 0.0
        
        # Physical constants (approximate for typical low temp plasma or normalized)
        # Using SI units for clarity unless we want to stick to normalized?
        # Let's stick to a consistent system. If previous was normalized, 
        # let's try to keep it simple but "Electrodes" implies Volts.
        # Let's use SI units effectively.
        self.e = 1.602e-19
        self.me = 9.109e-31
        self.eps0 = 8.854e-12
        
        # Grid arrays
        self.rho = np.zeros(N_grid)  # Charge density
        self.phi = np.zeros(N_grid)  # Potential
        self.E = np.zeros(N_grid)    # Electric field
        
        # Particle arrays (Phase space)
        # Initialize with a uniform distribution in z
        self.z = np.random.uniform(0, L, N_particles)
        
        # Velocity initialization (Maxwellian)
        self.Te_eV = 2.0 # Electron temperature in eV
        v_thermal = np.sqrt(2 * self.e * self.Te_eV / self.me)
        
        self.v = np.random.normal(0, v_thermal, N_particles)
        
        # Pre-compute matrix for Poisson solver (Finite Difference)
        # d2phi/dz2 = -rho/eps0
        # Discrete: (phi[i+1] - 2phi[i] + phi[i-1]) / dz^2 = -rho[i] / eps0
        # A * phi = b
        self.A = np.zeros((N_grid, N_grid))
        self.A[0, 0] = 1.0 # Boundary condition at z=0 (Ground)
        self.A[N_grid-1, N_grid-1] = 1.0 # Boundary condition at z=L (Driven)
        
        for i in range(1, N_grid-1):
            self.A[i, i-1] = 1.0
            self.A[i, i] = -2.0
            self.A[i, i+1] = 1.0
            
        self.A_inv = np.linalg.inv(self.A) # Pre-invert for speed (small grid)

    def weight_particles_to_grid(self):
        """
        Deposit particle charge onto the grid using 1st order linear weighting (PIC).
        """
        self.rho.fill(0.0)
        
        # Normalize positions to grid units
        z_norm = self.z / self.dz
        
        # Grid indices
        j = np.floor(z_norm).astype(int)
        
        # Filter out particles that are out of bounds (absorbed)
        # Although push_particles handles absorption, some might be initialized or numerical error
        mask = (j >= 0) & (j < self.Ng - 1)
        
        j = j[mask]
        w1 = z_norm[mask] - j 
        w0 = 1.0 - w1
        
        # Charge per simulation particle (superparticle)
        # Assume density ne ~ 1e16 m^-3
        n0 = 1e16
        q_real = n0 * self.L / self.Np * self.e # Area = 1 m^2 implicitly
        # Actually, let's just perform weighting.
        # To get real units, we need superparticle charge.
        # Let's define superparticle weight:
        w_sp = n0 * self.L / self.Np 
        q_sp = -self.e * w_sp # Electron charge
        
        # Add charge to grid
        # Note: self.rho is charge density * epsilon_0 factor? 
        # Poisson: d2phi/dx2 = -rho/eps0
        # We will compute rho as charge density [C/m^3]
        # rho[j] = sum(q_sp) / volume_cell = sum(q_sp) / (Area * dz)
        # Let Area = 1.
        
        np.add.at(self.rho, j, w0 * q_sp / self.dz)
        np.add.at(self.rho, j + 1, w1 * q_sp / self.dz)
        
        # Add background ion density (Uniform)
        rho_ion = self.e * n0
        self.rho += rho_ion

    def solve_fields(self):
        """
        Solve Poisson equation for potential phi and Electric field E.
        Using Finite Difference Method.
        """
        # RHS vector b
        # A * phi = b
        # Inner points: b[i] = -rho[i] / eps0 * dz^2
        # Boundary points: b[0] = V_left, b[-1] = V_right
        
        b = -self.rho * (self.dz**2) / self.eps0
        
        # Boundary Conditions
        V_left = 0.0
        V_right = self.voltage_amp * np.sin(self.omega * self.current_time)
        
        b[0] = V_left
        b[-1] = V_right
        
        # Solve A * phi = b
        self.phi = np.dot(self.A_inv, b)
        
        # Calculate Electric Field E = -dphi/dz
        # Central difference for inner points
        self.E[1:-1] = -(self.phi[2:] - self.phi[:-2]) / (2 * self.dz)
        # Forward/Backward for boundaries
        self.E[0] = -(self.phi[1] - self.phi[0]) / self.dz
        self.E[-1] = -(self.phi[-1] - self.phi[-2]) / self.dz

    def interpolate_fields_to_particles(self):
        """
        Interpolate E-field from grid back to particles.
        """
        z_norm = self.z / self.dz
        j = np.floor(z_norm).astype(int)
        
        # Handle particles exactly at the boundary or slightly out to avoid index error before removal
        j = np.clip(j, 0, self.Ng - 2)
        
        w1 = z_norm - j
        w0 = 1.0 - w1
        
        self.E_part = w0 * self.E[j] + w1 * self.E[j + 1]

    def push_particles(self):
        """
        Move particles using Leapfrog integration and apply boundary conditions.
        """
        # q/m for electron
        q_m = -self.e / self.me
        
        self.v += q_m * self.E_part * self.dt
        self.z += self.v * self.dt
        
        # Boundary Conditions: Absorb particles (remove)
        # Create a mask for particles inside the domain
        inside_mask = (self.z >= 0) & (self.z <= self.L)
        
        # Keep only particles inside
        self.z = self.z[inside_mask]
        self.v = self.v[inside_mask]
        self.E_part = self.E_part[inside_mask] # Although E_part is recalculated next step
        
        # Update particle count (for info, not strictly needed for logic if using arrays)
        # self.Np = len(self.z) 
        
        # Simple Reinjection (Optional - to keep simulation going for now)
        # If simulation empties, it's boring. Let's reinject absorbed particles at a random position with thermal velocity?
        # Or just let them die. Let's implement valid "maintain number" for stability if user didn't ask for discharge physics specifically.
        # User asked for "1D Z-axis with power". Real plasma would need ionization to sustain. 
        # Since we don't have ionization yet, let's keep Np constant by reinfecting Lost particles?
        # NO, user has another chat "Plasma Chemistry Simulation". Maybe they will add it later.
        # For now, let's just let them be absorbed. If they disappear, it's correct physics for vacuum.
        # BUT, to see anything, let's re-inject them as "new source" or assume periodic reinjection?
        # Let's reinject them thermally at the center or random to fake a source.
        
        num_lost = self.Np - len(self.z)
        if num_lost > 0:
            # Reinject
            z_new = np.random.uniform(0, self.L, num_lost)
            v_thermal = np.sqrt(2 * self.e * self.Te_eV / self.me)
            v_new = np.random.normal(0, v_thermal, num_lost)
            
            self.z = np.concatenate((self.z, z_new))
            self.v = np.concatenate((self.v, v_new))

    def step(self):
        """
        One complete time step of the simulation.
        """
        self.current_time += self.dt
        self.weight_particles_to_grid()
        self.solve_fields()
        self.interpolate_fields_to_particles()
        self.push_particles()

def run_simulation():
    # Simulation parameters
    # L = 0.05m (5cm), 100 grids. 
    # RF 13.56 MHz, 100V.
    sim = Plasma1D(L=0.05, N_grid=100, N_particles=20000, dt=1e-11, frequency=13.56e6, voltage=200.0)
    diag = Diagnostics()
    
    # Setup plotting
    if not os.environ.get('DISPLAY') or '--headless' in sys.argv:
        print("Running in headless mode (no display). Simulating 1000 steps...")
        max_steps = 1000
        for i in range(max_steps):
            sim.step()
            diag.calculate(sim, 0)
            if i % 100 == 0:
                print(f"Step {i}, Time: {sim.current_time:.3e} s")
                
        # create final static plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        ax1.set_xlim(0, sim.L)
        ax1.set_ylim(-3e6, 3e6) 
        ax1.set_xlabel('Position (z) [m]')
        ax1.set_ylabel('Velocity (v) [m/s]')
        ax1.set_title(f'Phase Space (z, v) - Final (Step {max_steps})')
        ax1.plot(sim.z, sim.v, 'k.', ms=0.2, alpha=0.2)
        
        times = diag.history['time']
        field_energy = diag.history['total_pe']
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Field Energy [J]')
        ax2.set_title('Field Energy vs Time')
        ax2.plot(times, field_energy, 'r-')
        if len(times) > 0:
            ax2.set_xlim(0, max(times[-1], 1e-9))
            ax2.set_ylim(0, max(max(field_energy), 1e-15) * 1.1)
            
        plt.tight_layout()
        plt.savefig('plasma_sim_result.png', dpi=150)
        print("Final plot saved to plasma_sim_result.png")
        return

    # Normal GUI Mode
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Phase Space Plot
    ax1.set_xlim(0, sim.L)
    # Estimate velocity range: v_th ~ sqrt(2*e*2/m) ~ 8e5 m/s. 
    # Oscillation v ~ eE/mw ~ e(V/L)/mw ~ 1.6e-19 * 4000 / (9e-31 * 8e7) ~ 8e6 m/s.
    # Set limit to +/- 5e6
    ax1.set_ylim(-3e6, 3e6) 
    ax1.set_xlabel('Position (z) [m]')
    ax1.set_ylabel('Velocity (v) [m/s]')
    ax1.set_title('Phase Space (z, v)')
    particles_plot, = ax1.plot([], [], 'k.', ms=0.2, alpha=0.2)
    
    # Diagnostics Plot (Field Energy History)
    ax2.set_xlim(0, 100 * sim.dt) # Initial view
    ax2.set_ylim(0, 1e-10) # Estimate
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Field Energy [J]')
    ax2.set_title('Field Energy vs Time')
    energy_line, = ax2.plot([], [], 'r-')
    
    # Text for time step
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)

    frames_per_update = 10 

    def init():
        particles_plot.set_data([], [])
        energy_line.set_data([], [])
        time_text.set_text('')
        return particles_plot, energy_line, time_text

    def animate(i):
        for _ in range(frames_per_update):
            sim.step()
            diag.calculate(sim, 0) # time_step argument is unused if sim.current_time is used
        
        # Update Phase Space
        particles_plot.set_data(sim.z, sim.v)
        
        # Update Diagnostics
        times = diag.history['time']
        field_energy = diag.history['total_pe'] # or 'avg_pe'
        
        energy_line.set_data(times, field_energy)
        
        # Dynamic scaling for energy plot
        if len(times) > 0:
            current_t = times[-1]
            # Keep last 5 RF cycles in view or expand? Let's just expand for now.
            ax2.set_xlim(0, max(current_t, 1e-9))
            ax2.set_ylim(0, max(max(field_energy), 1e-15) * 1.1)
        
        time_text.set_text(f'Time: {times[-1]:.3e} s')
        return particles_plot, energy_line, time_text

    # Create animation
    ani = animation.FuncAnimation(fig, animate, init_func=init, interval=30, blit=False) # blit=False for auto-scaling
    
    plt.show()

if __name__ == "__main__":
    run_simulation()
