import numpy as np

class Diagnostics:
    def __init__(self):
        """
        Initialize diagnostics storage.
        """
        self.history = {
            'time': [],
            'total_ke': [],     # Total Kinetic Energy
            'avg_ke': [],       # Average Kinetic Energy per particle
            'total_pe': [],     # Total Field Energy (Potential Energy)
            'avg_pe': [],       # Average Field Energy per grid point
            'drift_v': [],      # Average velocity (Drift)
            'temperature': []   # Temperature (Variance of velocity)
        }

    def calculate(self, sim, time_step):
        """
        Calculate diagnostics for the current time step.
        
        Parameters:
        sim (Plasma1D): The simulation object.
        time_step (int): Current time step index.
        """
        # Kinetic Energy: 0.5 * m * v^2
        # Use simple mass if available, else 1.0
        mass = getattr(sim, 'me', 1.0)
        ke_per_particle = 0.5 * mass * sim.v**2
        total_ke = np.sum(ke_per_particle)
        avg_ke = np.mean(ke_per_particle)
        
        # Field Energy (Potential Energy): 0.5 * epsilon_0 * E^2 * Volume
        # In 1D: Integral(0.5 * eps0 * E^2) dA dz. Assume Unit Area.
        eps0 = getattr(sim, 'eps0', 1.0)
        dz = getattr(sim, 'dz', getattr(sim, 'dx', 1.0)) # Support both z and x/dx
        
        pe_density = 0.5 * eps0 * sim.E**2
        total_pe = np.sum(pe_density) * dz
        avg_pe = np.mean(pe_density)
        
        # Drift Velocity (Mean velocity)
        drift_v = np.mean(sim.v)
        
        # Temperature (Variance of velocity)
        # T in Kelvin ~ m * var(v) / kB? Or just T_eV?
        # Let's store T in eV if constants are physical, or variance if normalized.
        # k_B = 1.38e-23
        # T = m * <(v - <v>)^2> / k_B / 11600 (to eV?) or just J?
        # Let's just store Half-Width Squared or Variance for now to analyze heating.
        temperature = np.var(sim.v) # This is v_th^2

        # Store history
        self.history['time'].append(sim.current_time if hasattr(sim, 'current_time') else time_step * sim.dt)
        self.history['total_ke'].append(total_ke)
        self.history['avg_ke'].append(avg_ke)
        self.history['total_pe'].append(total_pe)
        self.history['avg_pe'].append(avg_pe)
        self.history['drift_v'].append(drift_v)
        self.history['temperature'].append(temperature)

    def get_last(self):
        """Return the most recently calculated values."""
        return {k: v[-1] if v else 0.0 for k, v in self.history.items()}
