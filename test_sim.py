from plasma_sim import Plasma1D
from diagnostics import Diagnostics
import numpy as np

def test_simulation():
    print("Initializing Plasma1D...")
    sim = Plasma1D(L=0.05, N_grid=100, N_particles=1000, dt=1e-11)
    diag = Diagnostics()
    
    print("Running 10000 steps...")
    max_steps = 10000
    for i in range(max_steps):
        sim.step()
        diag.calculate(sim, i)
        
        # Check for NaNs
        if np.any(np.isnan(sim.rho)) or np.any(np.isnan(sim.phi)) or np.any(np.isnan(sim.E)):
            print(f"NaN detected at step {i}!")
            return
        
        # Check boundary conditions for potential
        # phi[0] should be 0, phi[-1] should match voltage source
        expected_V = sim.voltage_amp * np.sin(sim.omega * sim.current_time)
        # Note: solve_fields sets phi[-1] = V_right. 
        # But step() increments time at the BEGINNING. 
        # So solve_fields uses current_time.
        
        # Let's check if phi[-1] is close to expected
        if not np.isclose(sim.phi[-1], expected_V):
             print(f"Step {i}: Boundary potential mismatch. Got {sim.phi[-1]}, expected {expected_V}")

        if i % 100 == 0:
            print(f"Step {i}, Time: {sim.current_time:.2e}s, Max Ne: {np.max(sim.rho):.2e}")

    print(f"Simulation {max_steps} steps completed successfully.")
    print("Diagnostics last values:", diag.get_last())

if __name__ == "__main__":
    test_simulation()
