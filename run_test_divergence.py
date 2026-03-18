import sys
sys.path.append(r"d:\Model Try\Plasma Simulation Start")
from simulation_engine import SimulationEngine
import numpy as np

def run_test():
    print("Initializing Engine...")
    engine = SimulationEngine()
    
    print("Running Steps...")
    for i in range(200000):
        data = engine.step()
        
        # Check for NaNs
        is_nan = False
        for key in data['densities']:
            if np.any(np.isnan(data['densities'][key])):
                print(f"NaN detected in density of {key} at step {i}!")
                is_nan = True
                break
        
        if np.any(np.isnan(data['phi'])):
             print(f"NaN detected in phi at step {i}!")
             is_nan = True
        if np.any(np.isnan(data['E'])):
             print(f"NaN detected in E at step {i}!")
             is_nan = True
             
        if is_nan:
             print("Diverged!")
             return
             
        if i % 1000 == 0:
            print(f"Step {i}, Time: {engine.t:.2e}, Max E: {np.max(np.abs(data['E'])):.2e}")

    print("Finished 200000 steps without diverging.")
    
if __name__ == "__main__":
    run_test()
