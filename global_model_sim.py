import sys
import os
import matplotlib

# Check if we are running in a headless environment (like Rocky Linux terminal) or explicitly requested
if not os.environ.get('DISPLAY') or '--headless' in sys.argv:
    matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Physical Constants (SI Units)
e = 1.602e-19        # Elementary charge [C]
m_e = 9.109e-31      # Electron mass [kg]
M_i = 6.63e-26       # Ion mass (Argon approx 40 amu) [kg]
k_B = 1.38e-23       # Boltzmann constant [J/K]
eps0 = 8.854e-12     # Vacuum permittivity [F/m]

# Reactor Parameters (0D Assumption)
R = 0.15             # Radius [m]
L = 0.30             # Length [m]
Volume = np.pi * R**2 * L   # Volume [m^3]
Area = 2 * np.pi * R**2 + 2 * np.pi * R * L  # Surface Area [m^2]

# Neutral Gas Parameters
p_Torr = 0.01        # Pressure [Torr] (10 mTorr)
p_Pa = p_Torr * 133.322 # Pressure [Pa]
T_g = 300            # Gas temperature [K]
n_g = p_Pa / (k_B * T_g) # Neutral gas density [m^-3]

# Power Input
P_abs = 500.0        # Absorbed Power [W]

def reaction_rates(Te):
    """
    Calculate rate coefficients based on electron temperature Te [eV].
    Arrhenius type approximations for Argon.
    """
    # Safeguard for Te
    Te = max(Te, 0.1) # Minimum 0.1 eV
    
    # Ionization Rate (Lieberman & Lichtenberg approx for Argon)
    # K_iz = A * exp(-E_iz / Te)
    # E_iz = 15.76 eV for Argon
    # This is a simplified fit
    K_iz = 5e-14 * np.exp(-15.76 / Te) # [m^3/s]
    
    # Excitation Loss Energy (Effective)
    # Energy loss per collision (Ionization + Excitation + Elastic)
    # E_loss = E_iz + E_exc + 3/2 Te
    E_loss = 15.76 + 12.0 * np.exp(-5.0/Te) + 3*Te # [eV]
    
    return K_iz, E_loss

def bohm_velocity(Te):
    """Calculate Bohm velocity u_B = sqrt(e*Te/Mi). Te in eV."""
    Te = max(Te, 0.01)
    Te_J = Te * e # Convert eV to Joules
    return np.sqrt(Te_J / M_i)

def model_equations(y, t):
    """
    Differential equations for 0D Global Model.
    y = [n, Te]
    n: Electron/Ion density [m^-3]
    Te: Electron temperature [eV]
    """
    n, Te = y
    
    # Safety barriers
    if n < 1e10: n = 1e10 # Floor density
    if Te < 0.1: Te = 0.1 # Floor temperature
    
    # 1. Reaction Rates & Transport
    K_iz, E_loss = reaction_rates(Te)
    u_B = bohm_velocity(Te)
    
    # Effective Area (A_eff) for ion loss
    # h_L = 0.86 * (3 + L/2/lambda_i)^-0.5 (ignoring detailed fitting for simplified model)
    # Simple approx: h factors ~ 0.5 at low pressure
    h_R = 0.5 
    h_L = 0.5
    A_eff = 2 * np.pi * R**2 * h_L + 2 * np.pi * R * L * h_R
    
    # 2. Particle Balance: dn/dt
    # Production = n * n_g * K_iz * Volume
    # Loss = n * u_B * A_eff
    # dn/dt = (Production - Loss) / Volume
    
    dn_dt = n * n_g * K_iz - n * u_B * A_eff / Volume
    
    # 3. Energy Balance: dE/dt -> derive dTe/dt
    # Total Energy W = 3/2 * n * e * Te * Volume (Joules if Te in eV * e)
    # Power Balance: dW/dt = P_abs - P_loss
    # P_loss = P_collonal + P_surface
    # P_coll is absorbed in E_loss (eV) * production rate? 
    # Standard form: P_in = P_out
    # d(3/2 n e Te)/dt = (P_abs - P_loss_total) / V
    # 3/2 e (n dTe/dt + Te dn/dt) = (P_abs - e n n_g K_iz E_loss V - e n u_B A_eff (2Te + V_s) ) / V ??
    
    # Simplified version:
    # dTe/dt = (2 / (3 * n * e)) * (P_abs/V - P_loss_density) - (Te/n)*dn_dt
    
    # Loss per electron-ion pair creation (Collisional) + Kinetic loss at wall
    # P_loss_total = e * n * n_g * K_iz * E_loss * V + e * n * u_B * A_eff * (2 * Te + 5.0) 
    # (Assume wall sheath drop + thermal energy loss roughly 5-7 Te or similar)
    # Let's use E_c (Collisional energy loss) + E_w (Wall energy loss)
    
    E_c = E_loss # Collisional loss per ionization event (approx)
    E_w = 5.0 * Te # Kinetic energy loss per ion lost to wall (Mean ion energy + electron thermal energy)
    
    # Power Loss Density [W/m^3]
    # Ionization rate * Energy lost per create-loss cycle
    # At steady state, creation = loss.
    # Dynamically:
    
    P_loss_coll = n * n_g * K_iz * e * E_c # Collisional loss density
    P_loss_wall = (n * u_B * A_eff / Volume) * e * E_w # Wall loss density
    
    # Energy Equation
    # 3/2 n e dTe/dt + 3/2 e Te dn/dt = P_abs/V - (P_loss_coll + P_loss_wall)
    
    term1 = (2.0 / (3.0 * n * e)) * (P_abs / Volume - (P_loss_coll + P_loss_wall))
    term2 = - (Te / n) * dn_dt
    
    dTe_dt = term1 + term2
    
    return [dn_dt, dTe_dt]

def run_global_sim():
    print("Starting Global Model Simulation...")
    print(f"Condition: P={P_abs}W, p={p_Torr}Torr, Gas=Argon")
    
    # Initial Conditions
    n0 = 1e14 # Start with low plasma density [m^-3]
    Te0 = 3.0 # Start with typical electron temperature [eV]
    y0 = [n0, Te0]
    
    # Time grid
    t = np.linspace(0, 5e-3, 1000) # 5 ms simulation
    
    # Solve ODE
    try:
        sol = odeint(model_equations, y0, t)
    except Exception as ie:
        print(f"Integration failed: {ie}")
        sys.exit(1)
        
    n_sol = sol[:, 0]
    Te_sol = sol[:, 1]
    
    print("Simulation Complete.")
    print(f"Final Density: {n_sol[-1]:.2e} m^-3")
    print(f"Final Temperature: {Te_sol[-1]:.2f} eV")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    ax1.plot(t * 1e3, n_sol, 'b-', linewidth=2)
    ax1.set_ylabel('Density ($n_e$) [m$^{-3}$]', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_title(f'0D Global Model (Argon, {p_Torr} Torr, {P_abs} W)')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(t * 1e3, Te_sol, 'r-', linewidth=2)
    ax2.set_ylabel('Temperature ($T_e$) [eV]', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_xlabel('Time [ms]')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if running headless, else show it
    if not os.environ.get('DISPLAY') or '--headless' in sys.argv:
        plt.savefig('global_model_result.png', dpi=150)
        print("Plot saved to global_model_result.png")
    else:
        try:
            plt.show()
        except Exception as e:
            print(f"Warning: Could not display plot ({e}). Saving to file instead.")
            plt.savefig('global_model_result.png', dpi=150)
            print("Plot saved to global_model_result.png")

if __name__ == "__main__":
    run_global_sim()
