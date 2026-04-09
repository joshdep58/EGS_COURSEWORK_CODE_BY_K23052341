"""
Case A: Smart Home Energy Management (PV + Battery)
=====================================================
EGS Individual Coursework

This script implements two dispatch policies for a smart home with PV,
battery storage, and grid connection under time-varying tariffs:
    Strategy 1: Self-Consumption First (greedy rule-based)
    Strategy 2: Cost-Optimised (linear programming with perfect foresight)

Author: [Your Name]
Date:   [Date]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import linprog

# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_data(filepath):
    """Load and validate the smart home dataset."""
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    
    # Basic validation
    assert df.isnull().sum().sum() == 0, "Missing values found!"
    assert len(df) == 1440, f"Expected 1440 rows, got {len(df)}"
    
    # Check timestep consistency (should be 30 min = 1800 seconds)
    dt_seconds = df['timestamp'].diff().dt.total_seconds().dropna()
    assert (dt_seconds == 1800).all(), "Inconsistent timestep detected!"
    
    print(f"Data loaded: {len(df)} timesteps, {df.timestamp.min()} to {df.timestamp.max()}")
    return df


# =============================================================================
# 2. BATTERY PARAMETERS
# =============================================================================

class BatteryParams:
    """Battery parameters from the coursework specification."""
    def __init__(self):
        self.E_cap = 5.0        # Usable capacity [kWh]
        self.P_ch_max = 2.5     # Max charge power [kW]
        self.P_dis_max = 2.5    # Max discharge power [kW]
        self.eta_ch = 0.95      # Charge efficiency [-]
        self.eta_dis = 0.95     # Discharge efficiency [-]
        self.SOC_min = 0.0      # Min SOC [kWh]
        self.SOC_max = 5.0      # Max SOC [kWh]
        self.SOC_init = 2.5     # Initial SOC (50%) [kWh]
        self.dt = 0.5           # Timestep [hours]
    
    def __repr__(self):
        return (f"BatteryParams(E_cap={self.E_cap} kWh, "
                f"P_ch_max={self.P_ch_max} kW, P_dis_max={self.P_dis_max} kW, "
                f"eta_ch={self.eta_ch}, eta_dis={self.eta_dis}, "
                f"SOC_init={self.SOC_init} kWh)")


# =============================================================================
# 3. STRATEGY 1: SELF-CONSUMPTION FIRST (GREEDY)
# =============================================================================

def policy_self_consumption(df, bp):
    """
    Greedy self-consumption dispatch strategy.
    
    Priority order at each timestep:
    1. PV serves load directly
    2. Surplus PV charges battery (up to limits)
    3. Remaining surplus exported to grid
    4. Deficit met by battery discharge (up to limits)
    5. Remaining deficit met by grid import
    
    Parameters
    ----------
    df : pd.DataFrame — dataset with pv_kw, base_load_kw, tariffs
    bp : BatteryParams — battery specification
    
    Returns
    -------
    results : dict of numpy arrays — all power flows and SOC
    """
    N = len(df)
    
    # Extract data as numpy arrays for speed
    P_pv = df['pv_kw'].values
    P_load = df['base_load_kw'].values
    pi_imp = df['import_tariff_gbp_per_kwh'].values
    pi_exp = df['export_price_gbp_per_kwh'].values
    
    # Initialise output arrays
    P_ch = np.zeros(N)       # Battery charge power [kW]
    P_dis = np.zeros(N)      # Battery discharge power [kW]
    P_grid_imp = np.zeros(N) # Grid import [kW]
    P_grid_exp = np.zeros(N) # Grid export [kW]
    SOC = np.zeros(N + 1)    # SOC at start of each interval (+1 for final)
    SOC[0] = bp.SOC_init
    
    for t in range(N):
        net = P_pv[t] - P_load[t]  # Net power at bus [kW]
        
        if net >= 0:
            # SURPLUS: PV exceeds load
            surplus = net
            
            # Step 1: Charge battery with surplus (respecting limits)
            # Max energy we can add to battery this timestep
            SOC_room = bp.SOC_max - SOC[t]                    # [kWh] space in battery
            max_charge_energy = bp.P_ch_max * bp.dt            # [kWh] power limit
            charge_energy = min(surplus * bp.dt, max_charge_energy, SOC_room / bp.eta_ch)
            
            P_ch[t] = charge_energy / bp.dt                   # [kW]
            remaining_surplus = surplus - P_ch[t]
            
            # Step 2: Export remaining surplus
            P_grid_exp[t] = remaining_surplus
            
        else:
            # DEFICIT: Load exceeds PV
            deficit = -net  # positive value [kW]
            
            # Step 1: Discharge battery to meet deficit (respecting limits)
            SOC_available = SOC[t] - bp.SOC_min                # [kWh] available energy
            max_discharge_energy = bp.P_dis_max * bp.dt        # [kWh] power limit
            # Energy delivered to bus = P_dis * eta_dis * dt, but we track P_dis at bus side
            # Actually: energy removed from battery = P_dis / eta_dis ... wait.
            # Let me be precise:
            # SOC update: SOC(t+1) = SOC(t) + eta_ch*P_ch*dt - P_dis*dt/eta_dis
            # So P_dis is the power DELIVERED to the bus.
            # Energy removed from battery = P_dis * dt / eta_dis
            # Energy available at bus from battery = SOC_available * eta_dis
            
            discharge_at_bus = min(
                deficit,                                        # don't discharge more than needed
                bp.P_dis_max,                                   # power limit [kW]
                SOC_available * bp.eta_dis / bp.dt              # SOC limit converted to bus power
            )
            
            P_dis[t] = discharge_at_bus                        # [kW] at bus
            remaining_deficit = deficit - P_dis[t]
            
            # Step 2: Import remaining deficit from grid
            P_grid_imp[t] = remaining_deficit
        
        # Update SOC
        SOC[t + 1] = SOC[t] + (bp.eta_ch * P_ch[t] - P_dis[t] / bp.eta_dis) * bp.dt
        
        # Safety clamp (should not be needed if logic is correct)
        SOC[t + 1] = np.clip(SOC[t + 1], bp.SOC_min, bp.SOC_max)
    
    return {
        'P_ch': P_ch, 'P_dis': P_dis,
        'P_grid_imp': P_grid_imp, 'P_grid_exp': P_grid_exp,
        'SOC': SOC, 'P_pv': P_pv, 'P_load': P_load,
        'pi_imp': pi_imp, 'pi_exp': pi_exp
    }


# =============================================================================
# 4. STRATEGY 2: COST-OPTIMISED (LINEAR PROGRAMMING)
# =============================================================================

def policy_cost_optimised(df, bp):
    """
    Cost-optimised dispatch using linear programming (perfect foresight).
    
    Minimises total electricity cost over the full horizon.
    Uses scipy.optimize.linprog.
    
    Decision variables per timestep (4 × N total):
        x = [P_ch(0), ..., P_ch(N-1),     indices 0       to N-1
             P_dis(0), ..., P_dis(N-1),    indices N       to 2N-1
             P_grid_imp(0), ...,           indices 2N      to 3N-1
             P_grid_exp(0), ...]           indices 3N      to 4N-1
    
    Parameters
    ----------
    df : pd.DataFrame
    bp : BatteryParams
    
    Returns
    -------
    results : dict
    """
    N = len(df)
    
    P_pv = df['pv_kw'].values
    P_load = df['base_load_kw'].values
    pi_imp = df['import_tariff_gbp_per_kwh'].values
    pi_exp = df['export_price_gbp_per_kwh'].values
    
    # --- Variable indexing ---
    # 4 variables per timestep: P_ch, P_dis, P_grid_imp, P_grid_exp
    n_vars = 4 * N
    
    idx_ch  = np.arange(0, N)           # P_ch indices
    idx_dis = np.arange(N, 2*N)         # P_dis indices
    idx_imp = np.arange(2*N, 3*N)       # P_grid_imp indices
    idx_exp = np.arange(3*N, 4*N)       # P_grid_exp indices
    
    # --- Objective: minimise cost ---
    # C = sum[ pi_imp(t) * P_grid_imp(t) - pi_exp(t) * P_grid_exp(t) ] * dt
    c = np.zeros(n_vars)
    c[idx_imp] = pi_imp * bp.dt         # cost of importing
    c[idx_exp] = -pi_exp * bp.dt        # revenue from exporting (negative cost)
    
    # --- Bounds on all variables ---
    bounds = []
    for t in range(N):
        bounds.append((0, bp.P_ch_max))     # P_ch
    for t in range(N):
        bounds.append((0, bp.P_dis_max))    # P_dis
    for t in range(N):
        bounds.append((0, None))            # P_grid_imp (unbounded above)
    for t in range(N):
        bounds.append((0, None))            # P_grid_exp (unbounded above)
    
    # --- Equality constraints: energy balance at bus ---
    # P_pv(t) + P_dis(t) + P_grid_imp(t) = P_load(t) + P_ch(t) + P_grid_exp(t)
    # Rearranged: -P_ch(t) + P_dis(t) + P_grid_imp(t) - P_grid_exp(t) = P_load(t) - P_pv(t)
    
    A_eq_balance = np.zeros((N, n_vars))
    b_eq_balance = np.zeros(N)
    
    for t in range(N):
        A_eq_balance[t, idx_ch[t]]  = -1    # -P_ch
        A_eq_balance[t, idx_dis[t]] = +1    # +P_dis
        A_eq_balance[t, idx_imp[t]] = +1    # +P_grid_imp
        A_eq_balance[t, idx_exp[t]] = -1    # -P_grid_exp
        b_eq_balance[t] = P_load[t] - P_pv[t]
    
    # --- Inequality constraints: SOC bounds ---
    # SOC(t+1) = SOC(t) + [eta_ch * P_ch(t) - P_dis(t) / eta_dis] * dt
    # Unrolling: SOC(t+1) = SOC_0 + sum_{k=0}^{t} [eta_ch * P_ch(k) - P_dis(k)/eta_dis] * dt
    #
    # We need: SOC_min <= SOC(t+1) <= SOC_max for t = 0, ..., N-1
    # And:     SOC(N) >= SOC_init  (end-of-horizon constraint)
    #
    # SOC(t+1) = SOC_0 + dt * sum_{k=0}^{t} [eta_ch * P_ch(k) - P_dis(k)/eta_dis]
    #
    # Upper bound: SOC(t+1) <= SOC_max
    #   dt * sum_{k=0}^{t} [eta_ch * P_ch(k) - P_dis(k)/eta_dis] <= SOC_max - SOC_0
    #
    # Lower bound: SOC(t+1) >= SOC_min
    #   -dt * sum_{k=0}^{t} [eta_ch * P_ch(k) - P_dis(k)/eta_dis] <= SOC_0 - SOC_min
    #
    # End-of-horizon: SOC(N) >= SOC_init
    #   -dt * sum_{k=0}^{N-1} [eta_ch * P_ch(k) - P_dis(k)/eta_dis] <= SOC_0 - SOC_init
    
    # Upper SOC bound: N constraints
    A_ub_soc_upper = np.zeros((N, n_vars))
    b_ub_soc_upper = np.zeros(N)
    
    # Lower SOC bound: N constraints
    A_ub_soc_lower = np.zeros((N, n_vars))
    b_ub_soc_lower = np.zeros(N)
    
    for t in range(N):
        for k in range(t + 1):
            # Upper: cumulative charge contribution <= SOC_max - SOC_0
            A_ub_soc_upper[t, idx_ch[k]]  = bp.eta_ch * bp.dt
            A_ub_soc_upper[t, idx_dis[k]] = -bp.dt / bp.eta_dis
            
            # Lower: negative cumulative <= SOC_0 - SOC_min
            A_ub_soc_lower[t, idx_ch[k]]  = -bp.eta_ch * bp.dt
            A_ub_soc_lower[t, idx_dis[k]] = bp.dt / bp.eta_dis
        
        b_ub_soc_upper[t] = bp.SOC_max - bp.SOC_init
        b_ub_soc_lower[t] = bp.SOC_init - bp.SOC_min
    
    # End-of-horizon constraint: SOC(N) >= SOC_init
    A_ub_eoh = np.zeros((1, n_vars))
    b_ub_eoh = np.zeros(1)
    for k in range(N):
        A_ub_eoh[0, idx_ch[k]]  = -bp.eta_ch * bp.dt
        A_ub_eoh[0, idx_dis[k]] = bp.dt / bp.eta_dis
    b_ub_eoh[0] = bp.SOC_init - bp.SOC_init  # = 0 (SOC_final >= SOC_init)
    
    # Stack all inequality constraints
    A_ub = np.vstack([A_ub_soc_upper, A_ub_soc_lower, A_ub_eoh])
    b_ub = np.concatenate([b_ub_soc_upper, b_ub_soc_lower, b_ub_eoh])
    
    print(f"LP size: {n_vars} variables, {N} eq constraints, {len(b_ub)} ineq constraints")
    print("Solving LP (this may take a minute)...")
    
    # --- Solve ---
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq_balance, b_eq=b_eq_balance,
                     bounds=bounds, method='highs')
    
    if not result.success:
        raise RuntimeError(f"LP solver failed: {result.message}")
    
    print(f"LP solved successfully. Optimal cost: £{result.fun:.2f}")
    
    # --- Extract results ---
    x = result.x
    P_ch = x[idx_ch]
    P_dis = x[idx_dis]
    P_grid_imp = x[idx_imp]
    P_grid_exp = x[idx_exp]
    
    # Reconstruct SOC
    SOC = np.zeros(N + 1)
    SOC[0] = bp.SOC_init
    for t in range(N):
        SOC[t + 1] = SOC[t] + (bp.eta_ch * P_ch[t] - P_dis[t] / bp.eta_dis) * bp.dt
    
    return {
        'P_ch': P_ch, 'P_dis': P_dis,
        'P_grid_imp': P_grid_imp, 'P_grid_exp': P_grid_exp,
        'SOC': SOC, 'P_pv': P_pv, 'P_load': P_load,
        'pi_imp': pi_imp, 'pi_exp': pi_exp,
        'lp_result': result
    }


# =============================================================================
# 5. VERIFICATION CHECKS
# =============================================================================

def verify_results(results, bp, strategy_name):
    """
    Run comprehensive verification checks on dispatch results.
    
    Checks:
    1. Energy balance at every timestep
    2. SOC bounds respected
    3. Power bounds respected
    4. No simultaneous charge/discharge
    5. No simultaneous import/export
    6. SOC dynamics consistency
    7. End-of-horizon SOC
    8. Global energy balance
    
    Returns True if all checks pass.
    """
    N = len(results['P_ch'])
    tol = 1e-6  # numerical tolerance
    all_passed = True
    
    print(f"\n{'='*60}")
    print(f"VERIFICATION: {strategy_name}")
    print(f"{'='*60}")
    
    # --- Check 1: Bus energy balance at every timestep ---
    supply = results['P_pv'] + results['P_dis'] + results['P_grid_imp']
    demand = results['P_load'] + results['P_ch'] + results['P_grid_exp']
    balance_error = np.abs(supply - demand)
    max_balance_err = balance_error.max()
    check1 = max_balance_err < tol
    print(f"[{'PASS' if check1 else 'FAIL'}] Energy balance: max error = {max_balance_err:.2e} kW")
    all_passed &= check1
    
    # --- Check 2: SOC bounds ---
    SOC = results['SOC']
    soc_min_ok = SOC.min() >= bp.SOC_min - tol
    soc_max_ok = SOC.max() <= bp.SOC_max + tol
    check2 = soc_min_ok and soc_max_ok
    print(f"[{'PASS' if check2 else 'FAIL'}] SOC bounds: min={SOC.min():.4f} kWh (≥{bp.SOC_min}), "
          f"max={SOC.max():.4f} kWh (≤{bp.SOC_max})")
    all_passed &= check2
    
    # --- Check 3: Power bounds ---
    ch_ok = results['P_ch'].min() >= -tol and results['P_ch'].max() <= bp.P_ch_max + tol
    dis_ok = results['P_dis'].min() >= -tol and results['P_dis'].max() <= bp.P_dis_max + tol
    imp_ok = results['P_grid_imp'].min() >= -tol
    exp_ok = results['P_grid_exp'].min() >= -tol
    check3 = ch_ok and dis_ok and imp_ok and exp_ok
    print(f"[{'PASS' if check3 else 'FAIL'}] Power bounds: "
          f"P_ch=[{results['P_ch'].min():.4f}, {results['P_ch'].max():.4f}], "
          f"P_dis=[{results['P_dis'].min():.4f}, {results['P_dis'].max():.4f}]")
    all_passed &= check3
    
    # --- Check 4: No simultaneous charge & discharge ---
    simul_cd = np.sum((results['P_ch'] > tol) & (results['P_dis'] > tol))
    check4 = simul_cd == 0
    print(f"[{'PASS' if check4 else 'FAIL'}] No simultaneous charge/discharge: "
          f"{simul_cd} violations")
    all_passed &= check4
    
    # --- Check 5: No simultaneous import & export ---
    simul_ie = np.sum((results['P_grid_imp'] > tol) & (results['P_grid_exp'] > tol))
    check5 = simul_ie == 0
    print(f"[{'PASS' if check5 else 'FAIL'}] No simultaneous import/export: "
          f"{simul_ie} violations")
    all_passed &= check5
    
    # --- Check 6: SOC dynamics consistency ---
    SOC_reconstructed = np.zeros(N + 1)
    SOC_reconstructed[0] = bp.SOC_init
    for t in range(N):
        SOC_reconstructed[t+1] = (SOC_reconstructed[t] 
                                   + (bp.eta_ch * results['P_ch'][t] 
                                      - results['P_dis'][t] / bp.eta_dis) * bp.dt)
    soc_recon_err = np.abs(SOC - SOC_reconstructed).max()
    check6 = soc_recon_err < tol
    print(f"[{'PASS' if check6 else 'FAIL'}] SOC dynamics consistency: "
          f"max reconstruction error = {soc_recon_err:.2e} kWh")
    all_passed &= check6
    
    # --- Check 7: End-of-horizon SOC ---
    check7 = SOC[-1] >= bp.SOC_init - tol
    print(f"[{'PASS' if check7 else 'WARN'}] End-of-horizon SOC: "
          f"SOC_final={SOC[-1]:.4f} kWh (SOC_init={bp.SOC_init} kWh)")
    # Note: for Strategy 1 this may not be enforced (it's greedy)
    
    # --- Check 8: Global energy balance ---
    # Total energy in = PV + grid import + battery SOC decrease
    # Total energy out = load + grid export + battery SOC increase + losses
    E_pv = np.sum(results['P_pv']) * bp.dt
    E_load = np.sum(results['P_load']) * bp.dt
    E_grid_imp = np.sum(results['P_grid_imp']) * bp.dt
    E_grid_exp = np.sum(results['P_grid_exp']) * bp.dt
    E_ch = np.sum(results['P_ch']) * bp.dt
    E_dis = np.sum(results['P_dis']) * bp.dt
    E_batt_loss = E_ch * (1 - bp.eta_ch) + E_dis * (1/bp.eta_dis - 1)
    delta_SOC = SOC[-1] - SOC[0]
    
    # Global balance: PV + Import = Load + Export + Losses + ΔSOC
    # More precisely: PV + Import + E_dis = Load + Export + E_ch
    # And: ΔSOC = eta_ch * E_ch - E_dis / eta_dis
    lhs = E_pv + E_grid_imp + E_dis
    rhs = E_load + E_grid_exp + E_ch
    global_err = abs(lhs - rhs)
    check8 = global_err < tol * N
    print(f"[{'PASS' if check8 else 'FAIL'}] Global energy balance: "
          f"error = {global_err:.4f} kWh")
    
    print(f"\n--- Energy Summary ({strategy_name}) ---")
    print(f"  PV generation:    {E_pv:.2f} kWh")
    print(f"  Grid import:      {E_grid_imp:.2f} kWh")
    print(f"  Grid export:      {E_grid_exp:.2f} kWh")
    print(f"  Load consumed:    {E_load:.2f} kWh")
    print(f"  Battery charged:  {E_ch:.2f} kWh")
    print(f"  Battery discharged: {E_dis:.2f} kWh")
    print(f"  Battery losses:   {E_batt_loss:.2f} kWh")
    print(f"  SOC change:       {delta_SOC:+.4f} kWh")
    
    # Cost
    cost_import = np.sum(results['pi_imp'] * results['P_grid_imp']) * bp.dt
    revenue_export = np.sum(results['pi_exp'] * results['P_grid_exp']) * bp.dt
    net_cost = cost_import - revenue_export
    print(f"\n--- Cost Summary ({strategy_name}) ---")
    print(f"  Import cost:      £{cost_import:.2f}")
    print(f"  Export revenue:   £{revenue_export:.2f}")
    print(f"  Net cost:         £{net_cost:.2f}")
    
    print(f"\n{'='*60}")
    print(f"OVERALL: {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
    print(f"{'='*60}\n")
    
    return all_passed, {
        'E_pv': E_pv, 'E_load': E_load, 'E_grid_imp': E_grid_imp,
        'E_grid_exp': E_grid_exp, 'E_ch': E_ch, 'E_dis': E_dis,
        'E_batt_loss': E_batt_loss, 'delta_SOC': delta_SOC,
        'cost_import': cost_import, 'revenue_export': revenue_export,
        'net_cost': net_cost
    }


# =============================================================================
# 6. VISUALISATION
# =============================================================================

def plot_results(df, results1, results2, summary1, summary2, bp):
    """Generate all figures for the report."""
    
    timestamps = df['timestamp'].values
    N = len(df)
    
    # --- Figure 1: SOC comparison (full 30 days) ---
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(timestamps, results1['SOC'][:N], label='Strategy 1: Self-Consumption', alpha=0.8)
    ax.plot(timestamps, results2['SOC'][:N], label='Strategy 2: Cost-Optimised', alpha=0.8)
    ax.axhline(y=bp.SOC_max, color='r', linestyle='--', alpha=0.3, label='SOC bounds')
    ax.axhline(y=bp.SOC_min, color='r', linestyle='--', alpha=0.3)
    ax.set_ylabel('SOC [kWh]')
    ax.set_xlabel('Time')
    ax.set_title('Battery State of Charge — Strategy Comparison')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d Jul'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.tight_layout()
    plt.savefig('fig1_soc_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # --- Figure 2: Zoomed-in view (first 3 days) for detailed dispatch ---
    zoom = 3 * 48  # 3 days × 48 half-hours
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    t_zoom = timestamps[:zoom]
    
    # Panel A: PV and Load
    axes[0].fill_between(t_zoom, results1['P_pv'][:zoom], alpha=0.4, label='PV', color='gold')
    axes[0].plot(t_zoom, results1['P_load'][:zoom], label='Load', color='navy', linewidth=1)
    axes[0].set_ylabel('Power [kW]')
    axes[0].set_title('PV Generation and Load (First 3 Days)')
    axes[0].legend()
    
    # Panel B: Import tariff
    axes[1].plot(t_zoom, results1['pi_imp'][:zoom], color='red', linewidth=1)
    axes[1].set_ylabel('Import Tariff\n[£/kWh]')
    axes[1].set_title('Time-of-Use Import Tariff')
    
    # Panel C: Strategy 1 dispatch
    axes[2].fill_between(t_zoom, results1['P_grid_imp'][:zoom], alpha=0.5, label='Grid Import', color='red')
    axes[2].fill_between(t_zoom, -results1['P_grid_exp'][:zoom], alpha=0.5, label='Grid Export', color='green')
    axes[2].fill_between(t_zoom, results1['P_dis'][:zoom], alpha=0.5, label='Batt Discharge', color='blue')
    axes[2].fill_between(t_zoom, -results1['P_ch'][:zoom], alpha=0.5, label='Batt Charge', color='purple')
    axes[2].set_ylabel('Power [kW]')
    axes[2].set_title('Strategy 1: Self-Consumption First — Dispatch')
    axes[2].legend(loc='upper right', fontsize=8)
    axes[2].axhline(y=0, color='k', linewidth=0.5)
    
    # Panel D: Strategy 2 dispatch
    axes[3].fill_between(t_zoom, results2['P_grid_imp'][:zoom], alpha=0.5, label='Grid Import', color='red')
    axes[3].fill_between(t_zoom, -results2['P_grid_exp'][:zoom], alpha=0.5, label='Grid Export', color='green')
    axes[3].fill_between(t_zoom, results2['P_dis'][:zoom], alpha=0.5, label='Batt Discharge', color='blue')
    axes[3].fill_between(t_zoom, -results2['P_ch'][:zoom], alpha=0.5, label='Batt Charge', color='purple')
    axes[3].set_ylabel('Power [kW]')
    axes[3].set_title('Strategy 2: Cost-Optimised — Dispatch')
    axes[3].legend(loc='upper right', fontsize=8)
    axes[3].axhline(y=0, color='k', linewidth=0.5)
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%d Jul %H:%M'))
    
    plt.tight_layout()
    plt.savefig('fig2_dispatch_detail.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # --- Figure 3: Energy breakdown bar chart ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Energy breakdown
    categories = ['PV\nGeneration', 'Grid\nImport', 'Grid\nExport', 'Load\nConsumed', 
                  'Battery\nCharged', 'Battery\nDischarged']
    values1 = [summary1['E_pv'], summary1['E_grid_imp'], summary1['E_grid_exp'],
               summary1['E_load'], summary1['E_ch'], summary1['E_dis']]
    values2 = [summary2['E_pv'], summary2['E_grid_imp'], summary2['E_grid_exp'],
               summary2['E_load'], summary2['E_ch'], summary2['E_dis']]
    
    x = np.arange(len(categories))
    width = 0.35
    axes[0].bar(x - width/2, values1, width, label='Strategy 1', color='steelblue')
    axes[0].bar(x + width/2, values2, width, label='Strategy 2', color='coral')
    axes[0].set_ylabel('Energy [kWh]')
    axes[0].set_title('Energy Breakdown Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories, fontsize=9)
    axes[0].legend()
    
    # Cost breakdown
    cost_cats = ['Import Cost', 'Export Revenue', 'Net Cost']
    costs1 = [summary1['cost_import'], summary1['revenue_export'], summary1['net_cost']]
    costs2 = [summary2['cost_import'], summary2['revenue_export'], summary2['net_cost']]
    
    x2 = np.arange(len(cost_cats))
    axes[1].bar(x2 - width/2, costs1, width, label='Strategy 1', color='steelblue')
    axes[1].bar(x2 + width/2, costs2, width, label='Strategy 2', color='coral')
    axes[1].set_ylabel('Cost [£]')
    axes[1].set_title('Cost Breakdown Comparison')
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(cost_cats)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('fig3_energy_cost_breakdown.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # --- Figure 4: No-battery baseline for reference ---
    # What would costs be without any battery?
    net_power = results1['P_pv'] - results1['P_load']
    no_batt_imp = np.maximum(-net_power, 0)
    no_batt_exp = np.maximum(net_power, 0)
    no_batt_cost = np.sum(results1['pi_imp'] * no_batt_imp - results1['pi_exp'] * no_batt_exp) * bp.dt
    
    fig, ax = plt.subplots(figsize=(8, 5))
    scenarios = ['No Battery', 'Strategy 1:\nSelf-Consumption', 'Strategy 2:\nCost-Optimised']
    costs = [no_batt_cost, summary1['net_cost'], summary2['net_cost']]
    colors = ['grey', 'steelblue', 'coral']
    bars = ax.bar(scenarios, costs, color=colors, width=0.5)
    ax.set_ylabel('Net Electricity Cost [£]')
    ax.set_title('30-Day Electricity Cost Comparison')
    for bar, cost in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'£{cost:.2f}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig4_cost_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("All figures saved.")
    return no_batt_cost


# =============================================================================
# 7. MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    
    # Load data
    df = load_data('caseA_smart_home_30min_summer.csv')
    bp = BatteryParams()
    print(bp)
    
    # Run Strategy 1
    print("\n--- Running Strategy 1: Self-Consumption First ---")
    results1 = policy_self_consumption(df, bp)
    passed1, summary1 = verify_results(results1, bp, "Strategy 1: Self-Consumption")
    
    # Run Strategy 2
    print("\n--- Running Strategy 2: Cost-Optimised (LP) ---")
    results2 = policy_cost_optimised(df, bp)
    passed2, summary2 = verify_results(results2, bp, "Strategy 2: Cost-Optimised")
    
    # Generate plots
    print("\n--- Generating Figures ---")
    no_batt_cost = plot_results(df, results1, results2, summary1, summary2, bp)
    
    # Final comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"  No battery baseline:      £{no_batt_cost:.2f}")
    print(f"  Strategy 1 (Self-Consume):  £{summary1['net_cost']:.2f}  "
          f"(saves £{no_batt_cost - summary1['net_cost']:.2f})")
    print(f"  Strategy 2 (Optimised):     £{summary2['net_cost']:.2f}  "
          f"(saves £{no_batt_cost - summary2['net_cost']:.2f})")
    print(f"  Value of optimisation:    £{summary1['net_cost'] - summary2['net_cost']:.2f} "
          f"saved vs Strategy 1")
    print(f"\n  Strategy 1 SOC final:       {results1['SOC'][-1]:.4f} kWh")
    print(f"  Strategy 2 SOC final:       {results2['SOC'][-1]:.4f} kWh")
