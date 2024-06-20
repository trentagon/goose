
import numpy as np
from scipy.integrate import solve_ivp
from functions import chemistry as cm
import pandas as pd
from pprint import pprint # pretty print

class ODE_full_output:
    def __init__(self) -> None:
        pass

def poststrat_evol_sensitivitytest(inp,ode_solver,initial_conditions, max_duration):
    """
    This function finds the time dependent evolution of the post-glacial Earth
    """

    global pCO2_prev # global variable for steady-state checking during ODE solve
    pCO2_prev = -999

    global t_prev # global variable for steady-state checking during ODE solve
    t_prev = -0.1

    global pCO2_track # keep track of pCO2 separately for speed
    pCO2_track = 0.1 # speeds up first guess
    
    global md # need to use max_duration in the ode solver
    md = max_duration # this is how I managed to get it in there

    res = solve_ivp(lambda t, x: masterODE_test(t, x, inp, 0), (0, max_duration), initial_conditions,  method=ode_solver, max_step = 100) # solve the ODE and check for steady state
    
    #Define data frame
    all_results = ODE_full_output()
    all_results.data = pd.DataFrame(columns = ["t","Cs","As","Cd","Ad","Cp","Ap","Ts","Td","Tp","pCO2_s","pCO2_d","pCO2_p","Omega_s","Omega_d","Omega_p","pH_s","pH_d","pH_p","V_air","V_ridge","P_shelf","P_pore","W_carb","W_sil","W_sea","O_sink"])
    
    all_results.sens = pd.DataFrame(columns = ["t","t_pct","m_pct","current_melt_mass","current_melt_s"]) # new data from sensitivity study
    #To get the full evolution we have to loop through all times
    for i,t in enumerate(res.t):
        
        r = masterODE_test(t,res.y[:,i],inp,1)
        all_results.data.loc[len(all_results.data.index)] = [t,r.Cs,r.As,r.Cd,r.Ad,r.Cp,r.Ap,r.Ts,r.Td,r.Tp,r.pCO2_s,r.pCO2_d,r.pCO2_p,r.omega_s,r.omega_d,r.omega_p,r.pH_s,r.pH_d,r.pH_p,r.V_air,r.V_ridge,r.P_shelf,r.P_pore,r.W_carb,r.W_sil,r.W_sea,r.O_sink]
        all_results.sens.loc[len(all_results.sens.index)] = [t,r.t_pct,r.m_pct,r.current_melt_mass,r.current_melt_s] # new data from sensitivity study

    all_results.input_file = inp

    return all_results

def masterODE_test(t,x,inp,full_output):

    Cs, As, Cd, Ad, Cp, Ap = x # unpack individual variables from input array

    t_pct = t/md # percentage of the way through the stratified phase

    if inp.melt_mode == 'linear':
        m_pct = (1-inp.start_pct)*t_pct + inp.start_pct # percentage of melting completed, linear increasing starting at 10%
        m_prime = (1-inp.start_pct)*inp.meltwater_mass/md
        
    elif inp.melt_mode == 'pulsed':
        q = 100
        t1 = 0.25
        t2 = 0.5
        t3 = 0.75
        
        m_pct = 0.1 + (0.9/3)*((1/(1+np.exp(-q*(t_pct - t1))))+(1/(1+np.exp(-q*(t_pct - t2))))+(1/(1+np.exp(-q*(t_pct - t3)))))
        p1 = (q*np.exp(-q*(t_pct - t1)))/((1+np.exp(-q*(t_pct - t1)))**2)
        p2 = (q*np.exp(-q*(t_pct - t2)))/((1+np.exp(-q*(t_pct - t2)))**2)
        p3 = (q*np.exp(-q*(t_pct - t3)))/((1+np.exp(-q*(t_pct - t3)))**2)
        
        m_prime = (0.9*inp.meltwater_mass/(3*md))*(p1 + p2 + p3)
        

    current_melt_mass = m_pct*inp.meltwater_mass # get the current mass of the meltwater layer
    current_melt_s = 1.8e20/current_melt_mass # recalculate mass balance correction factor
    m = current_melt_mass
    
    ## Calculate equilibrium ocean chemistry - must iterate between temperature and pCO2 for surface
    global pCO2_track # keep track of the best pCO2 guess
    pCO2_s = pCO2_track # use the most recent value as the starting guess for iteration

    T_threshold = 1e-5  # threshold for equilibrium temperature-pCO2 iteration
    max_itter = 100  # max iteration before temperature-pCO2 timeout
    tcount = 0  # count how many times temperature iterates
    Ts_old = 0.1  # arbitrary start value for comparison

    while 1:

        Ts_calc = cm.climate_model_JKT(pCO2_s, inp.neoproterozoic_luminosity)  # calculate surface temperature
        Ts = inp.snowball_ocean_temp + (m_pct*(Ts_calc - inp.snowball_ocean_temp))
        
        omega_s, pCO2_s, pH_s = cm.equilibriumChemistryOcean(Ts, As, Cs, current_melt_s, inp.meltwater_alk, inp.meltwater_Ca)  # calculate surface ocean chemistry

        tcount += 1  # add iteration
        if abs(Ts_old - Ts) / Ts_old < T_threshold or tcount >= max_itter: # check for equilibrium or max iteration limit
            break
        Ts_old = Ts  # for comparison on next iteration

    pCO2_track = pCO2_s # record the new pCO2 value to use as initial guess for next time
    
    # Deep ocean temp calculations:
    normal_Td = cm.deepOceanTemp(Ts, inp.a_grad) # deep ocean temperature in non-stratified ocean
    snowball_Td = inp.snowball_ocean_temp # deep ocean temp during the snowball
    Td = (t_pct*normal_Td) + ((1-t_pct)*snowball_Td) # linearly warm the deep ocean until it is back to non-stratified

    Tp = cm.poreSpaceTemp(Td) # calculate pore space temperature


    omega_d, pCO2_d, pH_d = cm.equilibriumChemistryOcean(Td, Ad, Cd, 0,
                                                         inp.modern_Ao,
                                                         inp.modern_Ca_ocean)  # calculate deep ocean chemistry

    omega_p, pCO2_p, pH_p = cm.equilibriumChemistryOcean(Tp, Ap, Cp, 0,
                                                         inp.modern_Ap,
                                                         inp.modern_Ca_pore)  # calculate pore space chemistry

    ## Calculate carbon and alkalinity fluxes
    V_air = inp.V_frac_modern*inp.modern_V # subaerial volcanic outgassing flux
    V_ridge = (1-inp.V_frac_modern)*inp.modern_V # ridge volcanic outgassing flux


    # note that rates are multiplied by the melt percentage - NOT RIGHT NOW THOUGH    
    W_carb = m_pct*inp.f_w * inp.modern_W_carb * ((pCO2_s / inp.modern_pCO2) ** inp.carb_xi) * np.exp(
        (Ts - inp.modern_Ts) / inp.cont_Te) # continental carbonate weathering flux

    W_sil = m_pct*inp.f_w * inp.modern_W_sil * ((pCO2_s / inp.modern_pCO2) ** inp.sil_alpha) * np.exp(
        (Ts - inp.modern_Ts) / inp.cont_Te) # continental silicate weathering flux

    R = 8.314 # universal gas constant
    Hp = 10 ** (-pH_p) # hydrogen ion concentration in pore space, calculated directly from pH
    W_sea = inp.k_Wsea * np.exp(-inp.E_bas/(R*Tp)) * (Hp/inp.modern_Hp)**inp.sea_gamma # seafloor weathering flux

    if omega_s > 1: # shelf carbonates only precipitate if shelf omega is larger than 1
        P_shelf = inp.k_shelf * inp.post_shelf_area * (omega_s-1)**inp.carb_n # carbonate precipitation flux on continental shelf
        P_pel = 0 # no carbonate precipitation flux in open ocean
    else:
        P_shelf = 0 # no precipitation if omega is less than 1
        P_pel = 0 # ^
    P_ocean = P_shelf + P_pel # total ocean precipitation

    if omega_p > 1: # pore carbonates only precipitate if pore omega is larger than 1
        P_pore = inp.k_pore*(omega_p-1)**inp.carb_n # carbonate precipitation flux in pore space
    else:
        P_pore = 0 # no precipitation if omega is less than 1

    # New organic matter production rates
    O_sink = inp.O_sink # organic carbon in pelagic surface ocean that sinks to the deep

    ## Set up differential equations

    # mixing flux between surface ocean and deep ocean:
    K = 0 # No mixing
    # K = t_pct*inp.modern_K # Linearly increase mixing toward the modern rate
    # K = inp.modern_K/50

    J = inp.modern_J # mixing flux between ocean and pore space

    DIC_s = Cs - (current_melt_s*pCO2_s) # subtract off the atmosphere because that carbon is not mixing with the pore space
    
    dCs_dt = (1 / m) * ((-K*(DIC_s - Cd)) + V_air + W_carb - P_ocean - O_sink) - (Cs*m_prime/m) # surface ocean carbon ODE
    dAs_dt = (1 / m) * ((-K * (As - Ad)) + (2*W_carb) + (2*W_sil) - (2*P_ocean)) + ((inp.meltwater_alk - As)*m_prime/(m)) # surface ocean alkalinity ODE

    dCd_dt = (1 / inp.snowball_ocean_mass) * ((-K*(Cd - DIC_s)) + (-J*(Cd - Cp)) + V_ridge + O_sink)  # deep ocean carbon ODE
    dAd_dt = (1 / inp.snowball_ocean_mass) * ((-K*(Ad - As)) + (-J*(Ad - Ap))) # deep ocean alkalinity ODE

    dCp_dt = (1 / inp.modern_pore_mass) * ((-J * (Cp - Cd)) - P_pore) # pore carbon ODE
    dAp_dt = (1 / inp.modern_pore_mass) * ((-J * (Ap - Ad)) + (2*W_sea) - (2*P_pore)) # pore alkalinity ODE

    ODEs = np.array([dCs_dt, dAs_dt, dCd_dt, dAd_dt, dCp_dt, dAp_dt])

    if any(abs(ODEs) > 1e-2):
        bp = 1

    if full_output:
        r = ODE_full_output()
        r.Cs = Cs
        r.As = As
        r.Cd = Cd
        r.Ad = Ad
        r.Cp = Cp
        r.Ap = Ap
        r.Ts = Ts
        r.Td = Td
        r.Tp = Tp
        r.omega_s = omega_s
        r.omega_d = omega_d
        r.omega_p = omega_p
        r.pH_s = pH_s
        r.pH_d = pH_d
        r.pH_p = pH_p
        r.pCO2_s = pCO2_s
        r.pCO2_d = pCO2_d
        r.pCO2_p = pCO2_p
        r.V_air = V_air
        r.V_ridge = V_ridge
        r.W_carb = W_carb
        r.W_sil = W_sil
        r.W_sea = W_sea
        r.P_shelf = P_shelf
        r.P_pore = P_pore
        r.O_sink = O_sink
        r.ODEs = ODEs
        
        r.t_pct = t_pct
        r.m_pct = m_pct
        r.current_melt_mass = current_melt_mass
        r.current_melt_s = current_melt_s
        return r
    else:

        return ODEs
