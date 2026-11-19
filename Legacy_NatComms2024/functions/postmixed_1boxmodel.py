
import numpy as np
from scipy.integrate import solve_ivp
from functions import chemistry as cm
import pandas as pd

class ODE_full_output:
    def __init__(self) -> None:
        pass

def postmixed_evol(inp,ode_solver,initial_conditions,max_duration):

    global pCO2_prev # global variable for steady-state checking during ODE solve
    pCO2_prev = -999

    global t_prev # global variable for steady-state checking during ODE solve
    t_prev = -0.1

    global pCO2_track # keep track of pCO2 separately for speed
    pCO2_track = 0.05 # speeds up first guess

    # Create steady state checker
    event = lambda t, x: steadyStateCheck(t, x)
    event.terminal = True

    res = solve_ivp(lambda t, x: masterODE(0, x, inp, 0), (0, max_duration), initial_conditions,  method=ode_solver, events=event, max_step=1000) # solve the ODE and check for steady state

    all_results = ODE_full_output()
    all_results.data = pd.DataFrame(columns = ["t","Co","Ao","Cp","Ap","Ts","Td","Tp","pCO2_o","pCO2_p","Omega_o","Omega_p","pH_o","pH_p","V","P_shelf","P_pore","W_carb","W_sil","W_sea"])

    #To get the full evolution we have to loop through all times
    for i,t in enumerate(res.t):
        r = masterODE(t,res.y[:,i],inp,1)
        all_results.data.loc[len(all_results.data.index)] = [t,r.Co,r.Ao,r.Cp,r.Ap,r.Ts,r.Td,r.Tp,r.pCO2_o,r.pCO2_p,r.omega_o,r.omega_p,r.pH_o,r.pH_p,r.V,r.P_shelf,r.P_pore,r.W_carb,r.W_sil,r.W_sea]

    all_results.input_file = inp
    all_results.res = res

    return all_results


def masterODE(t,x,inp,full_output):

    Co, Ao, Cp, Ap = x # unpack individual variables from input array
    global pCO2_track # keep track of the best pCO2 guess
    pCO2_o = pCO2_track # use the most recent value as the starting guess for iteration

    ## Calculate equilibrium ocean chemistry - must iterate between temperature and pCO2 for surface

    T_threshold = 1e-5  # threshold for equilibrium temperature-pCO2 iteration
    max_itter = 100  # max iteration before temperature-pCO2 timeout
    tcount = 0  # count how many times temperature iterates
    Ts_old = 0.1  # arbitrary start value for comparison
    while 1:

        Ts = cm.climate_model_JKT(pCO2_o, inp.neoproterozoic_luminosity)  # calculate surface temperature
        omega_o, pCO2_o, pH_o = cm.equilibriumChemistryOcean(Ts, Ao, Co, inp.modern_s,
                                                             inp.modern_Ao,
                                                             inp.modern_Ca_ocean)  # calculate surface ocean chemistry

        tcount += 1  # add iteration
        if abs(Ts_old - Ts) / Ts_old < T_threshold or tcount >= max_itter: # check for equilibrium or max iteration limit
            break
        Ts_old = Ts  # for comparison on next iteration

    pCO2_track = pCO2_o # record the new pCO2 value to use as initial guess for next time

    Td = cm.deepOceanTemp(Ts, inp.a_grad) # calculate deep ocean temperature
    Tp = cm.poreSpaceTemp(Td) # calculate pore space temperature

    omega_p, pCO2_p, pH_p = cm.equilibriumChemistryOcean(Tp, Ap, Cp, 0,
                                                         inp.modern_Ap,
                                                         inp.modern_Ca_pore)  # calculate pore space chemistry


    ## Calculate carbon and alkalinity fluxes

    V = inp.modern_V # volcanic outgassing flux

    W_carb = inp.f_w * inp.modern_W_carb * ((pCO2_o / inp.modern_pCO2) ** inp.carb_xi) * np.exp(
        (Ts - inp.modern_Ts) / inp.cont_Te) # continental carbonate weathering flux

    W_sil = inp.f_w * inp.modern_W_sil * ((pCO2_o / inp.modern_pCO2) ** inp.sil_alpha) * np.exp(
        (Ts - inp.modern_Ts) / inp.cont_Te) # continental silicate weathering flux

    R = 8.314 # universal gas constant
    Hp = 10 ** (-pH_p) # hydrogen ion concentration in pore space, calculated directly from pH
    W_sea = inp.k_Wsea * np.exp(-inp.E_bas/(R*Tp)) * (Hp/inp.modern_Hp)**inp.sea_gamma # seafloor weathering flux

    if omega_o > 1: # ocean carbonates only precipitate if ocean omega is larger than 1
        P_shelf = inp.k_shelf * inp.shelf_area * (omega_o-1)**inp.carb_n # carbonate precipitation flux on continental shelf
        P_pel = 0 # no carbonate precipitation flux in open ocean
    else:
        P_shelf = 0 # no precipitation if omega is less than 1
        P_pel = 0 # ^

    if omega_p > 1: # pore carbonates only precipitate if pore omega is larger than 1
        P_pore = inp.k_pore*(omega_p-1)**inp.carb_n # carbonate precipitation flux in pore space
    else:
        P_pore = 0 # no precipitation if omega is less than 1

    ## Set up differential equations

    J = inp.modern_J # mixing flux between ocean and pore space
    DIC_o = Co - (inp.modern_s*pCO2_o) # subtract off the atmosphere because that carbon is not mixing with the pore space

    dCo_dt = (1 / inp.modern_ocean_mass) * ((-J*(DIC_o - Cp)) + V + W_carb - P_shelf) # ocean carbon ODE
    dAo_dt = (1 / inp.modern_ocean_mass) * ((-J * (Ao - Ap)) + (2*W_carb) + (2*W_sil) - (2*P_shelf)) # ocean alkalinity ODE

    dCp_dt = (1 / inp.modern_pore_mass) * ((J * (DIC_o - Cp)) - P_pore) # pore carbon ODE
    dAp_dt = (1 / inp.modern_pore_mass) * ((J * (Ao - Ap)) + (2*W_sea) - (2*P_pore)) # pore alkalinity ODE

    ODEs = np.array([dCo_dt, dAo_dt, dCp_dt, dAp_dt])

    if any(abs(ODEs) > 1e-2):
        bp = 1

    if full_output:
        r = ODE_full_output()
        r.Co = Co
        r.Ao = Ao
        r.Cp = Cp
        r.Ap = Ap
        r.Ts = Ts
        r.Td = Td
        r.Tp = Tp
        r.omega_o = omega_o
        r.omega_p = omega_p
        r.pH_o = pH_o
        r.pH_p = pH_p
        r.pCO2_o = pCO2_o
        r.pCO2_p = pCO2_p
        r.V = V
        r.W_carb = W_carb
        r.W_sil = W_sil
        r.W_sea = W_sea
        r.P_shelf = P_shelf
        r.P_pore = P_pore
        r.ODEs = ODEs
        return r
    else:
        return ODEs

def steadyStateCheck(t,x):

    # get global variables from inside function
    global pCO2_track
    global pCO2_prev
    global t_prev

    dpCO2 = pCO2_track - pCO2_prev # change in pCO2 between timesteps
    dt = t - t_prev # change in time between timesteps

    t_future = 1e9  # how far in the future to test?
    pctChangeThreshold = 1  # allowable percent change in pCO2 at t_future

    pctChange = 100 * abs((dpCO2 * t_future) / (dt * pCO2_track)) # projected percent change at t_future

    if pctChange < pctChangeThreshold: # steady state is reached if the projected change is less than the allowed
        result = 0
    else:
        result = -1

    # track global variables again for next iteration
    pCO2_prev = pCO2_track
    t_prev = t

    return result





