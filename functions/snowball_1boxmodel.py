
import numpy as np
from scipy.integrate import solve_ivp
from functions import chemistry as cm
import pandas as pd

class ODE_full_output:
    def __init__(self) -> None:
        pass

def snowball_evol(inp,ode_solver,initial_conditions,max_duration,co2_threshold):
    """
    This function finds the time dependent evolution of the ice covered Earth
    """

    global pCO2_track # keep track of pCO2 separately for deglaciation check
    pCO2_track = 200e-6 # doesn't matter

    if co2_threshold:
        deglaciation_event = lambda t, x: deglaciationCheck(t, x, co2_threshold)
        deglaciation_event.terminal = True
    else:
        deglaciation_event=None

    res = solve_ivp(lambda t, x: masterODE(0, x, inp, 0), (0, max_duration), initial_conditions,  method=ode_solver, events=deglaciation_event) # solve the ODE and check for steady state

    #Define data frame
    all_results = ODE_full_output()
    all_results.data = pd.DataFrame(columns = ["t","Co","Ao","Cp","Ap","Ts","Td","Tp","pCO2_o","pCO2_p","Omega_o","Omega_p","pH_o","pH_p","V","P_ocean","P_pore","W_carb","W_sil","W_sea"])

    #To get the full evolution we have to loop through all times
    for i,t in enumerate(res.t):
        r = masterODE(t,res.y[:,i],inp,1)
        all_results.data.loc[len(all_results.data.index)] = [t,r.Co,r.Ao,r.Cp,r.Ap,r.Ts,r.Td,r.Tp,r.pCO2_o,r.pCO2_p,r.omega_o,r.omega_p,r.pH_o,r.pH_p,r.V,r.P_ocean,r.P_pore,r.W_carb,r.W_sil,r.W_sea]

    all_results.input_file = inp
    all_results.snow_max_duration = max_duration

    return all_results

def masterODE(t,x,inp,full_output):

    Co, Ao, Cp, Ap = x # unpack individual variables from input array

    ## Calculate equilibrium ocean chemistry - must iterate between temperature and pCO2 for surface

        ## Fixed ocean temperature
    Ts = inp.snowball_ocean_temp
    Td = inp.snowball_ocean_temp
    Tp = cm.poreSpaceTemp(Td)

    omega_o, pCO2_o, pH_o = cm.equilibriumChemistryOcean(Ts, Ao, Co, inp.snowball_s,
                                                         inp.modern_Ao,
                                                         inp.modern_Ca_ocean)  # calculate surface ocean chemistry

    global pCO2_track  # keep track of the best pCO2 guess
    pCO2_track = pCO2_o # record the new pCO2 value to use as initial guess for next time

    omega_p, pCO2_p, pH_p = cm.equilibriumChemistryOcean(Tp, Ap, Cp, 0,
                                                         inp.modern_Ap,
                                                         inp.modern_Ca_pore)  # calculate pore space chemistry

    ## Calculate carbon and alkalinity fluxes

    V = inp.modern_V # volcanic outgassing flux

    W_carb = inp.snowball_W_carb # continental carbonate weathering flux

    W_sil = inp.snowball_W_sil # continental silicate weathering flux

    R = 8.314 # universal gas constant
    Hp = 10 ** (-pH_p) # hydrogen ion concentration in pore space, calculated directly from pH
    W_sea = inp.hyalo * inp.k_Wsea * np.exp(-inp.E_bas/(R*Tp)) * (Hp/inp.modern_Hp)**inp.sea_gamma # seafloor weathering flux

    if omega_o > 1: # ocean carbonates only precipitate if ocean omega is larger than 1
        P_shelf = inp.k_shelf * inp.snowball_shelf_area * (omega_o-1)**inp.carb_n # carbonate precipitation flux on continental shelf
        P_pel = 0 # no carbonate precipitation flux in open ocean
    else:
        P_shelf = 0 # no precipitation if omega is less than 1
        P_pel = 0 # ^
    P_ocean = P_shelf + P_pel # total ocean precipitation

    if omega_p > 1: # pore carbonates only precipitate if pore omega is larger than 1
        P_pore = inp.k_pore*(omega_p-1)**inp.carb_n # carbonate precipitation flux in pore space
    else:
        P_pore = 0 # no precipitation if omega is less than 1

    ## Set up differential equations

    J = inp.modern_J # mixing flux between ocean and pore space
    DIC_o = Co - (inp.snowball_s*pCO2_o) # subtract off the atmosphere because that carbon is not mixing with the pore space
    
    dCo_dt = (1 / inp.snowball_ocean_mass) * ((-J*(DIC_o - Cp)) + V + W_carb - P_ocean) # ocean carbon ODE
    dAo_dt = (1 / inp.snowball_ocean_mass) * ((-J * (Ao - Ap)) + (2*W_carb) + (2*W_sil) - (2*P_ocean)) # ocean alkalinity ODE

    dCp_dt = (1 / inp.modern_pore_mass) * ((J * (DIC_o - Cp)) - P_pore) # pore carbon ODE
    dAp_dt = (1 / inp.modern_pore_mass) * ((J * (Ao - Ap)) + (2*W_sea) - (2*P_pore)) # pore alkalinity ODE

    ODEs = np.array([dCo_dt, dAo_dt, dCp_dt, dAp_dt])

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
        r.P_ocean = P_ocean
        r.P_pore = P_pore
        r.ODEs = ODEs
        return r
    else:
        return ODEs

def deglaciationCheck(t, x, co2_threshold):

    # get global variables from inside function
    global pCO2_track

    if pCO2_track > co2_threshold: # steady state is reached if pCO2 is high enough to cause deglaciation
        result = 0
    else:
        result = -1

    return result
