
import numpy as np

def climate_model_JKT(pCO2, L):
    #From JKT et al., 2018 (Supplementary)

    x = np.log10(pCO2) #pCO2 is in bars
    y = L #Luminosity relative to modern

    T = (3145.89) - (894.196*x) - (14231.92*y) - (3791.699*x**2) + (18527.77*y*x**2) - (33560.74*x**2 * y**2) + 26297.61*y**2 \
        - (7674.76*x*y**2) + (4461.16*x*y) - (1569.25*x**3) + (11329.25*x**3 *y**3) - (21270.62*y**3) - (14022.32*x**3 * y**2) \
        + (26860.83*x**2*y**3) + (7684.55*x**3*y) + (5722.64*x*y**3) - (178.264*x**4) - (396.147*x**4 *y**4) + (6399.091*y**4) \
        + (875.423*x**4 *y) - (1605.364*x**4*y**2) + (1304.944*x**4 *y**3) - (1569.3007*y**4*x) - (8012.67*y**4*x**2) - (3415.93*y**4*x**3)

    return T

def deepOceanTemp(Ts, gradient, min_temp=271.15):
    """
    Determine the deep ocean temperature based on the surface temperature. The
    intercept term is chosen so that gradient*Ts+intercept gives the correct
    surface temperature. In the case of the modern Earth, that would be the
    modern average surface temperature. This function corresponds to equation
    S20.

    Inputs:
        Ts        - surface temperature [K]
        gradient  - total temperature gradient in the ocean [dimensionless]
        min_temp  - the minimum allowable temperature at the bottom of the
                    ocean. For an Earth-like planet below 271.15 K (the default
                    value) the ocean would freeze.

    Returns:
        Td - the temperature at the bottom of the ocean [K]
    """

    # intercept chosen to reproduce initial (modern) temperature
    intercept = 274.037 - gradient*285
    Td = np.max([np.min([gradient*Ts+intercept, Ts]), min_temp])

    return Td

def poreSpaceTemp(Td):
    return Td+9

def equilibriumRateConstants(T):
    """
    Calculates the carbon chemistry equilibrium constants as a function of
    temperature following the method in Appendix A of JKT 2018 (you actually
    have to look at their 2017 paper for these equations).

    Inputs:
        T - the temperature of the system [K]

    Returns:
        K1    - the first apparent dissociation rate constant of carbonic acid
        K2    - the second apparent dissociation rate constant of carbonic acid
        H_CO2 - Henry's law constant for CO2
    """

    pK1 = 17.788 - .073104 * T - .0051087 * 35 + 1.1463 * 10 ** -4 * T ** 2
    pK2 = 20.919 - .064209 * T - .011887 * 35 + 8.7313 * 10 ** -5 * T ** 2
    H_CO2 = np.exp(9345.17 / T - 167.8108 + 23.3585 * np.log(T) +
                (.023517 - 2.3656 * 10 ** -4 * T + 4.7036 * 10 ** -7 * T ** 2) * 35)

    K1 = 10.0 ** -pK1
    K2 = 10.0 ** -pK2

    return [K1, K2, H_CO2]

def carbonateSolubility(T):
    """
    Calculates carbonate solubility rate constant as a function of temperature.
    See Appendix A of JKT 2018 for further details (you'll need to look at
    their 2017 paper for these actual equations - but Appendix A tells you
    that).

    Inputs:
        T - the temperature of the system [K]

    Returns:
        result - the solubility rate constant
    """
    bo = -0.77712
    b1 = 0.0028426
    b2 = 178.34
    co = -0.07711
    do = 0.0041249
    S = 35.0
    logK0=-171.9065-0.077993*T+2839.319/T+71.595*np.log10(T)
    logK=logK0+(bo+b1*T+b2/T)*S**0.5+co*S+do*S**1.5

    result = 10.0**logK

    return result

def equilibriumChemistryOcean(T, alk, carb, s, alk_init, Ca_init):
    """
    Calculate the carbonate equilibrium and alkalinity for ocean reservoirs. This can be used for
    either the atmosphere-ocean or the pore space. This function represents
    equations S11-S18.

    Inputs:
        T        - the temperature of the system [K]
        alk      - the alkalinity of the system [mol eq]
        carb     - the carbon abundance in the system [mol]
        s        - correction factor for mass balance
        alk_init - initial alkalinity of the system [mol eq] - literally the value you start from when evolving (can be a guess)
        Ca_init  - initial calcium ion concentration in the system [mol]

    Returns:
        omega - the saturation state of the system
        pCO2  - the partial pressure of CO2 [bar]
        pH    - pH of the system
    """

    # get the rate constants and Henry's constant

    [K1, K2, H_CO2] = equilibriumRateConstants(T)

    # use equation S15 to first calculate the H+ ion concentration
    roots = np.roots([alk / (K1 * K2) * (1.0 + (s / H_CO2)),
                      (alk - carb) / K2,
                      alk - 2.0 * carb])

    H_ion = np.max(roots)  # just take the positive root
    pH = -np.log10(H_ion)  # equation S16 (aka pH definition)

    CO3 = alk / (2.0 + H_ion / K2)  # S14 with S11
    HCO3 = alk - 2.0 * CO3  # S11
    CO2_aq = H_ion * HCO3 / K1  # S13
    pCO2 = CO2_aq / H_CO2  # S12
    Ca_ion = 0.5 * (alk - alk_init) + Ca_init  # S17
    K_sp = carbonateSolubility(T)
    omega = Ca_ion * CO3 / K_sp  # S18

    return [omega, pCO2, pH]

