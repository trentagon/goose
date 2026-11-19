
import time
import numpy as np
from functions import chemistry as cm

class inputs:
    def __init__(self,name):
        self.name = name
    
    def shuffle(self, seed=None):

        if seed:
            np.random.seed(seed) # Allow for seed setting for reproducability
            
        #Modern ocean parameters
        self.modern_Ts = 285  # [K] pre-industrial modern value
        self.modern_pCO2 = 0.000280  # [bar] pre-industrial modern value
        self.modern_pH = 8.2  # [unitless] pre-industrial modern value for surface ocean
        self.modern_ocean_mass = 1.35e21 # [kg] ocean mass
        self.modern_s = 1.8e20/self.modern_ocean_mass # mass balance for ocean and atmosphere
        self.modern_pore_mass = 1.35e19 # [kg] mass of pore space water
        self.modern_Ca_ocean = 0.01027  # [mol kg-1] modern Ca molality in ocean
        
        # Modern mixing
        self.modern_J = np.random.uniform(5.4e15,2.0e16) # [kg yr-1] mixing flux of ocean and pore [70-250 kyr mixing time from Coogan and Gillis 2018]

        # Modern C and A fluxes
        self.modern_V = np.random.uniform(6.5e12,10.5e12)  # [mol C yr-1] total volcanism rate (varies in range 6-10e12)
        self.modern_W_carb = np.random.uniform(7e12,14e12)  # [mol C yr-1] continental carbonate weathering rate (varies in range 7-14 e12)
        self.modern_P_pore = 0.45e12  # [mol C yr-1] pore space carbonate deposition rate (used to calculate seafloor weathering)
        self.modern_pvw_x = np.random.uniform(0.5,1.5)  # [unitless] relationship between modern pore space precipitation and seafloor weathering (varies in range 0.5-1.5)

        #Modern Parameters
        self.a_grad = np.random.uniform(0.8,1.4) # [unitless] determines relationship between deep and surface temp (varies in range 0.8-1.4)
        self.carb_n = np.random.uniform(1.0,2.5) # [unitless] omega dependency exponent for shelf and pore carbonate deposition (varies in range 1-2.5)
        self.carb_xi = np.random.uniform(0.1, 0.5) # [unitless] carbonate weathering dependence exponent on pCO2 (varies in range 0.1-0.5)
        self.sil_alpha = np.random.uniform(0.1, 0.5) # [unitless] silicate weathering dependence exponent on pCO2 (varies in range 0.1-0.5)
        self.cont_Te = np.random.uniform(10.0,40.0) # [K] e-folding temperature for continental weathering rates (varies in range 10-40)
        self.E_bas = np.random.uniform(60.0,100.0) # [kJ mol-1] activation energy for basalt dissolution during seafloor weathering (varies in range 60-100)
        self.sea_gamma = np.random.uniform(0.0,0.5) # [unitless] ph dependence exponent for seafloor weathering (varies in range 0-0.5)
        self.modern_frac_pel = np.random.uniform(0.4,0.6) # [unitless] fraction of ocean carbonates that are pelagic instead of shelf (0 = all shelf, 1 = all pelagic)

        #Neoproterozoic Parameters
        self.neoproterozoic_luminosity = np.random.uniform(0.93,0.94) # [unitless] solar luminosity relative to modern
        self.shelf_area = 1 # [unitless] area of continental shelves relative to modern
        self.f_w = np.random.uniform(0.5,1.5) # [unitless] biological weathering factor (varies in range 0.1-1)

        #Snowball Parameters
        self.snowball_ocean_temp = 269.5 # [K] fixed snowball ocean temperature
        self.snowball_ice_volume = np.random.uniform(0.1,0.3) # [unitless] fraction of ocean that is frozen in the ice sheet
        self.snowball_ocean_mass = self.modern_ocean_mass*(1-self.snowball_ice_volume) # [kg] ocean mass under the ice sheet
        self.snowball_s = 1.8e20/self.snowball_ocean_mass # mass balance for ocean and atmosphere
        self.snowball_shelf_area = np.random.uniform(0.3,0.5) # [unitless] snowball continental shelf area relative to modern
        self.snowball_W_carb = np.random.uniform(8e8,18e8) # [mol C yr-1] continental carbonate weathering rate (varies in range 8-18 e8)
        self.snowball_W_sil = np.random.uniform(3.3e8,22e8) # [mol eq yr-1] continental silicate weathering rate (varies in range 3.3-22 e8)
        self.hyalo = 1 # [unitless] multiplier on the seafloor weathering rate to simulate hyaloclastite

        #Post-glacial Parameters
        self.meltwater_alk = np.random.uniform(700e-6,1000e-6) # [mol eq kg-1] alkalinity of modern glacial meltwater
        self.meltwater_Ca = np.random.uniform(211e-6, 307e-6) # [mol kg-1] calcium molality of modern glacial meltwater
        self.meltwater_mass = self.modern_ocean_mass*self.snowball_ice_volume # [kg] mass of the glacial meltwater
        self.meltwater_s = 1.8e20/self.meltwater_mass # mass balance for ocean and atmosphere
        self.V_frac_modern = 0.75 # [unitless] (subaerial volcanism)/(total volcanism) on modern earth
        self.post_shelf_area = np.random.uniform(1,2) # [unitless] post-glacial continental shelf area relative to modern
        self.O_sink = np.random.uniform(50e12,200e12) # [mol yr-1] global rate of organic carbon export in post-glacial ocean
        self.modern_K = 9.6e17 # [kg yr-1] modern mixing flux between surface and deep ocean
        self.t_strat = np.random.uniform(1e3,1e5) # [yr] duration of the stratified post-glacial ocean phase

        #Values that need calibration to the modern day
        self.modern_W_sea = -1
        self.modern_W_sil = -1
        self.modern_other_salt = -1
        self.modern_Ca_pore = -1
        self.modern_Hp = -1
        self.k_shelf = -1
        self.k_pel = -1
        self.k_pore = -1
        self.k_Wsea = -1

    def redo_calculations(self):
        self.modern_s = 1.8e20/self.modern_ocean_mass # mass balance for ocean and atmosphere
        self.snowball_ocean_mass = self.modern_ocean_mass*(1-self.snowball_ice_volume)
        self.snowball_s = 1.8e20/self.snowball_ocean_mass

        self.meltwater_mass = self.modern_ocean_mass*self.snowball_ice_volume
    
    def calibrate_to_modern(self):

        # Calculate initial climate and ocean chemistry
        Ts = cm.climate_model_JKT(self.modern_pCO2, 0.9906) # set luminosity to recreate modern temp (effictively changing albed within error)
        #Ts = self.modern_Ts
        Td = cm.deepOceanTemp(Ts, self.a_grad)
        Tp = cm.poreSpaceTemp(Td)
        [K1o, K2o, H_CO2] = cm.equilibriumRateConstants(Ts)

        # Use steady state assumption, modern outgassing rate, modern pore space carbonate precipitation rate, and modern ratio
        # of seafloor weathering to pore space precipitation to calculate seafloor and continental weathering rates
        self.modern_W_sea = self.modern_pvw_x*self.modern_P_pore
        self.modern_W_sil = self.modern_V - self.modern_W_sea

        # Get initial chemical conditions for modern surface reservoir
        CO2aq_o = H_CO2 * self.modern_pCO2 # dissolved CO2 in ocean
        HCO3_o = K1o * CO2aq_o / (10.0 ** -self.modern_pH) # dissolved bicarbonate
        CO3_o = K2o * HCO3_o / (10.0 ** -self.modern_pH) # dissolved carbonate
        DIC_o = CO3_o + HCO3_o + CO2aq_o  # total dissolved inorganic carbon
        ALK_o = (2*CO3_o) + HCO3_o
        self.modern_other_salt = ALK_o - (2.0*self.modern_Ca_ocean)  # other salt in the ocean besides calcium, consistent with modern alk and Ca

        # Modern Co and Ao values from ocean chemistry and rates
        self.modern_Co = DIC_o + (self.modern_s*self.modern_pCO2)
        self.modern_Ao = ALK_o

        # Solve the rest of the equations with steady state assumption to get pore space chemistry and carbonate precipitation
        DIC_p = DIC_o - (self.modern_P_pore/self.modern_J)
        ALK_p = ALK_o - ((2.0/self.modern_J)*(self.modern_P_pore-self.modern_W_sea))
        self.modern_P_ocean = -(self.modern_J*(DIC_o-DIC_p)) + self.modern_V + self.modern_W_carb

        # Modern Cp and Ap values from ocean chemistry and rates
        self.modern_Cp = DIC_p
        self.modern_Ap = ALK_p

        # Calculate chemistry in pore space
        [K1p, K2p, H_CO2p] = cm.equilibriumRateConstants(Tp)

        roots_pore = np.roots([self.modern_Ap / (K1p * K2p),
                               (self.modern_Ap - self.modern_Cp) / K2p,
                               self.modern_Ap - (2.0 * self.modern_Cp)])
        self.modern_Hp = np.max(roots_pore) # H ions in pore space
        pH_p = -np.log10(self.modern_Hp) # pH in the pore space


        CO2aq_p = DIC_p/(1 + (K1p/self.modern_Hp) + (K1p*K2p/(self.modern_Hp**2))) # dissolved CO2 in pore space
        HCO3_p = K1p * CO2aq_p / (10.0 ** -pH_p)  # dissolved bicarbonate in pore space
        CO3_p = K2p * HCO3_p / (10.0 ** -pH_p)  # dissolved carbonate in pore space
        self.modern_Ca_pore = 0.5 * (self.modern_Ap - self.modern_other_salt) # calcium in pore space

        # Saturation states
        omega_o = self.modern_Ca_ocean * CO3_o / cm.carbonateSolubility(Ts)
        omega_p = self.modern_Ca_pore * CO3_p / cm.carbonateSolubility(Tp)

        # Get proportionality constants for carbonate precipitation
        self.k_shelf = (1-self.modern_frac_pel)*self.modern_P_ocean/ (omega_o - 1)**self.carb_n
        self.k_pel = (self.modern_frac_pel)*self.modern_P_ocean / omega_o**2.84 # pelagic deposition is 0 during the neoproterozoic, so this value is not used
        self.k_pore = self.modern_P_pore / (omega_p - 1) ** self.carb_n

        # Get proportionality constants
        R = 8.314
        self.k_Wsea = self.modern_W_sea / np.exp(-self.E_bas / (R * Tp))

        #Recalculate downstream values just in case a parameter was changed (this is not calibration to modern)
        self.modern_s = 1.8e20/self.modern_ocean_mass # mass balance for ocean and atmosphere
        self.snowball_ocean_mass = self.modern_ocean_mass*(1-self.snowball_ice_volume)
        self.snowball_s = 1.8e20/self.snowball_ocean_mass 

        return