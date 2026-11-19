import numpy as np
import matplotlib.pyplot as plt

# This file contains functions that are useful for visualizing model outputs.
# It is called "solo" because these functions work for single model runs.

def plotFullClimEvol_solo(model_output):

    # Extract data
    neop_data = model_output.neo_output.data
    snow_data = model_output.snow_output.data
    stra_data = model_output.strat_output.data
    mixe_data = model_output.mixed_output.data
    
    # Combine the times
    t_snow = snow_data['t'].to_numpy()
    t_stra = stra_data['t'].to_numpy() + t_snow[-1] + 1
    t_mixe = mixe_data['t'].to_numpy() + t_stra[-1] + 1
    
    t = np.concatenate((t_snow,t_stra,t_mixe))/1e6 # Concatenate and convert to Myrs
    t = np.concatenate(([-5],t,[20])) # Pad data for visualizing steady states at start and end
    
    # Combine the variables
    pCO2  = np.concatenate((neop_data["pCO2_o"].to_numpy(),snow_data["pCO2_o"].to_numpy(),stra_data["pCO2_s"].to_numpy(),
                            mixe_data["pCO2_o"].to_numpy(),[mixe_data["pCO2_o"].to_numpy()[-1]]))
    
    SST   = np.concatenate((neop_data["Ts"].to_numpy(),snow_data["Ts"].to_numpy(),stra_data["Ts"].to_numpy(),
                            mixe_data["Ts"].to_numpy(),[mixe_data["Ts"].to_numpy()[-1]]))
    
    pH    = np.concatenate((neop_data["pH_o"].to_numpy(),snow_data["pH_o"].to_numpy(),stra_data["pH_d"].to_numpy(),
                            mixe_data["pH_o"].to_numpy(),[mixe_data["pH_o"].to_numpy()[-1]]))
    
    omega = np.concatenate((neop_data["Omega_o"].to_numpy(),snow_data["Omega_o"].to_numpy(),stra_data["Omega_s"].to_numpy(),
                            mixe_data["Omega_o"].to_numpy(),[mixe_data["Omega_o"].to_numpy()[-1]]))
    
    alk   = np.concatenate((neop_data["Ao"].to_numpy(),snow_data["Ao"].to_numpy(),stra_data["As"].to_numpy(),
                            mixe_data["Ao"].to_numpy(),[mixe_data["Ao"].to_numpy()[-1]]))
    
    
    # Plot it
    plt.rcParams['font.family'] = 'DIN Alternate'
    
    # Colors to use
    color1 = 'goldenrod'
    color2 = 'seagreen'
    color3 = 'royalblue'
    color4 = 'firebrick'
    color5 = 'rebeccapurple'
    
    # Create the figure
    fig, ax = plt.subplots(5, 1, sharex=True)
    fig.canvas.manager.set_window_title('Full Evolution')
    fig.set_size_inches(7,8)
    
    # Merge the plots
    fig.subplots_adjust(hspace=0, wspace=0.2)
    ax[0].spines['bottom'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    ax[2].spines['top'].set_visible(False)
    ax[2].spines['bottom'].set_visible(False)
    ax[3].spines['top'].set_visible(False)
    ax[3].spines['bottom'].set_visible(False)
    ax[4].spines['top'].set_visible(False)
    

    # Add vertical lines
    degt = model_output.snow_output.snow_max_duration/1e6
    strt = degt + (model_output.snow_output.input_file.t_strat/1e6)
    vline_color = "lightsteelblue"
    for j in range(5):
        ax[j].axvspan(0,degt, color='aliceblue', alpha=1)
        # ax[j].axvspan(degt,stratt, color='aliceblue', alpha=1)
        ax[j].axvline(x=0, color=vline_color, linestyle='--', linewidth=1)
        ax[j].axvline(x=degt, color=vline_color, linestyle='--', linewidth=1)
        ax[j].axvline(x=strt, color='k', linestyle='--', linewidth=0.5)
    
    # Add text for phases
    ax[0].text(0,    1.1, 'Phase 1', transform=ax[0].transAxes, fontsize=10, color="slategrey", va='top')
    ax[0].text(0.25, 1.1, 'Phase 2', transform=ax[0].transAxes, fontsize=10, color="slategrey", va='top')
    ax[0].text(0.5,  1.1, 'Phase 3', transform=ax[0].transAxes, fontsize=10, color="slategrey", va='top')
    ax[0].text(0.75, 1.1, 'Phase 4', transform=ax[0].transAxes, fontsize=10, color="slategrey", va='top')
    
    hline_color = 'slategray'
    hline_width = 1
    ax[0].axhline(y=4e-4, color = hline_color,linewidth = hline_width, linestyle = '--')
    ax[2].axhline(y=8.1, color = hline_color,linewidth = hline_width, linestyle = '--')
    ax[4].axhline(y=2320e-6, color = hline_color,linewidth = hline_width, linestyle = '--')
    
    ax[0].text(0.28, 0.26, 'modern', transform=ax[0].transAxes, fontsize=10, color=hline_color, va='top')
    ax[2].text(0.28, 0.72, 'modern', transform=ax[2].transAxes, fontsize=10, color=hline_color, va='top')
    ax[4].text(0.28, 0.2, 'modern', transform=ax[4].transAxes, fontsize=10, color=hline_color, va='top')
    
    yfs = 13 # Y-axis font size
    xmin = -2 # X-axis minimum (Myr)
    xmax = 20 # X-axis maximum (Myr)
    
    # Plot pCO2
    ax[0].plot(t, pCO2,color = color1, linewidth=2.5)
    ax[0].set_yscale('log')
    ax[0].set_ylabel(r'pCO$_2$ (bar)', fontsize=yfs)
    ax[0].set_xlim([xmin, xmax])
    ax[0].set_yticks(np.logspace(-4,0,5))
    ax[0].minorticks_off()
    ax[0].set_ylim([1e-5,1e1])
    
    # Plot SST
    ax[1].plot(t, SST,color = color2, linewidth=2.5)
    ax[1].set_ylabel(r'SST (K)', fontsize=yfs)
    ax[1].set_xlim([xmin, xmax])
    ax[1].set_yticks(np.linspace(265,315,6))
    ax[1].set_ylim([260,320])
    
    # Plot pH
    ax[2].plot(t, pH,color = color3, linewidth=2.5)
    ax[2].set_ylabel(r'pH', fontsize=yfs)
    ax[2].set_xlim([xmin, xmax])
    ax[2].set_yticks([6,7,8,9])
    ax[2].set_ylim([5,10])
    
    # Plot Omega
    ax[3].plot(t, omega,color = color4, linewidth=2.5)
    ax[3].set_ylabel('Saturation \n State, $\Omega$', fontsize=yfs)
    ax[3].set_xlim([xmin, xmax])
    ax[3].set_yticks([0,10,20,30,40,50])
    ax[3].set_xticks(np.linspace(0,20,11))
    ax[3].set_ylim([-5,55])
    
    # Plot Alk
    ax[4].plot(t,alk,color = color5, linewidth=2.5)
    ax[4].set_ylabel('Alkalinity \n (mol eq kg$^{-1}$)', fontsize=yfs)
    ax[4].set_xlabel('Time Since Glaciation (Myr)', fontsize=yfs)
    ax[4].set_xlim([xmin, xmax])
    ax[4].set_yticks([0,0.01,0.02,0.03,0.04])
    ax[4].set_xticks(np.linspace(0,20,11))
    ax[4].set_ylim([-0.01,0.05])
    
    fig.align_ylabels()
    plt.show()
    return



def plotStratClimEvol_solo(model_output):

    # Extract data
    neop_data = model_output.neo_output.data
    snow_data = model_output.snow_output.data
    stra_data = model_output.strat_output.data
    mixe_data = model_output.mixed_output.data
    
    # Combine the times
    t_snow = snow_data['t'].to_numpy()
    t_stra = stra_data['t'].to_numpy() + t_snow[-1]
    t_mixe = mixe_data['t'].to_numpy() + t_stra[-1] + 1
    
    t = np.concatenate(([9e6],t_stra,[11e6]))/1e6 # Concatenate and convert to Myrs
    t_melt = t_stra/1e6
    
    # Combine the variables
    pCO2  = np.concatenate(([snow_data["pCO2_o"].to_numpy()[1]],stra_data["pCO2_s"].to_numpy(),[mixe_data["pCO2_o"].to_numpy()[0]]))
    
    T_deep_time = np.concatenate(([9e6],t_stra))/1e6 # Need custom time for temperature
    T_surf  = stra_data["Ts"].to_numpy()
    T_deep  = np.concatenate(([snow_data["Td"].to_numpy()[1]],stra_data["Td"].to_numpy()))
    
    pH_surf  = stra_data["pH_s"].to_numpy()
    pH_deep  = np.concatenate(([snow_data["pH_o"].to_numpy()[1]],stra_data["pH_d"].to_numpy(),[mixe_data["pH_o"].to_numpy()[0]]))
    
    omega_surf  = stra_data["Omega_s"].to_numpy()
    omega_deep  = np.concatenate(([snow_data["Omega_o"].to_numpy()[1]],stra_data["Omega_d"].to_numpy(),[mixe_data["Omega_o"].to_numpy()[0]]))
    
    alk_surf  = stra_data["As"].to_numpy()
    alk_deep  = np.concatenate(([snow_data["Ao"].to_numpy()[1]],stra_data["Ad"].to_numpy(),[mixe_data["Ao"].to_numpy()[0]]))
    
    fig, ax = plt.subplots(2, 3)
    fig.canvas.manager.set_window_title('Stratified Evolution')
    fig.subplots_adjust(hspace=0.35, wspace=0.2)
    fig.set_tight_layout("tight")
    fig.set_size_inches(12,7)

    # Colors to use
    color1 = 'goldenrod'
    color2 = 'seagreen'
    color3 = 'royalblue'
    color4 = 'firebrick'
    color5 = 'rebeccapurple'

    stratt = model_output.snow_output.input_file.t_strat/1e6
    
    l1 = model_output.snow_output.snow_max_duration/1e6
    l2 = model_output.snow_output.snow_max_duration/1e6 + stratt

    xticks1 = np.round(np.linspace(l1,l2,5),4)
    
    xmin2 = l1 - (stratt/4)
    xmax2 = l2 + (stratt/4)

    melt_color = 'dimgrey'
    vline_color = "lightsteelblue"
    for j in range(2):
        for k in range(3):
            if j == 1 and k == 0:
                pass
            else:
                ax[j,k].axvspan(xmin2,l1, color='aliceblue', alpha=1)
                ax[j,k].axvline(x=l1, color=vline_color, linestyle='--', linewidth=1)
                ax[j,k].axvline(x=l2, color="k", linestyle='--', linewidth=1)
    
    ax[0, 0].text(0.43, 1.05, 'Phase 3', transform=ax[0, 0].transAxes, fontsize=10, color="slategrey", va='top')
    ax[0, 1].text(0.43, 1.05, 'Phase 3', transform=ax[0, 1].transAxes, fontsize=10, color="slategrey", va='top')
    ax[0, 2].text(0.43, 1.05, 'Phase 3', transform=ax[0, 2].transAxes, fontsize=10, color="slategrey", va='top')
    ax[1, 1].text(0.43, 1.05, 'Phase 3', transform=ax[1, 1].transAxes, fontsize=10, color="slategrey", va='top')
    ax[1, 2].text(0.43, 1.05, 'Phase 3', transform=ax[1, 2].transAxes, fontsize=10, color="slategrey", va='top')
    
    letter_pos_x2 = 0.03
    
    yfs2 = 14 # y-axis font size
    xfs2 = 13 # x-axis font size
    
    # Plot pco2
    ax[0, 0].plot(t, pCO2,color = color1, linewidth=2.5)
    ax[0, 0].set_yscale('log')
    ax[0, 0].set_ylabel(r'pCO$_2$ (bar)', fontsize=yfs2)
    ax[0, 0].set_xlabel(r"Time Since Glaciation (Myr)", fontsize=xfs2)
    ax[0, 0].set_xlim([xmin2, xmax2])
    ax[0, 0].set_xticks(xticks1)
    ax[0, 0].set_xticklabels(xticks1)
    ax[0, 0].set_ylim([1e-4,1e1])
    ax[0, 0].text(letter_pos_x2, 0.98, 'A', transform=ax[0,0].transAxes, fontsize=16, fontweight='bold', va='top')
    
    # Plot temperature
    ax[0, 1].plot(t_melt, T_surf,color = melt_color, linewidth=2.5, label = 'Meltwater Layer')
    ax[0, 1].plot(T_deep_time, T_deep,color = color2, linewidth=2.5, label = 'Deep Ocean')
    ax[0, 1].set_ylabel("Temperature (K)",fontsize=yfs2)
    ax[0, 1].set_xlabel(r"Time Since Glaciation (Myr)", fontsize=xfs2)
    ax[0, 1].set_xlim([xmin2, xmax2])
    ax[0, 1].set_xticks(xticks1)
    ax[0, 1].set_xticklabels(xticks1)
    ax[0, 1].set_ylim([265,335])
    ax[0, 1].legend()
    ax[0, 1].text(letter_pos_x2, 0.98, 'B', transform=ax[0,1].transAxes, fontsize=16, fontweight='bold', va='top')
    
    # Plot pH
    ax[0, 2].plot(t_melt, pH_surf,color = melt_color, linewidth=2.5, label='Meltwater Layer')
    ax[0, 2].plot(t, pH_deep,color = color3, linewidth=2.5, label='Ocean')
    ax[0, 2].set_ylabel(r'pH', fontsize=yfs2)
    ax[0, 2].set_xlabel(r"Time Since Glaciation (Myr)", fontsize=xfs2)
    ax[0, 2].set_xlim([xmin2,xmax2])
    ax[0, 2].set_xticks(xticks1)
    ax[0, 2].set_xticklabels(xticks1)
    ax[0, 2].set_ylim([4.5, 10])
    ax[0, 2].legend(loc='upper center')
    ax[0, 2].text(letter_pos_x2, 0.98, 'C', transform=ax[0,2].transAxes, fontsize=16, fontweight='bold', va='top')
    
    # Turn this one off
    ax[1, 0].axis('off')
    
    # Plot omega
    ax[1, 1].plot(t_melt, omega_surf,color = melt_color, linewidth=2.5, label='Meltwater Layer')
    ax[1, 1].plot(t, omega_deep,color = color4, linewidth=2.5, label='Ocean')
    ax[1, 1].set_ylabel(r'Saturation State, $\Omega$', fontsize=yfs2)
    ax[1, 1].set_xlabel(r"Time Since Glaciation (Myr)", fontsize=xfs2)
    ax[1, 1].set_xlim([xmin2,xmax2])
    ax[1, 1].set_xticks(xticks1)
    ax[1, 1].set_xticklabels(xticks1)
    ax[1, 1].set_ylim([-5, 70])
    ax[1, 1].legend()
    ax[1, 1].text(letter_pos_x2, 0.98, 'D', transform=ax[1,1].transAxes, fontsize=16, fontweight='bold', va='top')
    
    # Plot alk
    ax[1, 2].plot(t_melt, alk_surf,color = melt_color, linewidth=2.5, label='Meltwater Layer')
    ax[1, 2].plot(t, alk_deep,color = color5, linewidth=2.5, label='Ocean')
    ax[1, 2].set_ylabel(r'Alkalinity (mol eq kg$^{-1}$)', fontsize=yfs2)
    ax[1, 2].set_xlabel(r"Time Since Glaciation (Myr)", fontsize=xfs2)
    ax[1, 2].set_xlim([xmin2,xmax2])
    ax[1, 2].set_xticks(xticks1)
    ax[1, 2].set_xticklabels(xticks1)
    #ax[1, 2].set_ylim([0, 70])
    ax[1, 2].legend()
    ax[1, 2].text(letter_pos_x2, 0.98, 'E', transform=ax[1,2].transAxes, fontsize=16, fontweight='bold', va='top')
    
    plt.show()
    return


def plotPostGlacialCarbMass_solo(model_output):

    # Extract data
    neop_data = model_output.neo_output.data
    snow_data = model_output.snow_output.data
    stra_data = model_output.strat_output.data
    mixe_data = model_output.mixed_output.data
    
    # Combine the times
    t_stra = stra_data['t'].to_numpy()
    t_mixe = mixe_data['t'].to_numpy() + t_stra[-1] + 1
    t = np.concatenate((t_stra,t_mixe))
    tsteps = np.diff(t)
    
    pshelf_g_per_yr  = np.concatenate((stra_data["P_shelf"].to_numpy(), mixe_data["P_shelf"].to_numpy()))*100.01
    cumulative_carbs = np.cumsum(pshelf_g_per_yr[:-1]*tsteps)
    
    plt.rcParams['font.family'] = 'DIN Alternate'
    
    fig, ax = plt.subplots(1, 2)
    fig.subplots_adjust(hspace=0.35, wspace=0.2)
    fig.set_tight_layout("tight")
    fig.set_size_inches(10,4)
    
    color1 = 'midnightblue'
    color2 = 'darkolivegreen'
    
    yfs = 13
    
    xmin = 1e2
    xmax = 30e6
    
    strat_line = model_output.snow_output.input_file.t_strat
    
    ax[0].plot(t, pshelf_g_per_yr,color = color1, linewidth=2.5)
    ax[0].axvline(x=strat_line, color='k', linestyle='--', linewidth=1)
    ax[0].set_ylabel('Carbonate Deposition \n Rate (g CaCO$_3$ yr$^{-1}$)', fontsize=yfs)
    ax[0].set_xlabel('Time Since Deglaciation (yr)', fontsize=yfs)
    ax[0].set_xlim([xmin, xmax])
    ax[0].set_ylim([1e13, 1e18])
    ax[0].set_xticks([1e2,1e3,1e4,1e5,1e6,1e7])
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].text(0.02, 0.98, 'A', transform=ax[0].transAxes, fontsize=16, fontweight='bold', va='top')
    ax[0].text(0.11, 1.06, 'Phase 3', transform=ax[0].transAxes, fontsize=10, color="grey", va='top')
    ax[0].text(0.6, 1.06, 'Phase 4', transform=ax[0].transAxes, fontsize=10, color="grey", va='top')
    
    ax[1].plot(t[:-1], cumulative_carbs,color = color2, linewidth=2.5, label="Model results")
    ax[1].axvline(x=strat_line, color='k', linestyle='--', linewidth=1)
    ax[1].axhline(y=4.2e21, color = 'grey', linestyle=':', label="Estimates for global mass \n of CCs (Yu et al. 2020)")
    ax[1].axhline(y=9.3e21, color = 'grey', linestyle=':')
    ax[1].axhline(y=14.4e21,color = 'grey', linestyle=':')
    ax[1].set_ylabel('Cumulative Carbonate \n Deposition (g CaCO$_3$)', fontsize=yfs)
    ax[1].set_xlabel('Time Since Deglaciation (yr)', fontsize=yfs)
    ax[1].set_xlim([xmin, xmax])
    ax[1].set_ylim([1e14, 1e26])
    ax[1].set_xticks([1e2,1e3,1e4,1e5,1e6,1e7])
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].legend(loc="lower right")
    ax[1].text(0.02, 0.98, 'B', transform=ax[1].transAxes, fontsize=16, fontweight='bold', va='top')
    ax[1].text(0.11, 1.06, 'Phase 3', transform=ax[1].transAxes, fontsize=10, color="grey", va='top')
    ax[1].text(0.6, 1.06, 'Phase 4', transform=ax[1].transAxes, fontsize=10, color="grey", va='top')
    
    plt.show()
    return

def plotPostGlacialCarbHeight_solo(model_output):
     # Extract data
    neop_data = model_output.neo_output.data
    snow_data = model_output.snow_output.data
    stra_data = model_output.strat_output.data
    mixe_data = model_output.mixed_output.data
    
    # Combine the times
    t_stra = stra_data['t'].to_numpy()
    t_mixe = mixe_data['t'].to_numpy() + t_stra[-1] + 1
    t = np.concatenate((t_stra,t_mixe))
    tsteps = np.diff(t)
    
    pshelf_g_per_yr  = np.concatenate((stra_data["P_shelf"].to_numpy(), mixe_data["P_shelf"].to_numpy()))*100.01
    cumulative_carbs = np.cumsum(pshelf_g_per_yr[:-1]*tsteps)
    
    rho = 2.71*1e15 # g/km3
    dep_area = 2*3.22e7 # km2
    
    pshelf_m_per_yr = 1e3*pshelf_g_per_yr/(rho*dep_area)
    cumulative_carbs_height = 1e3*cumulative_carbs/(rho*dep_area)
    
    plt.rcParams['font.family'] = 'DIN Alternate'
    
    fig, ax = plt.subplots(1, 2)
    fig.subplots_adjust(hspace=0.35, wspace=0.2)
    fig.set_tight_layout("tight")
    fig.set_size_inches(10,4)
    
    color1 = 'darkred'
    color2 = 'darkolivegreen'
    
    yfs = 13
    
    ax[0].plot(np.trim_zeros(pshelf_m_per_yr[:-1], 'f'), np.trim_zeros(cumulative_carbs_height, 'f'),color = color1, linewidth=2.5)
    ax[0].set_ylabel('Thickness of Carbonate \n Deposit (m)', fontsize=yfs)
    ax[0].set_xlabel('Carbonate Deposition Rate (m yr$^{-1}$)', fontsize=yfs)
    ax[0].set_xscale('log')
    ax[0].set_xlim([1e-7,1e-2])
    ax[0].set_yscale('log')
    ax[0].set_ylim([0.01,1000])
    ax[0].get_yaxis().get_major_formatter().labelOnlyBase = False
    ax[0].set_yticks([1e-2,1e-1,1e0,1e1,1e2,1000])
    ax[0].set_yticklabels(['0.01','0.1','1','10','100','1000'])
    ax[0].text(0.02, 0.98, 'A', transform=ax[0].transAxes, fontsize=16, fontweight='bold', va='top')
    
    ax[1].plot(t[:-1], cumulative_carbs_height,color = color2, linewidth=2.5)
    ax[1].set_ylabel('Thickness of Carbonate \n Deposit (m)', fontsize=yfs)
    ax[1].set_xlabel('Time Since Deglaciation (yr)', fontsize=yfs)
    ax[1].set_xscale('log')
    ax[1].set_xlim([1e2,3e7])
    ax[1].set_yscale('log')
    ax[1].set_ylim([0.01,1000])
    ax[1].get_yaxis().get_major_formatter().labelOnlyBase = False
    ax[1].set_yticks([1e-2,1e-1,1e0,1e1,1e2,1000])
    ax[1].set_yticklabels(['0.01','0.1','1','10','100','1000'])
    ax[1].text(0.02, 0.98, 'B', transform=ax[1].transAxes, fontsize=16, fontweight='bold', va='top')
    
    strat_line = model_output.snow_output.input_file.t_strat
    ax[1].axvline(x=strat_line, color='k', linestyle='--', linewidth=1)
    ax[1].text(0.11, 1.06, 'Phase 3', transform=ax[1].transAxes, fontsize=10, color="grey", va='top')
    ax[1].text(0.65, 1.06, 'Phase 4', transform=ax[1].transAxes, fontsize=10, color="grey", va='top')
    
    plt.show()
    return

def plotFluxes_solo(model_output):
    # Extract data
    neop_data = model_output.neo_output.data
    snow_data = model_output.snow_output.data
    stra_data = model_output.strat_output.data
    mixe_data = model_output.mixed_output.data
    
    # Combine the times
    t_snow = snow_data['t'].to_numpy()
    t_stra = stra_data['t'].to_numpy() + t_snow[-1] + 1
    t_mixe = mixe_data['t'].to_numpy() + t_stra[-1] + 1
    
    t = np.concatenate((t_snow,t_stra,t_mixe))/1e6 # Concatenate and convert to Myrs
    t = np.concatenate(([-5],t,[20])) # Pad data for visualizing steady states at start and end
    
    # Combine the variables
    V_total = np.concatenate((neop_data["V"].to_numpy(),snow_data["V"].to_numpy(),stra_data["V_air"].to_numpy() + stra_data["V_ridge"].to_numpy(),mixe_data["V"].to_numpy(),[mixe_data["V"].to_numpy()[-1]]))
    V_ridge = stra_data["V_ridge"].to_numpy()
    V_air = stra_data["V_air"].to_numpy()
    
    P_shelf  = np.concatenate((neop_data["P_shelf"].to_numpy(),snow_data["P_ocean"].to_numpy(),stra_data["P_shelf"].to_numpy(),mixe_data["P_shelf"].to_numpy(),[mixe_data["P_shelf"].to_numpy()[-1]]))
    P_pore  = np.concatenate((neop_data["P_pore"].to_numpy(),snow_data["P_pore"].to_numpy(),stra_data["P_pore"].to_numpy(),mixe_data["P_pore"].to_numpy(),[mixe_data["P_pore"].to_numpy()[-1]]))
    
    W_carb  = np.concatenate((neop_data["W_carb"].to_numpy(),snow_data["W_carb"].to_numpy(),stra_data["W_carb"].to_numpy(),mixe_data["W_carb"].to_numpy(),[mixe_data["W_carb"].to_numpy()[-1]]))
    W_sil  = np.concatenate((neop_data["W_sil"].to_numpy(),snow_data["W_sil"].to_numpy(),stra_data["W_sil"].to_numpy(),mixe_data["W_sil"].to_numpy(),[mixe_data["W_sil"].to_numpy()[-1]]))
    W_sea  = np.concatenate((neop_data["W_sea"].to_numpy(),snow_data["W_sea"].to_numpy(),stra_data["W_sea"].to_numpy(),mixe_data["W_sea"].to_numpy(),[mixe_data["W_sea"].to_numpy()[-1]]))
    W_carb[1] = W_carb[0] # makes the plot look better at transition
    W_sil[1] = W_carb[0] # makes the plot look better at transition
    
    
    O_sink = stra_data["O_sink"].to_numpy()
    
    color1 = 'goldenrod'
    color2 = 'seagreen'
    color3 = 'royalblue'
    color4 = 'firebrick'
    color5 = 'rebeccapurple'
    color6 = 'black'
    
    # Create the figure
    fig, ax = plt.subplots(1, 1, sharex=True)
    fig.canvas.manager.set_window_title('Full Evolution')
    fig.set_size_inches(10,4)
    
    # Merge the plots
    fig.subplots_adjust(hspace=0, wspace=0.2)
    
    # Add vertical lines
    degt = model_output.snow_output.snow_max_duration/1e6
    strt = degt + (model_output.snow_output.input_file.t_strat/1e6)
    vline_color = "lightsteelblue"
    
    ax.axvspan(0,degt, color='aliceblue', alpha=1)
    # ax.axvspan(degt,stratt, color='aliceblue', alpha=1)
    ax.axvline(x=0, color=vline_color, linestyle='--', linewidth=1)
    ax.axvline(x=degt, color=vline_color, linestyle='--', linewidth=1)
    ax.axvline(x=strt, color='k', linestyle='--', linewidth=0.5)
    
    # Add text for phases
    ax.text(0,    1.05, 'Phase 1', transform=ax.transAxes, fontsize=10, color="slategrey", va='top')
    ax.text(0.25, 1.05, 'Phase 2', transform=ax.transAxes, fontsize=10, color="slategrey", va='top')
    ax.text(0.5,  1.05, 'Phase 3', transform=ax.transAxes, fontsize=10, color="slategrey", va='top')
    ax.text(0.75, 1.05, 'Phase 4', transform=ax.transAxes, fontsize=10, color="slategrey", va='top')
    
    
    yfs = 13 # Y-axis font size
    xmin = -2 # X-axis minimum (Myr)
    xmax = 20 # X-axis maximum (Myr)
    
    # Plot pCO2
    ax.plot(t, V_total,color = color1, linewidth=2, label ="Volcanism")
    ax.plot(t, P_shelf,color = color2, linewidth=2, label="Precip: Shelf/Slope")
    ax.plot(t, P_pore,color = color3, linewidth=2, label ="Precip: Pore space")
    ax.plot(t, W_carb,color = color4, linewidth=2, label="Weath: Cont. Carb.")
    ax.plot(t, W_sil,color = color5, linewidth=2, label ="Weath: Cont. Sil.")
    ax.plot(t, W_sea,color = color6, linewidth=2, label ="Weath: Seafloor")
    ax.set_yscale('log')
    ax.set_ylabel('Flux (mol C yr$^{-1}$)', fontsize=yfs)
    ax.set_xlabel('Time Since Glaciation (Myr)', fontsize=yfs)
    ax.set_xlim([xmin, xmax])
    ax.minorticks_off()
    ax.legend()
    plt.show()
    return
