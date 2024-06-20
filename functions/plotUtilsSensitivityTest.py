import numpy as np
import matplotlib.pyplot as plt

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

    stra_sens = model_output.strat_output.sens
    t_pct = stra_sens["t_pct"].to_numpy()
    m_pct = stra_sens["m_pct"].to_numpy()
    current_melt_mass = stra_sens["current_melt_mass"].to_numpy()
    current_melt_s = stra_sens["current_melt_s"].to_numpy()

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
            ax[j,k].axvspan(xmin2,l1, color='aliceblue', alpha=1)
            ax[j,k].axvline(x=l1, color=vline_color, linestyle='--', linewidth=1)
            ax[j,k].axvline(x=l2, color="k", linestyle='--', linewidth=1)
    
    ax[0, 0].text(0.43, 1.05, 'Phase 3', transform=ax[0, 0].transAxes, fontsize=10, color="slategrey", va='top')
    ax[0, 1].text(0.43, 1.05, 'Phase 3', transform=ax[0, 1].transAxes, fontsize=10, color="slategrey", va='top')
    ax[0, 2].text(0.43, 1.05, 'Phase 3', transform=ax[0, 2].transAxes, fontsize=10, color="slategrey", va='top')
    ax[1, 0].text(0.43, 1.05, 'Phase 3', transform=ax[1, 0].transAxes, fontsize=10, color="slategrey", va='top')
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
    
    # Plot meltwater layer mass
    ax[1, 0].plot(t_melt, current_melt_mass,color = melt_color, linewidth=2.5)
    ax[1, 0].set_ylabel(r'Meltwater Layer Mass (kg)', fontsize=yfs2)
    ax[1, 0].set_xlabel(r"Time Since Glaciation (Myr)", fontsize=xfs2)
    ax[1, 0].set_xlim([xmin2,xmax2])
    ax[1, 0].set_xticks(xticks1)
    ax[1, 0].set_xticklabels(xticks1)
    #ax[1, 0].set_ylim([-5, 70])
    ax[1, 0].text(letter_pos_x2, 0.98, 'D', transform=ax[1,0].transAxes, fontsize=16, fontweight='bold', va='top')
    
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
    ax[1, 1].text(letter_pos_x2, 0.98, 'E', transform=ax[1,1].transAxes, fontsize=16, fontweight='bold', va='top')
    
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
    ax[1, 2].text(letter_pos_x2, 0.98, 'F', transform=ax[1,2].transAxes, fontsize=16, fontweight='bold', va='top')
    
    plt.show()
    return

def plotStratClimEvol_multi(goose_rec):
    dlength = len(goose_rec)

    points_per_myr_full  = 1e3
    points_per_myr_inset = 1e4
    stratt = goose_rec[1].snow_output.input_file.t_strat/1e6
    stratp = int(stratt*points_per_myr_inset)
    tgrid_stra = np.linspace(0,stratt,stratp)

    mixedt = 1
    mixedp = int(mixedt*points_per_myr_full)
    tgrid_mixe = np.linspace(0,mixedt,mixedp)

    melt_mass = np.zeros([stratp,1,dlength])
    
    stra_dict = {0:"pCO2_s",1:"Ts",2:"Td",3:"pH_s",4:"pH_d",5:"Omega_s",6:"Omega_d",7:"As",8:"Ad"}
    stra_vars = np.zeros([stratp,len(stra_dict),dlength])
    
    snow_dict = {0:"pCO2_o",1:"Td",2:"pH_o",3:"Omega_o",4:"Ao"}
    snow_vars = np.zeros([len(snow_dict),dlength])
    
    mixe_dict = {0:"pCO2_o",1:"pH_o",2:"Omega_o",3:"Ao"}
    mixe_vars = np.zeros([mixedp,len(mixe_dict),dlength])

    # stra_sens = model_output.strat_output.sens
    # t_pct = stra_sens["t_pct"].to_numpy()
    # m_pct = stra_sens["m_pct"].to_numpy()
    # current_melt_mass = stra_sens["current_melt_mass"].to_numpy()
    # current_melt_s = stra_sens["current_melt_s"].to_numpy()

    for i in range(dlength):
        rec = goose_rec[i]
        melt_mass[:,0,i] = np.interp(tgrid_stra,rec.strat_output.data["t"]/1e6,rec.strat_output.sens["current_melt_mass"])
        for j in range(len(stra_dict)):
            stra_vars[:,j,i] = np.interp(tgrid_stra,rec.strat_output.data["t"]/1e6,rec.strat_output.data[stra_dict[j]])
        
        for k in range(len(snow_dict)):
            snow_vars[k,i] = rec.snow_output.data[snow_dict[k]].iloc[-1]
    
        for l in range(len(mixe_dict)):
            mixe_vars[:,l,i] = np.interp(tgrid_mixe,rec.mixed_output.data["t"]/1e6,rec.mixed_output.data[mixe_dict[l]])
            # mixe_vars[l,i] = rec.mixed_output.data[mixe_dict[l]].iloc[0]
        
    intervals = [2.5,50,97.5]
    stra_cons = np.nanpercentile(stra_vars,intervals,axis=2)
    snow_cons = np.nanpercentile(snow_vars,intervals,axis=1)
    mixe_cons = np.nanpercentile(mixe_vars,intervals,axis=2)
    melt_mass_cons = np.nanpercentile(melt_mass,intervals,axis=2)
    
    melt_t = tgrid_stra + (goose_rec[1].snow_output.snow_max_duration/1e6)
    melt_cons = stra_cons
    
    full_t = np.concatenate(([-5,-1/1e6],melt_t,melt_t[-1]+tgrid_mixe))
    
    temp1 = np.zeros((3,1,1))
    temp1[:,0,0] = snow_cons[:,0]
    co2_con = np.concatenate((temp1[:,:,0],temp1[:,:,0],stra_cons[:,:,0],mixe_cons[:,:,0]),axis=1)
    
    temp1 = np.zeros((3,1,1))
    temp1[:,0,0] = snow_cons[:,2]
    pH_con = np.concatenate((temp1[:,:,0],temp1[:,:,0],stra_cons[:,:,4],mixe_cons[:,:,1]),axis=1)
    
    temp1 = np.zeros((3,1,1))
    temp1[:,0,0] = snow_cons[:,3]
    omega_con = np.concatenate((temp1[:,:,0],temp1[:,:,0],stra_cons[:,:,6],mixe_cons[:,:,2]),axis=1)
    
    temp1 = np.zeros((3,1,1))
    temp1[:,0,0] = snow_cons[:,4]
    alk_con = np.concatenate((temp1[:,:,0],temp1[:,:,0],stra_cons[:,:,8],mixe_cons[:,:,3]),axis=1)
    
    
    td_t = np.concatenate(([-5,-1/1e6],melt_t))
    temp1 = np.zeros((3,1,1))
    temp1[:,0,0] = snow_cons[:,1]
    td_con = np.concatenate((temp1[:,:,0],temp1[:,:,0],stra_cons[:,:,2]),axis=1)
    
    
    fig, ax = plt.subplots(2, 3)
    fig.canvas.manager.set_window_title('Stratified Evolution')
    fig.subplots_adjust(hspace=0.35, wspace=0.2)
    fig.set_tight_layout("tight")
    fig.set_size_inches(12,7)
    
    letter_pos_x2 = 0.03
    
    l1 = goose_rec[1].snow_output.snow_max_duration/1e6
    l2 = goose_rec[1].snow_output.snow_max_duration/1e6 + stratt

    xticks1 = np.round(np.linspace(l1,l2,5),4)
    
    xmin2 = l1 - (stratt/4)
    xmax2 = l2 + (stratt/4)
    
    color1 = 'goldenrod'
    color2 = 'seagreen'
    color3 = 'royalblue'
    color4 = 'firebrick'
    color5 = 'rebeccapurple'
    melt_color = 'dimgrey'
    vline_color = "lightsteelblue"
    
    for j in range(2):
        for k in range(3):
            ax[j,k].axvspan(xmin2,l1, color='aliceblue', alpha=1)
            ax[j,k].axvline(x=l1, color=vline_color, linestyle='--', linewidth=1)
            ax[j,k].axvline(x=l2, color="k", linestyle='--', linewidth=1)
    
    
    ax[0, 0].text(0.43, 1.05, 'Phase 3', transform=ax[0, 0].transAxes, fontsize=10, color="slategrey", va='top')
    ax[0, 1].text(0.43, 1.05, 'Phase 3', transform=ax[0, 1].transAxes, fontsize=10, color="slategrey", va='top')
    ax[0, 2].text(0.43, 1.05, 'Phase 3', transform=ax[0, 2].transAxes, fontsize=10, color="slategrey", va='top')
    ax[1, 0].text(0.43, 1.05, 'Phase 3', transform=ax[1, 0].transAxes, fontsize=10, color="slategrey", va='top')
    ax[1, 1].text(0.43, 1.05, 'Phase 3', transform=ax[1, 1].transAxes, fontsize=10, color="slategrey", va='top')
    ax[1, 2].text(0.43, 1.05, 'Phase 3', transform=ax[1, 2].transAxes, fontsize=10, color="slategrey", va='top')
    
    yfs2 = 14
    xfs2 = 13
    
    pd = {v: k for k, v in stra_dict.items()}
    
    ax[0, 0].fill_between(full_t, co2_con[0,:],  co2_con[2,:], color = color1, alpha = 0.2)
    ax[0, 0].plot(full_t, co2_con[1,:],color = color1, linewidth=2.5)
    ax[0, 0].set_yscale('log')
    ax[0, 0].set_ylabel(r'pCO$_2$ (bar)', fontsize=yfs2)
    ax[0, 0].set_xlabel(r"Time Since Glaciation (Myr)", fontsize=xfs2)
    ax[0, 0].set_xlim([xmin2, xmax2])
    ax[0, 0].set_xticks(xticks1)
    ax[0, 0].set_xticklabels(xticks1)
    ax[0, 0].set_ylim([1e-4,1e1])
    ax[0, 0].text(letter_pos_x2, 0.98, 'A', transform=ax[0,0].transAxes, fontsize=16, fontweight='bold', va='top')
    
    ax[0, 1].fill_between(melt_t, melt_cons[0,:,pd["Ts"]],  melt_cons[2,:,pd["Ts"]], color = melt_color, alpha = 0.2)
    ax[0, 1].plot(melt_t, melt_cons[1,:,pd["Ts"]],color = melt_color, linewidth=2.5, label = 'Meltwater Layer')
    ax[0, 1].fill_between(td_t, td_con[0,:],  td_con[2,:], color = color2, alpha = 0.2)
    ax[0, 1].plot(td_t, td_con[1,:],color = color2, linewidth=2.5, label = 'Deep Ocean')
    ax[0, 1].set_ylabel("Temperature (K)",fontsize=yfs2)
    ax[0, 1].set_xlabel(r"Time Since Glaciation (Myr)", fontsize=xfs2)
    ax[0, 1].set_xlim([xmin2, xmax2])
    ax[0, 1].set_xticks(xticks1)
    ax[0, 1].set_xticklabels(xticks1)
    ax[0, 1].set_ylim([265,335])
    ax[0, 1].legend()
    ax[0, 1].text(letter_pos_x2, 0.98, 'B', transform=ax[0,1].transAxes, fontsize=16, fontweight='bold', va='top')
    
    ax[0, 2].fill_between(melt_t, melt_cons[0,:,pd["pH_s"]],  melt_cons[2,:,pd["pH_s"]], color = melt_color, alpha = 0.2)
    ax[0, 2].plot(melt_t, melt_cons[1,:,pd["pH_s"]],color = melt_color, linewidth=2.5, label='Meltwater Layer')
    ax[0, 2].fill_between(full_t, pH_con[0,:],  pH_con[2,:], color = color3, alpha = 0.2)
    ax[0, 2].plot(full_t, pH_con[1,:],color = color3, linewidth=2.5, label='Ocean')
    ax[0, 2].set_ylabel(r'pH', fontsize=yfs2)
    ax[0, 2].set_xlabel(r"Time Since Glaciation (Myr)", fontsize=xfs2)
    ax[0, 2].set_xlim([xmin2,xmax2])
    ax[0, 2].set_xticks(xticks1)
    ax[0, 2].set_xticklabels(xticks1)
    ax[0, 2].set_ylim([4.5, 10])
    ax[0, 2].legend(loc='upper center')
    ax[0, 2].text(letter_pos_x2, 0.98, 'C', transform=ax[0,2].transAxes, fontsize=16, fontweight='bold', va='top')
    
    ax[1, 0].fill_between(melt_t, melt_mass_cons[0,:,0],  melt_mass_cons[2,:,0], color = melt_color, alpha = 0.2)
    ax[1, 0].plot(melt_t, melt_mass_cons[1,:,0],color = melt_color, linewidth=2.5)
    #ax[1, 0].set_yscale('log')
    ax[1, 0].set_ylabel(r'Meltwater Layer Mass (kg)', fontsize=yfs2)
    ax[1, 0].set_xlabel(r"Time Since Glaciation (Myr)", fontsize=xfs2)
    ax[1, 0].set_xlim([xmin2, xmax2])
    ax[1, 0].set_xticks(xticks1)
    ax[1, 0].set_xticklabels(xticks1)
    #ax[1, 0].set_ylim([1e-4,1e1])
    ax[1, 0].text(letter_pos_x2, 0.98, 'D', transform=ax[1,0].transAxes, fontsize=16, fontweight='bold', va='top')
    
    ax[1, 1].fill_between(melt_t, melt_cons[0,:,pd["Omega_s"]],  melt_cons[2,:,pd["Omega_s"]], color = melt_color, alpha = 0.2)
    ax[1, 1].plot(melt_t, melt_cons[1,:,pd["Omega_s"]],color = melt_color, linewidth=2.5, label='Meltwater Layer')
    ax[1, 1].fill_between(full_t, omega_con[0,:],  omega_con[2,:], color = color4, alpha = 0.2)
    ax[1, 1].plot(full_t, omega_con[1,:],color = color4, linewidth=2.5, label='Ocean')
    ax[1, 1].set_ylabel(r'Saturation State, $\Omega$', fontsize=yfs2)
    ax[1, 1].set_xlabel(r"Time Since Glaciation (Myr)", fontsize=xfs2)
    ax[1, 1].set_xlim([xmin2,xmax2])
    ax[1, 1].set_xticks(xticks1)
    ax[1, 1].set_xticklabels(xticks1)
    ax[1, 1].set_ylim([-5, 90])
    ax[1, 1].legend()
    ax[1, 1].text(letter_pos_x2, 0.98, 'E', transform=ax[1,1].transAxes, fontsize=16, fontweight='bold', va='top')
    
    ax[1, 2].fill_between(melt_t, melt_cons[0,:,pd["As"]],  melt_cons[2,:,pd["As"]], color = melt_color, alpha = 0.2)
    ax[1, 2].plot(melt_t, melt_cons[1,:,pd["As"]],color = melt_color, linewidth=2.5, label='Meltwater Layer')
    ax[1, 2].fill_between(full_t, alk_con[0,:],  alk_con[2,:], color = color5, alpha = 0.2)
    ax[1, 2].plot(full_t, alk_con[1,:],color = color5, linewidth=2.5, label='Ocean')
    ax[1, 2].set_ylabel(r'Alkalinity (mol eq kg$^{-1}$)', fontsize=yfs2)
    ax[1, 2].set_xlabel(r"Time Since Glaciation (Myr)", fontsize=xfs2)
    ax[1, 2].set_xlim([xmin2,xmax2])
    ax[1, 2].set_xticks(xticks1)
    ax[1, 2].set_xticklabels(xticks1)
    #ax[1, 2].set_ylim([0, 70])
    ax[1, 2].legend()
    ax[1, 2].text(letter_pos_x2, 0.98, 'F', transform=ax[1,2].transAxes, fontsize=16, fontweight='bold', va='top')

    ax[0,0].minorticks_on()
    ax[0,1].minorticks_on()
    ax[0,2].minorticks_on()
    ax[1,0].minorticks_on()
    ax[1,1].minorticks_on()
    ax[1,2].minorticks_on()
    
    plt.show()
    return
