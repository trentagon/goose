import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from scipy.integrate import cumtrapz
import pandas as pd

# This file contains functions that are useful for visualizing model outputs.
# It is called "solo" because these functions work for single model runs.

def plotFullClimEvol_multi(goose_rec):
    dlength = len(goose_rec)

    points_per_myr_full  = 1e2
    points_per_myr_inset = 1e4
    
    snowt = goose_rec[1].snow_output.snow_max_duration/1e6
    snowp = int(snowt*points_per_myr_full)
    tgrid_snow = np.linspace(0,snowt,snowp)
    
    stratt = goose_rec[1].snow_output.input_file.t_strat/1e6
    stratp = int(stratt*points_per_myr_inset)
    tgrid_stra = np.linspace(0,stratt,stratp)
    
    mixedt = 20
    mixedp = int(mixedt*points_per_myr_full)
    tgrid_mixe = np.linspace(0,mixedt,mixedp)
    
    snow_dict = {0:"pCO2_o",1:"Ts",2:"pH_o",3:"Omega_o",4:"Ao"}
    stra_dict = {0:"pCO2_s",1:"Ts",2:"pH_d",3:"Omega_d",4:"Ad"}
    mixe_dict = {0:"pCO2_o",1:"Ts",2:"pH_o",3:"Omega_o",4:"Ao"}
    neop_dict = {0:"pCO2_o",1:"Ts",2:"pH_o",3:"Omega_o",4:"Ao"}
    
    snow_vars = np.zeros([snowp,len(snow_dict),dlength])
    stra_vars = np.zeros([stratp,len(stra_dict),dlength])
    mixe_vars = np.zeros([mixedp,len(mixe_dict),dlength])
    neop_vars = np.zeros([len(neop_dict),dlength])
    
    for i in range(dlength):
        rec = goose_rec[i]
        for j in range(len(snow_dict)):
            neop_vars[j,i] = rec.neo_output.data[neop_dict[j]]
            snow_vars[:,j,i] = np.interp(tgrid_snow,rec.snow_output.data["t"]/1e6,rec.snow_output.data[snow_dict[j]])
            stra_vars[:,j,i] = np.interp(tgrid_stra,rec.strat_output.data["t"]/1e6,rec.strat_output.data[stra_dict[j]])
            mixe_vars[:,j,i] = np.interp(tgrid_mixe,rec.mixed_output.data["t"]/1e6,rec.mixed_output.data[mixe_dict[j]])
    
        
    intervals = [2.5,50,97.5]
    
    neop_cons = np.nanpercentile(neop_vars,intervals,axis=1)
    snow_cons = np.nanpercentile(snow_vars,intervals,axis=2)
    stra_cons = np.nanpercentile(stra_vars,intervals,axis=2)
    mixe_cons = np.nanpercentile(mixe_vars,intervals,axis=2)
    
    t = np.concatenate((tgrid_snow, tgrid_stra + tgrid_snow[-1] + (1/1e6), tgrid_mixe + tgrid_stra[-1] + tgrid_snow[-1] + (2/1e6)))
    cons = np.concatenate((snow_cons,stra_cons,mixe_cons), axis=1)
    
    # Add padding to the front to visualize the neoproterozoic steady state
    t = np.concatenate(([-5,-1/1e6],t))
    
    temp = np.zeros((3,1,5))
    for i in range(5):
        temp[:,0,i] = neop_cons[:,i]
    cons = np.concatenate((temp,temp,cons),axis=1)
    
    # Plot
    plt.rcParams['font.family'] = 'DIN Alternate'
    
    glact = 0
    degt = goose_rec[1].snow_output.snow_max_duration/1e6
    stratt = degt + goose_rec[1].snow_output.input_file.t_strat/1e6
    
    xmin = -2
    xmax = 20
    
    letter_pos_x = 0.12
    
    color1 = 'goldenrod'
    color2 = 'seagreen'
    color3 = 'royalblue'
    color4 = 'firebrick'
    color5 = 'rebeccapurple'
    
    fig, ax = plt.subplots(5, 1, sharex=True)
    fig.canvas.manager.set_window_title('Full Evolution')
    fig.subplots_adjust(hspace=0, wspace=0.2)
    fig.set_size_inches(7,8)
    
    # Merge plots
    ax[0].spines['bottom'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    ax[2].spines['top'].set_visible(False)
    ax[2].spines['bottom'].set_visible(False)
    ax[3].spines['top'].set_visible(False)
    ax[3].spines['bottom'].set_visible(False)
    ax[4].spines['top'].set_visible(False)
    
    # Add vertical lines
    vline_color = "lightsteelblue"
    for j in range(5):
        ax[j].axvspan(glact,degt, color='aliceblue', alpha=1)
        ax[j].axvline(x=glact, color=vline_color, linestyle='--', linewidth=1)
        ax[j].axvline(x=degt, color=vline_color, linestyle='--', linewidth=1)
        ax[j].axvline(x=stratt, color='k', linestyle='--', linewidth=0.5)
    
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
    ax[4].text(0.28, 0.13, 'modern', transform=ax[4].transAxes, fontsize=10, color=hline_color, va='top')
    
    yfs = 13
    pd = {"pCO2":0,"T":1,"pH":2,"Omega":3,"Alk":4}
    
    
    ax[0].fill_between(t,cons[0,:,pd["pCO2"]],  cons[2,:,pd["pCO2"]], color = color1, alpha = 0.2)
    ax[0].plot(t, cons[1,:,pd["pCO2"]],color = color1, linewidth=2.5)
    ax[0].set_yscale('log')
    ax[0].set_ylabel(r'pCO$_2$ (bar)', fontsize=yfs)
    ax[0].set_xlim([xmin, xmax])
    ax[0].set_yticks(np.logspace(-4,0,5))
    ax[0].minorticks_off()
    ax[0].set_ylim([1e-5,1e1])
    
    ax[1].fill_between(t,cons[0,:,pd["T"]],  cons[2,:,pd["T"]], color = color2, alpha = 0.2)
    ax[1].plot(t, cons[1,:,pd["T"]],color = color2, linewidth=2.5)
    ax[1].set_ylabel(r'SST (K)', fontsize=yfs)
    ax[1].set_xlim([xmin, xmax])
    ax[1].set_yticks(np.linspace(265,315,6))
    ax[1].set_ylim([260,320])
    
    ax[2].fill_between(t,cons[0,:,pd["pH"]],  cons[2,:,pd["pH"]], color = color3, alpha = 0.2)
    ax[2].plot(t, cons[1,:,pd["pH"]],color = color3, linewidth=2.5)
    ax[2].set_ylabel(r'pH', fontsize=yfs)
    ax[2].set_xlim([xmin, xmax])
    ax[2].set_yticks([6,7,8,9])
    ax[2].set_ylim([5,10])
    
    ax[3].fill_between(t,cons[0,:,pd["Omega"]],  cons[2,:,pd["Omega"]], color = color4, alpha = 0.2)
    ax[3].plot(t, cons[1,:,pd["Omega"]],color = color4, linewidth=2.5)
    ax[3].set_ylabel('Saturation \n State, $\Omega$', fontsize=yfs)
    ax[3].set_xlim([xmin, xmax])
    ax[3].set_yticks([0,20,40,60,80])
    ax[3].set_xticks(np.linspace(0,20,11))
    ax[3].set_ylim([-5,90])
    
    ax[4].fill_between(t,cons[0,:,pd["Alk"]],  cons[2,:,pd["Alk"]], color = color5, alpha = 0.2)
    ax[4].plot(t, cons[1,:,pd["Alk"]],color = color5, linewidth=2.5)
    ax[4].set_ylabel('Alkalinity \n (mol eq kg$^{-1}$)', fontsize=yfs)
    ax[4].set_xlabel('Time Since Glaciation (Myr)', fontsize=yfs)
    ax[4].set_xlim([xmin, xmax])
    ax[4].set_yticks([0,0.02,0.04,0.06])
    ax[4].set_xticks(np.linspace(0,20,11))
    ax[4].set_ylim([-0.01,0.075])
    
    fig.align_ylabels()
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
    
    stra_dict = {0:"pCO2_s",1:"Ts",2:"Td",3:"pH_s",4:"pH_d",5:"Omega_s",6:"Omega_d",7:"As",8:"Ad"}
    stra_vars = np.zeros([stratp,len(stra_dict),dlength])
    
    snow_dict = {0:"pCO2_o",1:"Td",2:"pH_o",3:"Omega_o",4:"Ao"}
    snow_vars = np.zeros([len(snow_dict),dlength])
    
    mixe_dict = {0:"pCO2_o",1:"pH_o",2:"Omega_o",3:"Ao"}
    mixe_vars = np.zeros([mixedp,len(mixe_dict),dlength])
    
    for i in range(dlength):
        rec = goose_rec[i]
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
    
    ax[1, 0].axis('off')
    
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
    ax[1, 1].text(letter_pos_x2, 0.98, 'D', transform=ax[1,1].transAxes, fontsize=16, fontweight='bold', va='top')
    
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
    ax[1, 2].text(letter_pos_x2, 0.98, 'E', transform=ax[1,2].transAxes, fontsize=16, fontweight='bold', va='top')

    ax[0,0].minorticks_on()
    ax[0,1].minorticks_on()
    ax[0,2].minorticks_on()
    ax[1,1].minorticks_on()
    ax[1,2].minorticks_on()
    
    plt.show()
    return


def plotPostGlacialCarbMass_multi(goose_rec):

    dlength = len(goose_rec)
        
    points_per_myr_full  = 1e3
    points_per_myr_inset = 1e4
    
    stratt = goose_rec[1].snow_output.input_file.t_strat
    stratp = int(stratt*1e-6*points_per_myr_inset)
    tgrid_stra = np.linspace(0,stratt,stratp)
    
    mixedt = 30*1e6
    mixedp = int(mixedt*1e-6*points_per_myr_full)
    tgrid_mixe = np.linspace(0,mixedt,mixedp)
    
    stra_rate = np.zeros([stratp,1,dlength])
    mixe_rate = np.zeros([mixedp,1,dlength])
    
    cums = np.zeros([stratp + mixedp - 1,1,dlength])
    
    tgrids = np.concatenate((tgrid_stra,tgrid_mixe[1:] + tgrid_stra[-1]))
    
    for i in range(dlength):
        rec = goose_rec[i]
    
        stra_rate[:,0,i] = np.interp(tgrid_stra,rec.strat_output.data["t"],rec.strat_output.data["P_shelf"]*100.01)
        mixe_rate[:,0,i] = np.interp(tgrid_mixe,rec.mixed_output.data["t"],rec.mixed_output.data["P_shelf"]*100.01)
    
        r1 = rec.strat_output.data["P_shelf"].to_numpy().real*100.01
        r2 = rec.mixed_output.data["P_shelf"].to_numpy().real*100.01
        raw_rates = np.concatenate((r1[:-1],r2))
    
        t1 = rec.strat_output.data["t"].to_numpy().real
        t2 = rec.mixed_output.data["t"].to_numpy().real
        raw_times = np.concatenate((t1,t2[1:]+t1[-1]))
        interp_times = np.concatenate((t1[1:],t2[1:]+t1[-1]))
    
        raw_cums = cumtrapz(raw_rates,raw_times)
        
        cums[:,0,i] = np.interp(tgrids,interp_times,raw_cums)
    
    intervals = [2.5,50,97.5]
        
    stra_rate_cons = np.nanpercentile(stra_rate,intervals,axis=2)
    mixe_rate_cons = np.nanpercentile(mixe_rate,intervals,axis=2)
    cum_cons = np.nanpercentile(cums,intervals,axis=2)
    
    t_rate = np.concatenate((tgrid_stra, tgrid_mixe + tgrid_stra[-1]))
    t_cum = tgrids
    
    rate_cons = np.concatenate((stra_rate_cons,mixe_rate_cons), axis=1)
    
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
    
    strat_line = stratt
    
    ax[0].fill_between(t_rate,rate_cons[0,:,0],  rate_cons[2,:,0], color = color1, alpha = 0.2)
    ax[0].plot(t_rate, rate_cons[1,:,0],color = color1, linewidth=2.5)
    # ax[0].plot(t, pshelf_g_per_yr,color = color1, linewidth=2.5)
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
    
    ax[1].fill_between(t_cum,cum_cons[0,:,0],  cum_cons[2,:,0], color = color2, alpha = 0.2)
    ax[1].plot(t_cum, cum_cons[1,:,0],color = color2, linewidth=2.5)
    #ax[1].plot(t[:-1], cumulative_carbs,color = color2, linewidth=2.5, label="Model results")
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


def plotPostGlacialCarbHeight_multi(goose_rec):

    dlength = len(goose_rec)
        
    points_per_myr_full  = 1e3
    points_per_myr_inset = 1e4
    
    stratt = goose_rec[1].snow_output.input_file.t_strat
    stratp = int(stratt*1e-6*points_per_myr_inset)
    tgrid_stra = np.linspace(0,stratt,stratp)
    
    mixedt = 30*1e6
    mixedp = int(mixedt*1e-6*points_per_myr_full)
    tgrid_mixe = np.linspace(0,mixedt,mixedp)
    
    stra_rate = np.zeros([stratp,1,dlength])
    mixe_rate = np.zeros([mixedp,1,dlength])
    
    cums = np.zeros([stratp + mixedp - 1,1,dlength])
    
    tgrids = np.concatenate((tgrid_stra,tgrid_mixe[1:] + tgrid_stra[-1]))
    
    for i in range(dlength):
        rec = goose_rec[i]
    
        stra_rate[:,0,i] = np.interp(tgrid_stra,rec.strat_output.data["t"],rec.strat_output.data["P_shelf"]*100.01)
        mixe_rate[:,0,i] = np.interp(tgrid_mixe,rec.mixed_output.data["t"],rec.mixed_output.data["P_shelf"]*100.01)
    
        r1 = rec.strat_output.data["P_shelf"].to_numpy().real*100.01
        r2 = rec.mixed_output.data["P_shelf"].to_numpy().real*100.01
        raw_rates = np.concatenate((r1[:-1],r2))
    
        t1 = rec.strat_output.data["t"].to_numpy().real
        t2 = rec.mixed_output.data["t"].to_numpy().real
        raw_times = np.concatenate((t1,t2[1:]+t1[-1]))
        interp_times = np.concatenate((t1[1:],t2[1:]+t1[-1]))
    
        raw_cums = cumtrapz(raw_rates,raw_times)
        
        cums[:,0,i] = np.interp(tgrids,interp_times,raw_cums)
    
    intervals = [2.5,50,97.5]
        
    stra_rate_cons = np.nanpercentile(stra_rate,intervals,axis=2)
    mixe_rate_cons = np.nanpercentile(mixe_rate,intervals,axis=2)
    cum_cons = np.nanpercentile(cums,intervals,axis=2)
    
    t_rate = np.concatenate((tgrid_stra, tgrid_mixe + tgrid_stra[-1]))
    t_cum = tgrids
    
    rate_cons = np.concatenate((stra_rate_cons[:,:-1,:],mixe_rate_cons), axis=1)
    
    rho = 2.71*1e15 # g/km3
    dep_area = 2*3.22e7 # km2
    
    rate_cons_height = 1e3*rate_cons/(rho*dep_area)
    cum_cons_height = 1e3*cum_cons/(rho*dep_area)
    
    plt.rcParams['font.family'] = 'DIN Alternate'
    
    fig, ax = plt.subplots(1, 2)
    fig.subplots_adjust(hspace=0.35, wspace=0.2)
    fig.set_tight_layout("tight")
    fig.set_size_inches(10,4)
    
    color1 = 'darkred'
    color2 = 'darkolivegreen'
    
    yfs = 13
    
    xf00 = np.concatenate([np.trim_zeros(rate_cons_height[0,:,0]), np.trim_zeros(rate_cons_height[2,::-1,0])])
    yf00 = np.concatenate([np.trim_zeros(cum_cons_height[0,:,0]), np.trim_zeros(cum_cons_height[2,::-1,0])])
    ax[0].fill(xf00,yf00, color = color1, alpha = 0.2)
    
    ax[0].plot(rate_cons_height[1,:,0], cum_cons_height[1,:,0],color = color1, linewidth=2.5)
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
    
    ax[1].fill_between(t_cum,cum_cons_height[0,:,0],  cum_cons_height[2,:,0], color = color2, alpha = 0.2)
    ax[1].plot(t_cum, cum_cons_height[1,:,0],color = color2, linewidth=2.5)
    # ax[1].plot(t[:-1], cumulative_carbs_height,color = color2, linewidth=2.5)
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
    
    strat_line = stratt
    ax[1].axvline(x=strat_line, color='k', linestyle='--', linewidth=1)
    ax[1].text(0.11, 1.06, 'Phase 3', transform=ax[1].transAxes, fontsize=10, color="grey", va='top')
    ax[1].text(0.65, 1.06, 'Phase 4', transform=ax[1].transAxes, fontsize=10, color="grey", va='top')
        
    plt.show()

    return

def plotFluxes_multi(goose_rec):
    dlength = len(goose_rec)

    points_per_myr_full  = 1e2
    points_per_myr_inset = 1e4
    
    snowt = goose_rec[1].snow_output.snow_max_duration/1e6
    snowp = int(snowt*points_per_myr_full)
    tgrid_snow = np.linspace(0,snowt,snowp)
    
    stratt = goose_rec[1].snow_output.input_file.t_strat/1e6
    stratp = int(stratt*points_per_myr_inset)
    tgrid_stra = np.linspace(0,stratt,stratp)
    
    mixedt = 20
    mixedp = int(mixedt*points_per_myr_full)
    tgrid_mixe = np.linspace(0,mixedt,mixedp)
    
    neop_dict = {0:"V",1:"P_shelf",2:"P_pore",3:"W_carb",4:"W_sil",5:"W_sea"}
    snow_dict = {0:"V",1:"P_ocean",2:"P_pore",3:"W_carb",4:"W_sil",5:"W_sea"}
    stra_dict = {0:"V",1:"P_shelf",2:"P_pore",3:"W_carb",4:"W_sil",5:"W_sea"}
    mixe_dict = {0:"V",1:"P_shelf",2:"P_pore",3:"W_carb",4:"W_sil",5:"W_sea"}
    
    neop_vars = np.zeros([len(neop_dict),dlength])
    snow_vars = np.zeros([snowp,len(snow_dict),dlength])
    stra_vars = np.zeros([stratp,len(stra_dict),dlength])
    mixe_vars = np.zeros([mixedp,len(mixe_dict),dlength])
    
    for i in range(dlength):
        rec = goose_rec[i]
        for j in range(len(neop_dict)):
            neop_vars[j,i] = rec.neo_output.data[neop_dict[j]]
            snow_vars[:,j,i] = np.interp(tgrid_snow,rec.snow_output.data["t"]/1e6,rec.snow_output.data[snow_dict[j]])
            mixe_vars[:,j,i] = np.interp(tgrid_mixe,rec.mixed_output.data["t"]/1e6,rec.mixed_output.data[mixe_dict[j]])
            if j == 0:
                stra_vars[:,j,i] = np.interp(tgrid_stra,rec.strat_output.data["t"]/1e6,rec.strat_output.data["V_air"] + rec.strat_output.data["V_ridge"])
            else:
                stra_vars[:,j,i] = np.interp(tgrid_stra,rec.strat_output.data["t"]/1e6,rec.strat_output.data[stra_dict[j]])
        
    intervals = [2.5,50,97.5]
    
    neop_cons = np.nanpercentile(neop_vars,intervals,axis=1)
    snow_cons = np.nanpercentile(snow_vars,intervals,axis=2)
    stra_cons = np.nanpercentile(stra_vars,intervals,axis=2)
    mixe_cons = np.nanpercentile(mixe_vars,intervals,axis=2)
    
    t = np.concatenate((tgrid_snow, tgrid_stra + tgrid_snow[-1] + (1/1e6), tgrid_mixe + tgrid_stra[-1] + tgrid_snow[-1] + (2/1e6)))
    cons = np.concatenate((snow_cons,stra_cons,mixe_cons), axis=1)
    
    # Add padding to the front to visualize the neoproterozoic steady state
    t = np.concatenate(([-5,-1/1e6],t))
    
    temp = np.zeros((3,1,6))
    for i in range(5):
        temp[:,0,i] = neop_cons[:,i]
    cons = np.concatenate((temp,temp,cons),axis=1)
    
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
    degt = goose_rec[0].snow_output.snow_max_duration/1e6
    strt = degt + (goose_rec[0].snow_output.input_file.t_strat/1e6)
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
    
    ax.plot(t, cons[1,:,0],color = color1, linewidth=2, label ="Volcanism")
    ax.plot(t, cons[1,:,1],color = color2, linewidth=2, label="Precip: Shelf/Slope")
    ax.plot(t, cons[1,:,2],color = color3, linewidth=2, label ="Precip: Pore space")
    ax.plot(t, cons[1,:,3],color = color4, linewidth=2, label="Weath: Cont. Carb.")
    ax.plot(t, cons[1,:,4],color = color5, linewidth=2, label ="Weath: Cont. Sil.")
    ax.plot(t, cons[1,:,5],color = color6, linewidth=2, label ="Weath: Seafloor")
    ax.set_yscale('log')
    ax.set_ylabel('Flux (mol C yr$^{-1}$)', fontsize=yfs)
    ax.set_xlabel('Time Since Glaciation (Myr)', fontsize=yfs)
    ax.set_xlim([xmin, xmax])
    ax.minorticks_off()
    ax.legend()
    plt.show()
    return

def plotStrontium(data_path):

    sr = pd.read_excel(data_path)
    
    offset = 0.003
    
    sr_ol = sr.iloc[:,2].dropna() - sr.iloc[:,2].min()
    h_ol = sr.iloc[:,3].dropna()
    
    sr_etoto = sr.iloc[:,4].dropna() - sr.iloc[:,4].min() + sr_ol.max() + offset
    h_etoto = sr.iloc[:,5].dropna()
    
    sr_jiu = sr.iloc[:,6].dropna() - sr.iloc[:,6].min() + sr_etoto.max() + offset
    h_jiu = sr.iloc[:,7].dropna()
    
    sr_nucc = sr.iloc[:,0].dropna() - sr.iloc[:,0].min() + sr_jiu.max() + offset
    h_nucc = sr.iloc[:,1].dropna()
    
    sr_wushi = sr.iloc[:,8].dropna() - sr.iloc[:,8].min() + sr_nucc.max() + offset
    h_wushi = sr.iloc[:,9].dropna()

    plt.rcParams['font.family'] = 'DIN Alternate'
    
    fig, ax = plt.subplots(1,2, figsize=(16,6), gridspec_kw={'width_ratios': [5, 1]})
    fig.set_tight_layout("tight")
    
    # Sr data
    offset = 0.003
    ax[0].plot(sr_ol.to_numpy(),    h_ol.to_numpy(),    'ko--')
    ax[0].plot(sr_etoto.to_numpy(), h_etoto.to_numpy(), 'ko--')
    ax[0].plot(sr_jiu.to_numpy(),   h_jiu.to_numpy(),   'ko--')
    ax[0].plot(sr_nucc.to_numpy(),  h_nucc.to_numpy(),  'ko--')
    ax[0].plot(sr_wushi.to_numpy(), h_wushi.to_numpy(), 'ko--')
    
    label_fs = 20
    tick_fs = 16
    ax[0].set_ylabel(r'Height (m)', fontsize=label_fs)
    ax[0].set_xlabel(r"Measured $^{87}$Sr/$^{86}$Sr offset", fontsize=label_fs)
    ax[0].set_ylim([0,60])
    ax[0].tick_params(axis='x', labelsize=tick_fs)
    ax[0].tick_params(axis='y', labelsize=tick_fs)
    ax[0].xaxis.set_minor_locator(AutoMinorLocator())
    ax[0].yaxis.set_minor_locator(AutoMinorLocator())
    ax[0].grid(axis='y')
    
    hline_col = 'red'
    hline_alpha = 0.5
    hline_width = 4
    hlo = 0.001
    
    ax[0].hlines(y=38, xmin = sr_ol.min()-hlo, xmax = sr_ol.max()+hlo, color = hline_col, alpha=hline_alpha, linewidth=hline_width)
    ax[0].hlines(y=5, xmin = sr_etoto.min()-hlo, xmax = sr_etoto.max()+hlo, color = hline_col, alpha=hline_alpha, linewidth=hline_width)
    ax[0].hlines(y=2, xmin = sr_jiu.min()-hlo, xmax = sr_jiu.max()+hlo, color = hline_col, alpha=hline_alpha, linewidth=hline_width)
    ax[0].hlines(y=2.9, xmin = sr_nucc.min()-hlo, xmax = sr_nucc.max()+hlo, color = hline_col, alpha=hline_alpha, linewidth=hline_width)
    ax[0].hlines(y=0.8, xmin = sr_wushi.min()-hlo, xmax = sr_wushi.max()+hlo, color = hline_col, alpha=hline_alpha, linewidth=hline_width)
    
    form_fs = 20
    ax[0].text(0.12, 0.9, 'Mongolia', transform=ax[0].transAxes, fontsize=form_fs, color="slategrey", va='top')
    ax[0].text(0.26, 0.3, 'Namibia', transform=ax[0].transAxes, fontsize=form_fs, color="slategrey", va='top')
    ax[0].text(0.46, 0.25, 'South China', transform=ax[0].transAxes, fontsize=form_fs, color="slategrey", va='top')
    ax[0].text(0.66, 0.25, 'Australia', transform=ax[0].transAxes, fontsize=form_fs, color="slategrey", va='top')
    ax[0].text(0.825, 0.13, 'North China', transform=ax[0].transAxes, fontsize=form_fs, color="slategrey", va='top')
    
    ax[0].text(0.205, 0.55, 'End of ocean stratification', transform=ax[0].transAxes, fontsize=form_fs, color=hline_col, va='top')
    ax[0].arrow(0.004, 32, -0.002, 5, width=0.0001, head_width=0.0007, head_length=1, length_includes_head=True, edgecolor=hline_col, facecolor=hline_col)
    ax[0].text(0.005, 0.98, 'A', transform=ax[0].transAxes, fontsize=16, fontweight='bold', va='top')
    
    # Model data
    
    color1 = 'royalblue'
    color2 = 'forestgreen'
    
    e4 = np.ones((2,1))
    e4[0] = 10
    e4[1] = 8.12 - 0.18
    ax[1].errorbar(1,0.18,yerr=e4, 
        marker="s", 
        markersize=15, 
        markerfacecolor=color1, 
        markeredgecolor=color1, 
        capsize=15, 
        capthick=4, 
        elinewidth=4, 
        ecolor=color1)
    
    e5 = np.ones((2,1))
    e5[0] = 14.36 - 1.84
    e5[1] = 53.68 - 14.36
    ax[1].errorbar(2,14.36,yerr=e5, 
        marker="s", 
        markersize=15, 
        markerfacecolor=color2, 
        markeredgecolor=color2, 
        capsize=15, 
        capthick=4, 
        elinewidth=4, 
        ecolor=color2)
    
    ts_fs = 12
    ax[1].text(0.15, 0.21, r't$_{strat}$ = $10^4$ yrs', transform=ax[1].transAxes, fontsize=ts_fs, color=color1, va='top')
    ax[1].text(0.46, 0.96, r't$_{strat}$ = $10^5$ yrs', transform=ax[1].transAxes, fontsize=ts_fs, color=color2, va='top')
    ax[1].text(0.02, 0.98, 'B', transform=ax[1].transAxes, fontsize=16, fontweight='bold', va='top')
    
    ax[1].set_xlabel('Predicted end of\nocean stratification',fontsize=label_fs, labelpad=10)
    ax[1].set_xlim([0,3])
    ax[1].set_ylim([0,60])
    ax[1].tick_params(axis='y', labelsize=tick_fs)
    ax[1].yaxis.set_minor_locator(AutoMinorLocator())
    ax[1].grid(axis='y')
    ax[1].tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    
    return
