import glob
import math
import obspy
import time
from obspy import UTCDateTime
from obspy.core import Stats
from obspy.clients.iris import Client
client = Client()
from matplotlib import pyplot as plt, patches
from matplotlib.backends.backend_pdf import PdfPages
from IPython.display import clear_output
import numpy as np
from sklearn.decomposition import PCA
from scipy import signal
from obspy.io.sac.sactrace import SACTrace
from obspy.io.sac.util import get_sac_reftime
import os
import glob
import numpy
from shutil import move
from obspy.taup import TauPyModel

def get_snr(st):
    #signal noise ratio
    TNoise1 = 15  # noise window_left before t1
    TNoise2 = 0   # noise window_right before t1
    TSignal1 = 5  # signal window_left before t1
    TSignal2 = 10 
    
    T = get_sac_reftime(st[0].stats.sac)
    # measure noise level
    tr = st.copy()
    tr.trim(starttime = T-TNoise1, endtime = T-TNoise2, fill_value=0)
    Noisamean = sum(abs(tr[0].data))/len(tr[0].data)
    # measure signal level
    tr = st.copy()
    tr.trim(starttime = T-TSignal1, endtime = T+TSignal2, fill_value=0)
    SignMax = max(abs(tr[0].data))
    # define the signal to noise value
    snr = SignMax/Noisamean
    return snr
def get_theoretical_arrival(st, Tp):
    model = TauPyModel(model=f"C:/Users/youzh/Documents/My documents(not synced)/sP_Workflow/Others/Path/Path/hk.npz")
    sachd = st[0].stats.sac
    distaz = client.distaz(sachd["stla"], sachd["stlo"], sachd["evla"], sachd["evlo"])
    gcarc = distaz['distance']

    arrival = model.get_travel_times(source_depth_in_km=0,distance_in_degree=gcarc, phase_list=["ttp"])
    p = float("{:.3f}".format(arrival[0].time))

    
    if sachd["evdp"] < 0:
        arrival = model.get_travel_times(source_depth_in_km=0,distance_in_degree=gcarc, phase_list=["sPvmP"])
    else:                                  
        arrival = model.get_travel_times(source_depth_in_km=sachd["evdp"],distance_in_degree=gcarc, phase_list=["sPvmP"])
    if len(arrival) != 0:
        PTsPmP = float("{:.3f}".format(arrival[0].time))
    else: 
        PTsPmP = "nan" #change this later
    arrival = model.get_travel_times(source_depth_in_km=sachd["evdp"],distance_in_degree=gcarc, phase_list=["sPn"])
    if len(arrival) != 0:
        PTsPn = float("{:.3f}".format(arrival[0].time))
    else:
        PTsPn = "nan"
    return [PTsPmP-p+Tp, PTsPn-p+Tp]
def get_particle_motion_angle(st, idxp1, idxp2):
    maxp = max(max(abs(st[2].data[idxp1:idxp2])),max(abs(st[0].data[idxp1:idxp2])),max(abs(st[1].data[idxp1:idxp2])))
    zpm_p = st[2].data[idxp1:idxp2]/maxp
    rpm_p = st[0].data[idxp1:idxp2]/maxp
    X = np.column_stack((zpm_p, rpm_p))
    pca = PCA(n_components=1)
    pca.fit(X)
    angle = math.degrees(math.atan(pca.components_[0][1]/pca.components_[0][0]))
    #angle of Z vs R
    if angle < 0:
        angle1 = int("{:.0f}".format(180+angle))
    else:
        angle1 = int("{:.0f}".format(angle))
    return angle1

def get_spectrum(st, idxp1, idxp2):
    dt = st[2].stats.sac.delta
    z_p = st[2].data[idxp1:idxp2]
    (freqs,Pxx) = signal.periodogram(z_p, nfft=2**8, fs=1/dt)
    df1 = float("%.1f" % freqs[np.argmax(Pxx)])   
    return df1
def auto_filter(st, Tp, p_len, sPmP_len, sPn_len, SRsPmP, SRsPn, snr):
    #SR is search range
    snr_val = get_snr(st)
    if snr_val < snr:
        print("Signal Noise Ratio lower than threshold: ", snr_val)
        return False
    #calculate predicted arrival time for sPmP and sPn
    PTsPmP, PTsPn = get_theoretical_arrival(st, Tp)
    if PTsPmP == "nan":
        print("no predicted sPmP arrival")
        return False
    if PTsPn == "nan":
        print("no predicted sPn arrival")
    #set window for p 
    shift = st[0].stats.sac.o
    idxp1 = int((Tp+shift-st[0].stats.sac.b)/st[0].stats.delta) #what is st[0].stats.sac.b?
    idxp2 = int((Tp+shift+p_len-st[0].stats.sac.b)/st[0].stats.delta)
    # amplitude (maximum amplitude within the interval idxp1 and idxp2)
    zamp_p = max(abs(st[2].data[idxp1:idxp2]))
    # particle motion
    angle1 = get_particle_motion_angle(st, idxp1, idxp2)
      #spectrum
    df1 = get_spectrum(st, idxp1, idxp2)
    #list of stats
    UTsPmP = []
    PMA1 = []
    DFreq1 = []
    ratio11 = []
    ratio12 = []
    ratio13 = []
    ssr1 = []
    PMAr = []
    for cTsPmP in np.arange(PTsPmP-SRsPmP,PTsPmP+SRsPmP+sPmP_len/2,0.05):
        idxspm1 = int((cTsPmP +shift-st[0].stats.sac.b)/st[0].stats.delta)
        idxspm2 = int((cTsPmP +shift+sPmP_len-st[0].stats.sac.b)/st[0].stats.delta)
        # amplitude
        zamp_spm = (max(abs(st[2].data[idxspm1:idxspm2])))
        ramp_spm = (max(abs(st[0].data[idxspm1:idxspm2])))
        tamp_spm = (max(abs(st[1].data[idxspm1:idxspm2])))
        ratio11_temp = zamp_spm/zamp_p #amplitude ratio z/z
        ratio12_temp = zamp_spm/ramp_spm #amplitude ratio z/r
        ratio13_temp = ramp_spm/tamp_spm #amplitude ratio z/t
        # particle motion
        angle2 = get_particle_motion_angle(st, idxspm1, idxspm2)
        # spectrum
        df2 = get_spectrum(st, idxspm1, idxspm2)
        if (ratio11_temp > 0.5) and (ratio12_temp > 0.5) and (ratio13_temp > 1.0) and (abs(angle1-angle2)<20) and (abs(df1-df2)<1.5):
            UTsPmP.append(cTsPmP) #if the above conditions are met then the arrival time of the wave is stored in UTsPmP
            PMA1.append(angle2)
            DFreq1.append(df2)
            ratio11.append(ratio11_temp)
            ratio12.append(ratio12_temp)
            ratio13.append(ratio13_temp)
            ssr1.append(ratio11_temp * ratio12_temp * ratio13_temp)
    if str(PTsPn) != "nan":
        #lists
        UTsPn = []
        PMA2 = []
        DFreq2 = []
        ratio21 = []
        ratio22 = []
        ratio23 = []
        ssr2 = []
        if PTsPn+SRsPn+sPn_len > PTsPmP:
            spn_window2 = PTsPmP-sPn_len
        else:
            spn_window2=PTsPn+SRsPn
        for cTsPn in np.arange(PTsPn-SRsPn,spn_window2,0.05):
            idxspn1 = int((cTsPn+shift-st[0].stats.sac.b)/st[0].stats.delta)
            idxspn2 = int((cTsPn+shift+sPn_len-st[0].stats.sac.b)/st[0].stats.delta)
            # amplitude
            zamp_spn = (max(abs(st[2].data[idxspn1:idxspn2])))
            ramp_spn = (max(abs(st[0].data[idxspn1:idxspn2])))
            tamp_spn = (max(abs(st[1].data[idxspn1:idxspn2])))
            ratio21_temp = zamp_spn/zamp_p
            ratio22_temp = zamp_spn/ramp_spn
            ratio23_temp = ramp_spn/tamp_spn
            # particle motion
            angle3 = get_particle_motion_angle(st, idxspn1, idxspn2)
            # spectrum
            df3 = get_spectrum(st, idxspn1, idxspn2)
            # composite test
            if (ratio21_temp > 0.5) and (ratio22_temp > 0.5) and (ratio23_temp > 1.0) and (abs(angle1-angle3)<20) and (abs(df1-df3)<1.5):
                #print("something")
                UTsPn.append(cTsPn)
                PMA2.append(angle3)
                DFreq2.append(df3)
                ratio21.append(ratio21_temp)
                ratio22.append(ratio22_temp)
                ratio23.append(ratio23_temp)
                ssr2.append(ratio21_temp * ratio22_temp * ratio23_temp)
        if len(ssr2) > 0:
                indx2 = np.argmax(np.array(ssr2)/np.max(ssr2)+1*(1-abs(np.array(PMA2)-angle1)/angle1))
                TsPn_autorefined = UTsPn[indx2]
        else:
            TsPn_autorefined = PTsPn
    else:
        TsPn_autorefined = PTsPn
    if len(ssr1) > 0:
        indx1 = np.argmax(np.array(ssr1)/np.max(ssr1)+1*(1-abs(np.array(PMA1)-angle1)/angle1))
        TsPmP_autorefined = float("{:.3f}".format(UTsPmP[indx1]) )
    else: return False
    return [Tp, p_len, PTsPmP, TsPmP_autorefined, sPmP_len, PTsPn, sPn_len, TsPn_autorefined]