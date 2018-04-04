# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:04:07 2017
 
@author: H1JDE02"""

import sys
import os
import inspect
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline,pchip_interpolate
import csv

sys.path.insert(0, os.path.abspath('/home/david/workspace/AOW/'))

execfile("discr_dist_F0_F1.py")

#read data from SCE:
SCEpctnetfnd = np.zeros((47,2))
with open('pctnetfnd_prob.csv', 'rb') as SCEcsv:
    csvSCEpctnetfnd = csv.reader(SCEcsv, delimiter=',')
    header = csvSCEpctnetfnd.next()
    for qi in range(0, 47):
        SCEpctnetfnd[qi,] = csvSCEpctnetfnd.next()



#Figure 9: adding in the data:
csPrNetFind = pchip(Gw[:-1],Pr_netfnd, extrapolate= True)
csSCEprnetfind = pchip(SCEpctnetfnd[:,1], SCEpctnetfnd[:,0], extrapolate= True)
plt.figure(facecolor="white")
plt.plot(SCEpctnetfnd[:,1],csSCEprnetfind(SCEpctnetfnd[:,1]),'k',SCEpctnetfnd[:,1], \
    csPrNetFind(SCEpctnetfnd[:,1])-csPrNetFind(np.mean(SCEpctnetfnd[:,1])),'k--',linewidth=2)
plt.xlabel("Wage quantile")
plt.legend(["SCE Data","Model"], loc="best")
plt.xticks(np.arange(0, .6, step=0.2*.6),np.arange(0, 1, step=0.2))
plt.grid()
plt.savefig("Pyfigs/py_Pr_netfnd_dat.png")
plt.savefig("Pyfigs/py_Pr_netfnd_dat.eps")

# Figure 1
#wgeven = np.linspace(np.min(wgrid),np.max(wgrid),10*wpts)
#wgevenBM = np.linspace(np.min(wgridBM),np.max(wgridBM),10*wpts)
#plt.figure(facecolor="white")
#plt.plot(wgrid, Fw, 'k', wgridBM, FBM, 'b:', linewidth=2)
#plt.title("Distribution of wage offers")
#plt.xlabel("Wage")
#plt.legend(["Equilibrium Distribution", "Burdett-Mortensen Distribution"], loc="best")
#plt.grid()
#plt.savefig("Pyfigs/py_F1_FBM_qtl.png")
#plt.savefig("Pyfigs/py_F1_FBM_qtl.eps", format="eps")

plt.figure(facecolor="white")
#plt.plot(Gw, Fw, 'k', Gw,np.interp(wgrid,wgridBM,FBM),  'k--', linewidth=2)
plt.plot(GwBM, np.interp(wgridBM,wgrid,Fw), 'k', GwBM,FBM,  'k--', linewidth=2)
plt.title("Distribution of wage offers")
plt.xlabel("BM wage quantile")
plt.legend(["'Equilibrium Offer Distribution", "Burdett-Mortensen Offer Distribution"], loc="best")
plt.grid()
plt.savefig("Pyfigs/py_F1_FBM_qtl_2.png")
plt.savefig("Pyfigs/py_F1_FBM_qtl_2.eps", format="eps")
#crossing point: 
np.interp(0.,GwBM,np.interp(wgridBM,wgrid,Fw)-FBM)


plt.figure(facecolor="white")
plt.plot(wgrid,Fw,'k',wgrid,Gtilde,'b--',linewidth=2)
plt.title("Distribution of wage offers")
plt.xlabel("w")
plt.ylabel("F(w)")
plt.legend(["Direct Offer Distribution","Network Offer Distribution"],loc="best")
plt.grid()
plt.savefig("Pyfigs/py_Fw_Gtilde.png")
plt.savefig("Pyfigs/py_Fw_Gtilde.eps", format="eps")
 
# Figure 2
ERz = simps(Omegaz*Rz,zgrid)/simps(Omegaz,zgrid)
cs_Rz = pchip(zgrid,Rz)
plt.figure(facecolor="white")
plt.plot(np.log(zgrid),cs_Rz(zgrid),'k',np.linspace(np.log(z0),np.log(zZ),zpts),ERz*np.ones(zpts),'k--',
         np.linspace(np.log(z0),np.log(zZ),zpts),RBM*np.ones(zpts),'b:o',linewidth=2)
yL = np.min(np.append(Rz,RBM))
yH = np.max(np.append(Rz,RBM))
plt.xlabel("Log(number of connections)")
plt.ylabel("R(z), % of max wage")
plt.legend(['Equilibrium Reservation','Average Reservation','Burdett-Mortensen Reservation'],loc="best")
plt.grid()
plt.savefig("Pyfigs/py_R1_RBM.png")
plt.savefig("Pyfigs/py_R1_RBM.eps", format="eps")
 
# Figure 3
csW_zpctile = pchip(zgrid,np.interp(w_z,wgrid,Gw))
plt.figure(facecolor="white")
fig, ax1 = plt.subplots()


ax1.set_xlabel('Log(number of connections)')
ax1.set_ylabel("Employment Rate", color='k')
l1= ax1.plot(np.log(zgrid), nz, color='k',linewidth=2,label="n(z), Left")
ax1.tick_params(axis='y', labelcolor='k')
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Wage Quantile', color='b')  # we already handled the x-label with ax1
l2 = ax2.plot(np.log(zgrid), csW_zpctile(np.log(zgrid)), 'b--',linewidth=2,label="E[w(z)], Right")
ax2.tick_params(axis='y', labelcolor='b')
ax1.grid()

plt.savefig("Pyfigs/py_nz_wz.png")
plt.savefig("Pyfigs/py_nz_wz.eps", format="eps")


plt.figure(facecolor="white")
plt.plot(zgrid,np.log(r0z),'k',zgrid,np.log(r1z),'b--',linewidth=2)
plt.title("Offer arrival rate through the network")
plt.xlabel("z")
plt.ylabel("Log arrival rate")
plt.legend(["Off the job", "On the job"],loc="best")
plt.grid()
plt.savefig("Pyfigs/py_rhoz.png")
plt.savefig("Pyfigs/py_rhoz.eps", format="eps")

# Figure 4
plt.figure(facecolor="white")
plt.plot(wgrid,Lw,'k',linewidth=2)
plt.title("Distribution of steady-state firm-size by wage")
plt.xlabel("w")
plt.ylabel(" L(w)")
plt.grid()
plt.savefig("Pyfigs/py_Lw1.png")
plt.savefig("Pyfigs/py_Lw1.eps", format="eps")

# Figure 5
plt.figure(facecolor="white")
plt.plot(Gw,Fw,'r-',Gw,FG,'k',Gw,Gtilde,'b--',linewidth=2)
plt.xlabel("Wage quantile")
plt.ylabel("Cumulative distribution")
plt.title("Offer Distribution")
plt.legend(["Direct Contact","Overall","Network"],loc="best")
plt.grid()
plt.savefig("Pyfigs/py_F0FGGtilde_linear.png")

csFw = pchip(Gw,Fw)
csFG = pchip(Gw,FG)
csGtilde = pchip(Gw,Gtilde)
qtls = np.linspace(0,1,wpts)
plt.figure(facecolor="white")
plt.plot(qtls,csFw(qtls),'r-',qtls,csFG(qtls),'k',qtls,csGtilde(qtls),'b--',linewidth=2)
plt.xlabel("Wage quantile")
plt.ylabel("Cumulative distribution")
plt.title("Offer Distribution")
plt.legend(["Direct Contact","Overall","Network"],loc="best")
plt.grid()
plt.savefig("Pyfigs/py_F0FGGtilde.png")
plt.savefig("Pyfigs/py_F0FGGtilde.eps", format="eps")


csFw = pchip(Gw,Fw) 
csFGz0 = pchip(Gw,FGz[:,0]) 
csFGzZ = pchip(Gw,FGz[:,zpts-1]) 
csGtilde = pchip(Gw,Gtilde)
qtls = np.linspace(0,1,wpts)
plt.figure(facecolor="white")
plt.plot(qtls,csFw(qtls),'r-',qtls,csFGz0(qtls),'k-',qtls,csFGzZ(qtls),'k--',qtls,csGtilde(qtls),'b--',linewidth=2)
plt.xlabel('Wage quantile')
plt.ylabel('Cumulative distribution')
plt.legend(["Direct Contact","Average, poorly connected","Average, well connected","Network"],loc="best")
plt.grid()
plt.savefig("Pyfigs/py_F0FGz0FGzZGtilde.png")
plt.savefig("Pyfigs/py_F0FGz0FGzZGtilde.eps", format="eps")

# Figure 6
plt.figure(facecolor="white")
plt.plot(zgrid,np.interp(w_z,wgrid,Gw),'k',zgrid,np.interp(w1_z,wgrid,Gw),'k--',linewidth=2)
plt.xlabel("z")
plt.ylabel("Wage quantile")
plt.legend(["Average Wage","Average Wage Out of Unemployment"],loc="best")
plt.grid()
plt.savefig("Pyfigs/py_w_zw1_z.png")
plt.savefig("Pyfigs/py_w_zw1_z.eps", format="eps")

# Figure 7
plt.figure(facecolor="white")
plt.plot(np.log(zgrid),halflife_wz[34,:],'k-.',np.log(zgrid),halflife_wz[24,:],'k--',np.log(zgrid),halflife_wz[1,:],'k',linewidth=2)
plt.xlabel("log(number connections)")
plt.ylabel("Half-life (months) to maximum wage")
plt.legend(["Offer = 70 pctile","Offer = 55 pctile","Offer = 9 pctile"])
plt.grid()
plt.savefig("Pyfigs/py_halflife_linear.png")
 
cs34 =  pchip(np.log(zgrid),halflife_wz[34,:])
cs24 =  pchip(np.log(zgrid),halflife_wz[24,:])
cs01 =  pchip(np.log(zgrid),halflife_wz[1,:])
logzgrid = np.linspace(np.log(zgrid[0]),np.log(zgrid[-1]) ,zpts)
plt.figure(facecolor="white")
plt.plot(logzgrid ,cs34(logzgrid),'k-.',logzgrid ,cs24(logzgrid),'k--',logzgrid ,cs01(logzgrid),'k',linewidth=2)
plt.xlabel("Log(number connections)")
plt.ylabel("Half-life (months) to maximum wage")
plt.legend(["64 pctile offer","50 pctile offer","10 pctile offer"])
plt.grid()
plt.savefig("Pyfigs/py_halflife.png")
plt.savefig("Pyfigs/py_halflife.eps", format="eps")


# Figure 8
#plt.figure(facecolor="white")
#plt.plot(np.linspace(0,1,wpts),LorenzW,'k',np.linspace(0,1,wpts),LorenzWBM,'k--',linewidth=2)
 
plt.figure(facecolor="white")
plt.plot(np.log(zgrid),halflife_wz[24,:],'k--',
         np.log(zgrid),np.ones(zpts)*np.interp(FG[24],FBM[0:-1],halflife_BM),'b--',
         np.log(zgrid),halflife_wz[1,:],'k',
         np.log(zgrid),np.ones(zpts)*np.interp(FG[1],FBM[0:-1],halflife_BM),'b',linewidth=2)
plt.xlabel("Log(number connections)")
plt.ylabel("Half-life (months) to maximum wage")
plt.legend(["50 pctile offer","50 pctile offer, BM","10 pctile offer","10 pctile offer, BM"])
plt.grid()
plt.savefig("Pyfigs/py_halflife_BM.png")
plt.savefig("Pyfigs/py_halflife_BM.eps", format="eps")
 
plt.figure(facecolor="white")
plt.plot(wgrid,np.log(lwz[:,1]/Omegaz[1]),'k',wgrid,np.log(lwz[:,zpts-2]/Omegaz[zpts-2]),'b--',linewidth=2)
plt.xlabel("z")
plt.ylabel("log(l(w,z)/\Omegaz(z))")
plt.legend(["z=1","z=61"])
plt.grid()
plt.savefig("Pyfigs/py_Pr_netfnd.png")
plt.savefig("Pyfigs/py_Pr_netfnd.eps", format="eps")
 

# Figure 10
plt.figure(facecolor="white")
plt.plot(wgrid,Gw,'k',wgrid,GwHS,'k--',linewidth=2)
 
plt.figure(facecolor="white")
plt.plot(np.linspace(0,1,wpts),LorenzW,'k',np.linspace(0,1,wpts),LorenzWHS,'k--',linewidth=2)
 
hl_wz_HS_25 = np.zeros(zpts)
hl_wz_HS_2 = np.zeros(zpts)
for zi in range(zpts):
    hl_wz_HS_25[zi] = np.interp(FG[24],FHS[0:wpts-1],halflife_wz_HS[:,zi])
    hl_wz_HS_2[zi]  = np.interp(FG[1],FHS[0:wpts-1],halflife_wz_HS[:,zi])
 
plt.figure(facecolor="white")
plt.plot(np.log(zgrid),halflife_wz[24,:],'k--',np.log(zgrid),hl_wz_HS_25,'b--',
         np.log(zgrid),halflife_wz[1,:],'k',np.log(zgrid),hl_wz_HS_2,'b',linewidth=2)
plt.xlabel("Log(number of connections)")
plt.ylabel("Half-life (months) to maximum wage")
plt.legend(["50 pctile offer",
"10 pctile offer",
"50 pctile offer, hetero arrival",
"10 pctile offer, hetero arrival"])
plt.grid()
plt.savefig("Pyfigs/py_halflife_HS.png")
plt.savefig("Pyfigs/py_halflife_HS.eps", format="eps")
 
# Figure 11
plt.figure(facecolor="white")
plt.plot(wgrid[0:wpts/2]/wgrid[wpts-1], np.log(Emdur_w_dir[0:wpts/2]),'k',
         wgrid[0:wpts/2]/wgrid[wpts-1], np.log(Emdur_w_net[0:wpts/2]),'b--', linewidth=2)
plt.xlabel("wage, % of maximum")
plt.ylabel("log Expected length of match duration")
plt.legend(["Expected duration of match, direct contact","Expected duration of match, network search"], loc="best")
plt.grid()
plt.savefig("Pyfigs/py_Emdur_w.png")
 
# Figure 12
plt.figure(facecolor="white")
plt.plot(np.log(zgrid),np.log(Distz_dir),'k',np.log(zgrid),np.log(Distz_net),'b--',linewidth=2)
plt.xlabel("Log(number of connections)")
plt.ylabel("Log(density)")
plt.legend(["Direct contact","Network search"], loc="best")
plt.grid()
plt.savefig("Pyfigs/py_Distz_dir_net.png")
plt.savefig("Pyfigs/py_Distz_dir_net.eps", format="eps")
 
