# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:04:07 2017
 
@author: H1JDE02
"""
 
import matplotlib.pyplot as plt
 
# Figure 1
plt.figure(facecolor="white")
plt.plot(wgrid, F0, 'k', wgridBM, FBM, 'b:o', linewidth=2)
plt.title("Distribution of wage offers")
plt.xlabel("Wage")
plt.legend(["Equilibrium Distribution", "Burdett-Mortensen Distribution"], loc="best")
plt.grid()
plt.savefig("Pyfigs/py_F1_FBM_qtl.png")

plt.figure(facecolor="white")
plt.plot(GwBM, np.interp(wgridBM, wgrid, F0), 'k', GwBM, FBM, 'k--', linewidth=2)
plt.title("Distribution of wage offers")
plt.xlabel("B-M wage quantile")
plt.legend(["'Equilibrium Offer Distribution", "Burdett-Mortensen Offer Distribution"], loc="best")
plt.grid()
plt.savefig("Pyfigs/py_F1_FBM_qtl_2.png")

plt.figure(facecolor="white")
plt.plot(wgrid,F0,'k',wgrid,Gtilde,'b--',linewidth=2)
plt.title("Distribution of wage offers")
plt.xlabel("w")
plt.ylabel("F(w)")
plt.legend(["Direct Offer Distribution","Network Offer Distribution"],loc="best")
plt.grid()
plt.savefig("Pyfigs/py_F1_Gtilde.png")
 
# Figure 2
ER1 = np.trapz(Omegaz*R1,zgrid)/np.trapz(Omegaz,zgrid)
plt.figure(facecolor="white")
plt.plot(zgrid,R1/wbar,'k',np.linspace(z0,zZ,zpts),ER1*np.ones(zpts),'k--',
         np.linspace(z0,zZ,zpts),RBM*np.ones(zpts)/wbar,'b:o',linewidth=2)
yL = np.min(np.append(R1,RBM)/wbar)
yH = np.max(np.append(R1,RBM)/wbar)
plt.xlabel("z")
plt.ylabel("R(z), % of max")
plt.legend(['Equilibrium Reservation','Average Reservation','Burdett-Mortensen Reservation'],loc="best")
plt.grid()
plt.savefig("Pyfigs/py_R1_RBM.png")
 
# Figure 3
plt.figure(facecolor="white")
plt.plot(zgrid,nz,'k',linewidth=2)
plt.title("Distribution of steady-state employment")
plt.xlabel("z")
plt.ylabel("n(z)")
plt.grid()
plt.savefig("Pyfigs/py_nz.png")
 
plt.figure(facecolor="white")
plt.plot(zgrid,np.log(r0z),'k',zgrid,np.log(r1z),'b--',linewidth=2)
plt.title("Offer arrival rate through the network")
plt.xlabel("z")
plt.ylabel("Log arrival rate")
plt.legend(["Off the job", "On the job"],loc="best")
plt.grid()
plt.savefig("Pyfigs/py_rhoz.png")
 
# Figure 4
plt.figure(facecolor="white")
plt.plot(wgrid,np.log(Lw),'k',linewidth=2)
plt.title("Distribution of steady-state employment by wage")
plt.xlabel("w")
plt.ylabel("log L(w)")
plt.grid()
plt.savefig("Pyfigs/py_Lw1.png")
 
# Figure 5
plt.figure(facecolor="white")
plt.plot(Gw,F0,'r-',Gw,FG,'k',Gw,Gtilde,'b--',linewidth=2)
plt.xlabel("Wage quantile")
plt.ylabel("Cumulative distribution")
plt.title("Offer Distribution")
plt.legend(["Direct Contact","Overall","Network"],loc="best")
plt.savefig("Pyfigs/py_F0FGGtilde.png")
 
plt.figure(facecolor="white")
plt.plot(Gw,F0,'r-',Gw,FGz[:,0],'k*',Gw,FGz[:,zpts-1],'ko',Gw,Gtilde,'b--',linewidth=2)
plt.xlabel('Wage quantile')
plt.ylabel('Cumulative distribution')
plt.legend(["Direct Contact","Average for low z","Average for high z","Network"],loc="best")
plt.grid()
plt.savefig("Pyfigs/py_F0FGz0FGzZGtilde.png")
 
# Figure 6
plt.figure(facecolor="white")
plt.plot(zgrid,np.interp(w_z,wgrid,Gw),'k',zgrid,np.interp(w1_z,wgrid,Gw),'k--',linewidth=2)
plt.xlabel("z")
plt.ylabel("Wage quantile")
plt.legend(["Average Wage","Average Wage Out of Unemployment"],loc="best")
plt.grid()
plt.savefig("Pyfigs/py_w_zw1_z.png")
 
# Figure 7
plt.figure(facecolor="white")
plt.plot(zgrid,halflife_wz[34,:],'k-.',zgrid,halflife_wz[24,:],'k--',zgrid,halflife_wz[1,:],'k',linewidth=2)
plt.xlabel("z")
plt.ylabel("Half-life (months) to maximum wage")
plt.legend(["Offer = 70 pctile","Offer = 55 pctile","Offer = 9 pctile"])
plt.grid()
plt.savefig("Pyfigs/py_halflife.png")
 
# Figure 8
plt.figure(facecolor="white")
plt.plot(np.linspace(0,1,wpts),LorenzW,'k',np.linspace(0,1,wpts),LorenzWBM,'k--',linewidth=2)
 
plt.figure(facecolor="white")
plt.plot(zgrid,halflife_wz[34,:],'k--',
         zgrid,np.ones(zpts)*np.interp(FG[34],FBM[0:-1],halflife_BM),'b--',
         zgrid,halflife_wz[1,:],'k',
         zgrid,np.ones(zpts)*np.interp(FG[1],FBM[0:-1],halflife_BM),'b',linewidth=2)
plt.xlabel("z")
plt.ylabel("Half-life (months) to maximum wage")
plt.grid()
plt.savefig("Pyfigs/py_halflife_BM.png")
 
plt.figure(facecolor="white")
plt.plot(wgrid,np.log(lwz[:,1]/Omegaz[1]),'k',wgrid,np.log(lwz[:,zpts-2]/Omegaz[zpts-2]),'b--',linewidth=2)
plt.xlabel("z")
plt.ylabel("log(l(w,z)/\Omegaz(z))")
plt.legend(["z=1","z=61"])
plt.grid()
plt.savefig("Pyfigs/py_Pr_netfnd.png")
 
# Figure 9: problems with Gw -> problems with F0
plt.figure(facecolor="white")
plt.plot(Gw[0:wpts-1],Pr_netfnd-np.mean(Pr_netfnd),'k',Gw[0:wpts-1],gw_meanz[0:wpts-1]/meanz-1,'k--',linewidth=2)
plt.xlabel("Wage quantile")
plt.legend(["Probability of network find","Average z",], loc="best")
plt.grid()
plt.savefig("Pyfigs/py_Pr_netfnd_meanz.png")
 
# Figure 10
plt.figure(facecolor="white")
plt.plot(wgrid,Gw,'k',wgrid,GwHS,'k--',linewidth=2)
 
plt.figure(facecolor="white")
plt.plot(np.linspace(0,1,wpts),LorenzW,'k',np.linspace(0,1,wpts),LorenzWHS,'k--',linewidth=2)
 
hl_wz_HS_35 = np.zeros(zpts)
hl_wz_HS_2 = np.zeros(zpts)
for zi in range(zpts):
    hl_wz_HS_35[zi] = np.interp(FG[34],FHS[0:wpts-1],halflife_wz_HS[:,zi])
    hl_wz_HS_2[zi]  = np.interp(FG[1],FHS[0:wpts-1],halflife_wz_HS[:,zi])
 
plt.figure(facecolor="white")
plt.plot(zgrid,halflife_wz[34,:],'k--',zgrid,hl_wz_HS_35,'b--',
         zgrid,halflife_wz[1,:],'k',zgrid,hl_wz_HS_2,'b',linewidth=2)
plt.xlabel("z")
plt.ylabel("Half-life (months) to maximum wage")
plt.legend(["Offer=70 pctile, with network",
"Offer=9 pctile, with network",
"Offer=70 pctile, heterogenous rates",
"Offer=9 pctile, heterogenous rates"])
plt.grid()
plt.savefig("Pyfigs/py_halflife_HS.png")
 
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
plt.plot(zgrid,np.log(Distz_dir),'k',zgrid,np.log(Distz_net),'b--',linewidth=2)
plt.xlabel("z")
plt.ylabel("Log density of number of peers")
plt.legend(["Direct contact","Network search"], loc="best")
plt.grid()
plt.savefig("Pyfigs/py_Distz_dir_net.png")
 
