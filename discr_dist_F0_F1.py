# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 10:15:59 2017

@author: h1jde02
@author: wiczerd
"""
import sys
import os

sys.path.insert(0, os.path.abspath('/home/david/workspace/AOW/'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize
from scipy.interpolate import pchip
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz

########## Solving Networks & Search


#wrapper for np.interp to enforce positive monotonicity (crudely)
def np_interp_mono(x,xp,fp):

    if np.all(np.diff(xp) > 0):
        f=np.interp(x,xp,fp)
    else:
        xxp_idx = np.argsort(xp)
        ffp = np.copy(fp[xxp_idx])
        xxp = np.copy(xp[xxp_idx])
        f=np.interp(x,xxp,ffp)

    return(f)

##### Functions
def rhoz0(z,nz):
    global Psis
    global gamma1
    global nu0
    global zgrid

    pzs = Psis*nz*gamma1*nu0/zgrid
    pz = np.trapz(pzs,zgrid)
    P = (1 - (1-pz)**z)

    return(P)

def rhoz1(z,nz):
    global Psis
    global gamma1
    global nu1
    global zgrid

    pzs = Psis*nz*gamma1*nu1/zgrid
    pz = np.trapz(pzs,zgrid)
    P = (1 - (1-pz)**z)

    return(P)

def Gwzdef(dist,R,F,mask,r0z,r1z):
    global gamma1
    global gamma0
    global delta
    global b
    global Psis
    global zgrid
    global wgrid
    global zpts
    global wpts

    # dist is Gwz stacked column-wise
    # R is the vector of reservation wages at each z
    # F is the initial guess for the wage distribution
    # mask is an indicator that, given Rz, we expect dist to be non-zero
    FRz = np.zeros(zpts)
    GtRz = np.zeros(zpts)

    onesz = np.ones(zpts)
    Gwz_imp = np.zeros((wpts,zpts))

    # Tricky to make masking work the same as in Matlab
    Gwz = np.zeros(wpts*zpts)
    mask = mask.flatten(order='F')
    Gwz[mask] = dist.copy()
    Gwz = Gwz.reshape(wpts,zpts,order='F')

    Gtilde = np.trapz(np.outer(Psis, np.ones(wpts)) * Gwz.T, zgrid,axis=0)

    Vp = 1/(delta + gamma1*np.outer(1-F,onesz) + (1-gamma1)*np.outer(1-Gtilde,r1z))

    for zi in range(0,zpts): #for(i=0;i<zpts;i++){}
        if R[zi] < wgrid[wpts-1]:
            FRz[zi] = np.interp(R[zi],wgrid,F,left=0,right=0)
            GtRz[zi] = np.interp(R[zi],wgrid,Gtilde,left=0,right=0)
        else:
            FRz[zi] = 1
            GtRz[zi] = 1

        v = (gamma0*(F - FRz[zi]) + (1-gamma0)*r0z[zi]*(Gtilde - GtRz[zi])) \
            /(delta + gamma1*(1-F) + (1-gamma1)*r1z[zi]*(1-Gtilde)) \
            * delta/(gamma0*(1-FRz[zi]) + (1-gamma0)*r0z[zi]*(1-GtRz[zi]))
        Gwz_imp[:,zi] = v.copy()

    Gwz_imp = Gwz_imp.flatten('F')
    Gwz_imp[~mask] = 0

    # MUST MAKE COPY, OTHERWISE CHANGES TO VP_RWBAR ALSO CHANGE VP.
    Vp_Rwbar = Vp.copy()

    VpGxs_sz = np.zeros(zpts)
    Vp_1F_sz = np.zeros(zpts)

    for zi in range(0,zpts):
        indic = (wgrid <= R[zi])
        Vp_Rwbar[indic,zi] = 0

        # VpGxz_fun giving different value than Matlab, check
        # This makes final results slightly different
        # VpGxz_fun = lambda x: np.interp(x,wgrid,(1-Gtilde)*Vp[:,zi],left=np.min(wgrid),right=np.min(wgrid))
        # Vp_Rwbar_fun = lambda x: np.interp(x,wgrid,(1-F)*Vp[:,zi],left=np.min(wgrid),right=np.min(wgrid))

        VpGxz_fun = interp1d(wgrid, (1-Gtilde)*Vp[:,zi], 'linear', bounds_error = False, fill_value=np.min(wgrid))
        Vp_Rwbar_fun = interp1d(wgrid, (1-F)*Vp[:,zi], 'linear', bounds_error = False, fill_value=np.min(wgrid))

        Rzi_wgrid = R[zi]
        Rzi_wgrid = np.max([wgrid[0],np.min([wgrid[wpts-1],Rzi_wgrid])])

        VpGxs_sz_tmpy = np.append(VpGxz_fun(Rzi_wgrid), (1-Gtilde[indic==0])*Vp[indic==0,zi])
        VpGxs_sz_tmpx = np.append(Rzi_wgrid, wgrid[indic==0])
        VpGxs_sz[zi] = np.trapz(VpGxs_sz_tmpy, VpGxs_sz_tmpx)

        Vp_1F_sz_tmpy = np.append(Vp_Rwbar_fun(Rzi_wgrid), (1-F[indic==0])*Vp[indic==0,zi])
        Vp_1F_sz_tmpx = np.append(Rzi_wgrid, wgrid[indic==0])
        Vp_1F_sz[zi] = np.trapz(Vp_1F_sz_tmpy, Vp_1F_sz_tmpx)

    # Reservation wage at each Z.  Not necessary to solve for G(w,z) but easy to compute and useful.
    Rz = b + (gamma0-gamma1)*Vp_1F_sz +  ( (1-gamma0)*r0z - (1-gamma1)*r1z )*VpGxs_sz

    Rz[Rz > wgrid[wpts-1]] = wgrid[wpts-1]

    resid = Gwz_imp[mask] - dist

    Gwz_imp = Gwz_imp.reshape(wpts,zpts,order='F')

    return [resid,Gwz_imp,Rz,Vp]

def sol_nu1(nu1_in):
    global nu0
    global nu1
    global gamma0
    global gamma1
    global zgrid
    global Omegaz
    global zpts

    nu0_old = nu0
    nu1_old = nu1
    nu1 = nu1_in
    r1z = np.ones(zpts)
    r0z = np.ones(zpts)
    for zi in range(0,zpts):
        r1z[zi] = rhoz1(zgrid[zi],0.95)
        r0z[zi] = rhoz0(zgrid[zi],0.95)

    resid = gamma1/gamma0 - np.trapz(Omegaz*(r1z/r0z),zgrid)

    nu1 = nu1_old
    nu0 = nu0_old

    return(resid)

def lowestw(R1):
    global Omegaz
    global wgrid
    global wpts
    global zpts
    global zgrid
    global p

    #need to solve for the wL such that wL = argmax (p-w)\int_1^R^{-1}(w)Omega(z)dz =>
    # (p-wL)Omega( R^{-1}(wL))  - \int_1^R^{-1}(wL)Omega(z)dz = 0

    # sort R1
    wgrid_R = np.sort(R1)
    # remove duplicates so that no flat spots
    for zi in range(1,zpts-1):
        if wgrid_R[zi]-wgrid_R[zi-1] < 5e-5:
            wgrid_R[zi] = wgrid_R[zi-1] + 5e-5

    if wgrid_R[zpts-1]-wgrid_R[zpts-2] < 5e-5:
        wgrid_R[zpts-1] = wgrid_R[zpts-2] + 5e-5

    wgrid_R = np.unique(wgrid_R)
    zpts_hr = np.max(len(wgrid_R))

    # ltRDist is the H() from B-M or J() in our draft
    # this will solve for the lowest offered wage
    ltRdist = np.zeros(zpts_hr)
    atRdist = np.zeros(zpts_hr)
    if zpts_hr > 2 and np.isfinite(zpts_hr):

        Om_adjfac = 1/np.trapz(Omegaz, zgrid)

        for zi in range(zpts_hr-1,0,-1):
            ltRdist[zi] = np.trapz(Omegaz[R1 <= wgrid_R[zi]], zgrid[R1 <= wgrid_R[zi]])*Om_adjfac
            zU = min(sum(R1 <= wgrid_R[zi]),zpts_hr-1 )
            atRdist[zi] = Omegaz[ zU ]
        ltRdist[0] = 0
        # now solve wL = argmax (p-w)*J(w) => p-J(w)/dJdw= w=> wL = -J(wL)/dJdw(wL)+p
        # this funciton is not the same as Matlab, check
        ppRdist = pchip(wgrid_R,ltRdist)
        pdRdist = pchip(wgrid_R,atRdist)


        resid = lambda wL: (p-wL)*pdRdist(wL) - ppRdist(wL)


        #####################################################
        # for some reason, this is not working right at R[0]
        lo = resid(wgrid_R[1])
        hi = resid(wgrid_R[zpts_hr-1])

        if lo*hi < 0:
            # fsolve gives a different result here than in Matlab, even with same resid func and wgrid_R
            wLoffered = fsolve(resid, np.mean(wgrid_R))
        elif abs(lo) < abs(hi):
            wLoffered = wgrid_R[0]
        else:
            wLoffered = wgrid_R[zpts_hr-1]

        if wLoffered > np.max(R1):
            wLoffered = np.max(R1)
        if wLoffered < np.min(R1):
            wLoffered = np.min(R1)

    else:

        wLoffered = np.max(wgrid_R)

    return(wLoffered)

def eqPi(F1in, pi_in,hwz0,r1z_endo,zg_endo,zused,Gtilde) :
#equalizes profit by manipulating F
    global wgrid
    global wpts
    global zpts
    global zgrid
    global p

    F1_resid = np.zeros(wpts)
    for wi in range(wpts-2,0,-1):
        F1_resid[wi] = pi_in/(p - wgrid[wi]) \
            - np.trapz(hwz0[wi,0:zused[wi]]/(delta + gamma1*(1-F1in[wi]) + (1-gamma1)*r1z_endo[wi,0:zused[wi]]*(1-Gtilde[wi])), zg_endo[wi,0:zused[wi]], axis=0)

    return(F1_resid)

def eqPi_jac(F1in, pi_in,hwz0,r1z_endo,zg_endo,zused,Gtilde) :
    #jacobian for eqPi condition solving for F1in
    global wgrid
    global wpts
    global zpts
    global zgrid
    global p

    F1_resid_jac = np.zeros(wpts)
    for wi in range(wpts-2,0,-1):
        F1_resid_jac[wi] = -gamma1*np.trapz(hwz0[wi,0:zused[wi]]/np.square(delta + gamma1*(1-F1in[wi]) + (1-gamma1)*r1z_endo[wi,0:zused[wi]]*(1-Gtilde[wi])), zg_endo[wi,0:zused[wi]], axis=0)

    return(F1_resid_jac)

##### Solving networks and search

nu0eqnu1 = 0

update_Gwz = 0.1
update_R = 0.05
update_wbar = 0.5
update_n = 0.5
update_wL = 0.2
update_F = 0.05
nstep =20
th_update = 5
wLtol = 1e-5
ntol=1e-6
RzTol=1e-6
RzConverged = 0

maxRziter = 2000
minRziter = 500
maxwbariter = 50
maxwLiter = 50
maxniter = 20
maxFiter = 200

gamma0 = 0.377
gamma1 = 0.25*gamma0
gamma0BM = gamma0
gamma1BM = gamma1
b = 0
p = 1
alpha = 2.1
M = 0.5
nu0 = 1.1
nu1 = nu0*gamma1/gamma0
delta = (gamma0 + gamma1*nu0)*0.06/(1.-0.06)

if (gamma0 - gamma1 < 1e-2 and nu0 - nu1 < 1e-2):
    maxRziter = 1
    minRziter = 1

z0 = 1
zZ = ((0.99*(1-alpha)/(alpha-1)+z0 )*z0**(-alpha))**(1/(1-alpha))

zpts = 30
wpts = 50

zpow = 1.5
wpow = 0.5

zeros = np.zeros(2)
zeros[0]+=1
zeros[1]+=2

# Set unevenly spaced grid points
zgrid = np.linspace(0,1,zpts+1)**zpow * (zZ-z0) + z0
zgrid = .5*(zgrid[1:] + zgrid[0:-1])
zgrid[0] = z0
zgrid[zpts-1] = zZ

zstep = np.linspace(0,1,zpts+1)**zpow * (zZ-z0) + z0
zstep = zstep[1:] - zstep[:-1]

# Setting up the wage grid: BurdMort solution
k1 = gamma1/delta
k0 = gamma0/delta
wL0 = ((1+k1)**2 * b + (k0-k1)*k1*p)/((1+k1)**2 + (k0-k1)*k1)
wL = wL0
wbar = p-(1-(gamma1/(delta+gamma1)))**2 * (p-wL)

# Put the points in the middle of spaces
wgrid = np.linspace(0,1,wpts)**wpow * (wbar-wL) + wL
wstep = np.zeros(len(wgrid))
wmid =  0.5*(wgrid[:-1]+wgrid[1:])
wstep[1:-1] = wmid[1:] - wmid[:-1]
wstep[0] = wmid[0] - wgrid[0]
wstep[-1] = wgrid[-1] - wmid[-1]

min_wi_Fpos = np.zeros(len(wgrid))

Omegaz = (alpha-1)*(zgrid/z0)**-alpha

# Renorm Omegaz to integrate to 1
Omegaz = Omegaz/np.sum(Omegaz*zstep)
# Mean is hypothetically alpha/(alpha-1)
# @ is matrix multiplication, assumes inner/dot product for 1-d arrays
# same as np.inner(zstep,Omegaz*zgrid)
meanz = np.inner(zstep,(Omegaz*zgrid) )

FBM = np.zeros(wpts)
# Guess F0: solve without any referrals
for wi in range(1,wpts):
    FBM[wi] = (delta + gamma1)/gamma1 * (1-((p-wgrid[wi])/(p-wL))**.5)
FBM[0] = np.min([1e-4, np.min(FBM)/10])

# Be sure it's a distribution
FBM = FBM/FBM[wpts-1]
F0 = FBM.copy()

# Average employment rate
n = 1 - delta/(delta + gamma0*(1-F0[0]))
nz = np.ones(zpts)*n/np.inner(zstep,Omegaz)
nz1 =np.copy(nz)

Psis = nz*Omegaz/np.inner(zstep,nz*Omegaz)
means = np.inner(zgrid, (Psis*zstep))


lwz = np.zeros((wpts,zpts))
Lw = np.zeros(wpts)
Lw1 = np.zeros(wpts)

thw = np.zeros(wpts)

r1z = np.zeros(zpts)
r0z = np.zeros(zpts)

r1z = rhoz1( zgrid, nz )
r0z = rhoz0( zgrid, nz )

R0 = np.ones(zpts)*wL
R1 = np.copy(R0)
mask = np.outer(np.ones(wpts),R0) <= np.outer(wgrid, np.ones(zpts))
#for i in range(0,np.shape(mask)[1]):
#    mask[i,i] = False
Gwzobj = lambda Gwz: Gwzdef(Gwz,R0,nz,F0,mask) #lambda delcares something an anonymous function

# Gwz0 = F0 @ np.ones((1,zpts))
Gwz0 = np.outer(F0, np.ones(zpts))
Gtilde = np.zeros(wpts)

refyield_wz = np.zeros((wpts,zpts))
diryield_wz = np.zeros((wpts,zpts))

# Setup homotopy steps

nu0steps = np.append(0.025, np.linspace(.05,1.,nstep-1)*nu0)
nu1steps = nu0steps*gamma1/gamma0

for ni in range(0,nstep):
    nu0_old = nu0
    nu0 = nu0steps[ni]
    nu1steps[ni] = fsolve(sol_nu1, nu1steps[ni])
    nu0 = nu0_old

if nu0eqnu1 == 1:
    # try with nu1 = nu0
    nu1steps = nu0steps.copy()

#%%

refRates = np.array([])
UErates= []
EErates = []

for homotop_i in range(0,nstep):

    nu0 = nu0steps[homotop_i]
    nu1 = nu1steps[homotop_i]
    nonmonoval = 0
    nonmonocount = 0
    err_wbarnew = 0

    gamma0 = gamma0BM - np.trapz(rhoz0(zgrid,nz), zstep)  #average finding rate adjusted
    gamma1 = gamma1BM - np.trapz(rhoz1(zgrid,nz), zstep)
    for Fi in range(0, maxFiter):

        nonmonolwz = 0
        min_wi_Fpos = np.zeros(len(wgrid))

        for wbar_i in range(0,maxwbariter):

            # Iterate on n, but it doesn't really matter what is the original guess
            for niter in range(0,maxniter):

                # Iterate on Rz until I get the lower bound on wages, actually not the lower bound given heterogeneity
                wL_i = 1
                wL1 = 0

                for wL_i in range(0,maxwLiter):

                    if wL_i > 1:
                        # Will have heterogeneous R
                        if nu0-nu1 > 1e-2 or (gamma0-gamma1 > 1e-2) :
                            wL1 = lowestw(R1)
                        else:
                            wL1 = np.max(R1)

                        wL = update_wL*wL1 + (1-update_wL)*wL

                        # Reconstruct wgrid
                        wstep_old = wstep.copy()
                        # Put the points in the middle of the space
                        wgrid = np.linspace(0,1,wpts)**wpow * (wbar-wL) + wL
                        wstep = np.zeros(len(wgrid))
                        wmid = 0.5*(wgrid[0:-1] + wgrid[1:])
                        wstep[1:-1] = wmid[1:] - wmid[0:-1] #wstep[1:end-1] is the Matlab-equivalent indexing
                        wstep[0] = wmid[0]-wgrid[0]
                        wstep[-1] = wgrid[-1]-wmid[-1]
                        # Rescale F0
                        F0 = F0*wstep_old/wstep
                        F0 = F0/F0[wpts-1]

                    mask = np.outer(np.ones(wpts), R0) <= np.outer(wgrid, np.ones(zpts))
                    mask = mask.flatten('F')

                    reset_flag = 0
                    Rdist = np.zeros((maxRziter,2))

                    for Rziter in range(0,maxRziter):

                        # Solve for the distribution of offers

                        for Gwziter in range(0,2000):

                            # print(homotop_i,Fi,wbar_i,niter,wL_i,Rziter,Gwziter)
                            Gtilde = np.trapz(np.outer(Psis,np.ones(wpts)) * Gwz0.T, zgrid, axis=0)
                            # Compute theta by taking derivatives of Gtilde
                            thw[1:wpts-1] = (Gtilde[2:wpts] - Gtilde[0:wpts-2])/(wgrid[2:wpts] - wgrid[0:wpts-2])
                            thw[0] = (Gtilde[1] - Gtilde[0])/(wgrid[1] - wgrid[0])
                            thw[wpts-1] = (Gtilde[wpts-1] - Gtilde[wpts-2])/(wgrid[wpts-1] - wgrid[wpts-2])
                            [resid,Gwz_1,C,D] = Gwzdef(Gwz0.flatten('F')[mask],R0,F0,mask,r0z,r1z)
                            Gwz_1 = Gwz_1.flatten('F')
                            Gwz_1[mask == 0] = 0
                            Gwz_1 = Gwz_1.reshape((wpts,zpts), order='F')
                            # Impose adding up to Gwz_1
                            Gwz_1 = Gwz_1/np.outer(np.ones(wpts),np.amax(Gwz_1,axis=0))
                            norm_resid = np.max(abs(Gwz_1 - Gwz0))
                            abs_resid = np.max(abs(resid))
                            Gwz0 = update_Gwz*Gwz_1 + (1-update_Gwz)*Gwz0
                            abs_tol = np.max([RzTol,(minRziter - Rziter - 1)/minRziter*1e-3 + (Rziter+1)/minRziter*RzTol])
                            norm_tol = abs_tol
                            if abs_resid < abs_tol or norm_resid < norm_tol:
                                break
                        if Gwziter >= 1999:
                            print("No Gwz converge",homotop_i, Fi, wbar_i,niter,wL_i,Rziter,Gwziter)
                        [resid,Gwz_1,R1,VpRz] = Gwzdef(Gwz0.flatten('F')[mask],R0,F0,mask,r0z,r1z)
                        Gwz_1 = Gwz_1/np.outer(np.ones(wpts),np.amax(Gwz_1,axis=0))
                        Rdist[Rziter,0] = np.max((R0-R1)**2/R0)
                        Rdist[Rziter,1] = np.argmax((R0-R1)**2/R0)+1

                        if Rdist[Rziter,0] < RzTol:
                            RzConverged = 1
                            break

                        if Rziter > minRziter:
                            if np.mean(Rdist[Rziter-100:Rziter-50,0]) <= (np.mean(Rdist[Rziter-50:Rziter,0]) + 5e-4):
                                RzConverged = 0
                                break

                        R0 = (1-update_R)*R0 + update_R*R1
                        mask = np.outer(np.ones(wpts), R0) <= np.outer(wgrid, np.ones(zpts))
                        mask = mask.flatten('F')

                        # Done with Rz iteration

                    if abs(wL1-wL) < wLtol:
                        # If done more than one F iteration
                        wstep_old = np.copy(wstep)
                        if nu0-nu1 > 1e-2 or gamma0-gamma1 > 1e-2:
                            # Will have heterogeneous R
                            wL1 = lowestw(R1)
                        else:
                            wL1 = np.min(R1)

                        wL = wL1*update_wL + wL*(1-update_wL)
                        wgrid = np.linspace(0,1,wpts)**wpow * (wbar-wL) + wL
                        wstep = np.zeros(len(wgrid))
                        wmid = 0.5*(wgrid[0:-1] + wgrid[1:])
                        wstep[1:-1] = wmid[1:] - wmid[0:-1]
                        wstep[0] = wmid[0]-wgrid[0]
                        wstep[-1] = wgrid[-1]-wmid[-1]

                        # Rescale the density
                        F0 = F0*wstep_old/wstep
                        F0 = F0/F0[wpts-1]

                        if RzConverged == 1:
                            break
                        else:
                            print("Rz did not converge, though wL did")

                # Done looking for the lower bound w_L

                Gtilde = np.trapz(np.outer(Psis,np.ones(wpts)) * Gwz_1.T, zgrid, axis=0)
                Gtilde = Gtilde/Gtilde[wpts-1]

                # Recover steady state n(z) and integrate to n
                for zi in range(0,zpts):
                    indic = wgrid <= R1[zi]
                    Rzi_disc = np.max([sum(indic),1])
                    Rzi = np.min([wgrid[wpts-1], np.max([wgrid[0],R1[zi]])])

                    FR = np.interp(Rzi,wgrid,F0,left=0,right=0)
                    GR = np.interp(Rzi,wgrid,Gtilde,left=0,right=0)

                    nz1[zi] = (gamma0*(1-FR)+(1-gamma0)*r0z[zi]*(1-GR) ) \
						/(delta+gamma0*(1-FR) + (1-gamma0)*r0z[zi]*(1-GR))

                n1 = np.inner(nz1*Omegaz, zstep)
                ndist = abs(n1-n)/n

                if ndist < ntol:
                    break
                else:
                    n = (1.-update_n)*n + update_n*n1
                    nz = (1.-update_n)*nz + update_n*nz1
                    Psis = nz * Omegaz / np.inner(zstep, nz * Omegaz)
                    Psis = Psis / np.sum(Psis * zstep)  # renorm to integrate to 1
                    means = np.inner(zgrid, (Psis * zstep))

                for zi in range(0,zpts):
                    r1z[zi] = rhoz1(zgrid[zi],nz)
                    r0z[zi] = rhoz0(zgrid[zi],nz)

            # Done solving for n

            #solve for l(w,z)
            zgrid_wz = np.zeros((wpts,zpts))
            hdir_wz = np.zeros((wpts,zpts))
            href_wz = np.zeros((wpts, zpts))
            r1z_wz  = np.zeros((wpts, zpts))
            zused   = np.ones(wpts,dtype=np.int)*zpts
            # Construct L(0), L(wpts) and hence profit

            # GR = 0  # everyone will take a dominating offer at the lowest wage
            # zused[0] =0
            # for zi in range(0,zpts):
            #     if( R1[zi]<=wL ):
            #         lwz[0,zi] = (Omegaz[zi]/M*gamma0*(1-nz[zi]))/(delta+gamma1 + (1-gamma1)* r1z[zi]*(1-GR))
            #         zused[0]+=1
            #         hdir_wz[0,zi] = Omegaz/M* gamma0*(1-nz[zi])
            #         href_wz[0,zi] = gamma1*Psis[zi]*(1-nz)*nu0
            #         r1z_wz[0,zi]  = r1z[zi]
            #     if( zi<zpts-1):
            #         if( R1[zi]<=wL and R1[zi+1]>wL ):
            #             wt = (np_interp_mono(wL,R1,zgrid) - zgrid[zi] )/(zgrid[zi+1]-zgrid[zi])
            #             lwz[0,zi+1] = lwz[0,zi]*wt
            #             zused[0] += 1
            #             hdir_wz[0, zi+1] = wt * Omegaz/M* gamma0 * (1 - nz[zi])
            #             href_wz[0, zi+1] = wt * gamma1 * Psis[zi] * (1 - nz) * nu0
            #             r1z_wz[0, zi+1] = r1z[zi]
            #     if( zi>0 ):
            #         if( R1[zi]<=wL and R1[zi-1]>wL ):
            #             wt = (zgrid[zi]-np_interp_mono(wL,R1,zgrid))/(zgrid[zi]-zgrid[zi-1])
            #             lwz[0,zi-1] = lwz[0,zi]*wt
            #             zused[0] += 1
            #             hdir_wz[0, zi-1] = wt * Omegaz/M* gamma0 * (1 - nz[zi])
            #             href_wz[0, zi-1] = wt * gamma1 * Psis[zi] * (1 - nz) * nu0
            #             r1z_wz[0, zi-1] = r1z[zi]
            #
            # Lw[0] = np.trapz(lwz[0,:],zgrid)

            if nu0 < 1e-6 and nu1 < 1e-6:
                # this is recovering the BM distribution when everyone has the
                # same reservation value and hence our assumption that lwz(1,:) is
                # nonzero only at z=1 is not held
                lwLz = lwz[0,:].copy()
                lwLz[1:] = lwLz[0]/Omegaz[0]*Omegaz[1:]
                Lw[0] = np.trapz(lwLz,zgrid)

            pi0 = Lw[0]*(p-wL)

            lbarw_resid = lambda lbarwH: lbarwH*delta - Omegaz/M*((1-nz)*gamma0+ nz*gamma1) - gamma1*Psis*(nu1*nz+(1-nz)*nu0)*np.trapz(lbarwH,zgrid, axis=0)

            lbarw, infodict, flag, mesg = fsolve(lbarw_resid, Omegaz/M, full_output=True)
            hdir_wz[wpts-1, :] = Omegaz/M*((1-nz)*gamma0+ nz*gamma1)
            href_wz[wpts-1,:] = gamma1*Psis*(nu1*nz+(1-nz)*nu0)
            r1z_wz[wpts-1,:] = 0.

            Lw[wpts-1] = np.trapz(lbarw, zgrid)

            pibarw = Lw[wpts-1]*(p-wgrid[wpts-1])
            if flag != 1:
                print("Labor force, lbarw, not solved")

            # Wage distribution for everyone
            lwz[wpts-1,:] = lbarw.copy()
            indicz = np.zeros((wpts,zpts),dtype = np.int)
            for wi in range(0,wpts):
                indicz[wi] = R1 <= wgrid[wi]

            for wi in range(0,wpts-1):
                beta_z = delta + gamma1*(1-F0[wi]) + (1-gamma1)*r1z*(1-Gtilde[wi])
                hdir_z = Omegaz/M * ((1-nz)*gamma0*indicz[wi] + nz*gamma1*Gwz_1[wi])
                href_z = gamma1*Psis*((1-nz)*nu0*indicz[wi] + nz*nu1*Gwz_1[wi])
                r1z_z  = r1z
                # solve the edge, where R1(z)<wgrid(wi)
                zRm1 = sum(indicz[wi])
                zRm1 = int(max([zRm1,1]))
                if zRm1 <= (zpts-1):
                    beta_z[R1>wgrid[wi]] = 0.
                    hdir_z[R1>wgrid[wi]] = 0.
                    href_z[R1>wgrid[wi]] = 0.
                    r1z_z[R1>wgrid[wi]] = 0.
                    Omegaz_endo = Omegaz.copy()
                    zg_endo = zgrid.copy()
                    zg_endo[R1>wgrid[wi]] = 0
                    Omegaz_endo[R1>wgrid[wi]] = 0.
                    zindif = -1
                    for zR1 in range(0,zpts):
                    # by linear interpolation, find points beyond the threshold
                        if R1[zR1] > wgrid[wi]:
                            if zR1>0:
                                if R1[zR1-1] <= wgrid[wi]:
                                    zindif = (wgrid[wi] - R1[zR1-1])*(zgrid[zR1] - zgrid[zR1-1])/(R1[zR1] - R1[zR1-1]) + zgrid[zR1-1]
                                    zindif = max([z0,zindif])
                            if zR1 < zpts-1:
                                if R1[zR1+1] <= wgrid[wi]:
                                    zindif = (wgrid[wi] - R1[zR1+1])*(zgrid[zR1] - zgrid[zR1+1])/(R1[zR1] - R1[zR1+1]) + zgrid[zR1+1]
                                    zindif = max([z0,zindif])
                            if zindif>-1:
                                zg_endo[zR1] = zindif
                                beta_zfun = lambda zR: np.interp(zR,zgrid,beta_z)
                                hdir_zfun = lambda zR: np.interp(zR,zgrid, Omegaz/M*((1-nz)*gamma0 + nz*gamma1*Gwz_1[wi]))
                                href_zfun = lambda zR: np.interp(zR,zgrid, gamma1*Psis*((1-nz)*nu0 + nu1*nz*Gwz_1[wi]))
                                beta_z[zR1] = beta_zfun(zindif)
                                hdir_z[zR1] = hdir_zfun(zindif)
                                href_z[zR1] = href_zfun(zindif)
                                Omegaz_endo[zR1] = np.interp(zindif, zgrid, Omegaz/M)
                                r1z_z[zR1] = np.interp(zindif, zgrid, r1z)
                                zindif = -1
                    #end interpolation for edge case
                    nonzero_lwz = zg_endo>0
                    zg_endo   = np.trim_zeros( zg_endo )
                    beta_z = np.trim_zeros( beta_z )
                    hdir_z = np.trim_zeros( hdir_z )
                    href_z = np.trim_zeros( href_z )
                    Omegaz_endo = np.trim_zeros(Omegaz_endo)
                    r1z_z = np.trim_zeros(r1z_z)
                    lwz_resid = lambda lwzH: lwzH*beta_z - hdir_z - np.trapz(lwzH,zg_endo, axis=0)*href_z

                    lbarwTemp, infodict, flag, mesg = fsolve(lwz_resid, Omegaz_endo/M, full_output=True)
                    lwz[wi,nonzero_lwz] = lbarwTemp.copy()
                    Lw[wi] = np.trapz(lwz[wi,nonzero_lwz])
                    zused[wi] = len(beta_z)

                else:
                    lwz_resid = lambda lwzH: lwzH*beta_z - hdir_z - np.trapz(lwzH*zgrid, axis=0)*href_z
                    lwzTemp, infodict, flag, mesg = fsolve(lwz_resid, Omegaz, full_output=True)
                    lwz[wi,:] = lwzTemp.copy()
                    Lw[wi] = np.trapz(lwz[wi,:],zgrid)
                    nonzero_lwz =  indicz[wi]
                    zg_endo = zgrid.copy()
                    zused[wi] = zpts

                diryield_wz[wi,nonzero_lwz] = hdir_z.copy()
                refyield_wz[wi,nonzero_lwz] = href_z*np.trapz(lwz[wi,nonzero_lwz],zg_endo, axis=0)

                if flag != 1:
                    print("Did not solve labor force correctly")
                zgrid_wz[wi,0:zused[wi]] = zg_endo
                hdir_wz[wi,0:zused[wi]] = hdir_z
                href_wz[wi,0:zused[wi]] = href_z
                r1z_wz[wi,0:zused[wi]] = r1z_z
            #end wi loop


            piz_0 = Lw*(p-wgrid)
            # impose monotonicity on Lw for interior points only
            for wi in range(2,wpts-1):
                if Lw[wi] < Lw[wi-1]:
                    nonmonolwz = nonmonolwz+1
                    Lw[wi] = np.max([Lw[wi] , Lw[wi-1]])

            # match wbar to minimum profit in interior points of wage distribution
            Epiz_0 = np.trapz(piz_0[1:wpts-1], wgrid[1:wpts-1])/(wgrid[wpts-2]-wgrid[1])
            minpiz_0 = np.min(piz_0[1:wpts-1])
            minpiz_0 = (1-update_wbar)*minpiz_0 + update_wbar*pi0  #mix in the profit from the very very lowest firm.
            refzi = np.argmin(piz_0[1:wpts-1])
            pibarw = Lw[wpts-1]*(p - wgrid[wpts-1])
            wbar_new = -(minpiz_0/Lw[wpts-1] - p)
            if wbar_new > p:
                wbar_new = p
                err_wbarnew = err_wbarnew +1
            # now that we have the ends of this distribution, return and reset the domain
            wbar_old = wbar
            wbar = update_wbar*wbar_new + (1 - update_wbar)*wbar
            wstep_old = wstep.copy()

            wgrid	= np.linspace(0,1,wpts)**wpow*(wbar-wL)+ wL
            wstep	= np.zeros(len(wgrid))
            wmid	= 0.5*(wgrid[:-1]+wgrid[1:])
            wstep[1:-1] = wmid[1:]-wmid[:-1]
            wstep[0] = wmid[0] - wgrid[0]
            wstep[-1] = wgrid[-1] - wmid[-1]

            F0 = F0*wstep_old/wstep
            F0 = F0/F0[wpts-1]

            if abs(wbar_old - wbar_new) < 5e-5 and abs(minpiz_0 -pibarw) <5e-5:
                break
            #end wbar_i

        # Now solve for F1 over all wi
        F1 = np.zeros(wpts)
        Lw_implied = Lw.copy()

        #use hdir_wz, href_wz, zg_endo, r1z_endo from above
        h_wz = np.zeros((wpts,zpts))
        for wi in range(1,wpts-1):
            zR1 = zused[wi]
            h_wz[wi,0:zR1] = hdir_wz[wi,0:zR1] + Lw[wi]*href_wz[wi,0:zR1]
                                                       
        eqPiobj = lambda F1in: sum(np.square(eqPi(F1in, minpiz_0, h_wz,r1z_wz,zgrid_wz,zused,Gtilde)))
        eqPiobj_J = lambda F1in: 2*eqPi(F1in, minpiz_0, h_wz,r1z_wz,zgrid_wz,zused,Gtilde) * eqPi_jac(F1in,minpiz_0, h_wz,r1z_wz,zgrid_wz,zused,Gtilde)

        bnds = np.zeros((wpts,2))
        bnds[:,1] = 1.
        bnds[wpts-1,0] = 1.
        cons = ({'type': 'ineq', 'fun': lambda x: x[1:] - x[:-1] })

        res_j = minimize(eqPiobj, F0, jac= eqPiobj_J,method='SLSQP', bounds=bnds, constraints = cons)

        # for wi in range(wpts-1,1,-1):
        #     hdir_wz = Omegaz/M*((1-nz)*gamma0*indicz[wi] + nz*gamma1*Gwz_1[wi])
        #     href_wz = gamma1*Psis*((1-nz)*nu0*indicz[wi] + nu1*nz*Gwz_1[wi])
        #     zRm1 = sum(indicz[wi,:])
        #     if zRm1 <= zpts-1 and zRm1 > 1:
        #         zR1 = zRm1 + 1
        #         # By linear interpolation
        #         zindif = (wgrid[wi] - R1[zRm1])*(zgrid[zR1]-zgrid[zRm1])/(R1[zR1]-R1[zRm1])+zgrid[zRm1]
        #         zg_endo = np.append(zgrid[0:zRm1], zindif)
        #         hdir_wzfun = lambda zR: np.interp(zR,zgrid,Omegaz/M*((1-nz)*gamma0 + nz*gamma1*Gwz_1[wi]))
        #         href_wzfun = lambda zR: np.interp(zR,np.arange(1,31), gamma1*Psis*((1-nz)*nu0 + nu1*nz*Gwz_1[wi]))
        #         hdir_wz[zR1] = hdir_wzfun(zindif)
        #         hdir_wz = hdir_wz[0:zR1].copy()
        #         href_wz[zR1] = href_wzfun(zindif)
        #         href_wz = href_wz[0:zR1].copy()
        #         r1z_endo = np.append(r1z[0:zRm1], np.interp(zindif,zgrid,r1z))
        #     else:
        #         zR1 = zpts
        #         zg_endo = zgrid.copy()
        #         r1z_endo = r1z.copy()
        #
        #     hwz0[wi,0:zR1] = hdir_wz + Lw[wi]*href_wz
        #
        #     F1_resid = lambda logF1w: minpiz_0/(p - wgrid[wi]) \
        #         - np.trapz(hwz0[wi,0:zR1]/(delta + gamma1*(1-np.exp(logF1w)) + (1-gamma1)*r1z_endo*(1-Gtilde[wi])), zg_endo, axis=0)
        #
        #     F0temp = np.log(F0[wi])
        #     if F0[wi] < 1e-3:
        #         F0temp = np.log(1e-3)
        #     #check corner solutions (i.e. 0)
        #     if F1_resid(0.)>0. :
        #         F1[wi] = 0.
        #     else:
        #         F1Temp, infodict, flag, mesg = fsolve(F1_resid,F0temp , full_output=True)
        #         F1[wi] = F1Temp.copy()
        #         F1[wi] = np.exp(F1[wi])
        #
        #     # resid output from fsolve?
        #     # F1w_normresid[wi] = np.sum(abs(F1w_resid))
        #     Lw_implied[wi] = np.trapz(hwz0[wi,0:zR1]/(delta + gamma1*(1-F1[wi]) + (1-gamma1)*r1z_endo*(1-Gtilde[wi])), zg_endo, axis=0)
        #     piz_implied = Lw_implied*(p-wgrid)
        #     if flag != 1:
        #         if F1[wi] < 1e-4 or abs(F1_resid(-10)) <= abs(F1_resid(-9)):
        #             F1[wi] = 0
        #             min_wi_Fpos[wi] = 1

        difF = (F1-F0)

        if np.max(abs(difF)) < 1e-4:
            break

        F1 = F1/F1[-1]
        F0 = update_F*F1 + (1-update_F)*F0

        # Impose monotonicity
        for wi in range(1,wpts):
            if F0[wi] < F0[wi-1]:
                nonmonocount = nonmonocount+1
                nonmonoval = F0[wi] - F0[wi-1] + nonmonoval
                F0[wi] = F0[wi-1]

        F0 = F0/F0[wpts-1]
        F0[0] = np.min([1e-4, np.min(F0)/10])
        if nonmonocount > 0:
            print("Nonmonotone F a total of %2.0d" % nonmonoval)

        if err_wbarnew > 0:
            print("wbar_new > 0 %2.0d" % err_wbarnew)

        if nonmonolwz > 0:
            print("Nonmonotone lwz a total of %2.0d" % nonmonolwz)

        delta = (gamma0 + np.trapz(Omegaz*r0z,zgrid,axis=0)) * 0.06/(1-0.06)

    totemp = np.trapz(np.trapz(lwz,wgrid,axis=0),zgrid, axis=0)

    # Vacancies filled by referral
    refyield = np.trapz(np.trapz(refyield_wz*lwz,wgrid, axis=0),zgrid,axis=0)
    # Vacancies fill by direct contact
    diryield = np.trapz(np.trapz(diryield_wz*lwz,wgrid, axis=0),zgrid,axis=0)
    # Unempoyed worker's finding rate
    UEfrt = np.trapz(Omegaz*(1-nz)*(gamma0 + (1-gamma0)*r0z),zgrid,axis=0)/np.trapz(Omegaz*(1-nz),zgrid,axis=0)
    # Employed worker's finding rate
    EEfrt = np.trapz(Omegaz*nz*(gamma1 + (1-gamma1)*r1z),zgrid,axis=0)/np.trapz(Omegaz*nz,zgrid,axis=0)
    # Employed worker's endogenous separation rate. fix this
    EEmrt = (1-gamma1)*np.trapz( Omegaz*nz*r1z*np.trapz(lwz*np.outer(1-Gtilde,np.ones(zpts)),wgrid,axis=0), zgrid, axis=0) + \
            gamma1*np.trapz(Omegaz*nz*np.trapz(lwz*np.outer(1-F1,np.ones(zpts)),wgrid, axis=0),zgrid,axis=0)/n1/np.trapz(np.trapz(lwz,wgrid, axis=0),zgrid, axis=0)

    print("nu0= %f" % nu0)
    print("Filled by referral: %f" % float(refyield/(diryield + refyield)))
    print("UE finding rate: %f" % UEfrt)
    print("EE hazard rate: %f" % EEmrt)
    print("F(wi)=0 at wi=", np.where(min_wi_Fpos==1))

    refRates = np.append(refRates, refyield/(diryield + refyield))
    UErates = np.append(UErates, UEfrt)
    EErates = np.append(EErates, EEmrt)

    if refyield/(diryield + refyield) > 0.359 - 1e-3:
        break

#%%

# Implied offer distribution and measurs of wage growth
FG = np.zeros(wpts)
FGz = np.zeros((wpts,zpts))

for wi in range(1, wpts):
    FG[wi] = np.trapz(Omegaz*((F0[wi]*gamma0 + (1-gamma0)*r0z*Gtilde[wi])*(1-nz) + (gamma1*F0[wi] + (1-gamma1)*r1z*Gtilde[wi])*nz), zgrid)
    FGz[wi,:] = (F0[wi]*gamma0 + (1-gamma0)*r0z*Gtilde[wi])*(1-nz) + (gamma1*F0[wi] + (1-gamma1)*r1z*Gtilde[wi])*nz
FG = FG/FG[wpts-1]
FGz = FGz/np.outer(np.ones(wpts),FGz[wpts-1,:])

# Type distribution by firm type
z_w = np.zeros(wpts)

for wi in range(wpts):
    z_w[wi] = np.trapz(zgrid*lwz[wi,:], zgrid)/np.trapz(lwz[wi],zgrid)

# Average wage by z
w_z = np.zeros(zpts)
for zi in range(zpts):
    w_z[zi] = np.trapz(lwz[:,zi]*wgrid, wgrid)/np.trapz(lwz[:,zi],wgrid)

# Average initial wage by z
w1_z = np.zeros(zpts)
f0 = np.zeros(wpts)
gtil = np.zeros(wpts)
wg_m = np.concatenate(([wgrid[0]], 0.5*(wgrid[1:] + wgrid[0:-1]), [wgrid[-1]]))
wi = 0
f0[wi] = (F0[wi+1] - F0[wi])/(wgrid[wi+1]-wgrid[wi])
gtil[wi] = (Gtilde[wi+1] - Gtilde[wi])/(wgrid[wi+1]-wgrid[wi])
for wi in range(1,wpts-1):
    f0[wi] = (F0[wi+1]-F0[wi-1])/(wgrid[wi+1]-wgrid[wi-1])
    gtil[wi] = (Gtilde[wi+1] - Gtilde[wi-1])/(wgrid[wi+1]-wgrid[wi-1])
wi = wpts-1
f0[wi] = (F0[wpts-1]-F0[wpts-2])/(wgrid[wpts-1]-wgrid[wpts-2])
gtil[wi] = (Gtilde[wpts-1] - Gtilde[wpts-2])/(wgrid[wpts-1] - wgrid[wpts-2])

Ew_F = np.trapz(wgrid*f0,wgrid)
Ew_Gtil = np.trapz(wgrid*gtil,wgrid)
for zi in range(0,zpts):
    w1_z[zi] = (gamma0*Ew_F + (1-gamma0)*r0z[zi]*Ew_Gtil)/(gamma0 + (1-gamma0)*r0z[zi])

# Average wage growth by z
wi = wpts
# Distribution such that above the current wage
Ew_F_trunc = np.ones(wpts)*wbar
Ew_Gtil_trunc = np.ones(wpts)*wbar
for wi in range(wpts-2,1,-1):
    Ew_F_trunc[wi] = np.trapz(wgrid[wi:wpts]*f0[wi:wpts], wgrid[wi:wpts])/np.trapz(f0[wi:wpts],wgrid[wi:wpts])
    Ew_Gtil_trunc[wi] = np.trapz(wgrid[wi:wpts]*f0[wi:wpts],wgrid[wi:wpts])/np.trapz(f0[wi:wpts],wgrid[wi:wpts])
Ew_F_trunc[0] = Ew_F
Ew_Gtil_trunc[0] = Ew_Gtil

Tper = 100
jobten = np.arange(0,Tper)
wt_z = np.zeros((zpts,Tper))
wt_z[:,0] = w1_z
for zi in range(zpts):
    for t in range(1,Tper):
        # Prob of wage that dominates
        FR = 1 - np.interp(wt_z[zi,t-1],wgrid,F0)
        GR = 1 - np.interp(wt_z[zi,t-1],wgrid,Gtilde)
        Ew_FR = np.interp(wt_z[zi,t-1],wgrid,Ew_F_trunc)
        Ew_GtilR = np.interp(wt_z[zi,t-1],wgrid,Ew_Gtil_trunc)
        wt_z[zi,t] = gamma1*FR*Ew_FR + (1-gamma1)*r1z[zi]*GR*Ew_GtilR + (1-gamma1*FR-(1-gamma1)*r1z[zi]*GR)*wt_z[zi,t-1]

# For each w, compute half-life to wbar by z for estimate of exponential decay
halflife_wz = np.zeros((wpts-1,zpts))

for wi in range(wpts-1):
    halfwbar = 0.5*(wgrid[wi] +wbar)
    for zi in range(zpts):
        if R1[zi] <= wgrid[wi]:
            # Probability of a wage that dominates
            FR = 1 - np.interp(wgrid[wi],wgrid,F0)
            GR = 1 - np.interp(wgrid[wi],wgrid,Gtilde)
            Ew_FR = np.interp(wgrid[wi],wgrid,Ew_F_trunc)
            Ew_GtilR = np.interp(wgrid[wi],wgrid,Ew_Gtil_trunc)
            Ewtp1 = gamma1*FR*Ew_FR + (1-gamma1)*r1z[zi]*GR*Ew_GtilR + (1 - gamma1*FR - (1-gamma1)*r1z[zi]*GR)*wgrid[wi]
            convergert = -np.log((wbar-Ewtp1)/(wbar- wgrid[wi]))
            halflife_wz[wi,zi] = 1/convergert*np.log(2)

#%% Compute paths for network and direct search
GRz = np.zeros(zpts)
FRz = np.zeros(zpts)
for zi in range(zpts):
    if R1[zi] <= np.min(wgrid):
        pchip_func1 = pchip(wgrid,Gtilde)
        GRz[zi] = pchip_func1(R1[zi])
        pchip_func2 = pchip(wgrid,F0)
        FRz[zi] = pchip_func2(R1[zi])
    else:
        GRz[zi] = 0
        FRz[zi] = 0

Distz_net = (Omegaz*(1-nz)*r0z*(1-GRz))/np.trapz(Omegaz*(1-nz)*r0z*(1-GRz), zgrid)
Ez_net = np.trapz(zgrid*Omegaz*(1-nz)*r0z*(1-GRz), zgrid)/np.trapz(Omegaz*(1-nz)*r0z*(1-GRz), zgrid)
Distz_dir = (Omegaz*(1-nz)*(1-FRz))/np.trapz(Omegaz*(1-nz)*(1-FRz), zgrid)
Ez_dir = np.trapz(zgrid*Omegaz*(1-nz)*(1-FRz),zgrid)/np.trapz(Omegaz*(1-nz)*(1-FRz),zgrid)
Distz_netdir = Omegaz*(1-nz)*(gamma0*(1-FRz) + (1-gamma0)*r0z*(1-GRz))/np.trapz(Omegaz*(1-nz)*(gamma0*(1-FRz) + (1-gamma0)*r0z*(1-GRz)),zgrid)
Ez_netdir = np.trapz(zgrid*Omegaz*(1-nz)*(gamma0*(1-FRz) + (1-gamma0)*r0z*(1-GRz)), zgrid)/np.trapz(Omegaz*(1-nz)*(gamma0*(1-FRz) + (1-gamma0)*r0z*(1-GRz)), zgrid)
Omegaz_cdf = cumtrapz(Omegaz,zgrid,initial=0)
pctile_Ez_net = np.interp(Ez_net, zgrid, Omegaz_cdf)
pctile_Ez_dir = np.interp(Ez_dir, zgrid, Omegaz_cdf)
pctile_Ez_netdir = np.interp(Ez_netdir, zgrid, Omegaz_cdf)

# Wage distribution, marginal over z
Gw = np.zeros(wpts)
for wi in range(wpts):
    Gw[wi] = np.trapz(Omegaz*nz*Gwz0[wi,:], zgrid)
Gw = Gw/Gw[-1]

# Densities
gtilde = np.zeros(wpts)
gw = np.zeros(wpts)
f0 = np.zeros(wpts)
gwz = np.zeros((wpts,zpts))
wi = 0
f0[wi] = (F0[wi+1]-F0[wi])/(wgrid[wi+1]-wgrid[wi])
gtilde[wi] = (Gtilde[wi+1]-Gtilde[wi])/(wgrid[wi+1]-wgrid[wi])
gw[wi] = (Gw[wi+1]-Gw[wi])/(wgrid[wi+1]-wgrid[wi])
for zi in range(zpts):
    gwz[wi,zi] = (Gwz0[wi+1,zi]-Gwz0[wi,zi])/(wgrid[wi+1]-wgrid[wi])
for wi in range(1,wpts-1):
    f0[wi] = (F0[wi+1]-F0[wi-1])/(wgrid[wi+1]-wgrid[wi-1])
    gtilde[wi] = (Gtilde[wi+1]-Gtilde[wi-1])/(wgrid[wi+1]-wgrid[wi-1])
    gw[wi] = (Gw[wi+1]-Gw[wi-1])/(wgrid[wi+1]-wgrid[wi-1])
    for zi in range(zi):
        gwz[wi,zi] = (Gwz0[wi+1,zi]-Gwz0[wi-1,zi])/(wgrid[wi+1]-wgrid[wi-1])

f0[wpts-1] = (F0[wpts-1]-F0[wpts-2])/(wgrid[wpts-1]-wgrid[wpts-2])
gtilde[wpts-1] = (Gtilde[wpts-1]-Gtilde[wpts-2])/(wgrid[wpts-1]-wgrid[wpts-2])
gw[wpts-1] = (Gw[wpts-1]-Gw[wpts-2])/(wgrid[wpts-1]-wgrid[wpts-2])
for zi in range(zpts):
    gwz[wpts-1,zi] = (Gwz0[wpts-1,zi]-Gwz0[wpts-2,zi])/(wgrid[wpts-1]-wgrid[wpts-2])
    gsum = np.trapz(gwz[:,zi],wgrid)
    gwz[:,zi] = gwz[:,zi]/gsum
fsum = np.trapz(f0,wgrid)
gtsum = np.trapz(gtilde,wgrid)
gsum = np.trapz(gw,wgrid)
gw = gw/gsum
f0 = f0/fsum
gtilde = gtilde/gtsum

# Average z by w
gw_meanz = np.zeros(wpts)
for wi in range(wpts):
    gw_meanz[wi] = np.trapz(zgrid*gwz[wi,:]/np.sum(gwz[wi,:]),zgrid)
tmp = np.trapz(gw_meanz*gw,wgrid)
gw_meanz = gw_meanz/tmp*meanz

# Lorenz curves:
# Compute Lorenz curve in each scenario
LorenzW = np.zeros(wpts)
SiW = np.zeros(wpts)
for wi in range(wpts):
    for wj in range(wi):
        SiW[wi] = f0[wj]*wgrid[wj]+SiW[wi]
for wi in range(wpts):
    LorenzW[wi] = SiW[wi]/SiW[wpts-1]

pctile_Ew_F = np.interp(Ew_F,wgrid,Gw)
pctile_Ew_Gtil = np.interp(Ew_Gtil,wgrid,Gw)

#%% HALF LIFES

# Network initial wage and network initial z
Ez_netLi = np.max(np.nonzero(zgrid < Ez_net))
Ez_netLwt= (zgrid[Ez_netLi+1] - Ez_net)/(zgrid[Ez_netLi+1]-zgrid[Ez_netLi])

h1_netw_netz = np.interp(Ew_Gtil,wgrid[:wpts-1],halflife_wz[:,Ez_netLi])*Ez_netLwt + \
                np.interp(Ew_Gtil,wgrid[:wpts-1],halflife_wz[:,Ez_netLi+1])*(1-Ez_netLwt)

convergert_netw_netz = (1/h1_netw_netz)/np.log(2)

# Network initial wage and average initial z
Ez_netdirLi = np.max(np.nonzero(zgrid < Ez_netdir))
Ez_netdirLwt = (zgrid[Ez_netdirLi +1] - Ez_netdir)/(zgrid[Ez_netdirLi+1]-zgrid[Ez_netdirLi])

hl_netw_netdirz = np.interp(Ew_Gtil, wgrid[:wpts-1],halflife_wz[:,Ez_netdirLi])*Ez_netdirLwt + \
                np.interp(Ew_Gtil,wgrid[:wpts-1],halflife_wz[:,Ez_netdirLi+1])*(1-Ez_netdirLwt)

convergert_netw_netdirz = (1/hl_netw_netdirz)/np.log(2)

# Direct initial wage and direct initial z
Ez_dirLi = np.max(np.nonzero(zgrid < Ez_dir))
Ez_dirLwt = (zgrid[Ez_dirLi +1] - Ez_dir)/(zgrid[Ez_dirLi+1]-zgrid[Ez_dirLi])

hl_dirw_dirz = np.interp(Ew_F, wgrid[:wpts-1],halflife_wz[:,Ez_dirLi])*Ez_dirLwt + \
                np.interp(Ew_F,wgrid[:wpts-1],halflife_wz[:,Ez_dirLi+1])*(1-Ez_dirLwt)

convergert_dirw_dirz = (1/hl_dirw_dirz)/np.log(2)

# Direction initial wage and average initial z
hl_dirw_netdirz = np.interp(Ew_F, wgrid[:wpts-1],halflife_wz[:,Ez_netdirLi])*Ez_netdirLwt + \
                np.interp(Ew_F,wgrid[:wpts-1],halflife_wz[:,Ez_netdirLi+1])*(1-Ez_netdirLwt)
convergert_dirw_netdirz = (1/hl_dirw_netdirz)/np.log(2)

# Finding rate for network finder,direct finder and average finder
# here compute the average unemployment duration
Eudur_net = np.trapz(Distz_net/((gamma0*(1-FRz) + (1-gamma0)*r0z*(1-GRz))),zgrid)
Eudur_dir = np.trapz(Distz_dir/((gamma0*(1-FRz) + (1-gamma0)*r0z*(1-GRz))),zgrid)
Eudur_netdir = np.trapz(Distz_netdir/((gamma0*(1-FRz) + (1-gamma0)*r0z*(1-GRz))),zgrid)

Eudur_net_ave = Eudur_net/Eudur_netdir
Eudur_dir_ave = Eudur_dir/Eudur_netdir

Gwz_dist = np.zeros((wpts,zpts))
for zi in range(zpts):
    Gwz_dist[0,zi] = (Gwz0[1,zi] - Gwz0[1,zi])/(wgrid[1]-wgrid[0])
    for wi in range(1,wpts-1):
        Gwz_dist[wi,zi] = (Gwz0[wi+1,zi] - Gwz0[wi-1,zi])/(wgrid[1]-wgrid[0])
    Gwz_dist[wpts-1,zi] = (Gwz0[wpts-1,zi] - Gwz0[wpts-2,zi])/(wgrid[wpts-1]-wgrid[wpts-2])
    Gwz_int = np.trapz(Gwz_dist[:,zi],wgrid)
    Gwz_dist[:,zi] = Gwz_dist[:,zi]/Gwz_int

# SS wage diff
lwz_netz = Ez_netLwt*(np.trapz(wgrid*Gwz_dist[:,Ez_netLi], wgrid)/np.trapz(Gwz_dist[:,Ez_netLi],wgrid)) + \
            (1-Ez_netLwt)*(np.trapz(wgrid*Gwz_dist[:,Ez_netLi+1],wgrid)/np.trapz(Gwz_dist[:,Ez_netLi+1],wgrid))
pctile_lwz_netz = np.interp(lwz_netz,wgrid,Gw)
lwz_dirz = Ez_dirLwt*np.trapz(wgrid*Gwz_dist[:,Ez_dirLi],wgrid)/np.trapz(Gwz_dist[:,Ez_dirLi],wgrid) + \
            (1-Ez_dirLwt)*np.trapz(wgrid*Gwz_dist[:,Ez_dirLi+1],wgrid)/np.trapz(Gwz_dist[:,Ez_dirLi+1],wgrid)
pctile_lwz_dirz = np.interp(lwz_dirz,wgrid,Gw)

#%% Average duration of match whether through network or directed search

# First average over z for each wage level, then integerate over wage levels
Emdur_w_net = np.zeros(wpts)
Emdur_w_dir = np.zeros(wpts)
for wi in range(wpts-1):
    Emdur_w_net[wi] = np.trapz(Distz_net/(gamma1*(1-F0[wi]) + (1-gamma1)*r1z*(1-Gtilde[wi])),zgrid)
    Emdur_w_dir[wi] = np.trapz(Distz_dir/(gamma1*(1-F0[wi]) + (1-gamma1)*r1z*(1-Gtilde[wi])),zgrid)
gtilde_1wptsM1 = gtilde[:wpts-1]/np.trapz(gtilde[:wpts-1],wgrid[:wpts-1])
f0_1wptsM1 = f0[:wpts-1]/np.trapz(f0[:wpts-1],wgrid[:wpts-1])
Emdur_net = np.trapz(gtilde_1wptsM1*Emdur_w_net[:wpts-1],wgrid[:wpts-1])/12
Emdur_dir = np.trapz(f0_1wptsM1*Emdur_w_dir[:wpts-1],wgrid[:wpts-1])/12

# Expected duration conditional on wage
Emdur_netVdir_condw = np.trapz(gtilde_1wptsM1[:wpts-1]*(Emdur_w_net[:wpts-1]/Emdur_w_dir[:wpts-1]),wgrid[:wpts-1])

#%% Probability of network search find

Pr_netfnd = np.zeros(wpts-1)
for wi in range(wpts-1):
    Pr_netfnd[wi] = np.trapz((1-gamma1)*r1z*(1-Gtilde[wi])/((1-gamma1)*r1z*(1-Gtilde[wi]) + gamma1*(1-F0[wi]))*Omegaz,zgrid)

#%% Find BM analogs
gamma0BM = np.trapz(Omegaz*(gamma0 + (1-gamma0)*r0z),zgrid)
gamma1BM = np.trapz(Omegaz*(gamma1 + (1-gamma1)*r1z),zgrid)
k0BM = gamma0BM/delta
k1BM = gamma1BM/delta
RBM = ((1+k1BM)**2*b+ (k0BM-k1BM)*k1BM*p)/( (1+k1BM)**2 + (k0BM-k1BM)*k1BM)
wLBM = RBM
wbarBM = p-(p-RBM)/(1+k1BM)**2
wgridBM = np.linspace(0,1,wpts)**wpow*(wbarBM-wLBM)+ wLBM
FBM = np.zeros(wpts)
# Guess F0: solve w/o any referrals
for wi in range(1,wpts):
    FBM[wi] = (delta + gamma1BM)/gamma1BM*(1-((p-wgridBM[wi])/(p-wLBM))**.5)
FBM[0] = np.min([1e-4, np.min(FBM)/10])
# Be sure it's a distribution
FBM = FBM/FBM[wpts-1]
# Compute earnings distribution
GwBM = FBM/(1+k1BM*(1-FBM))

# Lorenz Curve in this scenario
fBM = np.zeros(wpts)
wi = 0
fBM[wi] = (FBM[wi+1]-FBM[wi])/(wgridBM[wi+1]-wgridBM[wi])
for wi in range(1,wpts-1):
    fBM[wi] = (FBM[wi+1]-FBM[wi-1])/(wgridBM[wi+1]-wgridBM[wi-1])
fBM[wpts-1] = (FBM[wpts-1]-FBM[wpts-2])/(wgridBM[wpts-1]-wgridBM[wpts-2])
fsum = np.trapz(fBM,wgrid)
fBM = fBM/fsum

# Compute Lorenz curve in hetero search scenario
LorenzWBM = np.zeros(wpts)
SiWBM = np.zeros(wpts)
for wi in range(wpts):
    for wj in range(wi):
        SiWBM[wi] = fBM[wj]*wgridBM[wj]+SiWBM[wi]
for wi in range(wpts):
    LorenzWBM[wi] = SiWBM[wi]/SiWBM[wpts-1]

Ew_FBM = np.trapz(wgridBM*fBM,wgridBM)
wi = wpts-1
# Distribution such that above the current wage
Ew_FBM_trunc = np.ones(wpts)*wbarBM
for wi in range(wpts-2,1,-1):
    Ew_FBM_trunc[wi] = np.trapz(wgridBM[wi:wpts]*fBM[wi:wpts], wgridBM[wi:wpts])/np.trapz(fBM[wi:wpts],wgridBM[wi:wpts])
Ew_FBM_trunc[0] = Ew_FBM

# For each w, compute half-life to wbar by z, for estimate of exponential decay
halflife_BM = np.zeros(wpts-1)

for wi in range(wpts-1):
    halfwbar = 0.5*(wgridBM[wi] + wbarBM)
    if RBM <= wgridBM[wi]:
        # Prob. of a wage that dominates
        FR = 1 - np.interp(wgridBM[wi],wgridBM,FBM)
        Ew_FR = np.interp(wgridBM[wi],wgridBM,Ew_FBM_trunc)
        Ewtp1 = gamma1BM*FR*Ew_FR + (1 - gamma1BM*FR)*wgridBM[wi]
        convergert = -np.log((wbarBM-Ewtp1)/(wbarBM- wgridBM[wi]))
        halflife_BM[wi] = 1/convergert*np.log(2)

#%% Find hetero-search analogs

gamma0HS = gamma0 + (1-gamma0)*r0z
gamma1HS = gamma1 + (1-gamma1)*r1z

wbarHS = wbar
wgridHS = wgrid.copy()
FHS0 = np.zeros(wpts)
FHS1 = np.zeros(wpts)
FHS = np.zeros(wpts)
for wi in range(wpts):
    FHS0[wi] = np.trapz(Omegaz*(F0[wi]*gamma0 + (1-gamma0)*r0z*Gtilde[wi]),zgrid)
    FHS1[wi] = np.trapz(Omegaz*(gamma1*F0[wi] + (1-gamma1)*r1z*Gtilde[wi]),zgrid)
    FHS[wi] = np.trapz(Omegaz*(nz*(gamma1*F0[wi] + (1-gamma1)*r1z*Gtilde[wi]) + (1-nz)*(F0[wi]*gamma0 + (1-gamma0)*r0z*Gtilde[wi])),zgrid)

FHS0 = FHS0/FHS0[-1]
FHS1 = FHS1/FHS1[-1]
GwzHS = np.zeros((wpts,zpts))
for zi in range(zpts):
    FHS0R_func = interp1d(wgrid,FHS0,'cubic',bounds_error=False,fill_value=0)
    FHS0R = FHS0R_func(R0[zi])
    FHS1R_func = interp1d(wgrid,FHS1,'cubic',bounds_error=False,fill_value=0)
    FHS1R = FHS0R_func(R0[zi])
    GwzHS[:,zi] = (1-nz[zi])*(gamma0HS[zi]*(FHS0 - FHS0R) )/(nz[zi]*(delta+gamma1HS[zi]*(1-FHS0)))

GwzHS[0,:] = 0
FHS = FHS1.copy()

# Wage distribution, marginal over z
GwHS = np.zeros(wpts)
for wi in range(wpts):
    GwHS[wi] = np.trapz(Omegaz*nz*GwzHS[wi,:],zgrid)
GwHS = GwHS/GwHS[-1]

fHS = np.zeros(wpts)
wi = 0
fHS[wi] = (FHS[wi+1] - FHS[wi])/(wgrid[wi+1] - wgrid[wi])
for wi in range(1,wpts-1):
    fHS[wi] = (FHS[wi+1] - FHS[wi-1])/(wgrid[wi+1] - wgrid[wi-1])
fHS[wpts-1] = (FHS[wpts-1]-FHS[wpts-2])/(wgrid[wpts-1]-wgrid[wpts-2])
fsum = np.trapz(fHS,wgrid)
fHS = fHS/fsum

# Compute Ew_FHS and Ew_FHS_trunc
Ew_FHS = np.trapz(wgridHS*fHS, wgridHS)
wi=wpts
Ew_FHS_trunc = np.ones(wpts)*wbarHS
for wi in range(wpts-1,1,-1):
    Ew_FHS_trunc[wi] = np.trapz(wgridHS[wi:wpts]*fHS[wi:wpts],wgridHS[wi:wpts])/np.trapz(fHS[wi:wpts],wgridHS[wi:wpts])
Ew_FHS_trunc[0] = Ew_FHS

# Compute Lorenz curve in hetero search scenario
LorenzWHS = np.zeros(wpts)
SiWHS = np.zeros(wpts)
for wi in range(wpts):
    for wj in range(wi):
        SiWHS[wi] = fHS[wj]*wgrid[wj]+SiWHS[wi]

for wi in range(wpts):
    LorenzWHS[wi] = SiWHS[wi]/SiWHS[wpts-1]

# Compute half-life

# For each w, compute half-life to wbar by z, for estimate of exponential decay
halflife_wz_HS = np.zeros((wpts-1,zpts))

for wi in range(wpts-1):
    halfwbar = 0.5*(wgrid[wi] + wbar)
    for zi in range(zpts):
        if R0[zi] <= wgrid[wi]:
            # Prob of wage that dominates
            FR = 1 - FHS[wi]
            Ew_FR = Ew_FHS_trunc[wi]
            Ewtp1 = gamma1HS[zi]*FR*Ew_FR + (1-gamma1HS[zi]*FR)*wgrid[wi]
            convergert = -np.log((wbar-Ewtp1)/(wbar-wgrid[wi]))
            halflife_wz_HS[wi,zi] = 1/convergert*np.log(2)

#%% Compute wage offer distribution without paradox of friends
FG_noFP = np.zeros(wpts)
for wi in range(1,wpts):
    FG_noFP[wi] = np.trapz(Omegaz*((F0[wi]*gamma0 + (1-gamma0)*r0z*Gw[wi])*(1-nz)+ \
        (gamma1*F0[wi] + (1-gamma1)*r1z*Gw[wi])*nz), zgrid)
FG_noFP = FG_noFP/FG_noFP[wpts-1]

# More densities
fg = np.zeros(wpts)
gw = np.zeros(wpts)
fg_nFP = np.zeros(wpts)
wi = 0
fg[wi] = (FG[wi+1]-FG[wi])/(wgrid[wi+1]-wgrid[wi])
fg_nFP[wi] = (FG_noFP[wi+1]-FG_noFP[wi])/(wgrid[wi+1]-wgrid[wi])
gw[wi] = (Gw[wi+1]-Gw[wi])/(wgrid[wi+1]-wgrid[wi])
gtilde[wi]= (Gtilde[wi+1]-Gtilde[wi])/(wgrid[wi+1]-wgrid[wi])
for wi in range(1,wpts-1):
    fg[wi] = (FG[wi+1]-FG[wi-1])/(wgrid[wi+1]-wgrid[wi-1])
    fg_nFP[wi] = (FG_noFP[wi+1]-FG_noFP[wi-1])/(wgrid[wi+1]-wgrid[wi-1])
    gw[wi] = (Gw[wi+1]-Gw[wi-1])/(wgrid[wi+1]-wgrid[wi-1])
    gtilde[wi]= (Gtilde[wi+1]-Gtilde[wi-1])/(wgrid[wi+1]-wgrid[wi-1])

fg[wpts-1] = (FG[wpts-1]-FG[wpts-2])/(wgrid[wpts-1]-wgrid[wpts-2])
fg_nFP[wpts-1] = (FG_noFP[wpts-1]-FG_noFP[wpts-2])/(wgrid[wpts-1]-wgrid[wpts-2])
gw[wpts-1] = (Gw[wpts-1]-Gw[wpts-2])/(wgrid[wpts-1]-wgrid[wpts-2])
gtilde[wpts-1]= (Gtilde[wpts-1]-Gtilde[wpts-2])/(wgrid[wpts-1]-wgrid[wpts-2])

gw = gw/np.trapz(gw,wgrid);
fg = fg/np.trapz(fg,wgrid);
fg_nFP= fg_nFP/np.trapz(fg_nFP,wgrid)

# Expected wages
Ew_G = np.trapz(wgrid*gw,wgrid)
Ew_FG = np.trapz(wgrid*fg,wgrid)
Ew_FG_nFP = np.trapz(wgrid*fg_nFP)
Ew_F = np.trapz(wgrid*f0,wgrid)
Ew_Gtil = np.trapz(wgrid*gtilde,wgrid)

