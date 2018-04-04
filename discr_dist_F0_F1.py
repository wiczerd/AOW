# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 10:15:59 2017

@author: h1jde02
@author: wiczerd
"""
import sys
import os
import inspect

sys.path.insert(0, os.path.abspath('/home/david/workspace/AOW/'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize
from scipy.interpolate import pchip
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz,simps
import dfogn
from mpi4py import MPI

########## Solving Networks & Search


print_lev = 1
calibrate_flag = False #run calibration or not?

nu0eqnu1 = 1

update_Gwz = 0.1
update_R = 0.05
update_wbar = 0.5
update_n = 0.1
update_wL = 0.1
update_F = 0.5
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

UEtarget = 0.249 #FALLICK & FLEISCHMAN Numbers (as of 2017:10)
EEtarget = 0.021
Utarget  = 0.055
NetfndTarget = 0.23
EENetfndTarget = 0.2481977
netfrtdirfrtTarget = 0.2431196   #<-full sample, 2013-2015 #1.11392 # <- using less than median, only 2013 # Using full sample -> 0.23
b = 0.
p = 1.
alpha = 2.5
M = 1.
delta = 0.013 # UEtarget*Utarget/(1.-Utarget)

#some variables that'll be over-written
gamma0 = UEtarget
gamma1 = EEtarget
nu0 = 0.5*gamma0
nu1 = nu0


if (gamma0 - gamma1 < 1e-2 and nu0 - nu1 < 1e-2):
    maxRziter = 1
    minRziter = 1

z0 = 1
zZ = ((0.999*(1-alpha)/(alpha-1)+z0 )*z0**(-alpha))**(1/(1-alpha))

zpts = 31 #gives an even number of intervals for Simpson's rule
wpts = 51

zpow = 1.5
wpow = 0.5 #this ends up getting more scrunched at the top when I solve for things

# Set unevenly spaced grid points
zgrid = np.linspace(0,1,zpts+1)**zpow * (zZ-z0) + z0
zgrid = .5*(zgrid[1:] + zgrid[0:-1])
zgrid[0] = z0
zgrid[zpts-1] = zZ

zstep = np.linspace(0,1,zpts+1)**zpow * (zZ-z0) + z0
zstep = zstep[1:] - zstep[:-1]


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
def rhoz0(z,nz,Psis,nu0,gamma1):
    global zgrid
    pzs = Psis*nz*gamma1*nu0/zgrid
    pz = np.trapz(pzs,zgrid)
    P = 1 - np.exp(-z*pz)
    return(P)

def rhoz1(z,nz,Psis,nu1,gamma1):

    global zgrid

    pzs = Psis*nz*gamma1*nu1/zgrid
    pz = np.trapz(pzs,zgrid)
    P = 1 - np.exp(-z*pz)

    return(P)

def Gwzdef(dist,R,F,Psis,wgrid,mask,r0z,r1z):
    
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

        v = (gamma0* (F - FRz[zi]) + (1-gamma0)*r0z[zi]*(Gtilde - GtRz[zi])) \
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
        #VpGxz_fun = pchip(wgrid, (1 - Gtilde) * Vp[:, zi], extrapolate=True)
        #Vp_Rwbar_fun = pchip(wgrid, (1-F)*Vp[:,zi], extrapolate=True)

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

def lowestw(Rz,Omegaz, Psis,nz,r1z):


    #need to solve for the wL such that wL = argmax (p-w)\int_1^R^{-1}(w)L(w,z)dz =>
    # (p-wL)Omega( R^{-1}(wL))  - \int_1^R^{-1}(wL)Omega(z)dz = 0

    # sort Rz
    wgrid_R = np.sort(Rz)
    # remove duplicates so that no flat spots
    for zi in range(1,zpts-1):
        if wgrid_R[zi]-wgrid_R[zi-1] < 5e-5:
            wgrid_R[zi] = wgrid_R[zi-1] + 5e-5

    if wgrid_R[zpts-1]-wgrid_R[zpts-2] < 5e-5:
        wgrid_R[zpts-1] = wgrid_R[zpts-2] + 5e-5

    wgrid_R = np.unique(wgrid_R)
    zpts_hr = np.max(len(wgrid_R))

    # this will solve for the lowest offered wage
    ltRdist = np.zeros(zpts_hr)
    atRdist = np.zeros(zpts_hr)
    LwL = np.zeros( zpts_hr )
    if zpts_hr > 2 and np.isfinite(zpts_hr):
        for wi in range(zpts_hr):
            indic = Rz<=wgrid_R[wi]
            beta_z = delta + gamma1 + (1 - gamma1) * r1z[indic==1]
            hdir_z = Omegaz[indic==1] / M    * ((1 - nz[indic==1]) * gamma0 )
            href_z = gamma1 * Psis[indic==1] * ((1 - nz[indic==1]) * nu0 )
            zg_endo = zgrid[indic==1]
            if( sum(indic)>1):
                lwz_resid = lambda lwzH: lwzH * beta_z - hdir_z - np.trapz(lwzH * zg_endo, zg_endo, axis=0) * href_z
                lwzTemp, infodict, flag, mesg = fsolve(lwz_resid, Omegaz[indic==1]/ M, full_output=True)

                LwL[wi] =  np.trapz(lwzTemp,zg_endo)
            else :
                lwz_resid = lambda lwzH: lwzH * beta_z - hdir_z - lwzH * zg_endo * href_z
                lwzTemp, infodict, flag, mesg = fsolve(lwz_resid, Omegaz[indic==1]/ M, full_output=True)

                LwL[wi] = lwzTemp
        ppRdist = pchip(wgrid_R,LwL)
        pdRdist = ppRdist.derivative()

        resid = lambda wL: (p-wL)*pdRdist(wL) - ppRdist(wL)

        #
        # Om_adjfac = 1/np.trapz(Omegaz, zgrid)
        #
        # for zi in range(zpts_hr-1,0,-1):
        #     ltRdist[zi] = np.trapz(Omegaz[Rz <= wgrid_R[zi]], zgrid[Rz <= wgrid_R[zi]])*Om_adjfac
        #     zU = min(sum(Rz <= wgrid_R[zi]),zpts_hr-1 )
        #     atRdist[zi] = Omegaz[ zU ]
        # ltRdist[0] = 0
        # # now solve wL = argmax (p-w)*J(w) => p-J(w)/dJdw= w=> wL = -J(wL)/dJdw(wL)+p
        # # this funciton is not the same as Matlab, check
        # ppRdist = pchip(wgrid_R,ltRdist)
        # pdRdist = pchip(wgrid_R,atRdist)
        #
        # resid = lambda wL: (p-wL)*pdRdist(wL) - ppRdist(wL)


        #####################################################
        # for some reason, this is not working right at R[0]
        lo = resid(wgrid_R[1])
        hi = resid(wgrid_R[zpts_hr-1])

        if lo*hi < 0:
            wLoffered = fsolve(resid, np.mean(wgrid_R))
        elif abs(lo) < abs(hi):
            wLoffered = wgrid_R[0]
        else:
            wLoffered = wgrid_R[zpts_hr-1]

        if wLoffered > np.max(Rz):
            wLoffered = np.max(Rz)
        if wLoffered < np.min(Rz):
            wLoffered = np.min(Rz)

    else:

        wLoffered = np.min(wgrid_R)

    return(wLoffered)

def dist2dens(Fin,gridin):
    npts = Fin.size
    fout = np.zeros(npts)
    #pchip for CDF:
    pcFin = pchip(gridin,Fin)
    pcfout = pcFin.derivative()
    fout = pcfout(gridin)
    #discrete:
    #wi = 0
    #fout[wi] = (Fin[wi + 1] - Fin[wi]) / (gridin[wi + 1] - gridin[wi])
    #for wi in range(1, npts - 1):
    #    fout[wi] = (Fin[wi + 1] - Fin[wi - 1]) / (gridin[wi + 1] - gridin[wi - 1])
    #wi = wpts - 1
    #fout[wi] = (Fin[wi] - Fin[wi - 1]) / (gridin[wi] - gridin[wi - 1])
    #fsum = np.trapz(fout, gridin)
    #fout = fout / fsum
    return(fout)

def initBM(UErt, EErt):

    global delta
    global b
    global zgrid
    global zpts
    global wpts

    gamma0BM = UErt
    gamma1BM = EErt

    for gammaiter in range(0,1000):

        # Setting up the wage grid: BurdMort solution
        k1 = gamma1BM/delta
        k0 = gamma0BM/delta
        wLBM = ((1+k1)**2 * b + (k0-k1)*k1*p)/((1+k1)**2 + (k0-k1)*k1)
        wbarBM = p-(1-(gamma1BM/(delta+gamma1BM)))**2 * (p-wLBM)
        # Put the points in the middle of spaces
        wgridBM = np.linspace(0,1,wpts)**wpow * (wbarBM-wLBM) + wLBM

        FBM = np.zeros(wpts)
        for wi in range(1, wpts):
            FBM[wi] = (delta + gamma1BM) / gamma1BM * (1 - ((p - wgridBM[wi]) / (p - wLBM)) ** .5)
        FBM[0] = np.min([1e-4, np.min(FBM) / 10])
        # Be sure it's a distribution
        FBM = FBM / FBM[wpts - 1]
        GwBM = ((FBM - FBM[0]) / (1 - FBM[0])) / (1 + k1 * (1 - FBM))

        gbm = dist2dens(GwBM,wgridBM)
                
        #gbm[0] = (GwBM [1] - GwBM [0]) / (wgridBM[1] - wgridBM[0])
        #for wi in range(1, wpts - 1):
        #    gbm[wi] = (GwBM [wi + 1] - GwBM [wi - 1]) / (wgridBM[wi + 1] - wgridBM[wi - 1])
        #gbm[wpts - 1] = (GwBM[wpts - 1] - GwBM[wpts - 2]) / (wgridBM[wpts - 1] - wgridBM[wpts - 2])

        gamma0BM = UErt/(1-FBM[0])
        EEhere = np.trapz(gamma1BM*(1-FBM)*gbm,wgridBM)/np.trapz(gbm,wgridBM)
        g1BMp = EErt/np.trapz((1-FBM)*gbm,wgridBM)*np.trapz(gbm,wgridBM)
        if( abs(EEhere - EErt)< 1e-6):
            break
        else:
            gamma1BM = 0.9*gamma1BM+ 0.1*g1BMp


    # Average employment rate
    n = 1 - delta / (delta + gamma0BM * (1 - FBM[0]))
    print(EEhere)
    print(gammaiter)
    R = np.ones(zpts) * wLBM

    return(n,FBM,GwBM,R,wgridBM,gamma0BM,gamma1BM)


def setOmega(alphain):
    global zgrid
    global z0
    global zZ
    global zstep

    z0 = 1
    zZ = ((-0.999 + z0) * z0 ** (-alphain)) ** (1 / (1 - alphain))
    zgrid = np.logspace(np.log(z0),np.log(zZ),num= zpts + 1,base=np.exp(1)) #np.linspace(0, 1,num= zpts + 1,base=np.exp(1)) ** zpow * (zZ - z0) + z0
    zgrid = .5 * (zgrid[1:] + zgrid[0:-1])
    zgrid[0] = z0
    zgrid[zpts - 1] = zZ

    zstep = np.logspace(np.log(z0),np.log(zZ),num= zpts + 1,base=np.exp(1)) #np.linspace(0, 1, num= zpts + 1,base=np.exp(1)) ** zpow * (zZ - z0) + z0
    zstep = zstep[1:] - zstep[:-1]

    Omegaz = (alphain-1)*(zgrid/z0)**-alphain
    # Renorm Omegaz to integrate to 1
    Omegaz = Omegaz/np.trapz(Omegaz,zgrid)
    return(Omegaz)

def solEcon(gamma0in, gamma1in, nu0in,nu1in,alphain,cal_flag=False):

    global zgrid
    global zpts
    global wpts
    global nu0,nu1,gamma1,gamma0,alpha

    nu0 = nu0in
    nu1 = nu1in
    gamma1 = gamma1in
    gamma0 = gamma0in
    alpha  = alphain

    Omegaz = setOmega(alphain)
    Psis   = Omegaz.copy() #just to initialize
    wgrid  = np.ones(wpts)

    #initialize F, R, wgrid with the BM versions

    [n,Fw,Gw,Rz,wgrid,gamma0BM,gamma1BM] = initBM(UEtarget,EEtarget)

    nz = np.ones(zpts)*n
    wbar = wgrid[wpts-1]
    wL = wgrid[0]

    r1z = rhoz1(zgrid, nz,Psis,nu1,gamma1)
    r0z = rhoz0(zgrid, nz,Psis,nu0,gamma1)

    # setup some objects
    Gwz0 = np.outer(Gw, np.ones(zpts))
    Gtilde = np.zeros(wpts)

    refyield_wz = np.zeros((wpts, zpts))
    diryield_wz = np.zeros((wpts, zpts))

    lwz = np.zeros((wpts, zpts))
    Lw = np.zeros(wpts)

    nz = np.ones(zpts) * n / np.inner(zstep, Omegaz)
    nz1 = nz.copy()
    n1  = n

    Psis = nz * Omegaz / np.inner(zstep, nz * Omegaz)
    means = np.inner(zgrid, (Psis * zstep))

    R1 = np.copy(Rz)
    mask = np.outer(np.ones(wpts), Rz) <= np.outer(wgrid, np.ones(zpts))

    if (print_lev > 0):
        print ("gamma0:   %6.6f, gamma1: %6.6f, nu1:   %6.6f, alpha:       %6.6f" % (gamma0in, gamma1in, nu1in, alphain))

    for Fi in range(0, maxFiter):

        # Iterate on n, but it doesn't really matter what is the original guess
        for niter in range(0,maxniter):

            mask = np.outer(np.ones(wpts), Rz) <= np.outer(wgrid, np.ones(zpts))
            mask = mask.flatten('F')

            Rdist = np.zeros((maxRziter,2))

            for Rziter in range(0,maxRziter):

                # Solve for the distribution of offers

                for Gwziter in range(0,2000):

                    Gtilde = np.trapz(np.outer(Psis,np.ones(wpts)) * Gwz0.T, zgrid, axis=0)

                    [resid,Gwz1,C,D] = Gwzdef(Gwz0.flatten('F')[mask],Rz,Fw,Psis,wgrid,mask,r0z,r1z)
                    Gwz1 = Gwz1.flatten('F')
                    Gwz1[mask == 0] = 0
                    Gwz1 = Gwz1.reshape((wpts,zpts), order='F')
                    # Impose adding up to Gwz1
                    Gwz1 = Gwz1/np.outer(np.ones(wpts),np.amax(Gwz1,axis=0))
                    norm_resid = np.max(abs(Gwz1 - Gwz0))
                    abs_resid = np.max(abs(resid))
                    Gwz0 = update_Gwz*Gwz1 + (1-update_Gwz)*Gwz0
                    abs_tol = np.max([RzTol,(minRziter - Rziter - 1)/minRziter*1e-3 + (Rziter+1)/minRziter*RzTol])
                    norm_tol = abs_tol
                    if abs_resid < abs_tol or norm_resid < norm_tol:
                        break
                if Gwziter >= 1999:
                    print("No Gwz converge", Fi, niter,Rziter,Gwziter)

                [resid,Gwz1,R1,VpRz] = Gwzdef(Gwz0.flatten('F')[mask],Rz,Fw,Psis,wgrid,mask,r0z,r1z)
                Gwz1 = Gwz1/np.outer(np.ones(wpts),np.amax(Gwz1,axis=0))
                Rdist[Rziter,0] = np.max((Rz-R1)**2/Rz)
                Rdist[Rziter,1] = np.argmax((Rz-R1)**2/Rz)+1

                if Rdist[Rziter,0] < RzTol:
                    RzConverged = 1
                    break

                if Rziter > minRziter:
                    if np.mean(Rdist[Rziter-100:Rziter-50,0]) <= (np.mean(Rdist[Rziter-50:Rziter,0]) + 5e-4):
                        RzConverged = 0
                        break

                Rz = (1-update_R)*Rz + update_R*R1
                mask = np.outer(np.ones(wpts), Rz) <= np.outer(wgrid, np.ones(zpts))
                mask = mask.flatten('F')
            # Done with Rz iteration

            Gtilde = np.trapz(np.tile(Psis,(wpts,1))*Gwz0, zgrid)
            Gtilde = Gtilde/Gtilde[wpts-1]
            Gw     = np.trapz(np.tile(Omegaz,(wpts,1))*Gwz0, zgrid)
            Gw     = Gw/Gw[wpts-1] #because int nz /= 1

            # Recover steady state n(z) and integrate to n
            for zi in range(0,zpts):
                indic = wgrid <= Rz[zi]

                Rzi = np.min([wgrid[wpts-1], np.max([wgrid[0],Rz[zi]])])

                FR = np.interp(Rzi,wgrid,Fw,left=0,right=0)
                GR = np.interp(Rzi,wgrid,Gtilde,left=0,right=0)

                nz1[zi] = (gamma0*(1-FR)+(1-gamma0)*r0z[zi]*(1-GR) ) \
                    /(delta+gamma0*(1-FR) + (1-gamma0)*r0z[zi]*(1-GR))

            n1 = np.inner(nz1 * Omegaz, zstep)
            ndist = abs(n1-n)/n

            if ndist < ntol:
                break
            else:
                nz = (1.-update_n)*nz + update_n*nz1
                n = np.inner(nz * Omegaz, zstep)
                Psis = nz * Omegaz / np.inner(zstep, nz * Omegaz)
                Psis = Psis / np.sum(Psis * zstep)  # renorm to integrate to 1
                FOSD_check = np.cumsum(Omegaz*zstep) - np.cumsum(Psis*zstep)
                means = np.inner(zgrid, (Psis * zstep))

            for zi in range(0,zpts):
                r1z[zi] = rhoz1(zgrid[zi],nz,Psis,nu1,gamma1)
                r0z[zi] = rhoz0(zgrid[zi],nz,Psis,nu0,gamma1)

        # Done solving for n

        #solve for l(w,z)
        zgrid_wz = np.zeros((wpts,zpts))
        hdir_wz = np.zeros((wpts,zpts))
        href_wz = np.zeros((wpts, zpts))
        r1z_wz  = np.zeros((wpts, zpts))
        zused   = np.ones(wpts,dtype=np.int)*zpts

        lbarw_resid = lambda lbarwH: lbarwH*delta - Omegaz/M*((1-nz)*gamma0+ nz*gamma1) - np.trapz(lbarwH*zgrid,zgrid, axis=0)*gamma1*Psis*(nu1*nz+(1-nz)*nu0)

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

        for wi in range(0,wpts-1):
            indicz[wi] = Rz <= wgrid[wi]
            zRm1 = sum(indicz[wi])
            if zRm1 == 0: # at least the lowest R-type will accept
                indicz[wi, np.argmin(Rz)] = 1
            beta_z = delta + gamma1*(1-Fw[wi]) + (1-gamma1)*r1z*(1-Gtilde[wi])*indicz[wi]
            hdir_z = Omegaz/M * ((1-nz)*gamma0*indicz[wi] + nz*gamma1*Gwz1[wi])
            href_z = gamma1*Psis*((1-nz)*nu0*indicz[wi] + nz*nu1*Gwz1[wi])
            r1z_z  = r1z.copy()
            # solve the edge, where Rz(z)<wgrid(wi)
            if zRm1 <= (zpts-1):
                beta_z[indicz[wi]==0] = 0.
                hdir_z[indicz[wi]==0] = 0.
                href_z[indicz[wi]==0] = 0.
                r1z_z[indicz[wi]==0] = 0.
                Omegaz_endo = Omegaz.copy()
                zg_endo = zgrid.copy()
                zg_endo[indicz[wi]==0] = 0.
                Omegaz_endo[indicz[wi]==0] = 0.
                zindif = -1
                if zRm1 >= 1: #<full but at least one accepts it
                    for zR1 in range(0,zpts):
                    # by linear interpolation, find points beyond the threshold
                        if Rz[zR1] > wgrid[wi]:
                            if zR1>0:
                                if Rz[zR1-1] <= wgrid[wi]:
                                    zindif = (wgrid[wi] - Rz[zR1-1])*(zgrid[zR1] - zgrid[zR1-1])/(Rz[zR1] - Rz[zR1-1]) + zgrid[zR1-1]
                                    zindif = min(max([z0,zindif]),zZ)
                            if zR1 < zpts-1:
                                if Rz[zR1+1] <= wgrid[wi]:
                                    zindif = (wgrid[wi] - Rz[zR1+1])*(zgrid[zR1] - zgrid[zR1+1])/(Rz[zR1] - Rz[zR1+1]) + zgrid[zR1+1]
                                    zindif = min(max([z0,zindif]),zZ)
                            if zindif>-1:
                                zg_endo[zR1] = zindif
                                beta_zfun = lambda zR: np.interp(zR,zgrid,beta_z)
                                hdir_zfun = lambda zR: np.interp(zR,zgrid, Omegaz/M*((1-nz)*gamma0 + nz*gamma1*Gwz1[wi]))
                                href_zfun = lambda zR: np.interp(zR,zgrid, gamma1*Psis*((1-nz)*nu0 + nu1*nz*Gwz1[wi]))
                                beta_z[zR1] = beta_zfun(zindif)
                                hdir_z[zR1] = hdir_zfun(zindif)
                                href_z[zR1] = href_zfun(zindif)
                                Omegaz_endo[zR1] = np.interp(zindif, zgrid, Omegaz/M)
                                r1z_z[zR1] = np.interp(zindif, zgrid, r1z)
                                zindif = -1

                #end interpolation for edge case
                beta_z = np.trim_zeros( beta_z )
                hdir_z = np.trim_zeros( hdir_z )
                href_z = np.trim_zeros( href_z )
                Omegaz_endo = np.trim_zeros(Omegaz_endo)
                if zRm1 >= 1:
                    zg_endo = np.trim_zeros(zg_endo)
                    r1z_z = rhoz1(zg_endo, nz,Psis,nu1,gamma1)
                    lwz_resid = lambda lwzH: lwzH*beta_z - hdir_z - np.trapz(lwzH*zg_endo,zg_endo, axis=0)*href_z
                    lwzTemp, infodict, flag, mesg = fsolve(lwz_resid, Omegaz_endo/M, full_output=True)
                    zused[wi] = len(beta_z)
                    if flag != 1:
                        if(print_lev>0):
                            print("Labor force flag is %d at wi=%d length is %d" % (flag, wi, zused[wi]))
                        #lwzTemp = Omegaz_endo * 0.
                    lwz[wi,0:zused[wi]] = lwzTemp.copy()
                    if zRm1>1:
                        Lw[wi] = np.trapz(lwz[wi,0:zused[wi]],zg_endo)
                    else:
                        Lw[wi] = lwz[wi,0]
                else: # zRm1 <1
                    zused[wi] = 2
                    zg_endo[0] = zgrid[indicz[wi]==1]
                    zg_endo[1] = zgrid[indicz[wi]==1]+1
                    zg_endo = zg_endo[0:2]
                    r1z_z = rhoz1(  zgrid[indicz[wi]==1], nz,Psis,nu1,gamma1)
                    lwz_resid = lambda lwzH: lwzH * beta_z - hdir_z - lwzH*zg_endo[0] * href_z
                    lwzTemp, infodict, flag, mesg = fsolve(lwz_resid, Omegaz_endo / M, full_output=True)
                    if flag != 1:
                        if(print_lev>0):
                            print("Labor force flag is %d at wi=%d length is %d" % (flag, wi, 1))
                        #lwzTemp = Omegaz_endo*0.
                    lwz[wi, 0:zused[wi]] = lwzTemp.copy()
                    Lw[wi] = lwz[wi, 0]
            else: # zRm1 =zpts
                lwz_resid = lambda lwzH: lwzH*beta_z - hdir_z - np.trapz(lwzH*zgrid,zgrid, axis=0)*href_z
                lwzTemp, infodict, flag, mesg = fsolve(lwz_resid, Omegaz, full_output=True)
                if flag != 1 and wi>0:
                    lwzTemp, infodict, flag, mesg = fsolve(lwz_resid, lwz[wi-1,:], full_output=True)
                lwz[wi,:] = lwzTemp.copy()
                Lw[wi] = np.trapz(lwz[wi,:],zgrid)
                zg_endo = zgrid.copy()
                zused[wi] = zpts
                if flag != 1:
                    print("Labor force flag is %d at wi= %d" % (flag,wi))

            diryield_wz[wi,0:zused[wi]] = hdir_z.copy()
            refyield_wz[wi,0:zused[wi]] = href_z.copy()*np.trapz(lwz[wi,0:zused[wi]]*zg_endo,zg_endo, axis=0)


            zgrid_wz[wi,0:zused[wi]] = zg_endo.copy()
            hdir_wz[wi,0:zused[wi]] = hdir_z.copy()
            href_wz[wi,0:zused[wi]] = href_z.copy()
            r1z_wz[wi,0:zused[wi]] = r1z_z.copy()
    #            lwz[wi,0:zused[wi]] = np.trim_zeros(lwzTemp)

        if( Lw[wpts-2]>Lw[wpts-1] ): #ensure Lw is non-decreasing at the top
            if wgrid[wpts-1]>wgrid[wpts-2]:            
                Lw[wpts-1] = Lw[wpts-2]+(Lw[wpts-2]-Lw[wpts-3])/(wgrid[wpts-2]-wgrid[wpts-3])*(wgrid[wpts-1]-wgrid[wpts-2])
            else:
                Lw[wpts-1] = Lw[wpts-3]+(Lw[wpts-2]-Lw[wpts-3])/(wgrid[wpts-2]-wgrid[wpts-3])*(wgrid[wpts-1]-wgrid[wpts-3])
        piz_0 = Lw*(p-wgrid)

        # match wbar to minimum profit in interior points of wage distribution
        Epiz_0 = np.trapz(piz_0[1:wpts-1], wgrid[1:wpts-1])/(wgrid[wpts-2]-wgrid[1])
        minpiz_0 = np.min(piz_0[1:wpts-2])
        var_pi = sum(np.square(piz_0 - Epiz_0)) / wpts

        pibarw = Lw[wpts-1]*(p - wgrid[wpts-1])
        pi_target = max(minpiz_0,1.e-5)

        #use hdir_wz, href_wz, zg_endo, r1z_endo from above
        h_wz = np.zeros((wpts,zpts))
        for wi in range(1,wpts-1):
            zR1 = zused[wi]
            h_wz[wi,0:zR1] = hdir_wz[wi,0:zR1] + lwz[wi,0:zR1]*href_wz[wi,0:zR1]

    #solve wbar
        wbar_new = p - pi_target/Lw[wpts-1]
        wbar_new = min(wbar_new,p)
        wbar = update_wbar * wbar_new + (1 - update_wbar) * wbar
    # solve wL
        if nu0-nu1 > 1e-2 or gamma0-gamma1 > 1e-2:
            # Will have heterogeneous R
            wL1 = lowestw(Rz,Omegaz,Psis,nz,r1z)
        else:
            wL1 = np.min(Rz)

        wgrid1 = wgrid.copy()

        for wi in range(wpts-1,0,-1):
            wgrid1[wi]= p - pi_target/Lw[wi]
            wgrid[wi] = wgrid1[wi]*update_wbar + (1.-update_wbar)*wgrid[wi]
        wL = wL1*update_wbar + (1.-update_wbar)*wgrid[0]
        wL = max(wL,b)        
        if wL< wgrid[1] :
            wgrid[0] = wL
        else:
            wgrid[0] = wgrid[1]-1.e-6

        wstep = np.zeros(len(wgrid))
        wmid = 0.5 * (wgrid[:-1] + wgrid[1:])
        wstep[1:-1] = wmid[1:] - wmid[:-1]
        wstep[0] = wmid[0] - wgrid[0]
        wstep[-1] = wgrid[-1] - wmid[-1]

        difw = wgrid1 - wgrid
        if (np.max(abs(difw)) < 1e-6) | (np.sqrt(var_pi)/pi_target <1e-2 ):
            break

        if (var_pi >1e-7) and print_lev>2 :
            print("variance in pi is %f" % var_pi)
        if print_lev >2 or wbar<.8 :
            print("wbar is %f compared to BM %f" % (wbar_new, p-(1-(gamma1/(delta+gamma1)))**2*(p-wL)) )
            print("profit is %f" % minpiz_0)
            print("-------------------------")

    #   compute some stats with it.
    #------------------------------------------------------------

    #densities (from the distributions)
    fw = dist2dens(Fw,wgrid)
    gw = dist2dens(Gw,wgrid)
    gtilde = dist2dens(Gtilde,wgrid)
    gwz= np.zeros((wpts,zpts))
    for zi in range(0,zpts):
        gwz[:,zi] =dist2dens(Gwz0[:,zi],wgrid)

    # Vacancies filled by referral
    refyield = np.trapz(np.trapz(refyield_wz*lwz,zgrid, axis=1)/Lw*Fw,wgrid,axis=0)
    # Vacancies fill by direct contact
    diryield = np.trapz(np.trapz(diryield_wz*lwz,zgrid, axis=1)/Lw*Fw,wgrid,axis=0)
    # Unempoyed worker's finding rate
    indicR_wz = np.ones((wpts,zpts))
    for zi in range(0,zpts):
        indicR_wz[:,zi] = wgrid>= Rz[zi]

    UEfrt = np.trapz(Omegaz*(1-nz)*np.trapz(np.outer( fw,gamma0 + (1-gamma0)*r0z)*indicR_wz, wgrid,axis=0) \
                     ,zgrid,axis=0)/np.trapz(Omegaz*(1-nz),zgrid,axis=0)
    # Employed worker's finding rate
    EEfrt = np.trapz(Omegaz * nz * np.trapz((np.outer(gamma1 * (1 - Fw), np.ones(zpts)) + \
                                       (1 - gamma1) * np.outer((1 - Gtilde), r1z)) * gwz, wgrid, axis=0), zgrid, axis=0) \
            / np.trapz(Omegaz * nz, zgrid, axis=0)
    EEnetpct = np.trapz(np.trapz((1 - gamma1) * np.outer((1 - Gtilde), r1z) * gwz, wgrid, axis=0) /
                     np.trapz((np.outer(gamma1 * (1 - Fw), np.ones(zpts)) + (1 - gamma1) * np.outer((1 - Gtilde),r1z)) * gwz, wgrid,
                           axis=0) \
                     * Omegaz * nz, zgrid, axis=0) / np.trapz(Omegaz * nz, zgrid, axis=0)


    # Probability of found by network/Probability of found by direct at each wage
    netfrtdirfrt_w = np.trapz( np.tile(Omegaz*nz,(wpts-1,1))* \
        np.outer((1-Gtilde[:-1])*(1-gamma1),r1z) / ( np.outer((1-Gtilde[:-1])*(1-gamma1),r1z)+np.tile((1-Fw[:-1])*gamma1,(zpts,1)).T ),zgrid,axis=1 )
    netfrtX = np.vstack((Gw[1:], np.ones(wpts-1))).T
    netfrtXpX = np.matmul(netfrtX.T, netfrtX)

    netfrtBeta   = np.linalg.solve(netfrtXpX,np.matmul(netfrtX.T, netfrtdirfrt_w))
    netfrtdirfrt = float(netfrtBeta[0]) #np.trapz(gw[:-1]*netfrtdirfrt_w,wgrid[:-1])

    if( print_lev>=2):
        print("-------------------------")
        print("paramvec: %f,%f,%f,%f" % (gamma0,gamma1,nu1,alpha) )

        print("Filled by referral: %f"    % float(refyield/(diryield + refyield)))
        print("EE Filled by referral: %f" % EEnetpct)
               
        print("UE finding rate: %f" % UEfrt)
        print("EE finding rate: %f" % EEfrt)
        print("Average increase in net frt: %f" % netfrtdirfrt)
        print("-------------------------")

    errvec = np.zeros(4)
#    errvec[0] = (EEnetpct - EENetfndTarget)/EENetfndTarget
    errvec[0] = (refyield/(diryield + refyield) - NetfndTarget)/NetfndTarget
    errvec[1] = (EEfrt - EEtarget)/EEtarget
    errvec[2] = (UEfrt - UEtarget) / UEtarget
    errvec[3] = (netfrtdirfrt - netfrtdirfrtTarget)/netfrtdirfrtTarget

    errwts = np.ones(4)
    errwts[0] = 0.5 #just arbitrary

    for si in range(0,4):
        errvec[si] = errvec[si]*errwts[si]

    if(print_lev>0):
    #    print ("gamma0:   %6.6f, gamma1: %6.6f, nu1:   %6.6f, alpha:       %6.6f" % (gamma0,gamma1,nu1,alpha))
        print ("refyield: %6.6f, EEfrt:  %6.6f, UEfrt: %6.6f, net_dir frt: %6.6f" % (\
            errvec[0]/errwts[0],errvec[1]/errwts[1],errvec[2]/errwts[2],errvec[3]/errwts[3]))

    if( cal_flag == True):
        return(errvec)
    else:
        return(errvec,Rz,wgrid, Fw,Gtilde, Gwz0,Psis,nz,lwz,Lw)
    # end SolEcon() ----------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------


# for ni in range(0,10):
#     print_lev = 2
#     nu1 = 0.01 + ni*0.01
#     [errvec,Rz,wgrid,Fw,Gtilde,Gwz,Psis,nz,lwz,Lw] = solEcon(nu1, nu1,alpha)
print_lev = 1

if calibrate_flag == True:

    print_lev = 0
    comm = MPI.COMM_WORLD
    Nnode = comm.Get_size()
    Nx = 4
    Nstartpernode = Nx # Nx**2
    fopthist = open("fopthist.txt","w")
    fopthist.write("opt f, ")
    for i in range(0, Nx - 1):
        fopthist.write("opt x %d, " % i)
    fopthist.write("opt x %d \n" % (Nx-1))
    fopthist.close()
    rankhr = comm.Get_rank()
    constr_lb =  np.array([1.e-3, 1.e-3, 1.e-3, 2.])
    constr_ub =  np.array([0.7  , 1.0  , 1.0  , 3.48])

    solEcon_obj = lambda xin: solEcon(xin[0]*(constr_ub[0]-constr_lb[0])+constr_lb[0],
                                      (xin[1]*(constr_ub[1]-constr_lb[1])+constr_lb[1]) * (xin[0]*(constr_ub[0]-constr_lb[0])+constr_lb[0]),
                                      xin[2]*(constr_ub[2]-constr_lb[2])+constr_lb[2],
                                      xin[2]*(constr_ub[2]-constr_lb[2])+constr_lb[2],
                                      xin[3]*(constr_ub[3]-constr_lb[3])+constr_lb[3],True)
    xin_arr  = np.zeros((Nstartpernode*Nnode,Nx))
    solx_arr = np.zeros(Nstartpernode*Nnode*Nx)
    solf_arr = np.zeros(Nstartpernode*Nnode)

    solx_arrhr = np.zeros(Nstartpernode*Nx)
    solf_arrhr = np.zeros(Nstartpernode)

    for xi in range(0,Nstartpernode*Nnode):
        for vi in range(0,Nx):
            xin_arr[xi,vi] = np.random.uniform(0.,1.)
    for si in range(0,Nstartpernode):
        
        soln = dfogn.solve(solEcon_obj, xin_arr[rankhr*Nstartpernode+si], lower=np.zeros(Nx), upper=np.ones(Nx), maxfun=100,
                   rhobeg=.1, rhoend=1e-8)
        solf_arrhr[si] = soln.f
        solx_arrhr[si*Nx:(si+1)*Nx] = soln.x
        fopthist = open("fopthist.txt", "a+")
        fopthist.write("%f, " % soln.f)
        for i in range(0, Nx - 1):
            fopthist.write("%f, " % soln.x[i])
        fopthist.write("%f \n" % soln.x[Nx - 1])
        fopthist.close()


    comm.Gather(solf_arrhr,solf_arr)
    comm.Gather(solx_arrhr,solx_arr)


    if rankhr==0:
        print solf_arr
        print solx_arr
        print_lev = 2
        optf = np.min(solf_arr)
        opti = np.argmin(solf_arr)
        optx = solx_arr[opti*Nx:(opti+1)*Nx]
        print optf
        print optx

        f = open('optfx.txt','w')
        optfx = (optf,optx[0],optx[1],optx[2],optx[3])
        f.write("opt f: %f," % optf)
        f.write("opt x: ")
        for i in range(0,Nx-1):
            f.write(" %f," % optx[i])
        f.write(" %f \n" % optx[Nx-1])
        f.close()

        gamma0 = optx[0]*(constr_ub[0]-constr_lb[0])+constr_lb[0]
        gamma1 =(optx[1]*(constr_ub[1]-constr_lb[1])+constr_lb[1])*gamma0
        nu0    = optx[2]*(constr_ub[2]-constr_lb[2])+constr_lb[2]
        nu1    = optx[2]*(constr_ub[2]-constr_lb[2])+constr_lb[2]
        alpha  = optx[3]*(constr_ub[3]-constr_lb[3])+constr_lb[3]

else:

    rankhr = 0 # just to have conistency with the MPI version
    print_lev = 2    
    
    # use calibration location gamma0: 0.236142, gamma1: 0.087374, nu1: 0.067709, alpha: 3.480000
    constr_lb =  np.array([1.e-3, 1.e-3, 1.e-3, 2.])
    constr_ub =  np.array([0.7  , 1.0  , 1.0  , 3.48])
    #optx = np.array((0.340765, 0.107869, 0.106933, 1.000000))
    #optx  = np.array((0.417114, 0.127935, 0.088333, 1.000000))
    #optx  = np.array((0.548626, 0.120896, 0.056056, 1.000000))
    optx  = np.array((0.345180, 0.400275, 0.041292, 0.232880))
    gamma0 = optx[0] * (constr_ub[0] - constr_lb[0]) + constr_lb[0]
    gamma1 = optx[1] * ((constr_ub[1] - constr_lb[1]) + constr_lb[1])*gamma0
    nu0 = optx[2] * (constr_ub[2] - constr_lb[2]) + constr_lb[2]*2
    nu1 = optx[2] * (constr_ub[2] - constr_lb[2]) + constr_lb[2]*2
    alpha = optx[3] * (constr_ub[3] - constr_lb[3]) + constr_lb[3]


if rankhr == 0:
    
    calibrate_flag = False
    print_lev = 2
    [errvec, Rz, wgrid, Fw, Gtilde, Gwz0, Psis, nz, lwz, Lw] = solEcon(gamma0,gamma1,nu0,nu1,alpha,False)
    print "Err vec here:"
    print errvec
    Omegaz = setOmega(alpha)
    
#%% Compute distributions and densities
    Gw = np.trapz(np.tile(Omegaz , (wpts, 1)) * Gwz0, zgrid)
    Gw = Gw / Gw[wpts - 1]  # because int nz /= 1
    fw = dist2dens(Fw,wgrid)
    gw = dist2dens(Gw,wgrid)
    gtilde = dist2dens(Gtilde,wgrid)
    gwz= np.zeros((wpts,zpts))
    for zi in range(0,zpts):
        gwz[:,zi] =dist2dens(Gwz0[:,zi],wgrid)

    wbar = np.max(wgrid)
    r1z = rhoz1(zgrid, nz, Psis, nu1, gamma1)
    r0z = rhoz0(zgrid, nz, Psis, nu0, gamma1)

    #Implied offer distribution and measurs of wage growth
    FG = np.zeros(wpts)
    FGz = np.zeros((wpts,zpts))

    for wi in range(1, wpts):
        FG[wi] = np.trapz(Omegaz*((Fw[wi]*gamma0 + (1-gamma0)*r0z*Gtilde[wi])*(1-nz) + (gamma1*Fw[wi] + (1-gamma1)*r1z*Gtilde[wi])*nz), zgrid)
        FGz[wi,:] = (Fw[wi]*gamma0 + (1-gamma0)*r0z*Gtilde[wi])*(1-nz) + (gamma1*Fw[wi] + (1-gamma1)*r1z*Gtilde[wi])*nz
    FG = FG/FG[wpts-1]
    FGz = FGz/np.outer(np.ones(wpts),FGz[wpts-1,:])


    EEhr = np.trapz(Omegaz * nz * np.trapz((np.outer(gamma1 * (1 - Fw), np.ones(zpts)) + \
                                       (1 - gamma1) * np.outer((1 - Gtilde), r1z)) * gwz, wgrid, axis=0), zgrid, axis=0) \
            / np.trapz(Omegaz * nz, zgrid, axis=0)
    #get the BM values associated
 
    [nBM,FBM,GwBM,RBM,wgridBM,gamma0BM,gamma1BM] = initBM(UEtarget, EEtarget)
 

#%% Half-lives and distributions
    
    # Type distribution by firm type
    z_w = np.zeros(wpts)

    for wi in range(wpts):
        z_w[wi] = np.trapz(zgrid*lwz[wi,:], zgrid)/np.trapz(lwz[wi],zgrid)

    # Average wage by z
    w_z = np.zeros(zpts)
    for zi in range(zpts):
        w_z[zi] = np.trapz(gwz[:,zi]*wgrid, wgrid)
        w_z[zi] = w_z[zi]/np.trapz(gwz[:,zi],wgrid)

    # Average initial wage by z
    w1_z = np.zeros(zpts)
    Ew_F = np.trapz(wgrid*fw,wgrid)/np.trapz(fw,wgrid)
    Ew_Gtil = np.trapz(wgrid*gtilde,wgrid)/np.trapz(gtilde,wgrid)
    for zi in range(0,zpts):
        w1_z[zi] = (gamma0*Ew_F + (1-gamma0)*r0z[zi]*Ew_Gtil)/(gamma0 + (1-gamma0)*r0z[zi])

    # Average wage growth by z
    wi = wpts
    # Distribution such that above the current wage
    Ew_F_trunc = np.ones(wpts)*wbar
    Ew_Gtil_trunc = np.ones(wpts)*wbar
    for wi in range(wpts-2,0,-1):
        Ew_F_trunc[wi] = np.trapz(wgrid[wi:wpts]*fw[wi:wpts], wgrid[wi:wpts])/np.trapz(fw[wi:wpts],wgrid[wi:wpts])
        Ew_Gtil_trunc[wi] = np.trapz(wgrid[wi:wpts]*gtilde[wi:wpts],wgrid[wi:wpts])/np.trapz(gtilde[wi:wpts],wgrid[wi:wpts])
    Ew_F_trunc[0] = Ew_F
    Ew_Gtil_trunc[0] = Ew_Gtil

    Tper = 100
    jobten = np.arange(0,Tper)
    wt_z = np.zeros((zpts,Tper))
    wt_z[:,0] = w1_z
    for zi in range(zpts):
        for t in range(1,Tper):
            # Prob of wage that dominates
            FR = 1 - np.interp(wt_z[zi,t-1],wgrid,Fw)
            GR = 1 - np.interp(wt_z[zi,t-1],wgrid,Gtilde)
            Ew_FR = np.interp(wt_z[zi,t-1],wgrid,Ew_F_trunc)
            Ew_GtilR = np.interp(wt_z[zi,t-1],wgrid,Ew_Gtil_trunc)
            wt_z[zi,t] = gamma1*FR*Ew_FR + (1-gamma1)*r1z[zi]*GR*Ew_GtilR + (1-gamma1*FR-(1-gamma1)*r1z[zi]*GR)*wt_z[zi,t-1]

    # For each w, compute half-life to wbar by z for estimate of exponential decay
    halflife_wz = np.zeros((wpts-1,zpts))

    for wi in range(wpts-1):
        for zi in range(zpts):
            if Rz[zi] <= wgrid[wi]:
                # Probability of a wage that dominates
                FR = 1 - np.interp(wgrid[wi],wgrid,Fw)
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
        if Rz[zi] <= np.min(wgrid):
            pchip_func1 = pchip(wgrid,Gtilde)
            GRz[zi] = pchip_func1(Rz[zi])
            pchip_func2 = pchip(wgrid,Fw)
            FRz[zi] = pchip_func2(Rz[zi])
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

    # Average z by w
    gw_meanz = np.zeros(wpts)
    for wi in range(wpts):
        gw_meanz[wi] = np.trapz(zgrid*gwz[wi,:]*Omegaz,zgrid)
    tmp = np.trapz(gw_meanz*gw,wgrid)
    gw_meanz = gw_meanz/tmp*np.trapz(Omegaz*zgrid,zgrid)

    # Lorenz curves:
    # Compute Lorenz curve in each scenario
    LorenzW = np.zeros(wpts)
    SiW = np.zeros(wpts)
    for wi in range(wpts):
        for wj in range(wi):
             SiW[wi] = fw[wj]*wgrid[wj]+SiW[wi]
    for wi in range(wpts):
        LorenzW[wi] = SiW[wi]/SiW[wpts-1]

    pctile_Ew_F = np.interp(Ew_F,wgrid,Gw)
    pctile_Ew_Gtil = np.interp(Ew_Gtil,wgrid,Gw)

    #%% Expected wage and durations

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
    # SS wage diff
    lwz_netz = Ez_netLwt*(np.trapz(wgrid*gwz[:,Ez_netLi], wgrid)/np.trapz(gwz[:,Ez_netLi],wgrid)) + \
             (1-Ez_netLwt)*(np.trapz(wgrid*gwz[:,Ez_netLi+1],wgrid)/np.trapz(gwz[:,Ez_netLi+1],wgrid))
    pctile_lwz_netz = np.interp(lwz_netz,wgrid,Gw)
    lwz_dirz = Ez_dirLwt*np.trapz(wgrid*gwz[:,Ez_dirLi],wgrid)/np.trapz(gwz[:,Ez_dirLi],wgrid) + \
             (1-Ez_dirLwt)*np.trapz(wgrid*gwz[:,Ez_dirLi+1],wgrid)/np.trapz(gwz[:,Ez_dirLi+1],wgrid)
    pctile_lwz_dirz = np.interp(lwz_dirz,wgrid,Gw)

    #%% Average duration of match whether through network or directed search

    # First average over z for each wage level, then integerate over wage levels
    Emdur_w_net = np.zeros(wpts)
    Emdur_w_dir = np.zeros(wpts)
    for wi in range(wpts-1):
         Emdur_w_net[wi] = np.trapz(Distz_net/(gamma1*(1-Fw[wi]) + (1-gamma1)*r1z*(1-Gtilde[wi])),zgrid)
         Emdur_w_dir[wi] = np.trapz(Distz_dir/(gamma1*(1-Fw[wi]) + (1-gamma1)*r1z*(1-Gtilde[wi])),zgrid)
    gtilde_1wptsM1 = gtilde[:wpts-1]/np.trapz(gtilde[:wpts-1],wgrid[:wpts-1])
    fw_1wptsM1 = fw[:wpts-1]/np.trapz(fw[:wpts-1],wgrid[:wpts-1])
    Emdur_net = np.trapz(gtilde_1wptsM1*Emdur_w_net[:wpts-1],wgrid[:wpts-1])/12
    Emdur_dir = np.trapz(fw_1wptsM1*Emdur_w_dir[:wpts-1],wgrid[:wpts-1])/12

    # Expected duration conditional on wage
    Emdur_netVdir_condw = np.trapz(gtilde_1wptsM1[:wpts-1]*(Emdur_w_net[:wpts-1]/Emdur_w_dir[:wpts-1]),wgrid[:wpts-1])

    #%% Probability of network search find

    Pr_netfnd = np.zeros(wpts-1)
    for wi in range(wpts-1):
        Pr_netfnd[wi] = np.trapz((1-gamma1)*r1z*(1-Gtilde[wi])/((1-gamma1)*r1z*(1-Gtilde[wi]) + gamma1*(1-Fw[wi]))*Omegaz*nz,zgrid)

    #%% Find BM analogs

    nBM,FBM,GwBM,RBM,wgridBM,gamma0BM,gamma1BM = initBM(UEtarget, EEhr)
    fBM = dist2dens(FBM,wgridBM)
    wbarBM = np.max(wgridBM)

    # Lorenz Curve in BM

    LorenzWBM = np.zeros(wpts)
    SiWBM = np.zeros(wpts)
    for wi in range(wpts):
        for wj in range(wi):
            SiWBM[wi] = fBM[wj] * wgridBM[wj] + SiWBM[wi]
    for wi in range(wpts):
        LorenzWBM[wi] = SiWBM[wi] / SiWBM[wpts - 1]

    Ew_FBM = np.trapz(wgridBM * fBM, wgridBM)/np.trapz(fBM,wgridBM)
    wi = wpts - 1
    # Distribution such that above the current wage
    Ew_FBM_trunc = np.ones(wpts) * wbarBM
    for wi in range(wpts - 2, 0, -1):
        Ew_FBM_trunc[wi] = np.trapz(wgridBM[wi:wpts] * fBM[wi:wpts], wgridBM[wi:wpts]) / np.trapz(fBM[wi:wpts],
                                                                                                  wgridBM[wi:wpts])
    Ew_FBM_trunc[0] = Ew_FBM

    # For each w, compute half-life to wbar by z, for estimate of exponential decay
    halflife_BM = np.zeros(wpts - 1)

    for wi in range(wpts - 1):
        if RBM[0] <= wgridBM[wi]:
            # Prob. of a wage that dominates
            FR = 1 - np.interp(wgridBM[wi], wgridBM, FBM)
            Ew_FR = np.interp(wgridBM[wi], wgridBM, Ew_FBM_trunc)
            Ewtp1 = gamma1BM * FR * Ew_FR + (1 - gamma1BM * FR) * wgridBM[wi]
            convergert = -np.log((wbarBM - Ewtp1) / (wbarBM - wgridBM[wi]))
            halflife_BM[wi] = 1 / convergert * np.log(2)

    ##############################################################
    #%% Find hetero-search analogs

    gamma0HS = gamma0 + (1 - gamma0) * r0z
    gamma1HS = gamma1 + (1 - gamma1) * r1z

    FHS0 = np.zeros(wpts)
    FHS1 = np.zeros(wpts)
    FHS = np.zeros(wpts)
    for wi in range(wpts):
        FHS0[wi] = np.trapz(Omegaz*(1-nz) * (Fw[wi] * gamma0 + (1 - gamma0) * r0z * Gtilde[wi]), zgrid)/np.trapz(Omegaz*(1-nz),zgrid)
        FHS1[wi] = np.trapz(Omegaz*   nz  * (gamma1 * Fw[wi] + (1 - gamma1) * r1z * Gtilde[wi]), zgrid)/np.trapz(Omegaz*nz,zgrid)
        FHS[wi] = np.trapz(Omegaz * (nz * (gamma1 * Fw[wi] + (1 - gamma1) * r1z * Gtilde[wi]) + (1 - nz) * (
                    Fw[wi] * gamma0 + (1 - gamma0) * r0z * Gtilde[wi])), zgrid)

    FHS0 = FHS0 / FHS0[-1]
    FHS1 = FHS1 / FHS1[-1]
    GwzHS = np.zeros((wpts, zpts))
    for zi in range(zpts):
        FHS0R_func = interp1d(wgrid, FHS0, 'cubic', bounds_error=False, fill_value=0)
        FHS0R = FHS0R_func(Rz[zi])
        FHS1R_func = interp1d(wgrid, FHS1, 'cubic', bounds_error=False, fill_value=0)
        FHS1R = FHS0R_func(Rz[zi])
        GwzHS[:, zi] = (1 - nz[zi]) * (gamma0HS[zi] * (FHS0 - FHS0R)) / (nz[zi] * (delta + gamma1HS[zi] * (1 - FHS0)))

    GwzHS[0, :] = 0
    FHS = FHS1.copy()

    # Wage distribution, marginal over z
    GwHS = np.zeros(wpts)
    for wi in range(wpts):
        GwHS[wi] = np.trapz(Omegaz * nz * GwzHS[wi, :], zgrid)
    GwHS = GwHS / GwHS[-1]

    fHS = dist2dens(FHS, wgrid)

    # Compute Ew_FHS and Ew_FHS_trunc
    Ew_FHS = np.trapz(wgrid * fHS, wgrid)/np.trapz(fHS,wgrid)
    wi = wpts
    Ew_FHS_trunc = np.ones(wpts) * wbar
    for wi in range(wpts - 1, 0, -1):
        Ew_FHS_trunc[wi] = np.trapz(wgrid[wi:wpts] * fHS[wi:wpts], wgrid[wi:wpts]) / \
                           np.trapz(fHS[wi:wpts],
                                    wgrid[wi:wpts])
    Ew_FHS_trunc[0] = Ew_FHS

    # Compute half-life

    # For each w, compute half-life to wbar by z, for estimate of exponential decay
    halflife_wz_HS = np.zeros((wpts - 1, zpts))
    for wi in range(wpts - 1):
        for zi in range(zpts):
            if Rz[zi] <= wgrid[wi]:
                # Prob of wage that dominates
                FR = 1 - FHS[wi]
                Ew_FR = Ew_FHS_trunc[wi]
                Ewtp1 = gamma1HS[zi] * FR * Ew_FR + (1 - gamma1HS[zi] * FR) * wgrid[wi]
                convergert = -np.log((wbar - Ewtp1) / (wbar - wgrid[wi]))
                halflife_wz_HS[wi, zi] = 1 / convergert * np.log(2)

#%% Compute wage offer distribution without paradox of friends
# FG_noFP = np.zeros(wpts)
# for wi in range(1,wpts):
#     FG_noFP[wi] = np.trapz(Omegaz*((Fw[wi]*gamma0 + (1-gamma0)*r0z*Gw[wi])*(1-nz)+ \
#         (gamma1*Fw[wi] + (1-gamma1)*r1z*Gw[wi])*nz), zgrid)
# FG_noFP = FG_noFP/FG_noFP[wpts-1]
#
# # More densities
# fg = np.zeros(wpts)
# gw = np.zeros(wpts)
# fg_nFP = np.zeros(wpts)
# wi = 0
# fg[wi] = (FG[wi+1]-FG[wi])/(wgrid[wi+1]-wgrid[wi])
# fg_nFP[wi] = (FG_noFP[wi+1]-FG_noFP[wi])/(wgrid[wi+1]-wgrid[wi])
# gw[wi] = (Gw[wi+1]-Gw[wi])/(wgrid[wi+1]-wgrid[wi])
# gtilde[wi]= (Gtilde[wi+1]-Gtilde[wi])/(wgrid[wi+1]-wgrid[wi])
# for wi in range(1,wpts-1):
#     fg[wi] = (FG[wi+1]-FG[wi-1])/(wgrid[wi+1]-wgrid[wi-1])
#     fg_nFP[wi] = (FG_noFP[wi+1]-FG_noFP[wi-1])/(wgrid[wi+1]-wgrid[wi-1])
#     gw[wi] = (Gw[wi+1]-Gw[wi-1])/(wgrid[wi+1]-wgrid[wi-1])
#     gtilde[wi]= (Gtilde[wi+1]-Gtilde[wi-1])/(wgrid[wi+1]-wgrid[wi-1])
#
# fg[wpts-1] = (FG[wpts-1]-FG[wpts-2])/(wgrid[wpts-1]-wgrid[wpts-2])
# fg_nFP[wpts-1] = (FG_noFP[wpts-1]-FG_noFP[wpts-2])/(wgrid[wpts-1]-wgrid[wpts-2])
# gw[wpts-1] = (Gw[wpts-1]-Gw[wpts-2])/(wgrid[wpts-1]-wgrid[wpts-2])
# gtilde[wpts-1]= (Gtilde[wpts-1]-Gtilde[wpts-2])/(wgrid[wpts-1]-wgrid[wpts-2])
#
# gw = gw/np.trapz(gw,wgrid);
# fg = fg/np.trapz(fg,wgrid);
# fg_nFP= fg_nFP/np.trapz(fg_nFP,wgrid)
#
# # Expected wages
# Ew_G = np.trapz(wgrid*gw,wgrid)
# Ew_FG = np.trapz(wgrid*fg,wgrid)
# Ew_FG_nFP = np.trapz(wgrid*fg_nFP)
# Ew_F = np.trapz(wgrid*fw,wgrid)
# Ew_Gtil = np.trapz(wgrid*gtilde,wgrid)
#
