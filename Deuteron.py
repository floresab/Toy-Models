"""
Author   : Abraham Flores
File     : Deuteron.py
Language : Python 3.6
Created  : 6/16/2018
Edited   : 7/18/2018

San Digeo State University 
Department of Physics and Astronomy

This code aims to compute the properties of the bound state Deuteron using 
the Numerov method with the argonneV14 potential. The potential code can 
be found in argonneV14.py

v14 -- https://doi.org/10.1103/PhysRevC.29.1207

--Experimental Values of Detueron Properties: 
    Ed: -2.22463(3)  (MeV)
    eta: 0.0265(5)
    As:  0.8846(8)
    Mu:  0.857441(2) (MuN)
    Pd:  4-7 (%)
    rd:  1.953(3)    (fm)
    Qd:  0.2860(15)  (fm^(2))
    
--V14 Values of Detueron Properties: 
    Ed: -2.2250  (MeV)
    eta: 0.0266
    As:  ?
    Mu:  0.845  (MuN)
    Pd:  6.08   (%)
    rd:  ?      (fm)
    Qd:  0.286  (fm^(2))
    
"""
import numpy as np
from scipy import optimize as opt
from scipy.special import gammaincc
from scipy.special import gamma as gm
from scipy.integrate import simps
import matplotlib.pyplot as plt
import seaborn as sns
import argonneV14 as av14

#https://journals.aps.org/prc/pdf/10.1103/PhysRevC.51.38

def Matrix_G(r,xi,B):
    """
        
    | G00 G01 |
    | G10 G11 | 
    
    """
    #Array of multipliers for operators in V14
    #----------------------------------------
    raw_pot = av14.V14(r)
    #----------------------------------------
    
    #Operator Values 
    #----------------------------------------
    operators00 = np.array([1,-3,1,-3,0,0,0,0,0,0,0,0,0,0])
    operators01 = np.sqrt(8)*np.array([0,0,0,0,1,-3,0,0,0,0,0,0,0,0])
    #operators10 = operators01
    operators11 = np.array([1,-3,1,-3,-2,6,-3,9,6,-18,6,-18,9,-27])
    
    #Matrix Values
    #----------------------------------------
    G00 = xi*(B + np.sum(operators00*raw_pot))
    G01 = xi*(np.sum(operators01*raw_pot))
    G10 = G01
    G11 = 6/r**2 + xi*(B + np.sum(operators11*raw_pot))
    #Generate and return (2x2)
    #----------------------------------------
    return np.array([[G00,G01],[G10,G11]])

def Numerov(U,B,xi,dx,R):
    #Identity Matrix 2x2
    I = np.identity(2)
    
    #Compute first step 
    #----------------------------------------
    #skip over r = 0
    #c_minus = 0
    c0 = 2*I + 5/6*dx**2*Matrix_G(dx,xi,B)
    c_plus = I - dx**2/12*Matrix_G(2*dx,xi,B)
    
    U0 = np.linalg.tensorsolve(c_plus,np.matmul(c0,U))
    U_minus = U
    #---------------------------------------
    
    #Data Storage of Wavefunctions and r
    #---------------------------------------
    r = [0,dx,2*dx]
    u = [0,U[0],U0[0]]
    w = [0,U[1],U0[1]]

    #---------------------------------------
    
    #Numerov Loop: Start Computing for r = 3*dx
    for i in range(2,R*128+1):
        #radius array
        #-------------------
        r_minus = (i-1)*dx
        r0 = i*dx
        r_plus = (i+1)*dx
        r.append(r_plus)
        #-------------------
        
        #Coefficent Matrcies 2x2
        #-------------------
        c_minus =  Matrix_G(r_minus,xi,B)*(dx**2/12) - I 
        c0      = 2*I + (5/6*dx**2)*Matrix_G(r0,xi,B)
        c_plus  = I - (dx**2/12)*Matrix_G(r_plus,xi,B)
        #-------------------
        
        #Compute Next Step
        #-------------------    
        U_plus = np.linalg.tensorsolve(c_plus,\
                            np.matmul(c0,U0) +\
                            np.matmul(c_minus,U_minus))
        #-------------------
        
        #proceed
        #-------------------
        U_minus = U0
        U0 = U_plus
        #-------------------        
        u.append(U0[0])
        w.append(U0[1])
 
    return np.array(r),np.array(u),np.array(w)

def Deuteron(B,minimized=False):
    #Constants
    #---------------------------------------    
    hbarC = 197.327054 #MeV fm
    m1 = 938.28  #Proton Mass
    m2 = 939.57  #Neutron Mass
    mu = m1*m2/(m1+m2)
    R = 20 #fm
    dx = 1/128
    xi = 2*mu/hbarC**2
    gamma = np.sqrt(xi*B)
    #---------------------------------------
    #Long Range Solutions VNN = 0
    # s-wave l=0
    #---------------------------------------
    f_s = np.exp(-gamma*R)
    f_s_prime = -gamma*f_s
    #---------------------------------------
    # d-wave l=2
    #---------------------------------------
    f_d = (1+3/(gamma*R)+3/(gamma**2*R**2))*f_s
    f_d_prime = -f_s*(gamma+3/R+6/(gamma*R**2)+6/(gamma**2*R**3))
    #---------------------------------------
    #Starting Value of Independent Solutions ~0 
    #---------------------------------------
    c = 10**-6
    U_zero = np.array([c,0])
    U_two = np.array([0,c])
    # Run Numerov    
    #--------------------------------------    
    r,u0,w0 = Numerov(U_zero,B,xi,dx,R) 
    r,u2,w2 = Numerov(U_two,B,xi,dx,R)
    #---------------------------------------    
    #Compute Derivatives -- Central Difference
    #---------------------------------------
    phi = [u0[-2],\
           w0[-2],\
           (u0[-1] - u0[-3])/(2*dx),\
           ((w0[-1] - w0[-3]))/(2*dx)]
    
    psi = [u2[-2],\
           w2[-2],\
           (u2[-1] - u2[-3])/(2*dx),\
           ((w2[-1] - w2[-3]))/(2*dx)]
    #---------------------------------------
    #Generate Matrix (4x4)
    #---------------------------------------
    FS = np.array([-f_s,0,-f_s_prime,0])
    FD = np.array([0,-f_d,0,-f_d_prime])
    
    M = np.array([phi,psi,FS,FD])
    #---------------------------------------
    #Compute Determanint -- Should be zero
    #---------------------------------------
    det = abs(np.linalg.det(M))
    #---------------------------------------
    ########################################
    #         Deuteron Properties          #
    ########################################
    if minimized == True:
        print("Binding Energy: " + str(-B))
        phi_c,phi_t,phi_c_prime,phi_t_prime = phi
        psi_c,psi_t,psi_c_prime,psi_t_prime = psi 
        
#        a = (f_s*psi_c_prime-f_s_prime*psi_c)/(phi_c*psi_c_prime-phi_c_prime*psi_c)
#        b = (f_s*phi_c_prime-f_s_prime*phi_c)/(psi_c*phi_c_prime-psi_c_prime*phi_c)
        
        b = (f_s*phi_c_prime-f_s_prime*phi_c)/(psi_c*phi_c_prime-psi_c_prime*phi_c)
        a = b*(f_d*psi_t_prime-psi_t*f_d_prime)/(phi_t*f_d_prime-phi_t_prime*f_d)
        
        u = a*u0 + b*u2 #s-wave Solution
        w = a*w0 + b*w2 #d-wave Solution

        #Proportion of Deuteron in d-wave
        eta = (a*phi_t + b*psi_t)/f_d
        
        print("Eta: "+ str(eta))
        #---------------------------------------
        Iu = simps(u**2,r)
        Iw = simps(w**2,r)
        
        fs_I = (np.exp(-2*gamma*R))/(2*gamma)
        fd_I = fs_I*(1+6/(gamma*R)+12/(gamma*R)**2+6/(gamma*R)**3)
        inf = fs_I + eta**2*fd_I
        
        As = Iu + Iw + inf
        As2 = 1/(As)
        print("As: "+str(np.sqrt(As2)))
        
        #---------------------------------------
        #Useful Calculation -- Total Probabilty of d-wave
        Pd =  Iw + eta**2*fd_I
        Pd *= As2
        print("Pd: " + str(Pd))
        #---------------------------------------
        muP = 2.793 #Magnetic Moment of Proton (Exp)
        muN = -1.913 #Magnetic Moment of Neutron (Exp)
        
        #Deuteron Magnetic Moment
        muD = (muN + muP)*(1 - 1.5*Pd) + .75*Pd
        print("Magnetic Moment: " + str(muD))
        #---------------------------------------
        #Useful Constants -- infinite integration of fs -- fd -- *r^2
        # integrand = exp(-2*gamma*R)*r^(2)/(gamma*r)^(pow)
        #factored out exp(-2*gamma*R)/(4*gamma**3)
        I0 = 2*gamma*R*(gamma*R+1)+1 #pow = 0
        I1 = 2*gamma*R+1             #pow = 1
        I2 = 2                       #pow = 2
        #----
        #Incomplete Gamma Function -- Scipy Uses Regularized -- Fixed
        a = 10**(-8)
        ginc = gm(a)*gammaincc(a,2*R*gamma)/np.exp(-2*gamma*R)
        #----
        I3 = 4*ginc 
        I4 = 4/(gamma*R) - 8*ginc
        #---------------------------------------
        #Deteron Radius: 
        integrand = (r**2)*(u**2 + w**2)
        rd_inf = I0 + eta**2*(I0 + 6*I1 + 15*I2 + 18*I3 + 9*I4)
        rd_inf *= np.exp(-2*gamma*R)/(4*gamma**3)
        rd = simps(integrand,r) + rd_inf
        rd = np.sqrt(As2*rd/4)
        print("Deteron Radius: "+str(rd))
        #---------------------------------------        
        #Quadrapole Moment: 
        integrand = (u*w*np.sqrt(8) - w**2)*r**2
        Qd_inf = eta*np.sqrt(8)*(I0 + 3*I1 + 3*I2) - eta**2*(I0 + 6*I1 + 15*I2 + 18*I3 + 9*I4)
        Qd_inf *= np.exp(-2*gamma*R)/(4*gamma**3)
        Qd = simps(integrand,r) + Qd_inf
        Qd *= As2/20
        print("Quadrapole Moment: " + str(Qd))
        #---------------------------------------
        
    ########################################
    else:
        print("Evaluated at B = " + str(B))
    return det

def PlotDet(low,high,n):
    dx = (high-low)/(n-1)
    B = [b*dx+low for b in range(n)]
    D = [Deuteron(b) for b in B]
    sns.set(font_scale = 2.0)
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    
    fig,ax = plt.subplots()
    fig.set_size_inches(14.4,9)
    
    plt.plot(B,D,linestyle="--",color="k",linewidth=3.0,label="Numerov")
    plt.xlabel('b (MeV)')
    plt.ylabel("Det")
    plt.legend(loc=0)
    plt.show()
    
if __name__ == "__main__":
#    PlotDet(2,2.5,20)
    res = opt.minimize_scalar(Deuteron)
    print("#"*20)
    print(res)
    print("#"*20)
    b = res.x
    Deuteron(b,True)
    print("#"*20)


