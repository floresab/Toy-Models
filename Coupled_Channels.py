"""
Author   : Abraham Flores
File     : Coupled_Channels.py
Language : Python 3.6
Created  : 7/17/2018
Edited   : 7/18/2018

San Digeo State University 
Department of Physics and Astronomy

"""

import numpy as np
import argonneV14 as av14
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import simps

def ReadFile(file_name,delim=" "):
    with open(file_name) as file:
        #grab header
        header = file.readline().split()
        data = [[] for i in range(len(header))]
        #iterate over data
        for line in file:
            line_list=line.split()
            #loop in parallel to append data to state
            for state, col in zip(data, line_list):
                state.append(float(col.strip()))
    file.close()
    return header,data

"""
  Bessel (int n, float z):
      computes spherical bessel functions and their derivative of 
      order n at z.
      
      Makes use of the Recurrence Relation for all spherical bessel functions.
      
      information at https://dlmf.nist.gov/10.47
      
      *******
      It should be noted that becasue bessel and neuman functions obey the 
      same recurrence relations a single function would suffice with an 
      additional parameter being the intial two values.
      *******
"""
def Bessel(n,z):
    #Intial Values
    j0 = np.sin(z)/z
    j1 = np.sin(z)/z**2 - np.cos(z)/z
    #spherical bessel function list
    sbf = [j0,j1]

    for i in range(1,n+1):
        #Recurrence Relation
        next_ = (2*i+1)/z*sbf[i]- sbf[i-1]
        sbf.append(next_)
     
    #reutrn nth bessel function and its derivative
    dz = -sbf[n+1] + n/z*sbf[n]
    return [sbf[n],dz]

"""
  Neumann (int n, float z):
      computes spherical neumann functions and their derivative of 
      order n at z.
      
      Makes use of the Recurrence Relation for all spherical bessel functions.
      
      information at https://dlmf.nist.gov/10.47
"""
def Neumann(n,z):
    #Intial Values
    n0 = -np.cos(z)/z
    n1 = -np.cos(z)/z**2 - np.sin(z)/z
    #spherical neumann function list
    snf = [n0,n1]
    
    for i in range(1,n+1):
        #Recurrence Relation
        next_ = (2*i+1)/z*snf[i]- snf[i-1]
        snf.append(next_)
     
    #reutrn nth order spherical neumann function and its derivative  
    dz = -snf[n+1] + n/z*snf[n]
    return [snf[n],dz]

def NodeCount(data):
    nodes = 0
    for prev,nxt in zip(data[:-1],data[1:]):
        if prev*nxt < 0:
            nodes += 1
    return nodes

def Operators(j,s):
    #Central Operator
    c = 1
    #IsoSpin
    tao = -3
    if j ==2:
        tao = 1
    #Spin
    sigma = 2*s*(s+1) - 3
    
    #Compute Lower Level L Channel
    l = j - 1 
    #Tensor (Sij) Operator
    T = -2*(j-1)/(2*j+1)
    #L dot S operator
    b = 0.5*(j*(j+1)-l*(l+1)-s*(s+1))
    #L^2 operator
    q = l*(l+1)
     
    op00 = [c,\
          tao,\
          sigma,\
          tao*sigma,\
          T,\
          T*tao,\
          b,\
          b*tao,\
          q,\
          q*tao,\
          q*sigma,\
          q*sigma*tao,\
          b*b,\
          b*b*tao]
    
    #Compute Higher level L channel
    l = j + 1
    #Tensor (Sij) Operator
    T = -2*(j+2)/(2*j+1)
    #L dot S operator
    b = 0.5*(j*(j+1)-l*(l+1)-s*(s+1))
    #L^2 operator
    q = l*(l+1)
    op11 = [c,\
          tao,\
          sigma,\
          tao*sigma,\
          T,\
          T*tao,\
          b,\
          b*tao,\
          q,\
          q*tao,\
          q*sigma,\
          q*sigma*tao,\
          b*b,\
          b*b*tao]
    
    #off Diagonal (Tensor Force mixing Channels)
    T = 6*np.sqrt(j*(j+1))/(2*j+1)
    op01 = np.array([0,0,0,0,T,T*tao,0,0,0,0,0,0,0,0])
    op10 = op01
    
    return [op00,op01,op10,op11]

def Matrix_G(r,xi,E,ops,j):
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
    op00,op01,op10,op11 = ops
    
    #Matrix Values
    #----------------------------------------
    G00 = (j-1)*j/r**2 + xi*(np.sum(op00*raw_pot) - E)
    G01 = xi*(np.sum(op01*raw_pot))
    G10 = G01#xi*(np.sum(operators10*raw_pot))
    G11 = (j+1)*(j+2)/r**2 + xi*(np.sum(op11*raw_pot) - E)
    #Generate and return (2x2)
    #----------------------------------------
    return np.array([[G00,G01],[G10,G11]])

def Numerov(U,E,xi,dx,R,j):
    #Identity Matrix 2x2
    I = np.identity(2)
    
    #Operator Values 
    #---------------------------------------- 
    ops = Operators(j,1)
    #----------------------------------------  
    #Compute first step 
    #----------------------------------------
    #skip over r = 0
    #c_minus = 0
    c0 = 2*I + 5/6*dx**2*Matrix_G(dx,xi,E,ops,j)
    c_plus = I - dx**2/12*Matrix_G(2*dx,xi,E,ops,j)
    
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
        c_minus =  Matrix_G(r_minus,xi,E,ops,j)*(dx**2/12) - I 
        c0      = 2*I + (5/6*dx**2)*Matrix_G(r0,xi,E,ops,j)
        c_plus  = I - (dx**2/12)*Matrix_G(r_plus,xi,E,ops,j)
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

def NeutronProton(E,j,zero_bool=False):
    #Constants
    #---------------------------------------    
    hbarC = 197.327054 #MeV fm
    mp = 938.28  #Proton Mass
    mn = 939.57  #Neutron Mass
    mu = mp*mn/(mn+mp) #Reduced Mass of the System
    R = 20 #fm
    dx = 1/128
    xi = 2*mu/hbarC**2
    k = np.sqrt(xi*E)
    #---------------------------------------
    #Long Range Solutions VNN = 0
    rho = k*R

    B_minus,B_minus_prime = Bessel(j-1,rho)
    B_plus,B_plus_prime = Bessel(j+1,rho)
    N_minus, N_minus_prime = Neumann(j-1,rho)
    N_plus, N_plus_prime = Neumann(j+1,rho)
    

    s = -N_minus +1j*B_minus
    d = -N_plus +1j*B_plus
    
    s_p = k*(s+rho*(-N_minus_prime +1j*B_minus_prime))
    d_p = k*(d+rho*(-N_plus_prime  +1j*B_plus_prime))
    
    s *= rho
    d *= rho
    out = np.array([s,d,s_p,d_p])

    #---------------------------------------
    #Starting Value of Independent Solutions 
    #---------------------------------------
    U1 = np.array([2,1])
    U2 = np.array([1,1])
    # Run Numerov    
    #--------------------------------------    
    r,u0,w0 = Numerov(U1,E,xi,dx,R,j) 
    r,u2,w2 = Numerov(U2,E,xi,dx,R,j)
    #---------------------------------------    
    #Compute Derivatives -- Central Difference
    #---------------------------------------
    one = [u0[-2],\
           w0[-2],\
           (u0[-1] - u0[-3])/(2*dx),\
           (w0[-1] - w0[-3])/(2*dx)]
    
    two = [u2[-2],\
           w2[-2],\
           (u2[-1] - u2[-3])/(2*dx),\
           (w2[-1] - w2[-3])/(2*dx)]
    #---------------------------------------
    """    
    | A00 A02 |
    | A20 A22 | 
    """    
    #Matrix Values
    #----------------------------------------
    A00 = one[0]*out[2] - one[2]*out[0]
    A02 = two[0]*out[2] - two[2]*out[0]
    A20 = one[1]*out[3] - one[3]*out[1]
    A22 = two[1]*out[3] - two[3]*out[1] 
    
    A = np.array([[A00,A02],[A20,A22]])*(-1j/(2*k))

    S = np.matmul(np.conjugate(A),np.linalg.inv(A))

    if zero_bool:        
        return S,NodeCount(u0),NodeCount(w0)
        
    print("Energy: "+str(E))
    
    return S

def PhaseShifts(S,low,high):
    C = np.absolute(S[0][1])
    mix = 0.5*np.degrees(np.arcsin(C))
    
    s_rad = 0.5*np.arctan2(np.imag(S[0][0]),np.real(S[0][0]))
    d_rad = 0.5*np.arctan2(np.imag(S[1][1]),np.real(S[1][1]))
    
    s = np.degrees(s_rad) + low
    d = np.degrees(d_rad) + high
    return (s,d,mix)

def Plot(E,data,E_comp,computed,title,y_label,file):
    sns.set(font_scale = 2.0)
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    
    fig,ax = plt.subplots()
    fig.set_size_inches(14.4,9)
    
    plt.scatter(E,data,color="k",label="GW")
    plt.plot(E_comp,computed,linestyle="--",color="r",linewidth=3.0,label="Numerov")
    plt.xlabel('Ecm (MeV)')
    plt.ylabel(y_label)
    plt.legend(loc=0)
    plt.title(title)
    plt.savefig(file+".png")
    
def GetShifts(E0,j,tol):
    S,n_s,n_d = NeutronProton(E0,j,True)
    s,d,mix = PhaseShifts(S,0,0)
    ds = s0 - s
    dd = d0 - d
    
    shift_s = abs(ds) > tol
    shift_d = abs(dd) > tol
    
#    if shift_s:
#        
#    else:
#        if shift_d and dd <0
    
    low,high = 1,2
    return low,high

if __name__ == "__main__":
    mp = 938.28  #Proton Mass
    mn = 939.57  #Neutron Mass
    mu = mp*mn/(mn+mp) #Reduced Mass of the System

    col1,datS1 = ReadFile("3P2.dat",2)
    col2,datD1 = ReadFile("3F2.dat",2)
    col3,datMix = ReadFile("e2.dat",2)
    T = np.array(datS1[0])*(mu/mn)
    ps_3s1 = datS1[1]
    ps_3d1 = datD1[1]
    mixing = datMix[1]
    
    dE = 20
    j = 1
    S = [NeutronProton(E,j) for E in T[1::dE]]

    tol = .1
    low,high = GetShifts(10**(-8),j,tol)
    
    data = [PhaseShifts(S_matrix,low,high) for S_matrix in S]
    num_3s1 = np.array([dat[0] for dat in data])
    num_3d1 = [dat[1] for dat in data]
    num_e1 = [dat[2] for dat in data]

    Plot(T,ps_3s1,T[1::dE],num_3s1,r"$3S_{1}$","Phase Shift","Figures/3s1")
    Plot(T,ps_3d1,T[1::dE],num_3d1,r"$3D_{1}$","Phase Shift","Figures/3d1")
    Plot(T,mixing,T[1::dE],num_e1,r"$J=1$","Mixing Parameter","Figures/e1")




