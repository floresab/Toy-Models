"""
File     : argonneV14.py
Language : Python 3.6
Created  : 7/13/2018
Edited   : 7/13/2018

San Digeo State University 
Department of Physics and Astronomy

#https://journals.aps.org/prc/pdf/10.1103/PhysRevC.51.38 --argonneV18
This code implements Argonne V14 potential outlined in ...

    --CONSTANTS --
    
    Hbar*c    |   197.33 MeV fm
    pion-Mass |   138.03 MeV
    
    Wood-Saxon|
        R     |   0.5 fm
        a     |   0.2 fm

    Operator   |       p       |       Ip      |      Sp      |  Index  |
    -----------------------------------------------------------
    central    |       c       |   -4.801125   |   2061.5625  |    0    |
  tao dot tao  |      tao      |    0.798925   |   -477.3125  |    1    |
sigma dot sigma|     sigma     |    1.189325   |   -502.3125  |    2    | 
  (sigma)(tao) |   sigma-tao   |    0.182875   |     97.0625  |    3    | 
      Sij      |       t       |     -0.1575   |      108.75  |    4    |
    Sij(tao)   |     t-tao     |     -0.7525   |      297.25  |    5    |
    L dot S    |       b       |      0.5625   |     -719.75  |    6    |
 L dot S (tao) |     b-tao     |      0.0475   |     -159.25  |    7    |
    L squared  |       q       |    0.070625   |       8.625  |    8    |
    L^2(tao)   |     q-tao     |   -0.148125   |       5.625  |    9    |
    L^2(sigma  |    q-sigma    |   -0.040625   |      17.375  |    10   | 
L^2(sigma)(tao)|  q-sigma-tao  |   -0.001875   |     -33.625  |    11   | 
  (L dot S)^2  |      bb       |     -0.5425   |       391.0  |    12   | 
  (LS)^2(tao)  |     bb-tao    |      0.0025   |       145.0  |    13   | 
  
"""

import numpy as np

def Yukawa(r):
    #mu = mc/hbar
    c = 2 #1/fm^2
    mu = 138.03/197.33
    return np.exp(-r*mu)/(mu*r)*(1-np.exp(-c*r**2))

def Tensor(r):
    c = 2 #1/fm^2
    mu = 138.03/197.33
    return (1+3/(mu*r) + 3/(mu*r)**2)*(np.exp(-mu*r)/(mu*r))*(1-np.exp(-c*r**2))**2

def V14(r): 
    Vst = np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0])
    Vttao = np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0])
    
    #Short Range Strengths
    Sp =    np.array([\
                      2061.5625,\
                      -477.3125,\
                      -502.3125,\
                      97.0625,\
                      108.75,\
                      297.25,\
                      -719.75,\
                      -159.25,\
                      8.625,\
                      5.625,\
                      17.375,\
                      -33.625,\
                      391.0,\
                      145.0])
    # phenomenological  Strengths
    Ip = np.array([\
                   -4.801125,\
                   0.798925,\
                   1.189325,\
                   0.182875,\
                   -0.1575,\
                   -0.7525,\
                   0.5625,\
                   0.0475,\
                   0.070625,\
                   -0.148125,\
                   -0.040625,\
                   -0.001875,\
                   -0.5425,\
                   0.0025])
    
    #woods-Saxon
    short_range = Sp*1.0/(1+np.exp((r-0.5)/.2))
    # phenomenological shape
    phenomenological = Ip*(Tensor(r)**2)
    #Explcit Pion Exchange 
    pion = 3.72681*(Vst*Yukawa(r) + Vttao*Tensor(r))
    
    #compute V14 potential -- of indivdual operators
    return short_range + phenomenological + pion


