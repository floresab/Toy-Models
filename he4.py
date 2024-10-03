import numpy as np
import random
#import scipy

def WoodsSaxon(r,V_0,R,a):
  return V_0/(1+np.exp((r-R)/a))

def Numerov(V, E, h, r, m, l):
  nfd=h**2/12 #numerov finite difference factor
  hbar2 = 197.32696**2
  lsq = l*(l+1)
  g=[0]+[-2*m/hbar2*(E-V[i]-lsq*hbar2/(2*m*r[i]**2)) for i in range(1,len(r))]
  # Initial conditions
  psi = np.zeros_like(r)
  psi[0] = 0
  psi[1] = 10**-6
  psi[2] = psi[1]*(2 + 10*g[1]*nfd)/(1-g[2]*nfd)
  for i in  range(2, len(r)-1):
    psi[i+1] =(psi[i]*(2 + 10*g[i]*nfd) + psi[i-1]*(g[i-1]*nfd-1))/(1-g[i+1]*nfd)
  return psi

def parity_of_permutation(perm):
    """
    Calculates the parity of a given permutation. 
    
    Args:
        perm: A list representing a permutation. 
    
    Returns:
        int: 1 if the permutation is even, -1 if odd. 
    """
    n = len(perm)
    inversions = 0
    for i in range(n):
        for j in range(i+1, n):
            if perm[i] > perm[j]:
                inversions += 1
    return 1 if inversions % 2 == 0 else -1 

# Function to generate all binary strings 
def generateAllBinaryStrings(n, arr, i,full_arr): 
  if i == n:
    full_arr.append(arr.copy()) 
    return
  # First assign "0" at ith position 
  # and try for all other permutations 
  # for remaining positions 
  arr[i] = 0
  generateAllBinaryStrings(n, arr, i + 1,full_arr) 
  # And then assign "1" at ith position 
  # and try for all other permutations 
  # for remaining positions 
  arr[i] = 1
  generateAllBinaryStrings(n, arr, i + 1,full_arr) 
  return

def GenerateWF(psi,Stot,Ttot,npart,spin_states,isospin_states,phi,R,swap=False):
  r0,r1,r2,r3 = R
  for i,s in enumerate(spin_states):
    for j,t in enumerate(isospin_states):
      s0,s1,s2,s3 = s
      t0,t1,t2,t3 = t
      amp=1
      if swap:
        r_swap = [r1,r0,r2,r3]
        s_swap = [s1,s0,s2,s3]
        t_swap = [t1,t0,t2,t3]
        spin_good = sum(s_swap) == Stot
        pauli_good = sum([ss*tt if tt==1 else 0 for ss,tt in zip(s_swap,t_swap)]) == 0
        if (spin_good and pauli_good):
          for n in range(npart):
            ylm = np.sqrt(1/(4*np.pi)) #just swave
            rval = 1
            r_idx = 10
            amp *= 1#phi[r_idx*n+1]*ylm
          amp *= parity_of_permutation(s_swap)
        else:
          amp=0
        psi[i][j]=amp
      else:
        spin_good = sum(s) == Stot
        pauli_good = sum([ss*tt if tt==1 else 0 for ss,tt in zip(s,t)]) == 0
        if (spin_good and pauli_good):
          for n in range(npart):
            ylm = np.sqrt(1/(4*np.pi)) #just swave
            rval = 1
            r_idx = 10
            amp *= 1#phi[r_idx*n+1]*ylm
          amp *= parity_of_permutation(s)
        else:
          amp=0
        psi[i][j]=amp

  return

def NextState(psi):
  return 1

if __name__=='__main__':
  npart = 4
  nprot = 2
  stot = 0
  ttot = 0
  arr=[0 for n in range(npart)]
  full_arr = []

  V0 = 50
  Rws = 1.2*npart**(1/3)
  a = 0.6
  wsE = -10
  h = 1/128
  r_max = 20

  r = np.arange(0, r_max+h, h)
  V = WoodsSaxon(r, V0, Rws, a)
  m = (npart-1)/npart
  swave_phi = Numerov(V, wsE, h, r, m, 0)
  generateAllBinaryStrings(npart, arr, 0,full_arr)

  spin_states =[]
  isospin_states =[]
  for perm in full_arr:
    up_down = [p if p==1 else -1 for p in perm]
    spin_states.append(up_down.copy())
    if sum(perm) == nprot:
      isospin_states.append(up_down.copy())

  ns = len(spin_states)    # 2**A
  nt = len(isospin_states) # (A Z)

  psi = [[0 for j in range(nt)] for i in range(ns)]
  psi_swap = [[0 for j in range(nt)] for i in range(ns)]
  r0 = [0,1,2]
  r1 = [-0.5,0,0]
  r2 = [0,0.5,0]
  r3 = [0,0,-0.5]
  R=[r0,r1,r2,r3]
  GenerateWF(psi,stot,ttot,npart,spin_states,isospin_states,swave_phi,R)
  GenerateWF(psi_swap,stot,ttot,npart,spin_states,isospin_states,swave_phi,R,swap=True)
  for i in range(ns):
    for j in range(nt):
      print(i,j,psi[i][j],-psi_swap[i][j])