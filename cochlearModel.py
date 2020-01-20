import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import scipy.linalg as la
import pickle
from tqdm import tqdm
import scipy.signal as sg
#from tqdm import tqdm

# Lots of efficiency warnings. Only activate for usage, not testing
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# Simulation parameters
# =============================================================================

# Excitation
fForcing = 370          #Hz
p0Forcing = 20e-2

ampMod = 0
fMod = fForcing/50

# Duration & Time-step
dt = 1e-5
T = 2000*dt
tS = np.arange(0, T, dt)

# Data to be considered
nSnap = 400

# =============================================================================
# #Experimental parameters from the paper
# =============================================================================

# Isolated elements
## Membrane parameters
N = 3500                        # Number of oscillators
L = 35e-3                       # Length (m)
b = 1.1e-3                      # Breadth (m)

omeg0 = 1e5                     #Hz
d = 7e-3                        #m

beta = 8e19

k = 3.9e-9                      #m2s-1
kP = -3e-8                      #m2s-1

## Tympan
sOw = 3.2*(1e-3)**2             #m2
sTy = 49*(1e-3)**2              #m2
m = 5.9e-5                      #kg
omegOw = 2*np.pi*1500           #rad s-1
gamma = 0.0295                  #Ns m-1

## Fluid
rho = 1e3                       #kg m-3
l = 1e-3                        #m

## Ornstein Uhlenbeck
lamb = 5e-3
sig = 58.5

# Coupling parameters
## Membrane & internal pressure
alpha = 5e3                    #Pa.s.m-1

## Pressure & Tympan
Gamma = 1.3

# =============================================================================
# Operators definition
# =============================================================================

# Define usefull arrays
xS = np.linspace(0,L, N+1)
dx = xS[1] - xS[0]
omeg = lambda x : omeg0*np.exp(-x/d)
eps = lambda x : 0

omegF = 2*np.pi*fForcing
amp = lambda t : p0Forcing*(1+ampMod*np.sin(2*np.pi*fMod*t))

nT = len(tS)
nSnap = min(nT, nSnap)
dnX = N//nSnap+1
xSnap = xS[::dnX]
nX = len(xSnap)

pIn = lambda t : p0Forcing*np.exp(-0.1/(0.1*fForcing**2*t**2+0.01)) \
                    * amp(t)*np.sin(omegF*t)
# pIn = lambda t : p0Forcing*np.sin(omegF*t)*100*t/T

# Orntein Uhlenbeck process
eps = np.zeros(len(xS))
sigmaBis = sig*np.sqrt(2./lamb)
sqrtdx = np.sqrt(dx)

# For deterministic results
np.random.seed(2020)
for i in range(len(xS)-1):
    eps[i+1] = eps[i] + dx*(-eps[i]/lamb) + sigmaBis*sqrtdx*np.random.randn()
    
# Initial conditions
q0 = 0
qdot0 = 0
z = np.zeros(N+1, dtype='complex')
p = np.zeros(N+1, dtype='complex')
Q = np.zeros(2, dtype='complex')
           
zSnap = np.zeros((nX, nSnap), dtype ='complex')
pSnap = np.zeros((nX, nSnap), dtype ='complex')
QSnap = np.zeros((2, nSnap), dtype='complex')
hSaved = np.zeros(nT)

nSnap = min(nSnap, nT)
tSnap  = np.linspace(T-nSnap*dt,T, nSnap)

cpt1 = 0

# ===============================================================1==============
# Operators definition
# https://www.overleaf.com/read/nrkktcmttkzx
# =============================================================================

# =============================================================================
# Membrane operators
# =============================================================================
diag0 = (eps + 1j*omeg(xS))
L_OH = sp.diags(diag0, 0, dtype='complex', shape=(N+1,N+1))

# Main diagonals
diag0 = -2*np.ones(N+1, dtype='complex')
diag1 = np.ones(N, dtype='complex')
diagM1 = np.ones(N, dtype='complex')

#Boundary conditions : 
diag0[0] = -1 #z[-1] - z[0] = 0 => DxxZ[0] = (z[1] - z[0])/dx**2
diag1[0] = 1  
diag0[-1] = -1 #z[end+1] - z[end] = 0 => DxxZ[-1] = (z[-2] - z[-1])/dx**2
diagM1[-1] = 1

diagonals = [diag0, diag1, diagM1]

DxxZ = sp.diags(diagonals, [0,1,-1], dtype='complex', shape=(N+1,N+1))
DxxZ = (k+1j*kP)/dx**2*DxxZ

Lz = L_OH + DxxZ
Iz = sp.eye(N+1, dtype='complex')

LUz = sla.splu(Iz- dt/2*Lz)

# =============================================================================
# Pressure operator
# =============================================================================
# Main diagonals
diag0 = -2*np.ones(N+1, dtype='complex')
diag1 = np.ones(N, dtype='complex')
diagM1 = np.ones(N, dtype='complex')

#Boundary conditions : 
diag0[0] = -1
diag1[0] = 1  
diag0[-1] = 1
diagM1[-1] = 0

diagonals = [diag0, diag1, diagM1]
Delta = sp.diags(diagonals, [0,1,-1], dtype='complex', shape=(N+1,N+1))
Delta = l/2/rho*Delta/dx**2

# F

def fhS(hS, uS):
    res = np.zeros(N+1)
    res += eps*hS -omeg(xS)*uS - beta*abs(hS + 1j*uS)**2*hS
    res[1:N] += k*(hS[2:N+1] + hS[0:N-1] - 2*hS[1:N])/dx**2
    res[1:N] += -kP*(uS[2:N+1] + uS[0:N-1] - 2*uS[1:N])/dx**2
    res[0] += k*(hS[1]-hS[0])/dx**2
    res[0] += -kP*(uS[1]-uS[0])/dx**2
    res[N] += k*(hS[N-1]-hS[N])/dx**2
    res[N] += -kP*(uS[N-1]-uS[N])/dx**2
    return res

def fuS0(hS, uS):
    res = np.zeros(N+1)
    res += eps*uS + omeg(xS)*hS - beta*abs(uS + 1j*hS)**2*uS
    res[1:N] += k*(uS[2:N+1] + uS[0:N-1] - 2*uS[1:N])/dx**2
    res[1:N] += kP*(hS[2:N+1] + hS[0:N-1] - 2*hS[1:N])/dx**2
    res[0] += k*(uS[1]-uS[0])/dx**2
    res[0] += kP*(hS[1]-hS[0])/dx**2
    res[N] += k*(uS[N-1]-uS[N])/dx**2
    res[N] += kP*(hS[N-1]-hS[N])/dx**2
    return res

def DhFhS(hS, uS):
    tmp = k/dx**2*np.ones(N)

    diag0 = eps*np.ones(N+1)
    diag0 +=  -beta*(3*hS**2 + uS**2)
    diag0[1:-1] += -2*tmp[1:]
    diag0[0] += -k/dx**2
    diag0[-1] += -k/dx**2
    
    diag1 = tmp
    diagM1 = tmp
    
    diagonals = [diag0, diag1, diagM1]
    res = sp.diags(diagonals, [0,1,-1], dtype='complex', shape=(N+1,N+1))
    return res

def DuFhS(hS, uS):
    tmp = -kP/dx**2*np.ones(N)

    diag0 = - omeg(xS)
    diag0 += -2*beta*hS*uS

    diag0[1:-1] += -2*tmp[1:]
    diag0[0] += kP/dx**2
    diag0[-1] += kP/dx**2

    diag1 = tmp
    diagM1 = tmp  
    
    diagonals = [diag0, diag1, diagM1]
    res = sp.diags(diagonals, [0,1,-1], dtype='complex', shape=(N+1,N+1))
    return res

# =============================================================================
# Coupling pressure-membrane
# =============================================================================
diag0 = np.ones((N+1), dtype='complex')
diag0[0] = 0
diag0[-1] = 0
diagonals = [diag0]
P_z_F =sp.diags(diagonals, [0], dtype='complex')

# =============================================================================
# Tympan
# =============================================================================
diag0 = np.array([0,-gamma/m], dtype='complex')
diag1 = np.array([1], dtype='complex')
diagM1 = np.array([-omegOw**2], dtype='complex')
diagonals = [diag0, diag1, diagM1]

L_Ty = sp.diags(diagonals, [0,1,-1], dtype='complex', shape=(2,2))
I2 = sp.eye(2, dtype='complex')

LUTy = sla.splu(I2-dt*L_Ty/2)

# =============================================================================
# Coupling tympan-pressure
# =============================================================================
diagM1 = np.array([-sOw/m], dtype='complex')

diagonals = [diagM1]
R_dtQ_p = sp.diags(diagonals, [-1], dtype='complex', shape=(2,N+1))

vec_dtQ_p = np.zeros(N+1, dtype='complex')
vec_dtQ_p[0] = -sOw/m

R_dtQ_pIn = np.array([0, Gamma*sTy/m], dtype='complex')

R_DxxP_dtQ = sp.diags([np.array([-sOw/b/dx])], [1], dtype='complex', shape=(N+1,2))

diag0 = np.ones((N+1))
diag0[0] = 0
diag0[-1] = 0

R_DxxP_F = sp.diags([diag0], [0], dtype='complex', shape=((N+1, N+1)))


# =============================================================================
# Integration using a semi-Crank-Nicolson  method
# Euler for complicated terms
# CN for simple non-linear terms
# =============================================================================

for iT in tqdm(range(len(tS)-1)):
    if tSnap[cpt1] <= tS[iT] :
        zSnap[:, cpt1] = z[::dnX]
        pSnap[:,cpt1] = p[::dnX]
        QSnap[:,cpt1] = Q
        cpt1 +=1
    t = tS[iT]
    h = np.real(z)
    u = np.imag(z)
    
    hSaved[iT] = h[2402]
    
    DxxP = Delta + 1/alpha*P_z_F.dot(DuFhS(h,u))

    LUDxxP = sla.splu(DxxP)
    
    Iq = I2 - R_dtQ_p.dot(LUDxxP.solve(R_DxxP_dtQ.toarray()))
    LUIq = sla.splu(Iq)
    
    F = DhFhS(h,u).dot(fhS(h,u)) + DuFhS(h,u).dot((fuS0(h,u)))
    

    Y = sla.spsolve(DxxP.T,vec_dtQ_p)
    Y[0] = 0
    Y[-1] = 0
    R_dtQ_F = np.zeros((2,N+1))
    R_dtQ_F[1,:] = Y
        
    dtQ = LUIq.solve(L_Ty.dot(Q) + R_dtQ_F.dot(F) + R_dtQ_pIn*pIn(t))
    
    p = LUDxxP.solve(R_DxxP_F.dot(F) + R_DxxP_dtQ.dot(dtQ))
    
    Q += dt*(R_dtQ_p.dot(p) + R_dtQ_pIn*pIn(t))
    Q = LUTy.solve((I2 + dt*L_Ty/2).dot(Q))
    
    z+= -dt*1j/alpha*p - dt*beta*np.abs(z)**2*z
    z = LUz.solve((Iz + dt/2*Lz).dot(z))
    

# %%
# =============================================================================
# Plots
# =============================================================================
plt.close('all')

n0 = 0

xDisp = xSnap[n0:]
zDisp = zSnap[n0:,:]
pDisp = pSnap[n0:,:]

nX = len(xSnap)
TT,XX = np.meshgrid(tSnap, xDisp)
# TT, XX = np.meshgrid(tSnap, np.linspace(0, nX-1, nX))
data = np.real(zDisp)


with open('objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([tSnap, xDisp, zDisp, pDisp, QSnap], f)
    
plt.figure(1)
plt.subplot(211)
plt.plot(tS, pIn(tS))
plt.title('Pressure as input')
plt.subplot(212)
plt.plot(tS,hSaved)
plt.title('Response at tuned oscillator')

fig = plt.figure(2)
# ax = plt.axes(projection='3d')
# ax.plot_surface(TT,XX, data, cmap='viridis', edgecolor='none')
# ax.set_title('h')
# ax.set_xlabel('t')
# ax.set_ylabel('x')
# ax.set_zlabel('h')
im = plt.pcolor(XX, TT,data) # drawing the function
plt.title('h')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')

fig = plt.figure(3)
# ax = plt.axes(projection='3d')
# ax.plot_surface(TT,XX, np.real(pDisp), cmap='viridis', edgecolor='none')
# im = plt.imshow(np.real(pDisp),cmap=plt.cm.RdBu) # drawing the function
im = plt.pcolor(XX, TT,np.real(pDisp)) # drawing the function
plt.title('p')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
# ax.set_xlabel('t')
# ax.set_ylabel('x')
# ax.set_zlabel('p')

plt.figure(4)
plt.subplot(211)
plt.plot(tSnap, QSnap[0,:])
plt.title('q')
plt.subplot(212)
plt.plot(tSnap, QSnap[1,:])
plt.title('qdot')

plt.figure(5)
plt.plot(xSnap, data[:,-60])

plt.show()

# %%
#amps = sg.hilbert(hSaved)
#plt.figure(6)
#plt.plot(tS, amps)
#plt.close('all')
#plt.show()