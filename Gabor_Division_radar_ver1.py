### Phase Updating Loop in GDSS
### Algorithm 2 in SITA2022 
### Written by Yutaka Jitsumatsu 
### last edit 2022 Nov. 26 

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
from scipy.stats import rankdata

Loop_MAX = 20 # Max number of Loops of PUL 
snr_in_dB = 20; 
snr = 10**(snr_in_dB/10);

L = 1024 # the number of samples 
dt = 1/np.sqrt(L) # sampling interval (T_s) 
df = 1/np.sqrt(L) # discrete frequency = the inverse of the period T = L T_s
# Note: dt and df are set equal because of symmetry.
# The np.fft is computed with norm = "ortho"option because of symmetry.

Tc = 1/dt # the number of samples in one chip time
Fc = 1/df # the number of samples in one subcarrier spacing

###########################################################################
# Every signal in this simulation is discrete and periodic with sequence length L
# In frequency, it is well-known that X[L-i] implies X[-i]. 
# Remind that, in time domain, x[L-i] implies x[-i] as well. 
###########################################################################


Nt = 16 # code length in time domain (N in the paper)
Nf = 16 # code length in frequency domain (N' in the paper)

t_d_max = L    # maximum delay [0,t_d_max]
f_D_max = int(Nt*Fc/2) # maximum Doppler frequency [-f_D_max, f_D_max]

print("f_D_max is ", f_D_max)

def gauss(t):
    y = np.power(2, 1/4) * np.exp( - np.pi * t**2 )
    return y

def TDcorrelator(r, z, t_d_hat):
    # r is the received signal of length L
    # z is the pulse or template waveform of length L
    # t_d_hat is the current estimation of the delay

    N = len(r)
    c = r*np.roll(z.conjugate(), t_d_hat)
    C = np.fft.fft(c,norm="ortho")
    
    i =np.arange(N)
    C = np.dot( C.T, np.diag( np.exp( -1j*np.pi/N * i * t_d_hat ) ) )
    
    return C
    
def FDcorrelator(R, Z, f_D_hat):
    # R is the FT (Fourier transform of received signal of length L
    # Z is the FT of pulse or template waveform of length L
    # f_D_hat is the current estimation of the Doppler shift

    N = len(R)
    C = R * np.roll(Z.conjugate(), f_D_hat)
    c = np.fft.ifft(C, norm = "ortho")
    
    i = np.arange(N)
    c = np.dot ( c.T, np.diag ( np.exp( 1j * np.pi/N * i * f_D_hat) ) )
  
    return c
    

# definition of a pulse 
# the pulse is centered at i=0. 
i = np.arange(L)
z = gauss( ( Nt / Nf ) * i * dt ) + gauss( ( Nt / Nf ) * (i-L) * dt )
Z = np.fft.fft(z, norm = "ortho")


# Spreading Sequences　{+1, -1}-valued sequence
# Using Hadamard matrix is an option. 
# It is observed that Hadamard matrix does not have good performance numerically.
# An example for Nt = Nf = 16 case.
# X = np.array([\
#  [-1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1, -1,  1],\
#  [-1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1,  1,  1,  1,  1],\
#  [ 1,  1,  1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1],\
#  [ 1,  1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1, -1,  1, -1],\
#  [ 1, -1, -1, -1,  1,  1, -1,  1,  1, -1, -1, -1, -1,  1,  1,  1],\
#  [ 1,  1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1],\
#  [-1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1, -1,  1, -1,  1,  1],\
#  [ 1,  1,  1, -1, -1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1],\
#  [ 1, -1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1, -1,  1,  1,  1],\
#  [ 1,  1,  1,  1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1,  1],\
#  [ 1, -1, -1,  1,  1, -1, -1,  1, -1, -1,  1,  1,  1,  1, -1,  1],\
#  [-1, -1,  1, -1,  1, -1,  1,  1, -1,  1, -1,  1, -1,  1, -1, -1],\
#  [-1 , 1, -1,  1,  1, -1, -1,  1,  1, -1,  1, -1, -1,  1,  1, -1],\
#  [ 1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1,  1,  1,  1,  1],\
#  [-1, -1, -1, -1,  1,  1,  1, -1,  1,  1, -1, -1, -1,  1, -1, -1],\
#  [ 1,  1, -1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1,  1,  1]] )
# X = hadamard(Nt,dtype=int)
X = np.random.randint(2, size=[Nt, Nf]) * 2-1 
    
print(X)


# B and b are rectangular shape DFT matrix.
B = [ [ np.exp( -2j * np.pi * n * Nf/Nt * i * df) for i in range (L) ] for n in range(Nt) ]
B = np.array(B)
b = [ [ np.exp(  2j * np.pi * m * Nt/Nf * i * dt) for i in range (L) ] for m in range(Nf) ]
b = np.array(b)

# generation of TD templates
uTD = np.zeros((Nf, L), dtype="complex64") # TD template  
UTD = np.zeros((Nf, L), dtype="complex64") # FT of TD template  

XB = np.dot(X.T, B) / Nt
# print(B.shape)
UTD = np.dot( XB , np.diag(Z) )
uTD = np.fft.ifft(UTD, norm="ortho")

uTD = b * uTD
UTD = np.fft.fft(uTD, norm="ortho")
v1 = np.sum(uTD, axis=0) / Nf
V1 = np.fft.fft(v1, norm="ortho")
# print(v1.shape)


# generation of FD templates
uFD = np.zeros((Nt, L), dtype="complex64") # FD template  
UFD = np.zeros((Nt, L), dtype="complex64") # FT of FD template  

Xb = np.dot(X, b) / Nf
# print(B.shape)
uFD = np.dot( Xb, np.diag(z) )
UFD = np.fft.fft(uFD, norm="ortho")

UFD = B * UFD 
uFD = np.fft.ifft(UFD, norm="ortho")
V2 = np.sum(UFD, axis=0) / Nt
v2 = np.fft.ifft(V2, norm="ortho")

# Check the correctness of the computation.
# print("difference is", np.linalg.norm(v1-v2))
# plt.plot(i, np.abs(v2) )


### channel 
t_d = np.random.randint(t_d_max)  # random delay
f_D = np.random.randint(-f_D_max, f_D_max) # random Doppler frequency 
phase = np.random.rand()*2*np.pi # random phase
# amplitude of the received signal is fixed to be unity.
print([t_d, f_D])


# additive noise 
w = np.random.randn(L) / np.sqrt( snr * L * dt)
r = v1 + w

# generation of the received signal
# A single path model is assumed.  
# You can change the channel model here.

r = np.roll(r, t_d) # delay
R = np.fft.fft(r, norm="ortho")
R = np.roll(R, f_D) # Doppler shift
R = R * np.exp(1j*phase) # phase shift
r = np.fft.ifft(R, norm="ortho")

fig1 = plt.figure()
plt.plot(i, abs(r))

#################################
#   Phase Updating Loop Nt = Nf
#################################

f_D_hat = np.zeros( (Loop_MAX, Nt), dtype = int)
t_d_hat = np.zeros( (Loop_MAX, Nf), dtype = int)
epsilon = 1e-4

CFD = np.zeros(Nt) 
cTD = np.zeros(Nf) 

n = np.arange(Nt)
f_D_hat[0] = ( n - (Nt - 1) / 2 ) * Fc; 
# Initial guess for f_D for n-th FD correlator.  
# Each correlator has differenct estimation so that 
# one of the correlator can obtain the correct t_d.


i = 0; itr = Loop_MAX-1;
while i < Loop_MAX - 1:
    for n in range(Nt): 
        CFD[n] = np.max ( abs ( FDcorrelator(R, UFD[n], f_D_hat[i][n] ) ) )
        # print('CFD[{a}]={b}'.format( a=n, b=CFD[n] ))
        t_d_hat[i + 1][n] = np.argmax ( abs ( FDcorrelator(R, UFD[n], f_D_hat[i][n] ) ) )
    rnk = rankdata(CFD) # the index of ranking of CFD 
    upper = np.where(rnk>Nt//2); lower = np.where(rnk<=Nt//2) 
    # replace the t_d_hat of lower ranking with that of upper ranking.
    print("t_d_hat from correlator output ",t_d_hat[i+1])
    t_d_hat[i + 1][lower] = t_d_hat[i + 1][upper]; 
    # t_d_hat[i+1] = t_d_hat[i + 1][np.argmax(CFD)] # replace with the t_d_hat of maximum CFD 
    print("Half of estimation are replaced", t_d_hat[i+1])
    for m in range(Nf):
        cTD[m] = np.max ( abs ( TDcorrelator(r, uTD[m], t_d_hat[i+1][m] ) ) )
        #print('cTD[{a}]={b}'.format( a=m, b=cTD[m] ))        
        f_D_hat[i + 1][m] = np.argmax ( abs ( TDcorrelator(r, uTD[m], t_d_hat[i+1][m] ) ) )
        if f_D_hat[i + 1][m] > L//2:
            f_D_hat[i + 1][m] -= L
    rnk = rankdata(cTD) # the index of ranking of CFD 
    upper = np.where(rnk>Nf//2); lower = np.where(rnk<=Nf//2) 
    # replace the t_d_hat of lower ranking with that of upper ranking.
    print("f_D_hat from correlator output ",f_D_hat[i+1])
    f_D_hat[i + 1][lower] = f_D_hat[i + 1][upper]; 
    print("Half of estimation are replaced",f_D_hat[i+1])
    #f_D_hat[i+1] = f_D_hat[i + 1][np.argmax(cTD)]

    i = i + 1
    print(np.linalg.norm( t_d_hat[i] - t_d_hat[i-1]) , np.linalg.norm( f_D_hat[i] - f_D_hat[i-1] ))
    if np.linalg.norm( t_d_hat[i] - t_d_hat[i-1]) < 1 and np.linalg.norm( f_D_hat[i] - f_D_hat[i-1] ) < 1:
        itr = i-1;
        break


#print("t_d_hat=", t_d_hat)
#print("f_D_hat=", f_D_hat)
print("iteration = ", itr)

print("TD correlator outputs", cTD)
print("FD correlator outputs", CFD)

print("[delay,Doppler]=", t_d, f_D)

arr = np.array([ t_d_hat[itr], f_D_hat[itr]]).T
unique, freq = np.unique(arr, return_counts=True, axis=0)
print("The estimate is", unique[ np.argmax( freq ) ] )
print("Estimated by", np.max( freq ), "out of ", Nt, "correlators")


# roll_Z = np.roll(Z, L//2)
# fig1 = plt.figure()
# plt.plot(i,abs(roll_Z) )





