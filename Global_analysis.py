#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 11:00:02 2020

@author: darrenhsu
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
#%% 
def Gaussian(x, nu, sigma):
    return np.exp(- (x-nu) * (x-nu) / 2 / sigma / sigma)

def Autocorrelation(u, step=1, len_s=None):
    if len_s == None:
        len_s = len(u)
    return np.sum(u[step:]*u[:-step],0)[:len_s]

def SVDplot(data,x,t,xMinMax=None,tMinMax=None,n_cmp=None):
    print(data.shape)

    if tMinMax is not None:
        tSel = (t >= tMinMax[0]) & (t <= tMinMax[1])
        t = t[tSel]
        data = data[:,tSel]
    if xMinMax is not None:
        xSel = (x >= xMinMax[0]) & (x <= xMinMax[1])
        x = x[xSel]
        data = data[xSel]

    [u, s, v] = np.linalg.svd(data);
    plt.figure(figsize=(7,5))
    plt.subplot(1,2,1)
    plt.title('Data')
    plt.plot(x, data)
    ax1 = plt.subplot(3,2,2)
    plt.plot(s,'-o',zorder=10)
    plt.yscale('log')
    ax2 = ax1.twinx()
    plt.plot(Autocorrelation(u,len_s=len(s)),'-rx')
    if n_cmp == None:
        n_cmp = np.sum(Autocorrelation(u,len_s=len(s)) > 0.8)
    plt.title('S and autocorr')
    plt.subplot(3,2,4)
    plt.plot(x, u[:,:n_cmp])
    plt.title('U')
    plt.subplot(3,2,6)
    plt.plot(t, v[:n_cmp].T)
    plt.title('V')
    
def SVD_GlobalAnalysis(x, t, data, xMinMax, tMinMax, i_cmp, funVec=None, initialGuess=None, fitFlag=True, plot_flag=False):
    if (funVec is None) or (initialGuess is None):
        raise ValueError()
        
    xSel = (x >= xMinMax[0]) & (x <= xMinMax[1])
    tSel = (t >= tMinMax[0]) & (t <= tMinMax[1])
    xROI = x[xSel]
    tROI = t[tSel]
    data = data[:,tSel]
    dataROI = data[xSel]
    #dataROI = data[xSel][:,tSel]
    [u, s, v] = np.linalg.svd(dataROI);
    n_cmp = len(i_cmp);
    
    SVDplot(dataROI, xROI, tROI)
    
    funVecTest = funVec(initialGuess, tROI).T;
    n_input  = funVecTest.shape[1];
    n_par = len(initialGuess);
    n_coef = n_input*n_cmp
    
    def Rfun(a):
        #print("a is")
        #print(a)
        return np.reshape(a[:n_coef], (n_input, n_cmp), 'C');
    def vFitFun(a, tvar): 
        #print(np.dot(funVec(a[:n_par], tROI).T, Rfun(a[n_par:n_par+n_coef])))
        return np.dot(funVec(a[:n_par], tROI).T, Rfun(a[n_par:n_par+n_coef]))
    def DataFitFun(a, tvar): 
        #print(np.dot(np.dot(u[:,i_cmp], s[i_cmp])[:,np.newaxis], vFitFun(a, tvar).T).shape)
        return np.dot(np.dot(u[:,i_cmp], np.diag(s[i_cmp])), vFitFun(a, tROI).T)
    
    initialGuess = np.array(initialGuess)
    
    RinitialGuess = np.linalg.lstsq(funVec(initialGuess, tROI).T,v[i_cmp,:].T, rcond=None)
    RinitialGuess = RinitialGuess[0].flatten()
    
    fullInitialGuess = np.hstack((initialGuess, RinitialGuess))
    
    numel = dataROI.size
    y_reshape = np.reshape(dataROI, numel, 'C')
    
    def fun_reshape(xvar, *args):
        a = np.array(args)
        return np.reshape(DataFitFun(a, xvar), numel, 'C')
    
    fullFinalGuess, pcov = curve_fit(fun_reshape, tROI, y_reshape, p0 = fullInitialGuess)
    
    #print(fullFinalGuess)
    perr = np.sqrt(np.diag(pcov))
    #print(perr)
    
    finalGuess = fullFinalGuess[:n_par]
    finalGuessErr = perr[:n_par]
    print(finalGuess)
    print(finalGuessErr)
    
    SAD = np.dot(np.dot(u[:,i_cmp], np.diag(s[i_cmp])), Rfun(fullFinalGuess[n_par:n_coef+n_par]).T)
    SAK = np.linalg.lstsq(Rfun(fullFinalGuess[n_par:n_coef+n_par]).T, v[i_cmp,:],rcond=None)[0]

    if plot_flag:
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(xROI, SAD)
        plt.subplot(2,1,2)
        plt.plot(tROI, SAK.T)
        
        plt.figure()
        plt.subplot(1,2,1)
        plt.title('Data')
        offset = 0
        for i in np.arange(0,len(tROI),3):
            plt.plot(xROI, dataROI[:,i] + offset)
            offset -= np.sqrt(np.sum(np.abs(dataROI[:,i]))/len(xROI))
        plt.subplot(2,2,2)
        plt.plot(xROI, SAD)
        plt.subplot(2,2,4)
        offset = 0
        for i in np.arange(len(i_cmp)):
            plt.plot(tROI, SAK[i] + offset,'x')
            plt.plot(tROI, funVec(finalGuess,tROI)[i] + offset,'r')
            #offset -= np.max(SAK[i])
    return xROI, SAD, tROI, SAK, finalGuess, perr
    
    
    
#%% Simulating data

x = np.linspace(400,700,301) # Wavelength
t, dt = np.linspace(0, 200, 2001, retstep=True) # Time


s1 = Gaussian(x, 450, 20)
s2 = Gaussian(x, 480, 35)
s3 = Gaussian(x, 520, 40)
spec = np.vstack((s1,s2,s3)).T
c0 = np.array([1,0,0])
#spec = np.vstack((s1,s2)).T
#c0 = np.array([1,0])

k1 = 2
k2 = 120

c_mat = np.array([[-1/k1, 0, 0],
                  [1/k1, -1/k2, 0],
                  [0, 1/k2, 0]])

#c_mat = np.array([[-1/k1, 0], 
#                  [1/k1,  0]])


cpop = np.zeros((len(t),len(spec.T)))

cpop[0] = c0

for i in np.arange(1,len(t)):
    cpop[i] = cpop[i-1] + np.dot(c_mat,cpop[i-1]) * dt
    
#%
plt.figure()
plt.subplot(2,1,1)
plt.plot(x, np.dot(spec, cpop.T))
plt.subplot(2,1,2)
plt.plot(t, cpop)
#% Data prep
sparse_sel = np.arange(1,len(t),30)
data_t = t[sparse_sel]
data = np.dot(spec, cpop.T)[:,sparse_sel]
data = (data.T - data[:,0]).T
data += np.random.normal(size=data.shape) * 0.1

#% Plot data
plt.figure()
plt.plot(x, data)

#% SVD analysis
u,s,v = np.linalg.svd(data)

SVDplot(data,x,data_t)

#%

xMinMax = [400,650]
tMinMax = [0.1, 1000]
i_cmp = [0,1]

def funVec(a, tvar):
    return np.array([np.exp(-tvar/a[1]) - np.exp(-tvar/a[0]), 1 - np.exp(-tvar/a[1])])

initialGuess = [5, 40]



xROI, SAD, tROI, SAK, finalGuess, perr =  SVD_GlobalAnalysis(x, data_t, data, xMinMax, tMinMax, i_cmp, funVec, initialGuess, True, True)

#%%

xSel = (x >= xMinMax[0]) & (x <= xMinMax[1])
tSel = (data_t >= tMinMax[0]) & (data_t <= tMinMax[1])
xROI = x[xSel]
tROI = data_t[tSel]
dataROI = data[xSel][:,tSel]
[u, s, v] = np.linalg.svd(dataROI);
n_cmp = len(i_cmp);

#SVDplot(dataROI, xROI, tROI)

funVecTest = funVec(initialGuess, tROI).T;
n_input  = funVecTest.shape[1];
n_par = len(initialGuess);
n_coef = n_input*n_cmp

def Rfun(a):
    #print("a is")
    #print(a)
    return np.reshape(a[:n_coef], (n_input, n_cmp), 'C');
def vFitFun(a, tvar): 
    #print(np.dot(funVec(a[:n_par], tROI).T, Rfun(a[n_par:n_par+n_coef])))
    return np.dot(funVec(a[:n_par], tROI).T, Rfun(a[n_par:n_par+n_coef]))
def DataFitFun(a, tvar): 
    #print(np.dot(np.dot(u[:,i_cmp], s[i_cmp])[:,np.newaxis], vFitFun(a, tvar).T).shape)
    return np.dot(np.dot(u[:,i_cmp], np.diag(s[i_cmp])), vFitFun(a, tROI).T)

initialGuess = np.array(initialGuess)

RinitialGuess = np.linalg.lstsq(funVec(initialGuess, tROI).T,v[i_cmp,:].T, rcond=None)
RinitialGuess = RinitialGuess[0].flatten()

fullInitialGuess = np.hstack((initialGuess, RinitialGuess))

numel = dataROI.size
y_reshape = np.reshape(dataROI, numel, 'C')

def fun_reshape(xvar, *args):
    a = np.array(args)
    return np.reshape(DataFitFun(a, xvar), numel, 'C')

fullFinalGuess, pcov = curve_fit(fun_reshape, tROI, y_reshape, p0 = fullInitialGuess)

print(fullFinalGuess)
perr = np.sqrt(np.diag(pcov))
print(perr)

finalGuess = fullFinalGuess[:n_par]
finalGuessErr = perr[:n_par]

SAD = np.dot(np.dot(u[:,i_cmp], np.diag(s[i_cmp])), Rfun(fullFinalGuess[n_par:n_coef+n_par]).T)
SAK = np.linalg.lstsq(Rfun(fullFinalGuess[n_par:n_coef+n_par]).T, v[i_cmp,:],rcond=None)[0]

plt.figure()
plt.subplot(2,1,1)
plt.plot(xROI, SAD+s1[xSel,None])
plt.subplot(2,1,2)
plt.plot(tROI, SAK.T)
#np.u[:,i_cmp]s[i_cmp]*Rfun(fullFinalGuess([1:n_coef]+n_par))';
#SAK = v(:,i_cmp)/Rfun(fullFinalGuess([1:n_coef]+n_par));


#SVD_GlobalAnalysis(x, data_t, data, xMinMax, tMinMax, i_cmp, funVec, initialGuess, 1)


#%% Try this on Nate's data

strk_data = np.loadtxt(fname = r"/Users/darrenhsu/Downloads/Nate/COF5_4mmol_half_Jittercorrected.dat", delimiter = '\t')

#%% Binning
raw_w_len = len(strk_data[0,:])
raw_t_len = len(strk_data[:,0])

w_bin = 3
t_bin = 3

w_length = int(raw_w_len/w_bin)
t_length = int(raw_t_len/t_bin)

print(w_length)
print(t_length)

w_bin_array = np.zeros((raw_t_len,w_length))

sum = 0

for c in range(raw_t_len): #looping thru all raw time indices
    for b in range(w_length):  #looping thru a binned wavelengths indices in the first new array
        sum = 0   #restting the sum
        for d in range((b*w_bin),((b+1)*w_bin)): #looping thru and summing the correct bin range of the raw array wavelength intensities
            sum += strk_data[c,d]  #adding all the values to the sum
        w_bin_array[c,b] = sum/w_bin #filling the first new array

#plt.matshow(w_bin_array)

final_bin_array = np.zeros((t_length,w_length))

sum = 0

for c in range(w_length): #looping thru all binned wavelength indices
    for b in range(t_length):  #looping thru a binned time indices in the second new array
        sum = 0   #restting the sum
        for d in range((b*t_bin),((b+1)*t_bin)): #looping thru and summing the correct bin range of the raw array time intensities
            sum += w_bin_array[d,c]  #adding all the values to the sum
        final_bin_array[b,c] = sum/t_bin #filling the final array

plt.matshow(final_bin_array)

flipped_data = np.fliplr(final_bin_array)

w_initial = 274.7715
w_final = 752.3447
t_initial = 2.344787
t_final = 123.2613

w_axis = np.linspace(w_initial, w_final, w_length)
t_axis = np.linspace(t_initial, t_final, t_length)
                
print(len(w_axis))
print(len(final_bin_array[0,:]))

#%%
wminmax = [400,600]
tminmax = [3,120]
i_cmp = [0,1]

SVDplot(final_bin_array.T, w_axis, t_axis, wminmax, tminmax)
#%%
def funVec2(a, tvar):
    return np.array([
        np.exp(-(tvar-a[2])*(a[0]+a[1])) * (1+erf((tvar-a[2])/a[3])) / 2, 
        (np.exp(-(tvar-a[2])*a[1]) - np.exp(-(tvar-a[2])*(a[0]))) * (1+erf((tvar-a[2])/a[3])) / 2
        ])
#%%
initialGuess = [1/10,1/100,19,3]
SVD_GlobalAnalysis(w_axis, t_axis, final_bin_array.T, wminmax, tminmax, i_cmp, funVec2, initialGuess=initialGuess, plot_flag=True)
