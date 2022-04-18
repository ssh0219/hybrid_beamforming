#import tensorflow as tf
import numpy as np
#import os
import math
import cmath
import scipy.io as sio
#from tensorflow.examples.tutorials.mnist import input_data
#import matplotlib.pyplot as plt


Rt, Nt, Rr, Nr, N, K, snr = 8, 64, 4, 8, 3, 2, 30
layer, batch_size, H_num, I_max = 10, 20, 1, 100
p_ = np.random.normal(loc=0.0, scale=math.sqrt(0.5), size=(2*K*Rt,N))
#data = sio.loadmat('p_.mat')
#p_ = data['p_']
p = p_[0:K*Rt,:]+1j*p_[K*Rt:,:]

w_ = np.random.normal(loc=0.0, scale=math.sqrt(0.5), size=(2*K*Rr,N))
#data = sio.loadmat('w_.mat')
#w_ = data['w_']
w = w_[0:K*Rr,:]+1j*w_[K*Rr:,:]

thef = np.random.uniform(size=(Nt,Rt))+1j*np.zeros((Nt,Rt))
F = np.exp(1j*thef)

theu = np.random.uniform(size=(Rr*K,Nr))+1j*np.zeros((Rr*K,Nr))
U = np.exp(1j*theu)

power_n = math.pow(10,-snr/10)
'''
n = np.zeros((Nr,K))+1j*np.zeros((Nr,K))
for k in range(0,K):
    for i in range(0,Nr):
        n[i,k] = np.random.normal(loc=0.0, scale=math.sqrt(power_n*0.5), size=(1,1)) + 1j*np.random.normal(loc=0.0, scale=math.sqrt(power_n*0.5), size=(1,1))
'''
data = sio.loadmat('n.mat')
n = data['n']

b = np.sign(np.random.uniform(-1, 1, (K*N,batch_size))) + 1j*np.sign(np.random.uniform(-1, 1, (K*N,batch_size)))

#H_ = np.random.normal(loc=0.0, scale=math.sqrt(0.5), size=(2*K*Rr,Rt))
#H = H_[0:K*Rr,:]+1j*H_[K*Rr:,:]
data = sio.loadmat('H.mat')
H = data['H']
'''
sim_num=H_num
N_t=Nr*K
g_xindao_mem=np.zeros((N_t,Nt))+1j*np.zeros((N_t,Nt))
for sim_ind in range(0, sim_num):
    ##channel
    Lc=6
    Lr=8
    phi_t=np.pi*(np.random.uniform(0, 1, (1,Lc))-1/2)
    phi_r=np.pi*(np.random.uniform(0, 1, (1,Lc))-1/2)
    
    Lc=6
    Lr=8
    g=np.zeros((N_t,Nt))+1j*np.zeros((N_t,Nt))
    a_t = np.zeros((16,Lc))+1j*np.zeros((16,Lc))
    a_r = np.zeros((1,Lc))
    alpha_g = np.zeros((Lc,Nt))+1j*np.zeros((Lc,Nt))
    for k in range(0,Nt):
        phi_r=np.pi*(np.random.uniform(0, 1, (1,Lc))-1/2)
        for lc in range(0,Lc):
            for lr in range(0,Lr):
                phi_t_lr=np.random.normal(loc=0.0, scale=1.0, size=(1,1))*(7.5/180)*np.pi
                for i in range(0,16):
                    a_t[i,lc]=1/np.sqrt(N_t)*np.exp(1j*np.pi*i*np.sin(phi_t[0,lc]+phi_t_lr))[0,0]  #omp1中ULA有exp(j*k*d*nt*sin(phi))，k应该是2*pi/mu其中mu是波长，d是天线间距离，取mu/2，则kd=pi
                #a_t(:,lc)=1/sqrt(N_t)*exp(1j*np.pi*(0:N_t-1)*(phi_t(lc)+phi_t_lr));  #special
                a_r[:,lc]=1
                alpha_g[lc,k]=(np.random.normal(loc=0.0, scale=1.0, size=(1,1))+1j*np.random.normal(loc=0.0, scale=1.0, size=(1,1)))[0,0]/np.sqrt(2)
                    #alpha_g(lc,k)=exp(-1j*2*np.pi*rand(1));
                #alpha_g(lc,k)=(np.random.normal(loc=0.0, scale=1.0, size=(1,1))+1j*np.random.normal(loc=0.0, scale=1.0, size=(1,1)))/sqrt(2);
                #g(:,k)=g(:,k)+ alpha_g(lc,k)*a_t*a_r';
                g[:,k]=g[:,k]+ alpha_g[lc,k]*a_t[:,lc]*a_r[0,lc]
        #g(:,k)=sqrt(N_t/(Lc*Lr))*g(:,k);
        g[:,k]=np.sqrt(N_t/((1+(Lc-1)/1)*Lr))*g[:,k]
    #h=h_xindao_mem(:,sim_ind);
    #g=g_xindao_mem(:,:,sim_ind);
    
    g_xindao_mem=g

H = g_xindao_mem
'''

lr_p = 0.05*np.ones((K*Rt,N))+1j*np.zeros((K*Rt,N))
lr_w = 0.05*np.ones((K*Rr,N))+1j*np.zeros((K*Rr,N))
lr_theu = 0.05*np.ones((K*Rr,Nr))+1j*np.zeros((K*Rr,Nr))
lr_thef = 0.05*np.ones((Nt,Rt))+1j*np.zeros((Nt,Rt))

error = []
test_num = 5000
test_b = np.sign(np.random.uniform(-1, 1, (K*N,test_num))) + 1j*np.sign(np.random.uniform(-1, 1, (K*N,test_num)))


#rol
sum2 = np.zeros((Rt*K,Rt))+1j*np.zeros(((Rt*K,Rt)))
rol = np.zeros((K,N))+1j*np.zeros((K,N))
for k in range(0,K):
    for d in range(0,N):
        sum2[k*Rt:(k+1)*Rt,:] = sum2[k*Rt:(k+1)*Rt,:]+np.dot(p[k*Rt:(k+1)*Rt,d], np.matrix(p[k*Rt:(k+1)*Rt,d]).H)
for k in range(0,K):
    for d in range(0,N):
        rol[k:(k+1),d:d+1] = cmath.sqrt(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.matrix(w[k*Rr:(k+1)*Rr,d:d+1]).H, 
           U[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]), F), sum2[k*Rt:(k+1)*Rt,:]), np.matrix(F).H), np.matrix(H[k*Nr:(k+1)*Nr,:]).H), 
           np.matrix(U[k*Rr:(k+1)*Rr,:]).H), w[k*Rr:(k+1)*Rr,d:d+1])*power_n)
										


def forward(p, w, U, F, theu, thef, lr_p, lr_w, lr_theu, lr_thef):
    b_hat = np.zeros((K*N,batch_size))+1j*np.zeros((K*N,batch_size))
    y = np.zeros((K*Nr,batch_size))+1j*np.zeros((K*Nr,batch_size))
    delta_p_real = np.zeros((K*Rt,N*batch_size))+1j*np.zeros((K*Rt,N*batch_size))
    delta_w_real = np.zeros((K*Rr,N*batch_size))+1j*np.zeros((K*Rr,N*batch_size))
    delta_p_real_avr = np.zeros((K*Rt,N))+1j*np.zeros((K*Rt,N))
    delta_w_real_avr = np.zeros((K*Rr,N))+1j*np.zeros((K*Rr,N))
    
    delta_p_imag = np.zeros((K*Rt,N*batch_size))+1j*np.zeros((K*Rt,N*batch_size))
    delta_w_imag = np.zeros((K*Rr,N*batch_size))+1j*np.zeros((K*Rr,N*batch_size))
    delta_p_imag_avr = np.zeros((K*Rt,N))+1j*np.zeros((K*Rt,N))
    delta_w_imag_avr = np.zeros((K*Rr,N))+1j*np.zeros((K*Rr,N))
    
    delta_U_real = np.zeros((K*Rr,Nr*N*batch_size))+1j*np.zeros((K*Rr,Nr*N*batch_size))
    delta_F_real = np.zeros((K*Nt,Rt*N*batch_size))+1j*np.zeros((K*Nt,Rt*N*batch_size))
    delta_U_real_avr = np.zeros((K*Rr,Nr))+1j*np.zeros((K*Rr,Nr))
    delta_F_real_avr1 = np.zeros((K*Nt,Rt))+1j*np.zeros((K*Nt,Rt))
    delta_F_real_avr = np.zeros((Nt,Rt))+1j*np.zeros((Nt,Rt))
    
    delta_U_conj_real = np.zeros((K*Rr,Nr*N*batch_size))+1j*np.zeros((K*Rr,Nr*N*batch_size))
    delta_F_conj_real = np.zeros((K*Nt,Rt*N*batch_size))+1j*np.zeros((K*Nt,Rt*N*batch_size))
    delta_U_conj_real_avr = np.zeros((K*Rr,Nr))+1j*np.zeros((K*Rr,Nr))
    delta_F_conj_real_avr1 = np.zeros((K*Nt,Rt))+1j*np.zeros((K*Nt,Rt))
    delta_F_conj_real_avr = np.zeros((Nt,Rt))+1j*np.zeros((Nt,Rt))
    
    delta_U_imag = np.zeros((K*Rr,Nr*N*batch_size))+1j*np.zeros((K*Rr,Nr*N*batch_size))
    delta_F_imag = np.zeros((K*Nt,Rt*N*batch_size))+1j*np.zeros((K*Nt,Rt*N*batch_size))
    delta_U_imag_avr = np.zeros((K*Rr,Nr))+1j*np.zeros((K*Rr,Nr))
    delta_F_imag_avr1 = np.zeros((K*Nt,Rt))+1j*np.zeros((K*Nt,Rt))
    delta_F_imag_avr = np.zeros((Nt,Rt))+1j*np.zeros((Nt,Rt))
    
    delta_U_conj_imag = np.zeros((K*Rr,Nr*N*batch_size))+1j*np.zeros((K*Rr,Nr*N*batch_size))
    delta_F_conj_imag = np.zeros((K*Nt,Rt*N*batch_size))+1j*np.zeros((K*Nt,Rt*N*batch_size))
    delta_U_conj_imag_avr = np.zeros((K*Rr,Nr))+1j*np.zeros((K*Rr,Nr))
    delta_F_conj_imag_avr1 = np.zeros((K*Nt,Rt))+1j*np.zeros((K*Nt,Rt))
    delta_F_conj_imag_avr = np.zeros((Nt,Rt))+1j*np.zeros((Nt,Rt))
    
    delta_p_avr = np.zeros((K*Rt,N))+1j*np.zeros((K*Rt,N))
    delta_w_avr = np.zeros((K*Rr,N))+1j*np.zeros((K*Rr,N))
    delta_theu = np.zeros((K*Rr,Nr))+1j*np.zeros((K*Rr,Nr))
    delta_thef = np.zeros((Nt,Rt))+1j*np.zeros((Nt,Rt))
    delta_U = np.zeros((K*Rr,Nr))+1j*np.zeros((K*Rr,Nr))
    delta_F = np.zeros((Nt,Rt))+1j*np.zeros((Nt,Rt))
    delta_U_conj = np.zeros((K*Rr,Nr))+1j*np.zeros((K*Rr,Nr))
    delta_F_conj = np.zeros((Nt,Rt))+1j*np.zeros((Nt,Rt))
    
    
    sum2_ = np.zeros((Rt*K,Rt))+1j*np.zeros(((Rt*K,Rt)))
    rol_ = np.zeros((K,N))+1j*np.zeros((K,N))
    for k in range(0,K):
        for d in range(0,N):
            sum2_[k*Rt:(k+1)*Rt,:] = sum2_[k*Rt:(k+1)*Rt,:]+np.dot(p[k*Rt:(k+1)*Rt,d], np.matrix(p[k*Rt:(k+1)*Rt,d]).H)
    for k in range(0,K):
        for d in range(0,N):
            rol_[k:(k+1),d:d+1] = cmath.sqrt(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.matrix(w[k*Rr:(k+1)*Rr,d:d+1]).H, 
               U[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]), F), sum2_[k*Rt:(k+1)*Rt,:]), np.matrix(F).H), np.matrix(H[k*Nr:(k+1)*Nr,:]).H), 
               np.matrix(U[k*Rr:(k+1)*Rr,:]).H), w[k*Rr:(k+1)*Rr,d:d+1])*power_n)
    
    #print(rol_.shape)
    #print(rol_)
    
    sum1 = np.zeros((K*Rt,batch_size))+1j*np.zeros((K*Rt,batch_size))
    for i in range(0,batch_size):
        for k in range(0,K):
            for d in range(0,N):
                sum1[k*Rt:(k+1)*Rt,i:(i+1)] = sum1[k*Rt:(k+1)*Rt,i:(i+1)]+np.multiply(p[k*Rt:(k+1)*Rt,d:d+1], b[k*N+d:k*N+d+1,i:(i+1)])
    #print(sum1)
    
    for i in range(0,batch_size):
        for k in range(0,K):
            for d in range(0,N):
                y[k*Nr:(k+1)*Nr,i:i+1] = np.dot(np.dot(H[k*Nr:(k+1)*Nr,:], F), sum1[k*Rt:(k+1)*Rt,i:(i+1)])
                b_hat[k*N+d:k*N+d+1,i:(i+1)] = np.dot(np.dot(np.matrix(w[k*Rr:(k+1)*Rr,d:d+1]).H, U[k*Rr:(k+1)*Rr,:]), y[k*Nr:(k+1)*Nr,i:i+1])
    
    for i in range(0,batch_size):
        for k in range(0,K):
            for d in range(0,N):
                delta_p_real[k*Rt:(k+1)*Rt,i*N+d:i*N+d+1] = np.matrix(b[k*N+d,i].real/(cmath.sqrt(2*3.14*rol_[k,d].real))
                    *math.exp(-math.pow(b_hat[k*N+d,i].real, 2)/(2*math.pow(rol_[k,d].real, 2)))
                    *(-np.matrix(np.dot(np.dot(np.dot(np.matrix(w[k*Rr:(k+1)*Rr,d:d+1]).H, U[k*Rr:(k+1)*Rr,:]), 
                                              H[k*Nr:(k+1)*Nr,:]), F*b[k*N+d,i])).H))
                
                delta_p_imag[k*Rt:(k+1)*Rt,i*N+d:i*N+d+1] = np.matrix(b[k*N+d,i].imag/(cmath.sqrt(2*3.14*rol_[k,d].real))
                    *math.exp(-math.pow(b_hat[k*N+d,i].imag, 2)/(2*math.pow(rol_[k,d].real, 2)))
                    *(-1j)*np.matrix(np.dot(np.dot(np.dot(np.matrix(w[k*Rr:(k+1)*Rr,d:d+1]).H, U[k*Rr:(k+1)*Rr,:]), 
                                                 H[k*Nr:(k+1)*Nr,:]), F*b[k*N+d,i])).H)
    
    delta_p_real_avr = np.mean([delta_p_real[:,i:i+N] for i in range(0,N*batch_size,N)], axis=0, keepdims=True)
    delta_p_imag_avr = np.mean([delta_p_imag[:,i:i+N] for i in range(0,N*batch_size,N)], axis=0, keepdims=True)
    delta_p_avr = delta_p_real_avr + delta_p_imag_avr
    delta_p_avr = np.reshape(delta_p_avr, (K*Rt,N))
    #delta_p_mean = np.abs(np.mean(delta_p_avr, axis=0, keepdims=True))
    #delta_p_nor = delta_p_avr/delta_p_mean
    p = p-np.multiply(lr_p, delta_p_avr)
    #print(delta_p_avr.shape)
    #print(p.shape)
    #p = p-np.multiply(lr_p, delta_p_nor)
    
    
    sum2_ = np.zeros((Rt*K,Rt))+1j*np.zeros(((Rt*K,Rt)))
    rol_ = np.zeros((K,N))+1j*np.zeros((K,N))
    for k in range(0,K):
        for d in range(0,N):
            sum2_[k*Rt:(k+1)*Rt,:] = sum2_[k*Rt:(k+1)*Rt,:]+np.dot(p[k*Rt:(k+1)*Rt,d], np.matrix(p[k*Rt:(k+1)*Rt,d]).H)
    for k in range(0,K):
        for d in range(0,N):
            rol_[k:(k+1),d:d+1] = cmath.sqrt(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.matrix(w[k*Rr:(k+1)*Rr,d:d+1]).H, 
               U[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]), F), sum2_[k*Rt:(k+1)*Rt,:]), np.matrix(F).H), np.matrix(H[k*Nr:(k+1)*Nr,:]).H), 
               np.matrix(U[k*Rr:(k+1)*Rr,:]).H), w[k*Rr:(k+1)*Rr,d:d+1])*power_n)
    
    sum1 = np.zeros((K*Rt,batch_size))+1j*np.zeros((K*Rt,batch_size))
    for i in range(0,batch_size):
        for k in range(0,K):
            for d in range(0,N):
                sum1[k*Rt:(k+1)*Rt,i:(i+1)] = sum1[k*Rt:(k+1)*Rt,i:(i+1)]+np.multiply(p[k*Rt:(k+1)*Rt,d:d+1], b[k*N+d:k*N+d+1,i:(i+1)])
    #print(sum1)
    
    for i in range(0,batch_size):
        for k in range(0,K):
            for d in range(0,N):
                y[k*Nr:(k+1)*Nr,i:i+1] = np.dot(np.dot(H[k*Nr:(k+1)*Nr,:], F), sum1[k*Rt:(k+1)*Rt,i:(i+1)])
                b_hat[k*N+d:k*N+d+1,i:(i+1)] = np.dot(np.dot(np.matrix(w[k*Rr:(k+1)*Rr,d:d+1]).H, U[k*Rr:(k+1)*Rr,:]), y[k*Nr:(k+1)*Nr,i:i+1])
    
    for i in range(0,batch_size):
        for k in range(0,K):
            for d in range(0,N):
                delta_w_real[k*Rr:(k+1)*Rr,i*N+d:i*N+d+1] = np.matrix(b[k*N+d,i].real/(cmath.sqrt(2*3.14*rol_[k,d].real))
                        *math.exp(-math.pow(b_hat[k*N+d,i].real, 2)/(2*math.pow(rol_[k,d].real, 2)))
                        *(-np.dot(U[k*Rr:(k+1)*Rr,:], y[k*Nr:(k+1)*Nr,i:(i+1)])))
                
                delta_w_imag[k*Rr:(k+1)*Rr,i*N+d:i*N+d+1] = np.matrix(b[k*N+d,i].imag/(cmath.sqrt(2*3.14*rol_[k,d].real))
                        *math.exp(-math.pow(b_hat[k*N+d,i].imag, 2)/(2*math.pow(rol_[k,d].real, 2)))
                        *(+1j)*np.dot(U[k*Rr:(k+1)*Rr,:], y[k*Nr:(k+1)*Nr,i:(i+1)]))
    
    delta_w_real_avr = np.mean([delta_w_real[:,i:i+N] for i in range(0,N*batch_size,N)], axis=0, keepdims=True)
    delta_w_imag_avr = np.mean([delta_w_imag[:,i:i+N] for i in range(0,N*batch_size,N)], axis=0, keepdims=True)
    delta_w_avr = delta_w_real_avr + delta_w_imag_avr
    delta_w_avr = np.reshape(delta_w_avr, (K*Rr,N))
    #delta_w_mean = np.abs(np.mean(delta_w_avr, axis=0, keepdims=True))
    #delta_w_nor = delta_w_avr/delta_w_mean
    w = w-np.multiply(lr_w, delta_w_avr)
    #w = w-np.multiply(lr_w, delta_w_nor)
    
    
    sum2_ = np.zeros((Rt*K,Rt))+1j*np.zeros(((Rt*K,Rt)))
    rol_ = np.zeros((K,N))+1j*np.zeros((K,N))
    for k in range(0,K):
        for d in range(0,N):
            sum2_[k*Rt:(k+1)*Rt,:] = sum2_[k*Rt:(k+1)*Rt,:]+np.dot(p[k*Rt:(k+1)*Rt,d], np.matrix(p[k*Rt:(k+1)*Rt,d]).H)
    for k in range(0,K):
        for d in range(0,N):
            rol_[k:(k+1),d:d+1] = cmath.sqrt(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.matrix(w[k*Rr:(k+1)*Rr,d:d+1]).H, 
               U[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]), F), sum2_[k*Rt:(k+1)*Rt,:]), np.matrix(F).H), np.matrix(H[k*Nr:(k+1)*Nr,:]).H), 
               np.matrix(U[k*Rr:(k+1)*Rr,:]).H), w[k*Rr:(k+1)*Rr,d:d+1])*power_n)
    
    sum1 = np.zeros((K*Rt,batch_size))+1j*np.zeros((K*Rt,batch_size))
    for i in range(0,batch_size):
        for k in range(0,K):
            for d in range(0,N):
                sum1[k*Rt:(k+1)*Rt,i:(i+1)] = sum1[k*Rt:(k+1)*Rt,i:(i+1)]+np.multiply(p[k*Rt:(k+1)*Rt,d:d+1], b[k*N+d:k*N+d+1,i:(i+1)])
    #print(sum1)
    
    for i in range(0,batch_size):
        for k in range(0,K):
            for d in range(0,N):
                y[k*Nr:(k+1)*Nr,i:i+1] = np.dot(np.dot(H[k*Nr:(k+1)*Nr,:], F), sum1[k*Rt:(k+1)*Rt,i:(i+1)])
                b_hat[k*N+d:k*N+d+1,i:(i+1)] = np.dot(np.dot(np.matrix(w[k*Rr:(k+1)*Rr,d:d+1]).H, U[k*Rr:(k+1)*Rr,:]), y[k*Nr:(k+1)*Nr,i:i+1])
    
    for i in range(0,batch_size):
        for k in range(0,K):
            for d in range(0,N):
                delta_U_real[k*Rr:(k+1)*Rr,i*N*Nr+d*Nr:i*N*Nr+(d+1)*Nr] = np.matrix(b[k*N+d,i].real/(cmath.sqrt(2*3.14*rol_[k,d].real))
                    *math.exp(-math.pow(b_hat[k*N+d,i].real, 2)/(2*math.pow(rol_[k,d].real, 2)))
                    *(-np.dot(w[k*Rr:(k+1)*Rr,d:d+1].conjugate(), np.matrix(y[k*Nr:(k+1)*Nr,i:i+1]).T)))
                
                delta_U_imag[k*Rr:(k+1)*Rr,i*N*Nr+d*Nr:i*N*Nr+(d+1)*Nr] = np.matrix(b[k*N+d,i].imag/(cmath.sqrt(2*3.14*rol_[k,d].real))
                    *math.exp(-math.pow(b_hat[k*N+d,i].imag, 2)/(2*math.pow(rol_[k,d].real, 2)))
                    *(+1j)*np.dot(w[k*Rr:(k+1)*Rr,d:d+1].conjugate(), np.matrix(y[k*Nr:(k+1)*Nr,i:i+1]).T))
                
                delta_U_conj_real[k*Rr:(k+1)*Rr,i*N*Nr+d*Nr:i*N*Nr+(d+1)*Nr] = np.matrix(b[k*N+d,i].real/(cmath.sqrt(2*3.14*rol_[k,d].real))
                    *math.exp(-math.pow(b_hat[k*N+d,i].real, 2)/(2*math.pow(rol_[k,d].real, 2)))
                    *(-np.dot(w[k*Rr:(k+1)*Rr,d:d+1], np.matrix(y[k*Nr:(k+1)*Nr,i:i+1]).H)))
                
                delta_U_conj_imag[k*Rr:(k+1)*Rr,i*N*Nr+d*Nr:i*N*Nr+(d+1)*Nr] = np.matrix(b[k*N+d,i].imag/(cmath.sqrt(2*3.14*rol_[k,d].real))
                    *math.exp(-math.pow(b_hat[k*N+d,i].imag, 2)/(2*math.pow(rol_[k,d].real, 2)))
                    *(-1j)*np.dot(w[k*Rr:(k+1)*Rr,d:d+1], np.matrix(y[k*Nr:(k+1)*Nr,i:i+1]).H))
    
    delta_U_real_avr = np.mean([delta_U_real[:,i:i+Nr] for i in range(0,Nr*N*batch_size,Nr)], axis=0, keepdims=True)
    delta_U_imag_avr = np.mean([delta_U_imag[:,i:i+Nr] for i in range(0,Nr*N*batch_size,Nr)], axis=0, keepdims=True)
    delta_U_conj_real_avr = np.mean([delta_U_conj_real[:,i:i+Nr] for i in range(0,Nr*N*batch_size,Nr)], axis=0, keepdims=True)
    delta_U_conj_imag_avr = np.mean([delta_U_conj_imag[:,i:i+Nr] for i in range(0,Nr*N*batch_size,Nr)], axis=0, keepdims=True)
    
    #print(delta_U_real_avr.shape)
    
    delta_U_real_avr = np.reshape(delta_U_real_avr, (K*Rr,Nr))
    delta_U_imag_avr = np.reshape(delta_U_imag_avr, (K*Rr,Nr))
    delta_U_conj_real_avr = np.reshape(delta_U_conj_real_avr, (K*Rr,Nr))
    delta_U_conj_imag_avr = np.reshape(delta_U_conj_imag_avr, (K*Rr,Nr))
    
    delta_U = delta_U_real_avr + delta_U_imag_avr
    delta_U_conj = delta_U_conj_real_avr + delta_U_conj_imag_avr
    delta_theu = np.multiply(delta_U_real_avr + delta_U_imag_avr, 1j*U) - np.multiply(delta_U_conj_real_avr + delta_U_conj_imag_avr, 1j*U.conjugate())
    theu = theu-np.multiply(lr_theu, delta_theu)
    
    
    sum2_ = np.zeros((Rt*K,Rt))+1j*np.zeros(((Rt*K,Rt)))
    rol_ = np.zeros((K,N))+1j*np.zeros((K,N))
    for k in range(0,K):
        for d in range(0,N):
            sum2_[k*Rt:(k+1)*Rt,:] = sum2_[k*Rt:(k+1)*Rt,:]+np.dot(p[k*Rt:(k+1)*Rt,d], np.matrix(p[k*Rt:(k+1)*Rt,d]).H)
    for k in range(0,K):
        for d in range(0,N):
            rol_[k:(k+1),d:d+1] = cmath.sqrt(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.matrix(w[k*Rr:(k+1)*Rr,d:d+1]).H, 
               U[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]), F), sum2_[k*Rt:(k+1)*Rt,:]), np.matrix(F).H), np.matrix(H[k*Nr:(k+1)*Nr,:]).H), 
               np.matrix(U[k*Rr:(k+1)*Rr,:]).H), w[k*Rr:(k+1)*Rr,d:d+1])*power_n)
    
    sum1 = np.zeros((K*Rt,batch_size))+1j*np.zeros((K*Rt,batch_size))
    for i in range(0,batch_size):
        for k in range(0,K):
            for d in range(0,N):
                sum1[k*Rt:(k+1)*Rt,i:(i+1)] = sum1[k*Rt:(k+1)*Rt,i:(i+1)]+np.multiply(p[k*Rt:(k+1)*Rt,d:d+1], b[k*N+d:k*N+d+1,i:(i+1)])
    #print(sum1)
    
    for i in range(0,batch_size):
        for k in range(0,K):
            for d in range(0,N):
                y[k*Nr:(k+1)*Nr,i:i+1] = np.dot(np.dot(H[k*Nr:(k+1)*Nr,:], F), sum1[k*Rt:(k+1)*Rt,i:(i+1)])
                b_hat[k*N+d:k*N+d+1,i:(i+1)] = np.dot(np.dot(np.matrix(w[k*Rr:(k+1)*Rr,d:d+1]).H, U[k*Rr:(k+1)*Rr,:]), y[k*Nr:(k+1)*Nr,i:i+1])
    
    for i in range(0,batch_size):
        for k in range(0,K):
            for d in range(0,N):
                delta_F_real[k*Nt:(k+1)*Nt,i*N*Rt+d*Rt:i*N*Rt+(d+1)*Rt] = np.matrix(b[k*N+d,i].real/(cmath.sqrt(2*3.14*rol_[k,d].real))
                    *math.exp(-math.pow(b_hat[k*N+d,i].real, 2)/(2*math.pow(rol_[k,d].real, 2)))
                    *(-np.dot(np.matrix(np.dot(np.dot(np.matrix(w[k*Rr:(k+1)*Rr,d:d+1]).H, U[k*Rr:(k+1)*Rr,:]), 
                                              H[k*Nr:(k+1)*Nr,:])).T, np.matrix(sum1[k*Rt:(k+1)*Rt,i:(i+1)]).T)))
                
                delta_F_imag[k*Nt:(k+1)*Nt,i*N*Rt+d*Rt:i*N*Rt+(d+1)*Rt] = np.matrix(b[k*N+d,i].imag/(cmath.sqrt(2*3.14*rol_[k,d].real))
                    *math.exp(-math.pow(b_hat[k*N+d,i].imag, 2)/(2*math.pow(rol_[k,d].real, 2)))
                    *(+1j)*np.dot(np.matrix(np.dot(np.dot(np.matrix(w[k*Rr:(k+1)*Rr,d:d+1]).H, U[k*Rr:(k+1)*Rr,:]), 
                                              H[k*Nr:(k+1)*Nr,:])).T, np.matrix(sum1[k*Rt:(k+1)*Rt,i:(i+1)]).T))
                
                delta_F_conj_real[k*Nt:(k+1)*Nt,i*N*Rt+d*Rt:i*N*Rt+(d+1)*Rt] = np.matrix(b[k*N+d,i].real/(cmath.sqrt(2*3.14*rol_[k,d].real))
                    *math.exp(-math.pow(b_hat[k*N+d,i].real, 2)/(2*math.pow(rol_[k,d].real, 2)))
                    *(-np.dot(np.matrix(np.dot(np.dot(np.matrix(w[k*Rr:(k+1)*Rr,d:d+1]).H, U[k*Rr:(k+1)*Rr,:]), 
                                              H[k*Nr:(k+1)*Nr,:])).H, np.matrix(sum1[k*Rt:(k+1)*Rt,i:(i+1)]).H)))
                
                delta_F_conj_imag[k*Nt:(k+1)*Nt,i*N*Rt+d*Rt:i*N*Rt+(d+1)*Rt] = np.matrix(b[k*N+d,i].imag/(cmath.sqrt(2*3.14*rol_[k,d].real))
                    *math.exp(-math.pow(b_hat[k*N+d,i].imag, 2)/(2*math.pow(rol_[k,d].real, 2)))
                    *(-1j)*np.dot(np.matrix(np.dot(np.dot(np.matrix(w[k*Rr:(k+1)*Rr,d:d+1]).H, U[k*Rr:(k+1)*Rr,:]), 
                                              H[k*Nr:(k+1)*Nr,:])).H, np.matrix(sum1[k*Rt:(k+1)*Rt,i:(i+1)]).H))
    
    delta_F_real_avr1 = np.mean([delta_F_real[:,i:i+Rt] for i in range(0,Rt*N*batch_size,Rt)], axis=0, keepdims=True)
    delta_F_imag_avr1 = np.mean([delta_F_imag[:,i:i+Rt] for i in range(0,Rt*N*batch_size,Rt)], axis=0, keepdims=True)
    delta_F_conj_real_avr1 = np.mean([delta_F_conj_real[:,i:i+Rt] for i in range(0,Rt*N*batch_size,Rt)], axis=0, keepdims=True)
    delta_F_conj_imag_avr1 = np.mean([delta_F_conj_imag[:,i:i+Rt] for i in range(0,Rt*N*batch_size,Rt)], axis=0, keepdims=True)
    
    delta_F_real_avr1 = np.reshape(delta_F_real_avr1, (K*Nt,Rt))
    delta_F_imag_avr1 = np.reshape(delta_F_imag_avr1, (K*Nt,Rt))
    delta_F_conj_real_avr1 = np.reshape(delta_F_conj_real_avr1, (K*Nt,Rt))
    delta_F_conj_imag_avr1 = np.reshape(delta_F_conj_imag_avr1, (K*Nt,Rt))
    
    delta_F_real_avr = np.mean([delta_F_real_avr1[i:i+Nt,:] for i in range(0,K*Nt,Nt)], axis=0, keepdims=True)
    delta_F_imag_avr = np.mean([delta_F_imag_avr1[i:i+Nt,:] for i in range(0,K*Nt,Nt)], axis=0, keepdims=True)
    delta_F_conj_real_avr = np.mean([delta_F_conj_real_avr1[i:i+Nt,:] for i in range(0,K*Nt,Nt)], axis=0, keepdims=True)
    delta_F_conj_imag_avr = np.mean([delta_F_conj_imag_avr1[i:i+Nt,:] for i in range(0,K*Nt,Nt)], axis=0, keepdims=True)
    
    #print(delta_F_real_avr1.shape)
    #print(delta_F_real_avr.shape)
    
    delta_F_real_avr = np.reshape(delta_F_real_avr, (Nt,Rt))
    delta_F_imag_avr = np.reshape(delta_F_imag_avr, (Nt,Rt))
    delta_F_conj_real_avr = np.reshape(delta_F_conj_real_avr, (Nt,Rt))
    delta_F_conj_imag_avr = np.reshape(delta_F_conj_imag_avr, (Nt,Rt))
    
    delta_F = delta_F_real_avr + delta_F_imag_avr
    delta_F_conj = delta_F_conj_real_avr + delta_F_conj_imag_avr
    delta_thef = np.multiply(delta_F_real_avr + delta_F_imag_avr, 1j*F) - np.multiply(delta_F_conj_real_avr + delta_F_conj_imag_avr, 1j*F.conjugate())
    thef = thef-np.multiply(lr_thef, delta_thef)
    
    return p, w, theu, thef, delta_p_avr, delta_w_avr, delta_U, delta_F, delta_U_conj, delta_F_conj, delta_theu, delta_thef



def back(p0, p1, w0, w1, w2, U0, U1, U2, F0, F1, G_p, G_w, G_U, G_F, G_U_conj, G_F_conj, lr_p, lr_w0, lr_w1, lr_U0, lr_U1, lr_F0, lr_F1):
    #sum1 = tf.complex(tf.zeros(shape=[Rt,1]), tf.zeros(shape=[Rt,1]))
    #b_hat = tf.complex(tf.zeros(shape=[K,1]), tf.zeros(shape=[K,1]))
    #for k in range(0,K):
    #    sum1 = sum1+np.dot(p[k*Rt:(k+1)*Rt,:], b[k:(k+1),i:(i+1)])
    
    sum2_ = np.zeros((Rt*K,Rt))+1j*np.zeros(((Rt*K,Rt)))
    rol_ = np.zeros((K,N))+1j*np.zeros((K,N))
    for k in range(0,K):
        for d in range(0,N):
            sum2_[k*Rt:(k+1)*Rt,:] = sum2_[k*Rt:(k+1)*Rt,:]+np.dot(p0[k*Rt:(k+1)*Rt,d], np.matrix(p0[k*Rt:(k+1)*Rt,d]).H)
    for k in range(0,K):
        for d in range(0,N):
            rol_[k:(k+1),d:d+1] = cmath.sqrt(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, 
               U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]), F1), sum2_[k*Rt:(k+1)*Rt,:]), np.matrix(F1).H), np.matrix(H[k*Nr:(k+1)*Nr,:]).H), 
               np.matrix(U1[k*Rr:(k+1)*Rr,:]).H), w1[k*Rr:(k+1)*Rr,d:d+1])*power_n)
    
    sum10 = np.zeros((K*Rt,batch_size))+1j*np.zeros((K*Rt,batch_size))
    sum11 = np.zeros((K*Rt,batch_size))+1j*np.zeros((K*Rt,batch_size))
    for i in range(0,batch_size):
        for k in range(0,K):
            for d in range(0,N):
                sum10[k*Rt:(k+1)*Rt,i:(i+1)] = sum10[k*Rt:(k+1)*Rt,i:(i+1)]+np.multiply(p0[k*Rt:(k+1)*Rt,d:d+1], b[k*N+d:k*N+d+1,i:(i+1)])
                sum11[k*Rt:(k+1)*Rt,i:(i+1)] = sum11[k*Rt:(k+1)*Rt,i:(i+1)]+np.multiply(p1[k*Rt:(k+1)*Rt,d:d+1], b[k*N+d:k*N+d+1,i:(i+1)])
    #print(sum1)

    b_hat0000 = np.zeros((K*N, batch_size)) + 1j * np.zeros((K*N, batch_size))
    b_hat0100 = np.zeros((K*N, batch_size)) + 1j * np.zeros((K*N, batch_size))
    b_hat0110 = np.zeros((K*N, batch_size)) + 1j * np.zeros((K*N, batch_size))
    b_hat0111 = np.zeros((K*N, batch_size)) + 1j * np.zeros((K*N, batch_size))
    b_hat1111 = np.zeros((K*N, batch_size)) + 1j * np.zeros((K*N, batch_size))
    b_hat1211 = np.zeros((K*N, batch_size)) + 1j * np.zeros((K*N, batch_size))
    b_hat1221 = np.zeros((K*N, batch_size)) + 1j * np.zeros((K*N, batch_size))
    y00 = np.zeros((K*Nr,batch_size))+1j*np.zeros((K*Nr,batch_size))
    y01 = np.zeros((K*Nr,batch_size))+1j*np.zeros((K*Nr,batch_size))
    y11 = np.zeros((K*Nr,batch_size))+1j*np.zeros((K*Nr,batch_size))
    for i in range(0,batch_size):
        for k in range(0,K):
            for d in range(0,N):
                y00[k*Nr:(k+1)*Nr,i:i+1] = np.dot(np.dot(H[k*Nr:(k+1)*Nr,:], F0), sum10[k*Rt:(k+1)*Rt,i:(i+1)])
                y01[k*Nr:(k+1)*Nr,i:i+1] = np.dot(np.dot(H[k*Nr:(k+1)*Nr,:], F1), sum10[k*Rt:(k+1)*Rt,i:(i+1)])
                y11[k*Nr:(k+1)*Nr,i:i+1] = np.dot(np.dot(H[k*Nr:(k+1)*Nr,:], F1), sum11[k*Rt:(k+1)*Rt,i:(i+1)])
                b_hat0000[k*N+d:k*N+d+1,i:(i+1)] = np.dot(np.dot(np.matrix(w0[k*Rr:(k+1)*Rr,d:d+1]).H, U0[k*Rr:(k+1)*Rr,:]), y00[k*Nr:(k+1)*Nr,i:i+1])
                b_hat0100[k*N+d:k*N+d+1,i:(i+1)] = np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, U0[k*Rr:(k+1)*Rr,:]), y00[k*Nr:(k+1)*Nr,i:i+1])
                b_hat0110[k*N+d:k*N+d+1,i:(i+1)] = np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, U1[k*Rr:(k+1)*Rr,:]), y00[k*Nr:(k+1)*Nr,i:i+1])
                b_hat0111[k*N+d:k*N+d+1,i:(i+1)] = np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, U1[k*Rr:(k+1)*Rr,:]), y01[k*Nr:(k+1)*Nr,i:i+1])
                b_hat1111[k*N+d:k*N+d+1,i:(i+1)] = np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, U1[k*Rr:(k+1)*Rr,:]), y11[k*Nr:(k+1)*Nr,i:i+1])
                b_hat1211[k*N+d:k*N+d+1,i:(i+1)] = np.dot(np.dot(np.matrix(w2[k*Rr:(k+1)*Rr,d:d+1]).H, U1[k*Rr:(k+1)*Rr,:]), y11[k*Nr:(k+1)*Nr,i:i+1])
                b_hat1221[k*N+d:k*N+d+1,i:(i+1)] = np.dot(np.dot(np.matrix(w2[k*Rr:(k+1)*Rr,d:d+1]).H, U2[k*Rr:(k+1)*Rr,:]), y11[k*Nr:(k+1)*Nr,i:i+1])


    cof10000 = np.zeros((K*N,batch_size))+1j*np.zeros((K*N,batch_size))
    cof10100 = np.zeros((K*N,batch_size))+1j*np.zeros((K*N,batch_size))
    cof10110 = np.zeros((K*N,batch_size))+1j*np.zeros((K*N,batch_size))
    cof10111 = np.zeros((K*N,batch_size))+1j*np.zeros((K*N,batch_size))
    cof11111 = np.zeros((K*N,batch_size))+1j*np.zeros((K*N,batch_size))
    cof11211 = np.zeros((K*N,batch_size))+1j*np.zeros((K*N,batch_size))
    cof11221 = np.zeros((K*N,batch_size))+1j*np.zeros((K*N,batch_size))
    cof20000 = np.zeros((K*N,batch_size))+1j*np.zeros((K*N,batch_size))
    cof20100 = np.zeros((K*N,batch_size))+1j*np.zeros((K*N,batch_size))
    cof20110 = np.zeros((K*N,batch_size))+1j*np.zeros((K*N,batch_size))
    cof20111 = np.zeros((K*N,batch_size))+1j*np.zeros((K*N,batch_size))
    cof21111 = np.zeros((K*N,batch_size))+1j*np.zeros((K*N,batch_size))
    cof21211 = np.zeros((K*N,batch_size))+1j*np.zeros((K*N,batch_size))
    cof21221 = np.zeros((K*N,batch_size))+1j*np.zeros((K*N,batch_size))
    for i in range(0,batch_size):
        for k in range(0,K):
            for d in range(0,N):
                cof10000[k*N+d:k*N+d+1,i:(i+1)] = np.matrix(b[k*N+d,i]/(math.sqrt(2*3.14)*rol_[k,d].real)
                                          *math.exp(-math.pow(b_hat0000[k*N+d,i].real,2)/(2*math.pow(rol_[k,d].real,2))))
                cof10100[k*N+d:k*N+d+1,i:(i+1)] = np.matrix(b[k*N+d,i]/(math.sqrt(2*3.14)*rol_[k,d].real)
                                          *math.exp(-math.pow(b_hat0100[k*N+d,i].real,2)/(2*math.pow(rol_[k,d].real,2))))
                cof10110[k*N+d:k*N+d+1,i:(i+1)] = np.matrix(b[k*N+d,i]/(math.sqrt(2*3.14)*rol_[k,d].real)
                                          *math.exp(-math.pow(b_hat0110[k*N+d,i].real,2)/(2*math.pow(rol_[k,d].real,2))))
                cof10111[k*N+d:k*N+d+1,i:(i+1)] = np.matrix(b[k*N+d,i]/(math.sqrt(2*3.14)*rol_[k,d].real)
                                          *math.exp(-math.pow(b_hat0111[k*N+d,i].real,2)/(2*math.pow(rol_[k,d].real,2))))
                cof11111[k*N+d:k*N+d+1,i:(i+1)] = np.matrix(b[k*N+d,i]/(math.sqrt(2*3.14)*rol_[k,d].real)
                                          *math.exp(-math.pow(b_hat1111[k*N+d,i].real,2)/(2*math.pow(rol_[k,d].real,2))))
                cof11211[k*N+d:k*N+d+1,i:(i+1)] = np.matrix(b[k*N+d,i]/(math.sqrt(2*3.14)*rol_[k,d].real)
                                          *math.exp(-math.pow(b_hat1211[k*N+d,i].real,2)/(2*math.pow(rol_[k,d].real,2))))
                cof11221[k*N+d:k*N+d+1,i:(i+1)] = np.matrix(b[k*N+d,i]/(math.sqrt(2*3.14)*rol_[k,d].real)
                                          *math.exp(-math.pow(b_hat1221[k*N+d,i].real,2)/(2*math.pow(rol_[k,d].real,2))))
    
                cof20000[k*N+d:k*N+d+1,i:(i+1)] = np.matrix(cof10000[k*N+d,i]*(b_hat0000[k*N+d,i]).real/math.pow(rol_[k,d].real,2))
                cof20100[k*N+d:k*N+d+1,i:(i+1)] = np.matrix(cof10100[k*N+d,i]*(b_hat0100[k*N+d,i]).real/math.pow(rol_[k,d].real,2))
                cof20110[k*N+d:k*N+d+1,i:(i+1)] = np.matrix(cof10110[k*N+d,i]*(b_hat0110[k*N+d,i]).real/math.pow(rol_[k,d].real,2))
                cof20111[k*N+d:k*N+d+1,i:(i+1)] = np.matrix(cof10111[k*N+d,i]*(b_hat0111[k*N+d,i]).real/math.pow(rol_[k,d].real,2))
                cof21111[k*N+d:k*N+d+1,i:(i+1)] = np.matrix(cof11111[k*N+d,i]*(b_hat1111[k*N+d,i]).real/math.pow(rol_[k,d].real,2))
                cof21211[k*N+d:k*N+d+1,i:(i+1)] = np.matrix(cof11211[k*N+d,i]*(b_hat1211[k*N+d,i]).real/math.pow(rol_[k,d].real,2))
                cof21221[k*N+d:k*N+d+1,i:(i+1)] = np.matrix(cof11221[k*N+d,i]*(b_hat1221[k*N+d,i]).real/math.pow(rol_[k,d].real,2))


    G_p_real_index = np.zeros((N*batch_size,K*Rt))+1j*np.zeros((N*batch_size,K*Rt))
    G_w_real_index = np.zeros((N*batch_size,K*Rr))+1j*np.zeros((N*batch_size,K*Rr))
    G_U_real_index = np.zeros((Nr*N*batch_size,K*Rr))+1j*np.zeros((Nr*N*batch_size,K*Rr))
    G_F_real_index = np.zeros((Rt*N*batch_size,K*Nt))+1j*np.zeros((Rt*N*batch_size,K*Nt))
    G_U_conj_real_index = np.zeros((Nr*N*batch_size,K*Rr))+1j*np.zeros((Nr*N*batch_size,K*Rr))
    G_F_conj_real_index = np.zeros((Rt*N*batch_size,K*Nt))+1j*np.zeros((Rt*N*batch_size,K*Nt))
    for i in range(0,batch_size):
        for k in range(0,K):
            for d in range(0,N):
                G_F_real_index[i*N*Rt+d*Rt:i*N*Rt+(d+1)*Rt,k*Nt:(k+1)*Nt] = ( - np.multiply(np.dot(np.multiply(G_p[d:d+1,k*Rt:(k+1)*Rt],
                         np.matrix(lr_p[k*Rt:(k+1)*Rt,d:d+1]).T), np.matrix(np.dot(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, 
                                   U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]), F1*b[k*N+d,i])).H)*cof20111[k*N+d,i]
                         , np.dot(np.dot(np.dot(sum10[k*Rt:(k+1)*Rt,i:(i+1)], np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H), 
                                        U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]))
                         
                         - np.multiply(np.dot(np.multiply(G_w[d:d+1,k*Rr:(k+1)*Rr], np.matrix(lr_w1[k*Rr:(k+1)*Rr,d:d+1]).T), 
                                  np.dot(U1[k*Rr:(k+1)*Rr,:],y11[k*Nr:(k+1)*Nr,i:(i+1)]))*cof21111[k*N+d,i]
                         , np.dot(np.dot(np.dot(sum11[k*Rt:(k+1)*Rt,i:(i+1)], np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H), 
                                        U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]))
                         +np.dot(np.dot(np.dot(sum11[k*Rt:(k+1)*Rt,i:(i+1)], cof11111[k*N+d,i]*np.multiply(G_w[d:d+1,k*Rr:(k+1)*Rr],
                                        np.matrix(lr_w1[k*Rr:(k+1)*Rr,d:d+1]).T)), U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:])
                        
                         - np.multiply(np.dot(np.dot(np.matrix(y11[k*Nr:(k+1)*Nr,i:i+1]).T, np.multiply(G_U[:,k*Rr:(k+1)*Rr], 
                                       np.matrix(lr_U1[k*Rr:(k+1)*Rr,:]).T)), w2[k*Rr:(k+1)*Rr,d:d+1].conjugate())*cof21211[k*N+d,i]
                         , np.dot(np.dot(np.dot(sum11[k*Rt:(k+1)*Rt,i:(i+1)], np.matrix(w2[k*Rr:(k+1)*Rr,d:d+1]).H), 
                                       U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]))
                         + np.matrix(np.dot(np.dot(np.dot(np.matrix(H[k*Nr:(k+1)*Nr,:]).T, cof11211[k*N+d,i]*np.multiply(G_U[:,k*Rr:(k+1)*Rr], 
                                       np.matrix(lr_U1[k*Rr:(k+1)*Rr,:]).T)), w2[k*Rr:(k+1)*Rr,d:d+1].conjugate()), np.matrix(sum11[k*Rt:(k+1)*Rt,i:(i+1)]).T)).T
                        
                         - np.multiply(np.dot(np.dot(np.matrix(y11[k*Nr:(k+1)*Nr,i:i+1]).H, np.multiply(G_U_conj[:,k*Rr:(k+1)*Rr], 
                                       np.matrix(lr_U1[k*Rr:(k+1)*Rr,:]).T)), w2[k*Rr:(k+1)*Rr,d:d+1])*cof21211[k*N+d,i]
                         , np.dot(np.dot(np.dot(sum11[k*Rt:(k+1)*Rt,i:(i+1)], np.matrix(w2[k*Rr:(k+1)*Rr,d:d+1]).H), 
                                       U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]))  
                         
                         + G_F - np.multiply(np.dot(np.dot(np.matrix(sum11[k*Rt:(k+1)*Rt,i:(i+1)]).T, np.multiply(G_F, np.matrix(lr_F1).T)), 
                                      np.matrix(np.dot(np.dot(np.matrix(w2[k*Rr:(k+1)*Rr,d:d+1]).H, U2[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:])).T)
                         *cof21221[k*N+d,i], np.dot(np.dot(np.dot(sum11[k*Rt:(k+1)*Rt,i:(i+1)], np.matrix(w2[k*Rr:(k+1)*Rr,d:d+1]).H), 
                                        U2[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]))
                         
                          - np.multiply(np.dot(np.dot(np.matrix(sum11[k*Rt:(k+1)*Rt,i:(i+1)]).H, np.multiply(G_F, np.matrix(lr_F1).T)), 
                                      np.matrix(np.dot(np.dot(np.matrix(w2[k*Rr:(k+1)*Rr,d:d+1]).H, U2[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:])).H)
                         *cof21221[k*N+d,i], np.dot(np.dot(np.dot(sum11[k*Rt:(k+1)*Rt,i:(i+1)], np.matrix(w2[k*Rr:(k+1)*Rr,d:d+1]).H), 
                                        U2[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:])) )
                
                
                G_F_conj_real_index[i*N*Rt+d*Rt:i*N*Rt+(d+1)*Rt,k*Nt:(k+1)*Nt] = ( - np.multiply(np.dot(np.multiply(G_p[d:d+1,k*Rt:(k+1)*Rt],
                         np.matrix(lr_p[k*Rt:(k+1)*Rt,d:d+1]).T), np.matrix(np.dot(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, 
                                   U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]), F1*b[k*N+d,i])).H)*cof20111[k*N+d,i]
                         , np.dot(np.dot(np.dot(sum10[k*Rt:(k+1)*Rt,i:(i+1)].conjugate(), np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).T), 
                                   U1[k*Rr:(k+1)*Rr,:].conjugate()), H[k*Nr:(k+1)*Nr,:].conjugate()))
                         + np.matrix(np.dot(np.dot(np.dot(np.matrix(H[k*Nr:(k+1)*Nr,:]).H, np.matrix(U1[k*Rr:(k+1)*Rr,:]).H), w1[k*Rr:(k+1)*Rr,d:d+1]), 
                                   cof10111[k*N+d,i]*b[k*N+d,i].conjugate()*np.multiply(G_p[d:d+1,k*Rt:(k+1)*Rt], np.matrix(lr_p[k*Rt:(k+1)*Rt,d:d+1]).T))).T
                        
                         - np.multiply(np.dot(np.multiply(G_w[d:d+1,k*Rr:(k+1)*Rr], np.matrix(lr_w1[k*Rr:(k+1)*Rr,d:d+1]).T), 
                                  np.dot(U1[k*Rr:(k+1)*Rr,:],y11[k*Nr:(k+1)*Nr,i:(i+1)]))*cof21111[k*N+d,i]
                         , np.dot(np.dot(np.dot(sum11[k*Rt:(k+1)*Rt,i:(i+1)].conjugate(), np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).T), 
                                   U1[k*Rr:(k+1)*Rr,:].conjugate()), H[k*Nr:(k+1)*Nr,:].conjugate()))
                        
                         - np.multiply(np.dot(np.dot(np.matrix(y11[k*Nr:(k+1)*Nr,i:i+1]).T, np.multiply(G_U[:,k*Rr:(k+1)*Rr], 
                                       np.matrix(lr_U1[k*Rr:(k+1)*Rr,:]).T)), w2[k*Rr:(k+1)*Rr,d:d+1].conjugate())*cof21211[k*N+d,i]
                         , np.dot(np.dot(np.dot(sum11[k*Rt:(k+1)*Rt,i:(i+1)].conjugate(), np.matrix(w2[k*Rr:(k+1)*Rr,d:d+1]).T), 
                                   U1[k*Rr:(k+1)*Rr,:].conjugate()), H[k*Nr:(k+1)*Nr,:].conjugate()))
                        
                         - np.multiply(np.dot(np.dot(np.matrix(y11[k*Nr:(k+1)*Nr,i:i+1]).H, np.multiply(G_U_conj[:,k*Rr:(k+1)*Rr], 
                                       np.matrix(lr_U1[k*Rr:(k+1)*Rr,:]).T)), w2[k*Rr:(k+1)*Rr,d:d+1])*cof21211[k*N+d,i]
                         , np.dot(np.dot(np.dot(sum11[k*Rt:(k+1)*Rt,i:(i+1)].conjugate(), np.matrix(w2[k*Rr:(k+1)*Rr,d:d+1]).T), 
                                   U1[k*Rr:(k+1)*Rr,:].conjugate()), H[k*Nr:(k+1)*Nr,:].conjugate()))
                         + np.matrix(np.dot(np.dot(np.dot(np.matrix(H[k*Nr:(k+1)*Nr,:]).H, cof11211[k*N+d,i]*np.multiply(G_U_conj[:,k*Rr:(k+1)*Rr], 
                                       np.matrix(lr_U1[k*Rr:(k+1)*Rr,:]).T)), w2[k*Rr:(k+1)*Rr,d:d+1]), np.matrix(sum11[k*Rt:(k+1)*Rt,i:(i+1)]).H)).T
                         
                         - np.multiply(np.dot(np.dot(np.matrix(sum11[k*Rt:(k+1)*Rt,i:(i+1)]).T, np.multiply(G_F, np.matrix(lr_F1).T)), 
                                      np.matrix(np.dot(np.dot(np.matrix(w2[k*Rr:(k+1)*Rr,d:d+1]).H, U2[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:])).T)
                         *cof21221[k*N+d,i], np.dot(np.dot(np.dot(sum11[k*Rt:(k+1)*Rt,i:(i+1)].conjugate(), np.matrix(w2[k*Rr:(k+1)*Rr,d:d+1]).T), 
                                   U2[k*Rr:(k+1)*Rr,:].conjugate()), H[k*Nr:(k+1)*Nr,:].conjugate()))
                         
                         + G_F_conj - np.multiply(np.dot(np.dot(np.matrix(sum11[k*Rt:(k+1)*Rt,i:(i+1)]).H, np.multiply(G_F_conj, np.matrix(lr_F1).T)), 
                                      np.matrix(np.dot(np.dot(np.matrix(w2[k*Rr:(k+1)*Rr,d:d+1]).H, U2[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:])).H)
                         *cof21221[k*N+d,i], np.dot(np.dot(np.dot(sum11[k*Rt:(k+1)*Rt,i:(i+1)].conjugate(), np.matrix(w2[k*Rr:(k+1)*Rr,d:d+1]).T), 
                                   U2[k*Rr:(k+1)*Rr,:].conjugate()), H[k*Nr:(k+1)*Nr,:].conjugate())) )
                
                
                G_U_real_index[i*N*Nr+d*Nr:i*N*Nr+(d+1)*Nr,k*Rr:(k+1)*Rr] = ( - np.multiply(np.dot(np.multiply(G_p[d:d+1,k*Rt:(k+1)*Rt],
                         np.matrix(lr_p[k*Rt:(k+1)*Rt,d:d+1]).T), np.matrix(np.dot(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, 
                                   U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]), F1*b[k*N+d,i])).H)*cof20111[k*N+d,i]
                         , np.dot(y01[k*Nr:(k+1)*Nr,i:i+1], np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H))
                         
                         - np.multiply(np.dot(np.multiply(G_w[d:d+1,k*Rr:(k+1)*Rr], np.matrix(lr_w1[k*Rr:(k+1)*Rr,d:d+1]).T), 
                                  np.dot(U1[k*Rr:(k+1)*Rr,:],y11[k*Nr:(k+1)*Nr,i:(i+1)]))*cof21111[k*N+d,i]
                         , np.dot(y11[k*Nr:(k+1)*Nr,i:i+1], np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H))
                         +np.dot(y11[k*Nr:(k+1)*Nr,i:i+1], cof11111[k*N+d,i]*np.multiply(G_w[d:d+1,k*Rr:(k+1)*Rr], np.matrix(lr_w1[k*Rr:(k+1)*Rr,d:d+1]).T))
                         
                         + G_U[:,k*Rr:(k+1)*Rr] - np.multiply(np.dot(np.dot(np.matrix(y11[k*Nr:(k+1)*Nr,i:i+1]).T, np.multiply(G_U[:,k*Rr:(k+1)*Rr], 
                                       np.matrix(lr_U1[k*Rr:(k+1)*Rr,:]).T)), w2[k*Rr:(k+1)*Rr,d:d+1].conjugate())*cof21211[k*N+d,i]
                         , np.dot(y11[k*Nr:(k+1)*Nr,i:i+1], np.matrix(w2[k*Rr:(k+1)*Rr,d:d+1]).H))
                         
                         - np.multiply(np.dot(np.dot(np.matrix(y11[k*Nr:(k+1)*Nr,i:i+1]).H, np.multiply(G_U_conj[:,k*Rr:(k+1)*Rr], 
                                       np.matrix(lr_U1[k*Rr:(k+1)*Rr,:]).T)), w2[k*Rr:(k+1)*Rr,d:d+1])*cof21211[k*N+d,i]
                         , np.dot(y11[k*Nr:(k+1)*Nr,i:i+1], np.matrix(w2[k*Rr:(k+1)*Rr,d:d+1]).H))
                         
                         - np.multiply(np.dot(np.dot(np.matrix(sum11[k*Rt:(k+1)*Rt,i:(i+1)]).T, np.multiply(G_F_real_index[i*N*Rt+d*Rt:i*N*Rt+(d+1)*Rt,k*Nt:(k+1)*Nt], 
                                                         np.matrix(lr_F0).T)), np.matrix(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, 
                                                                   U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:])).T)*cof20110[k*N+d,i]
                         , np.dot(y00[k*Nr:(k+1)*Nr,i:i+1], np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H))
                         + np.matrix(np.dot(np.dot(np.dot(w1[k*Rr:(k+1)*Rr,d:d+1].conjugate(), np.matrix(sum11[k*Rt:(k+1)*Rt,i:(i+1)]).T), 
                                                   cof10110[k*N+d,i]*np.multiply(G_F_real_index[i*N*Rt+d*Rt:i*N*Rt+(d+1)*Rt,k*Nt:(k+1)*Nt], 
                                                         np.matrix(lr_F0).T)), np.matrix(H[k*Nr:(k+1)*Nr,:]).T)).T
                         
                         - np.multiply(np.dot(np.dot(np.matrix(sum11[k*Rt:(k+1)*Rt,i:(i+1)]).H, np.multiply(G_F_conj_real_index[i*N*Rt+d*Rt:i*N*Rt+(d+1)*Rt,k*Nt:(k+1)*Nt], 
                                                         np.matrix(lr_F0).T)), np.matrix(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, 
                                                                   U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:])).H)*cof20110[k*N+d,i]
                         , np.dot(y00[k*Nr:(k+1)*Nr,i:i+1], np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H)) )
                
                
                G_U_conj_real_index[i*N*Nr+d*Nr:i*N*Nr+(d+1)*Nr,k*Rr:(k+1)*Rr] = ( - np.multiply(np.dot(np.multiply(G_p[d:d+1,k*Rt:(k+1)*Rt],
                         np.matrix(lr_p[k*Rt:(k+1)*Rt,d:d+1]).T), np.matrix(np.dot(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, 
                                   U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]), F1*b[k*N+d,i])).H)*cof20111[k*N+d,i]
                         , np.dot(y01[k*Nr:(k+1)*Nr,i:i+1].conjugate(), np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).T))
                          + np.matrix(np.dot(np.dot(np.dot(w1[k*Rr:(k+1)*Rr,d:d+1], cof10111[k*N+d,i]*np.multiply(G_p[d:d+1,k*Rt:(k+1)*Rt],
                         np.matrix(lr_p[k*Rt:(k+1)*Rt,d:d+1]).T)), b[k*N+d,i].conjugate()*np.matrix(F1).H), np.matrix(H[k*Nr:(k+1)*Nr,:]).H)).T
                        
                         - np.multiply(np.dot(np.multiply(G_w[d:d+1,k*Rr:(k+1)*Rr], np.matrix(lr_w1[k*Rr:(k+1)*Rr,d:d+1]).T), 
                                  np.dot(U1[k*Rr:(k+1)*Rr,:],y11[k*Nr:(k+1)*Nr,i:(i+1)]))*cof21111[k*N+d,i]
                         , np.dot(y11[k*Nr:(k+1)*Nr,i:i+1].conjugate(), np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).T))
                         
                         - np.multiply(np.dot(np.dot(np.matrix(y11[k*Nr:(k+1)*Nr,i:i+1]).T, np.multiply(G_U[:,k*Rr:(k+1)*Rr], 
                                       np.matrix(lr_U1[k*Rr:(k+1)*Rr,:]).T)), w2[k*Rr:(k+1)*Rr,d:d+1].conjugate())*cof21211[k*N+d,i]
                         , np.dot(y11[k*Nr:(k+1)*Nr,i:i+1].conjugate(), np.matrix(w2[k*Rr:(k+1)*Rr,d:d+1]).T))
                        
                         + G_U_conj[:,k*Rr:(k+1)*Rr] - np.multiply(np.dot(np.dot(np.matrix(y11[k*Nr:(k+1)*Nr,i:i+1]).H, np.multiply(G_U_conj[:,k*Rr:(k+1)*Rr], 
                                       np.matrix(lr_U1[k*Rr:(k+1)*Rr,:]).T)), w2[k*Rr:(k+1)*Rr,d:d+1])*cof21211[k*N+d,i]
                         , np.dot(y11[k*Nr:(k+1)*Nr,i:i+1].conjugate(), np.matrix(w2[k*Rr:(k+1)*Rr,d:d+1]).T))
                        
                         - np.multiply(np.dot(np.dot(np.matrix(sum11[k*Rt:(k+1)*Rt,i:(i+1)]).T, np.multiply(G_F_real_index[i*N*Rt+d*Rt:i*N*Rt+(d+1)*Rt,k*Nt:(k+1)*Nt], 
                                                         np.matrix(lr_F0).T)), np.matrix(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, 
                                                                   U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:])).T)*cof20110[k*N+d,i] 
                         , np.dot(y00[k*Nr:(k+1)*Nr,i:i+1].conjugate(), np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).T))
                        
                         - np.multiply(np.dot(np.dot(np.matrix(sum11[k*Rt:(k+1)*Rt,i:(i+1)]).H, np.multiply(G_F_conj_real_index[i*N*Rt+d*Rt:i*N*Rt+(d+1)*Rt,k*Nt:(k+1)*Nt], 
                                                         np.matrix(lr_F0).T)), np.matrix(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, 
                                                                   U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:])).H)*cof20110[k*N+d,i]
                         , np.dot(y00[k*Nr:(k+1)*Nr,i:i+1].conjugate(), np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).T))
                         + np.matrix(np.dot(np.dot(np.dot(w1[k*Rr:(k+1)*Rr,d:d+1], np.matrix(sum11[k*Rt:(k+1)*Rt,i:(i+1)]).H), 
                                                   cof10110[k*N+d,i]*np.multiply(G_F_conj_real_index[i*N*Rt+d*Rt:i*N*Rt+(d+1)*Rt,k*Nt:(k+1)*Nt], 
                                                         np.matrix(lr_F0).T)), np.matrix(H[k*Nr:(k+1)*Nr,:]).H)).T )
                
                
                G_w_real_index[i*N+d:i*N+d+1,k*Rr:(k+1)*Rr] = ( - np.multiply(np.dot(np.multiply(G_p[d:d+1,k*Rt:(k+1)*Rt], 
                              np.matrix(lr_p[k*Rt:(k+1)*Rt,d:d+1]).T), np.matrix(np.dot(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, U1[k*Rr:(k+1)*Rr,:]), 
                                        H[k*Nr:(k+1)*Nr,:]), F1*b[k*N+d,i])).H)*cof20111[k*N+d,i]
                         , np.dot(np.matrix(y01[k*Nr:(k+1)*Nr,i:i+1]).H, np.matrix(U1[k*Rr:(k+1)*Rr,:]).H))
                         + np.dot(np.multiply(G_p[d:d+1,k*Rt:(k+1)*Rt], np.matrix(lr_p[k*Rt:(k+1)*Rt,d:d+1]).T)
                         *cof10111[k*N+d,i], np.matrix(np.dot(np.dot(U1[k*Rr:(k+1)*Rr,:], H[k*Nr:(k+1)*Nr,:]), F1*b[k*N+d,i])).H)
                         
                         + G_w[d:d+1,k*Rr:(k+1)*Rr] - np.multiply(np.dot(np.multiply(G_w[d:d+1,k*Rr:(k+1)*Rr], 
                         np.matrix(lr_w1[k*Rr:(k+1)*Rr,d:d+1]).T), np.dot(U1[k*Rr:(k+1)*Rr,:],y11[k*Nr:(k+1)*Nr,i:(i+1)]))
                         *cof21111[k*N+d,i], np.dot(np.matrix(y11[k*Nr:(k+1)*Nr,i:i+1]).H, np.matrix(U1[k*Rr:(k+1)*Rr,:]).H))
                         
                         - np.multiply(np.dot(np.dot(np.matrix(y00[k*Nr:(k+1)*Nr,i:i+1]).T, np.multiply(G_U_real_index[i*N*Nr+d*Nr:i*N*Nr+(d+1)*Nr,k*Rr:(k+1)*Rr], 
                                        np.matrix(lr_U0[k*Rr:(k+1)*Rr,:]).T)), w1[k*Rr:(k+1)*Rr,d:d+1].conjugate())*cof20100[k*N+d,i]
                         , np.dot(np.matrix(y00[k*Nr:(k+1)*Nr,i:i+1]).H, np.matrix(U0[k*Rr:(k+1)*Rr,:]).H))
                         
                         - np.multiply(np.dot(np.dot(np.matrix(y00[k*Nr:(k+1)*Nr,i:i+1]).H, np.multiply(G_U_conj_real_index[i*N*Nr+d*Nr:i*N*Nr+(d+1)*Nr,k*Rr:(k+1)*Rr], 
                                        np.matrix(lr_U0[k*Rr:(k+1)*Rr,:]).T)), w1[k*Rr:(k+1)*Rr,d:d+1])*cof20100[k*N+d,i]
                         , np.dot(np.matrix(y00[k*Nr:(k+1)*Nr,i:i+1]).H, np.matrix(U0[k*Rr:(k+1)*Rr,:]).H))
                         + np.dot(np.matrix(y00[k*Nr:(k+1)*Nr,i:i+1]).H, cof10100[k*N+d,i]*np.multiply(G_U_conj_real_index[i*N*Nr+d*Nr:i*N*Nr+(d+1)*Nr,k*Rr:(k+1)*Rr], 
                                  np.matrix(lr_U0[k*Rr:(k+1)*Rr,:]).T))
                         
                         - np.multiply(np.dot(np.dot(np.matrix(sum10[k*Rt:(k+1)*Rt,i:(i+1)]).T, np.multiply(G_F_real_index[i*N*Rt+d*Rt:i*N*Rt+(d+1)*Rt,k*Nt:(k+1)*Nt], 
                                                        np.matrix(lr_F0).T)), np.matrix(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, 
                                                                  U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:])).T)*cof20110[k*N+d,i]
                         , np.dot(np.matrix(y00[k*Nr:(k+1)*Nr,i:i+1]).H, np.matrix(U1[k*Rr:(k+1)*Rr,:]).H))
                         
                         - np.multiply(np.dot(np.dot(np.matrix(sum10[k*Rt:(k+1)*Rt,i:(i+1)]).H, np.multiply(G_F_conj_real_index[i*N*Rt+d*Rt:i*N*Rt+(d+1)*Rt,k*Nt:(k+1)*Nt], 
                                                        np.matrix(lr_F0).T)), np.matrix(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, 
                                                                  U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:])).H)*cof20110[k*N+d,i]
                         , np.dot(np.matrix(y00[k*Nr:(k+1)*Nr,i:i+1]).H, np.matrix(U1[k*Rr:(k+1)*Rr,:]).H))
                         + np.dot(np.dot(np.dot(np.matrix(sum10[k*Rt:(k+1)*Rt,i:(i+1)]).H, cof10110[k*N+d,i]*np.multiply(G_F_conj_real_index[i*N*Rt+d*Rt:i*N*Rt+(d+1)*Rt,k*Nt:(k+1)*Nt], 
                                                  np.matrix(lr_F0).T)), np.matrix(H[k*Nr:(k+1)*Nr,:]).H), np.matrix(U1[k*Rr:(k+1)*Rr,:]).H))
    			
    			
                G_p_real_index[i*N+d:i*N+d+1,k*Rt:(k+1)*Rt] = (G_p[d:d+1,k*Rt:(k+1)*Rt] - np.multiply(np.dot(np.multiply(G_p[d:d+1,k*Rt:(k+1)*Rt], 
                         np.matrix(lr_p[k*Rt:(k+1)*Rt,d:d+1]).T), np.matrix(np.dot(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, 
                                   U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]), F1*b[k*N+d,i])).H)*cof20111[k*N+d,i]
                         , np.dot(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]), F1*b[k*N+d,i]))
                         
                         - np.multiply(np.dot(np.multiply(G_w_real_index[i*N+d:i*N+d+1,k*Rr:(k+1)*Rr], np.matrix(lr_w0[k*Rr:(k+1)*Rr,d:d+1]).T), 
                                  np.dot(U0[k*Rr:(k+1)*Rr,:],y00[k*Nr:(k+1)*Nr,i:(i+1)]))*cof20000[k*N+d,i]
                         , np.dot(np.dot(np.dot(np.matrix(w0[k*Rr:(k+1)*Rr,d:d+1]).H, U0[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]), F0*b[k*N+d,i]))
                         + np.dot(np.multiply(G_w_real_index[i*N+d:i*N+d+1,k*Rr:(k+1)*Rr], np.matrix(lr_w0[k*Rr:(k+1)*Rr,d:d+1]).T)
                         *cof10000[k*N+d,i]*b[k*N+d,i], np.dot(np.dot(U0[k*Rr:(k+1)*Rr,:], H[k*Nr:(k+1)*Nr,:]), F0))
                         
                         - np.multiply(np.dot(np.dot(np.matrix(y00[k*Nr:(k+1)*Nr,i:i+1]).T, np.multiply(G_U_real_index[i*N*Nr+d*Nr:i*N*Nr+(d+1)*Nr,k*Rr:(k+1)*Rr], 
                                        np.matrix(lr_U0[k*Rr:(k+1)*Rr,:]).T)), w1[k*Rr:(k+1)*Rr,d:d+1].conjugate())*cof20100[k*N+d,i]
                         , np.dot(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, U0[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]), F0*b[k*N+d,i]))
                         + np.matrix(np.dot(np.dot(np.dot(np.matrix(F0).T, np.matrix(H[k*Nr:(k+1)*Nr,:]).T), np.multiply(G_U_real_index[i*N*Nr+d*Nr:i*N*Nr+(d+1)*Nr,k*Rr:(k+1)*Rr], 
                                        np.matrix(lr_U0[k*Rr:(k+1)*Rr,:]).T)*cof10100[k*N+d,i]), w1[k*Rr:(k+1)*Rr,d:d+1].conjugate()*b[k*N+d,i])).T
                         
                         - np.multiply(np.dot(np.dot(np.matrix(y00[k*Nr:(k+1)*Nr,i:i+1]).H, np.multiply(G_U_conj_real_index[i*N*Nr+d*Nr:i*N*Nr+(d+1)*Nr,k*Rr:(k+1)*Rr], 
                                        np.matrix(lr_U0[k*Rr:(k+1)*Rr,:]).T)), w1[k*Rr:(k+1)*Rr,d:d+1])*cof20100[k*N+d,i]
                         , np.dot(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, U0[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]), F0*b[k*N+d,i]))
                         
                         - np.multiply(np.dot(np.dot(np.matrix(sum10[k*Rt:(k+1)*Rt,i:(i+1)]).T, np.multiply(G_F_real_index[i*N*Rt+d*Rt:i*N*Rt+(d+1)*Rt,k*Nt:(k+1)*Nt], 
                                                        np.matrix(lr_F0).T)), np.matrix(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, 
                                                                  U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:])).T)*cof20110[k*N+d,i]
                         , np.dot(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]), F0*b[k*N+d,i]))
                         + np.matrix(np.dot(np.dot(np.dot(cof10110[k*N+d,i]*np.multiply(G_F_real_index[i*N*Rt+d*Rt:i*N*Rt+(d+1)*Rt,k*Nt:(k+1)*Nt], 
                                                        np.matrix(lr_F0).T), np.matrix(H[k*Nr:(k+1)*Nr,:]).H), np.matrix(U1[k*Rr:(k+1)*Rr,:]).H), 
                         w1[k*Rr:(k+1)*Rr,d:d+1].conjugate()*b[k*N+d,i])).T
                         
                         - np.multiply(np.dot(np.dot(np.matrix(sum10[k*Rt:(k+1)*Rt,i:(i+1)]).H, np.multiply(G_F_conj_real_index[i*N*Rt+d*Rt:i*N*Rt+(d+1)*Rt,k*Nt:(k+1)*Nt], 
                                                        np.matrix(lr_F0).T)), np.matrix(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, 
                                                                  U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:])).H)*cof20110[k*N+d,i]
                         , np.dot(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]), F0*b[k*N+d,i])))

            
    G_p_real_index = np.mean([G_p_real_index[i:i+N,:] for i in range(0,N*batch_size,N)], axis=0, keepdims=True)
    G_p_real_index = np.reshape(G_p_real_index, (N,K*Rt))
    
    G_w_real_index = np.mean([G_w_real_index[i:i+N,:] for i in range(0,N*batch_size,N)], axis=0, keepdims=True)
    G_w_real_index = np.reshape(G_w_real_index, (N,K*Rr))
    
    G_U_real_index = np.mean([G_U_real_index[i:i+Nr,:] for i in range(0,Nr*N*batch_size,Nr)], axis=0, keepdims=True)
    G_U_real_index = np.reshape(G_U_real_index, (Nr,K*Rr))
    
    G_U_conj_real_index = np.mean([G_U_conj_real_index[i:i+Nr,:] for i in range(0,Nr*N*batch_size,Nr)], axis=0, keepdims=True)
    G_U_conj_real_index = np.reshape(G_U_conj_real_index, (Nr,K*Rr))
    
    G_F_real_index = np.mean([G_F_real_index[i:i+Rt,:] for i in range(0,Rt*N*batch_size,Rt)], axis=0, keepdims=True)
    G_F_real_index = np.reshape(G_F_real_index, (Rt,K*Nt))
    
    G_F_real_index = np.mean([G_F_real_index[:,i:i+Nt] for i in range(0,K*Nt,Nt)], axis=0, keepdims=True)
    G_F_real_index = np.reshape(G_F_real_index, (Rt,Nt))
    
    G_F_conj_real_index = np.mean([G_F_conj_real_index[i:i+Rt,:] for i in range(0,Rt*N*batch_size,Rt)], axis=0, keepdims=True)
    G_F_conj_real_index = np.reshape(G_F_conj_real_index, (Rt,K*Nt))
    
    G_F_conj_real_index = np.mean([G_F_conj_real_index[:,i:i+Nt] for i in range(0,K*Nt,Nt)], axis=0, keepdims=True)
    G_F_conj_real_index = np.reshape(G_F_conj_real_index, (Rt,Nt))
    
    for i in range(0,batch_size):
        for k in range(0,K):
            for d in range(0,N):
                cof10000[k*N+d:k*N+d+1,i:(i+1)] = (b[k*N+d,i]/(math.sqrt(2*3.14)*rol_[k,d].real)
                                          *math.exp(-math.pow(b_hat0000[k*N+d,i].imag,2)/(2*math.pow(rol_[k,d].real,2))))
                cof10100[k*N+d:k*N+d+1,i:(i+1)] = (b[k*N+d,i]/(math.sqrt(2*3.14)*rol_[k,d].real)
                                          *math.exp(-math.pow(b_hat0100[k*N+d,i].imag,2)/(2*math.pow(rol_[k,d].real,2))))
                cof10110[k*N+d:k*N+d+1,i:(i+1)] = (b[k*N+d,i]/(math.sqrt(2*3.14)*rol_[k,d].real)
                                          *math.exp(-math.pow(b_hat0110[k*N+d,i].imag,2)/(2*math.pow(rol_[k,d].real,2))))
                cof10111[k*N+d:k*N+d+1,i:(i+1)] = (b[k*N+d,i]/(math.sqrt(2*3.14)*rol_[k,d].real)
                                          *math.exp(-math.pow(b_hat0111[k*N+d,i].imag,2)/(2*math.pow(rol_[k,d].real,2))))
                cof11111[k*N+d:k*N+d+1,i:(i+1)] = (b[k*N+d,i]/(math.sqrt(2*3.14)*rol_[k,d].real)
                                          *math.exp(-math.pow(b_hat1111[k*N+d,i].imag,2)/(2*math.pow(rol_[k,d].real,2))))
                cof11211[k*N+d:k*N+d+1,i:(i+1)] = (b[k*N+d,i]/(math.sqrt(2*3.14)*rol_[k,d].real)
                                          *math.exp(-math.pow(b_hat1211[k*N+d,i].imag,2)/(2*math.pow(rol_[k,d].real,2))))
                cof11221[k*N+d:k*N+d+1,i:(i+1)] = (b[k*N+d,i]/(math.sqrt(2*3.14)*rol_[k,d].real)
                                          *math.exp(-math.pow(b_hat1221[k*N+d,i].imag,2)/(2*math.pow(rol_[k,d].real,2))))
    
                cof20000[k*N+d:k*N+d+1,i:(i+1)] = cof10000[k*N+d,i]*(b_hat0000[k*N+d,i]).imag/math.pow(rol_[k,d].real,2)
                cof20100[k*N+d:k*N+d+1,i:(i+1)] = cof10100[k*N+d,i]*(b_hat0100[k*N+d,i]).imag/math.pow(rol_[k,d].real,2)
                cof20110[k*N+d:k*N+d+1,i:(i+1)] = cof10110[k*N+d,i]*(b_hat0110[k*N+d,i]).imag/math.pow(rol_[k,d].real,2)
                cof20111[k*N+d:k*N+d+1,i:(i+1)] = cof10111[k*N+d,i]*(b_hat0111[k*N+d,i]).imag/math.pow(rol_[k,d].real,2)
                cof21111[k*N+d:k*N+d+1,i:(i+1)] = cof11111[k*N+d,i]*(b_hat1111[k*N+d,i]).imag/math.pow(rol_[k,d].real,2)
                cof21211[k*N+d:k*N+d+1,i:(i+1)] = cof11211[k*N+d,i]*(b_hat1211[k*N+d,i]).imag/math.pow(rol_[k,d].real,2)
                cof21221[k*N+d:k*N+d+1,i:(i+1)] = cof11221[k*N+d,i]*(b_hat1221[k*N+d,i]).imag/math.pow(rol_[k,d].real,2)
    
    
    G_p_imag_index = np.zeros((N*batch_size,K*Rt))+1j*np.zeros((N*batch_size,K*Rt))
    G_w_imag_index = np.zeros((N*batch_size,K*Rr))+1j*np.zeros((N*batch_size,K*Rr))
    G_U_imag_index = np.zeros((Nr*N*batch_size,K*Rr))+1j*np.zeros((Nr*N*batch_size,K*Rr))
    G_F_imag_index = np.zeros((Rt*N*batch_size,K*Nt))+1j*np.zeros((Rt*N*batch_size,K*Nt))
    G_U_conj_imag_index = np.zeros((Nr*N*batch_size,K*Rr))+1j*np.zeros((Nr*N*batch_size,K*Rr))
    G_F_conj_imag_index = np.zeros((Rt*N*batch_size,K*Nt))+1j*np.zeros((Rt*N*batch_size,K*Nt))
    for i in range(0,batch_size):
        for k in range(0,K):
            for d in range(0,N):
                G_F_imag_index[i*N*Rt+d*Rt:i*N*Rt+(d+1)*Rt,k*Nt:(k+1)*Nt] = ( - (1)*np.multiply(np.dot(np.multiply(G_p[d:d+1,k*Rt:(k+1)*Rt],
                         np.matrix(lr_p[k*Rt:(k+1)*Rt,d:d+1]).T), np.matrix(np.dot(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, 
                                   U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]), F1*b[k*N+d,i])).H)*cof20111[k*N+d,i]
                         , np.dot(np.dot(np.dot(sum10[k*Rt:(k+1)*Rt,i:(i+1)], np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H), 
                                        U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]))
                         
                         - (-1)*np.multiply(np.dot(np.multiply(G_w[d:d+1,k*Rr:(k+1)*Rr], np.matrix(lr_w1[k*Rr:(k+1)*Rr,d:d+1]).T), 
                                  np.dot(U1[k*Rr:(k+1)*Rr,:],y11[k*Nr:(k+1)*Nr,i:(i+1)]))*cof21111[k*N+d,i]
                         , np.dot(np.dot(np.dot(sum11[k*Rt:(k+1)*Rt,i:(i+1)], np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H), 
                                        U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]))
                         + (-1j)*np.dot(np.dot(np.dot(sum11[k*Rt:(k+1)*Rt,i:(i+1)], cof11111[k*N+d,i]*np.multiply(G_w[d:d+1,k*Rr:(k+1)*Rr],
                                        np.matrix(lr_w1[k*Rr:(k+1)*Rr,d:d+1]).T)), U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:])
                        
                         - (-1)*np.multiply(np.dot(np.dot(np.matrix(y11[k*Nr:(k+1)*Nr,i:i+1]).T, np.multiply(G_U[:,k*Rr:(k+1)*Rr], 
                                       np.matrix(lr_U1[k*Rr:(k+1)*Rr,:]).T)), w2[k*Rr:(k+1)*Rr,d:d+1].conjugate())*cof21211[k*N+d,i]
                         , np.dot(np.dot(np.dot(sum11[k*Rt:(k+1)*Rt,i:(i+1)], np.matrix(w2[k*Rr:(k+1)*Rr,d:d+1]).H), 
                                       U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]))
                         + (-1j)*np.matrix(np.dot(np.dot(np.dot(np.matrix(H[k*Nr:(k+1)*Nr,:]).T, cof11211[k*N+d,i]*np.multiply(G_U[:,k*Rr:(k+1)*Rr], 
                                       np.matrix(lr_U1[k*Rr:(k+1)*Rr,:]).T)), w2[k*Rr:(k+1)*Rr,d:d+1].conjugate()), np.matrix(sum11[k*Rt:(k+1)*Rt,i:(i+1)]).T)).T
                        
                         - (1)*np.multiply(np.dot(np.dot(np.matrix(y11[k*Nr:(k+1)*Nr,i:i+1]).H, np.multiply(G_U_conj[:,k*Rr:(k+1)*Rr], 
                                       np.matrix(lr_U1[k*Rr:(k+1)*Rr,:]).T)), w2[k*Rr:(k+1)*Rr,d:d+1])*cof21211[k*N+d,i]
                         , np.dot(np.dot(np.dot(sum11[k*Rt:(k+1)*Rt,i:(i+1)], np.matrix(w2[k*Rr:(k+1)*Rr,d:d+1]).H), 
                                       U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]))  
                         
                         + G_F - (-1)*np.multiply(np.dot(np.dot(np.matrix(sum11[k*Rt:(k+1)*Rt,i:(i+1)]).T, np.multiply(G_F, np.matrix(lr_F1).T)), 
                                      np.matrix(np.dot(np.dot(np.matrix(w2[k*Rr:(k+1)*Rr,d:d+1]).H, U2[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:])).T)
                         *cof21221[k*N+d,i], np.dot(np.dot(np.dot(sum11[k*Rt:(k+1)*Rt,i:(i+1)], np.matrix(w2[k*Rr:(k+1)*Rr,d:d+1]).H), 
                                        U2[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]))
                         
                          - (1)*np.multiply(np.dot(np.dot(np.matrix(sum11[k*Rt:(k+1)*Rt,i:(i+1)]).H, np.multiply(G_F, np.matrix(lr_F1).T)), 
                                      np.matrix(np.dot(np.dot(np.matrix(w2[k*Rr:(k+1)*Rr,d:d+1]).H, U2[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:])).H)
                         *cof21221[k*N+d,i], np.dot(np.dot(np.dot(sum11[k*Rt:(k+1)*Rt,i:(i+1)], np.matrix(w2[k*Rr:(k+1)*Rr,d:d+1]).H), 
                                        U2[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:])) )
                
                
                G_F_conj_imag_index[i*N*Rt+d*Rt:i*N*Rt+(d+1)*Rt,k*Nt:(k+1)*Nt] = ( - (-1)*np.multiply(np.dot(np.multiply(G_p[d:d+1,k*Rt:(k+1)*Rt],
                         np.matrix(lr_p[k*Rt:(k+1)*Rt,d:d+1]).T), np.matrix(np.dot(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, 
                                   U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]), F1*b[k*N+d,i])).H)*cof20111[k*N+d,i]
                         , np.dot(np.dot(np.dot(sum10[k*Rt:(k+1)*Rt,i:(i+1)].conjugate(), np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).T), 
                                   U1[k*Rr:(k+1)*Rr,:].conjugate()), H[k*Nr:(k+1)*Nr,:].conjugate()))
                         + (1j)*np.matrix(np.dot(np.dot(np.dot(np.matrix(H[k*Nr:(k+1)*Nr,:]).H, np.matrix(U1[k*Rr:(k+1)*Rr,:]).H), w1[k*Rr:(k+1)*Rr,d:d+1]), 
                                   cof10111[k*N+d,i]*b[k*N+d,i].conjugate()*np.multiply(G_p[d:d+1,k*Rt:(k+1)*Rt], np.matrix(lr_p[k*Rt:(k+1)*Rt,d:d+1]).T))).T
                        
                         - (1)*np.multiply(np.dot(np.multiply(G_w[d:d+1,k*Rr:(k+1)*Rr], np.matrix(lr_w1[k*Rr:(k+1)*Rr,d:d+1]).T), 
                                  np.dot(U1[k*Rr:(k+1)*Rr,:],y11[k*Nr:(k+1)*Nr,i:(i+1)]))*cof21111[k*N+d,i]
                         , np.dot(np.dot(np.dot(sum11[k*Rt:(k+1)*Rt,i:(i+1)].conjugate(), np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).T), 
                                   U1[k*Rr:(k+1)*Rr,:].conjugate()), H[k*Nr:(k+1)*Nr,:].conjugate()))
                        
                         - (1)*np.multiply(np.dot(np.dot(np.matrix(y11[k*Nr:(k+1)*Nr,i:i+1]).T, np.multiply(G_U[:,k*Rr:(k+1)*Rr], 
                                       np.matrix(lr_U1[k*Rr:(k+1)*Rr,:]).T)), w2[k*Rr:(k+1)*Rr,d:d+1].conjugate())*cof21211[k*N+d,i]
                         , np.dot(np.dot(np.dot(sum11[k*Rt:(k+1)*Rt,i:(i+1)].conjugate(), np.matrix(w2[k*Rr:(k+1)*Rr,d:d+1]).T), 
                                   U1[k*Rr:(k+1)*Rr,:].conjugate()), H[k*Nr:(k+1)*Nr,:].conjugate()))
                        
                         - (-1)*np.multiply(np.dot(np.dot(np.matrix(y11[k*Nr:(k+1)*Nr,i:i+1]).H, np.multiply(G_U_conj[:,k*Rr:(k+1)*Rr], 
                                       np.matrix(lr_U1[k*Rr:(k+1)*Rr,:]).T)), w2[k*Rr:(k+1)*Rr,d:d+1])*cof21211[k*N+d,i]
                         , np.dot(np.dot(np.dot(sum11[k*Rt:(k+1)*Rt,i:(i+1)].conjugate(), np.matrix(w2[k*Rr:(k+1)*Rr,d:d+1]).T), 
                                   U1[k*Rr:(k+1)*Rr,:].conjugate()), H[k*Nr:(k+1)*Nr,:].conjugate()))
                         + (1j)*np.matrix(np.dot(np.dot(np.dot(np.matrix(H[k*Nr:(k+1)*Nr,:]).H, cof11211[k*N+d,i]*np.multiply(G_U_conj[:,k*Rr:(k+1)*Rr], 
                                       np.matrix(lr_U1[k*Rr:(k+1)*Rr,:]).T)), w2[k*Rr:(k+1)*Rr,d:d+1]), np.matrix(sum11[k*Rt:(k+1)*Rt,i:(i+1)]).H)).T
                         
                         - (1)*np.multiply(np.dot(np.dot(np.matrix(sum11[k*Rt:(k+1)*Rt,i:(i+1)]).T, np.multiply(G_F, np.matrix(lr_F1).T)), 
                                      np.matrix(np.dot(np.dot(np.matrix(w2[k*Rr:(k+1)*Rr,d:d+1]).H, U2[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:])).T)
                         *cof21221[k*N+d,i], np.dot(np.dot(np.dot(sum11[k*Rt:(k+1)*Rt,i:(i+1)].conjugate(), np.matrix(w2[k*Rr:(k+1)*Rr,d:d+1]).T), 
                                   U2[k*Rr:(k+1)*Rr,:].conjugate()), H[k*Nr:(k+1)*Nr,:].conjugate()))
                         
                         + G_F_conj - (-1)*np.multiply(np.dot(np.dot(np.matrix(sum11[k*Rt:(k+1)*Rt,i:(i+1)]).H, np.multiply(G_F_conj, np.matrix(lr_F1).T)), 
                                      np.matrix(np.dot(np.dot(np.matrix(w2[k*Rr:(k+1)*Rr,d:d+1]).H, U2[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:])).H)
                         *cof21221[k*N+d,i], np.dot(np.dot(np.dot(sum11[k*Rt:(k+1)*Rt,i:(i+1)].conjugate(), np.matrix(w2[k*Rr:(k+1)*Rr,d:d+1]).T), 
                                   U2[k*Rr:(k+1)*Rr,:].conjugate()), H[k*Nr:(k+1)*Nr,:].conjugate())) )
                
                
                G_U_imag_index[i*N*Nr+d*Nr:i*N*Nr+(d+1)*Nr,k*Rr:(k+1)*Rr] = ( - (1)*np.multiply(np.dot(np.multiply(G_p[d:d+1,k*Rt:(k+1)*Rt],
                         np.matrix(lr_p[k*Rt:(k+1)*Rt,d:d+1]).T), np.matrix(np.dot(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, 
                                   U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]), F1*b[k*N+d,i])).H)*cof20111[k*N+d,i]
                         , np.dot(y01[k*Nr:(k+1)*Nr,i:i+1], np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H))
                         
                         - (-1)*np.multiply(np.dot(np.multiply(G_w[d:d+1,k*Rr:(k+1)*Rr], np.matrix(lr_w1[k*Rr:(k+1)*Rr,d:d+1]).T), 
                                  np.dot(U1[k*Rr:(k+1)*Rr,:],y11[k*Nr:(k+1)*Nr,i:(i+1)]))*cof21111[k*N+d,i]
                         , np.dot(y11[k*Nr:(k+1)*Nr,i:i+1], np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H))
                         + (-1j)*np.dot(y11[k*Nr:(k+1)*Nr,i:i+1], cof11111[k*N+d,i]*np.multiply(G_w[d:d+1,k*Rr:(k+1)*Rr], np.matrix(lr_w1[k*Rr:(k+1)*Rr,d:d+1]).T))
                         
                         + G_U[:,k*Rr:(k+1)*Rr] - (-1)*np.multiply(np.dot(np.dot(np.matrix(y11[k*Nr:(k+1)*Nr,i:i+1]).T, np.multiply(G_U[:,k*Rr:(k+1)*Rr], 
                                       np.matrix(lr_U1[k*Rr:(k+1)*Rr,:]).T)), w2[k*Rr:(k+1)*Rr,d:d+1].conjugate())*cof21211[k*N+d,i]
                         , np.dot(y11[k*Nr:(k+1)*Nr,i:i+1], np.matrix(w2[k*Rr:(k+1)*Rr,d:d+1]).H))
                         
                         - (1)*np.multiply(np.dot(np.dot(np.matrix(y11[k*Nr:(k+1)*Nr,i:i+1]).H, np.multiply(G_U_conj[:,k*Rr:(k+1)*Rr], 
                                       np.matrix(lr_U1[k*Rr:(k+1)*Rr,:]).T)), w2[k*Rr:(k+1)*Rr,d:d+1])*cof21211[k*N+d,i]
                         , np.dot(y11[k*Nr:(k+1)*Nr,i:i+1], np.matrix(w2[k*Rr:(k+1)*Rr,d:d+1]).H))
                         
                         - (-1)*np.multiply(np.dot(np.dot(np.matrix(sum11[k*Rt:(k+1)*Rt,i:(i+1)]).T, np.multiply(G_F_imag_index[i*N*Rt+d*Rt:i*N*Rt+(d+1)*Rt,k*Nt:(k+1)*Nt], 
                                                         np.matrix(lr_F0).T)), np.matrix(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, 
                                                                   U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:])).T)*cof20110[k*N+d,i]
                         , np.dot(y00[k*Nr:(k+1)*Nr,i:i+1], np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H))
                         + (-1j)*np.matrix(np.dot(np.dot(np.dot(w1[k*Rr:(k+1)*Rr,d:d+1].conjugate(), np.matrix(sum11[k*Rt:(k+1)*Rt,i:(i+1)]).T), 
                                                   cof10110[k*N+d,i]*np.multiply(G_F_imag_index[i*N*Rt+d*Rt:i*N*Rt+(d+1)*Rt,k*Nt:(k+1)*Nt], 
                                                         np.matrix(lr_F0).T)), np.matrix(H[k*Nr:(k+1)*Nr,:]).T)).T
                         
                         - (1)*np.multiply(np.dot(np.dot(np.matrix(sum11[k*Rt:(k+1)*Rt,i:(i+1)]).H, np.multiply(G_F_conj_imag_index[i*N*Rt+d*Rt:i*N*Rt+(d+1)*Rt,k*Nt:(k+1)*Nt], 
                                                         np.matrix(lr_F0).T)), np.matrix(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, 
                                                                   U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:])).H)*cof20110[k*N+d,i]
                         , np.dot(y00[k*Nr:(k+1)*Nr,i:i+1], np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H)) )
                
                
                G_U_conj_imag_index[i*N*Nr+d*Nr:i*N*Nr+(d+1)*Nr,k*Rr:(k+1)*Rr] = ( - (-1)*np.multiply(np.dot(np.multiply(G_p[d:d+1,k*Rt:(k+1)*Rt],
                         np.matrix(lr_p[k*Rt:(k+1)*Rt,d:d+1]).T), np.matrix(np.dot(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, 
                                   U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]), F1*b[k*N+d,i])).H)*cof20111[k*N+d,i]
                         , np.dot(y01[k*Nr:(k+1)*Nr,i:i+1].conjugate(), np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).T))
                         + (1j)*np.matrix(np.dot(np.dot(np.dot(w1[k*Rr:(k+1)*Rr,d:d+1], cof10111[k*N+d,i]*np.multiply(G_p[d:d+1,k*Rt:(k+1)*Rt],
                         np.matrix(lr_p[k*Rt:(k+1)*Rt,d:d+1]).T)), b[k*N+d,i].conjugate()*np.matrix(F1).H), np.matrix(H[k*Nr:(k+1)*Nr,:]).H)).T
                        
                         - (1)*np.multiply(np.dot(np.multiply(G_w[d:d+1,k*Rr:(k+1)*Rr], np.matrix(lr_w1[k*Rr:(k+1)*Rr,d:d+1]).T), 
                                  np.dot(U1[k*Rr:(k+1)*Rr,:],y11[k*Nr:(k+1)*Nr,i:(i+1)]))*cof21111[k*N+d,i]
                         , np.dot(y11[k*Nr:(k+1)*Nr,i:i+1].conjugate(), np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).T))
                         
                         - (1)*np.multiply(np.dot(np.dot(np.matrix(y11[k*Nr:(k+1)*Nr,i:i+1]).T, np.multiply(G_U[:,k*Rr:(k+1)*Rr], 
                                       np.matrix(lr_U1[k*Rr:(k+1)*Rr,:]).T)), w2[k*Rr:(k+1)*Rr,d:d+1].conjugate())*cof21211[k*N+d,i]
                         , np.dot(y11[k*Nr:(k+1)*Nr,i:i+1].conjugate(), np.matrix(w2[k*Rr:(k+1)*Rr,d:d+1]).T))
                        
                         + G_U_conj[:,k*Rr:(k+1)*Rr] - (-1)*np.multiply(np.dot(np.dot(np.matrix(y11[k*Nr:(k+1)*Nr,i:i+1]).H, np.multiply(G_U_conj[:,k*Rr:(k+1)*Rr], 
                                       np.matrix(lr_U1[k*Rr:(k+1)*Rr,:]).T)), w2[k*Rr:(k+1)*Rr,d:d+1])*cof21211[k*N+d,i]
                         , np.dot(y11[k*Nr:(k+1)*Nr,i:i+1].conjugate(), np.matrix(w2[k*Rr:(k+1)*Rr,d:d+1]).T))
                        
                         - (1)*np.multiply(np.dot(np.dot(np.matrix(sum11[k*Rt:(k+1)*Rt,i:(i+1)]).T, np.multiply(G_F_imag_index[i*N*Rt+d*Rt:i*N*Rt+(d+1)*Rt,k*Nt:(k+1)*Nt], 
                                                         np.matrix(lr_F0).T)), np.matrix(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, 
                                                                   U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:])).T)*cof20110[k*N+d,i] 
                         , np.dot(y00[k*Nr:(k+1)*Nr,i:i+1].conjugate(), np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).T))
                        
                         - (-1)*np.multiply(np.dot(np.dot(np.matrix(sum11[k*Rt:(k+1)*Rt,i:(i+1)]).H, np.multiply(G_F_conj_imag_index[i*N*Rt+d*Rt:i*N*Rt+(d+1)*Rt,k*Nt:(k+1)*Nt], 
                                                         np.matrix(lr_F0).T)), np.matrix(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, 
                                                                   U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:])).H)*cof20110[k*N+d,i]
                         , np.dot(y00[k*Nr:(k+1)*Nr,i:i+1].conjugate(), np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).T))
                         + (1j)*np.matrix(np.dot(np.dot(np.dot(w1[k*Rr:(k+1)*Rr,d:d+1], np.matrix(sum11[k*Rt:(k+1)*Rt,i:(i+1)]).H), 
                                                   cof10110[k*N+d,i]*np.multiply(G_F_conj_imag_index[i*N*Rt+d*Rt:i*N*Rt+(d+1)*Rt,k*Nt:(k+1)*Nt], 
                                                         np.matrix(lr_F0).T)), np.matrix(H[k*Nr:(k+1)*Nr,:]).H)).T )
                
                
                G_w_imag_index[i*N+d:i*N+d+1,k*Rr:(k+1)*Rr] = ( - (-1)*np.multiply(np.dot(np.multiply(G_p[d:d+1,k*Rt:(k+1)*Rt], 
                              np.matrix(lr_p[k*Rt:(k+1)*Rt,d:d+1]).T), np.matrix(np.dot(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, U1[k*Rr:(k+1)*Rr,:]), 
                                        H[k*Nr:(k+1)*Nr,:]), F1*b[k*N+d,i])).H)*cof20111[k*N+d,i]
                         , np.dot(np.matrix(y01[k*Nr:(k+1)*Nr,i:i+1]).H, np.matrix(U1[k*Rr:(k+1)*Rr,:]).H))
                         + (1j)*np.dot(np.multiply(G_p[d:d+1,k*Rt:(k+1)*Rt], np.matrix(lr_p[k*Rt:(k+1)*Rt,d:d+1]).T)
                         *cof10111[k*N+d,i], np.matrix(np.dot(np.dot(U1[k*Rr:(k+1)*Rr,:], H[k*Nr:(k+1)*Nr,:]), F1*b[k*N+d,i])).H)
                         
                         + G_w[d:d+1,k*Rr:(k+1)*Rr] - (1)*np.multiply(np.dot(np.multiply(G_w[d:d+1,k*Rr:(k+1)*Rr], 
                         np.matrix(lr_w1[k*Rr:(k+1)*Rr,d:d+1]).T), np.dot(U1[k*Rr:(k+1)*Rr,:],y11[k*Nr:(k+1)*Nr,i:(i+1)]))
                         *cof21111[k*N+d,i], np.dot(np.matrix(y11[k*Nr:(k+1)*Nr,i:i+1]).H, np.matrix(U1[k*Rr:(k+1)*Rr,:]).H))
                         
                         - (1)*np.multiply(np.dot(np.dot(np.matrix(y00[k*Nr:(k+1)*Nr,i:i+1]).T, np.multiply(G_U_imag_index[i*N*Nr+d*Nr:i*N*Nr+(d+1)*Nr,k*Rr:(k+1)*Rr], 
                                        np.matrix(lr_U0[k*Rr:(k+1)*Rr,:]).T)), w1[k*Rr:(k+1)*Rr,d:d+1].conjugate())*cof20100[k*N+d,i]
                         , np.dot(np.matrix(y00[k*Nr:(k+1)*Nr,i:i+1]).H, np.matrix(U0[k*Rr:(k+1)*Rr,:]).H))
                         
                         - (-1)*np.multiply(np.dot(np.dot(np.matrix(y00[k*Nr:(k+1)*Nr,i:i+1]).H, np.multiply(G_U_conj_imag_index[i*N*Nr+d*Nr:i*N*Nr+(d+1)*Nr,k*Rr:(k+1)*Rr], 
                                        np.matrix(lr_U0[k*Rr:(k+1)*Rr,:]).T)), w1[k*Rr:(k+1)*Rr,d:d+1])*cof20100[k*N+d,i]
                         , np.dot(np.matrix(y00[k*Nr:(k+1)*Nr,i:i+1]).H, np.matrix(U0[k*Rr:(k+1)*Rr,:]).H))
                         + (1j)*np.dot(np.matrix(y00[k*Nr:(k+1)*Nr,i:i+1]).H, cof10100[k*N+d,i]*np.multiply(G_U_conj_imag_index[i*N*Nr+d*Nr:i*N*Nr+(d+1)*Nr,k*Rr:(k+1)*Rr], 
                                  np.matrix(lr_U0[k*Rr:(k+1)*Rr,:]).T))
                         
                         - (1)*np.multiply(np.dot(np.dot(np.matrix(sum10[k*Rt:(k+1)*Rt,i:(i+1)]).T, np.multiply(G_F_imag_index[i*N*Rt+d*Rt:i*N*Rt+(d+1)*Rt,k*Nt:(k+1)*Nt], 
                                                        np.matrix(lr_F0).T)), np.matrix(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, 
                                                                  U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:])).T)*cof20110[k*N+d,i]
                         , np.dot(np.matrix(y00[k*Nr:(k+1)*Nr,i:i+1]).H, np.matrix(U1[k*Rr:(k+1)*Rr,:]).H))
                         
                         - (-1)*np.multiply(np.dot(np.dot(np.matrix(sum10[k*Rt:(k+1)*Rt,i:(i+1)]).H, np.multiply(G_F_conj_imag_index[i*N*Rt+d*Rt:i*N*Rt+(d+1)*Rt,k*Nt:(k+1)*Nt], 
                                                        np.matrix(lr_F0).T)), np.matrix(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, 
                                                                  U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:])).H)*cof20110[k*N+d,i]
                         , np.dot(np.matrix(y00[k*Nr:(k+1)*Nr,i:i+1]).H, np.matrix(U1[k*Rr:(k+1)*Rr,:]).H))
                         + (1j)*np.dot(np.dot(np.dot(np.matrix(sum10[k*Rt:(k+1)*Rt,i:(i+1)]).H, cof10110[k*N+d,i]*np.multiply(G_F_conj_imag_index[i*N*Rt+d*Rt:i*N*Rt+(d+1)*Rt,k*Nt:(k+1)*Nt], 
                                                  np.matrix(lr_F0).T)), np.matrix(H[k*Nr:(k+1)*Nr,:]).H), np.matrix(U1[k*Rr:(k+1)*Rr,:]).H))
    			
    			
                G_p_imag_index[i*N+d:i*N+d+1,k*Rt:(k+1)*Rt] = (G_p[d:d+1,k*Rt:(k+1)*Rt] - (1)*np.multiply(np.dot(np.multiply(G_p[d:d+1,k*Rt:(k+1)*Rt], 
                         np.matrix(lr_p[k*Rt:(k+1)*Rt,d:d+1]).T), np.matrix(np.dot(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, 
                                   U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]), F1*b[k*N+d,i])).H)*cof20111[k*N+d,i]
                         , np.dot(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]), F1*b[k*N+d,i]))
                         
                         - (-1)*np.multiply(np.dot(np.multiply(G_w_imag_index[i*N+d:i*N+d+1,k*Rr:(k+1)*Rr], np.matrix(lr_w0[k*Rr:(k+1)*Rr,d:d+1]).T), 
                                  np.dot(U0[k*Rr:(k+1)*Rr,:],y00[k*Nr:(k+1)*Nr,i:(i+1)]))*cof20000[k*N+d,i]
                         , np.dot(np.dot(np.dot(np.matrix(w0[k*Rr:(k+1)*Rr,d:d+1]).H, U0[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]), F0*b[k*N+d,i]))
                         + (-1j)*np.dot(np.multiply(G_w_imag_index[i*N+d:i*N+d+1,k*Rr:(k+1)*Rr], np.matrix(lr_w0[k*Rr:(k+1)*Rr,d:d+1]).T)
                         *cof10000[k*N+d,i]*b[k*N+d,i], np.dot(np.dot(U0[k*Rr:(k+1)*Rr,:], H[k*Nr:(k+1)*Nr,:]), F0))
                         
                         - (-1)*np.multiply(np.dot(np.dot(np.matrix(y00[k*Nr:(k+1)*Nr,i:i+1]).T, np.multiply(G_U_imag_index[i*N*Nr+d*Nr:i*N*Nr+(d+1)*Nr,k*Rr:(k+1)*Rr], 
                                        np.matrix(lr_U0[k*Rr:(k+1)*Rr,:]).T)), w1[k*Rr:(k+1)*Rr,d:d+1].conjugate())*cof20100[k*N+d,i]
                         , np.dot(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, U0[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]), F0*b[k*N+d,i]))
                         + (-1j)*np.matrix(np.dot(np.dot(np.dot(np.matrix(F0).T, np.matrix(H[k*Nr:(k+1)*Nr,:]).T), np.multiply(G_U_imag_index[i*N*Nr+d*Nr:i*N*Nr+(d+1)*Nr,k*Rr:(k+1)*Rr], 
                                        np.matrix(lr_U0[k*Rr:(k+1)*Rr,:]).T)*cof10100[k*N+d,i]), w1[k*Rr:(k+1)*Rr,d:d+1].conjugate()*b[k*N+d,i])).T
                         
                         - (1)*np.multiply(np.dot(np.dot(np.matrix(y00[k*Nr:(k+1)*Nr,i:i+1]).H, np.multiply(G_U_conj_imag_index[i*N*Nr+d*Nr:i*N*Nr+(d+1)*Nr,k*Rr:(k+1)*Rr], 
                                        np.matrix(lr_U0[k*Rr:(k+1)*Rr,:]).T)), w1[k*Rr:(k+1)*Rr,d:d+1])*cof20100[k*N+d,i]
                         , np.dot(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, U0[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]), F0*b[k*N+d,i]))
                         
                         - (-1)*np.multiply(np.dot(np.dot(np.matrix(sum10[k*Rt:(k+1)*Rt,i:(i+1)]).T, np.multiply(G_F_imag_index[i*N*Rt+d*Rt:i*N*Rt+(d+1)*Rt,k*Nt:(k+1)*Nt], 
                                                        np.matrix(lr_F0).T)), np.matrix(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, 
                                                                  U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:])).T)*cof20110[k*N+d,i]
                         , np.dot(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]), F0*b[k*N+d,i]))
                         + (-1j)*np.matrix(np.dot(np.dot(np.dot(cof10110[k*N+d,i]*np.multiply(G_F_imag_index[i*N*Rt+d*Rt:i*N*Rt+(d+1)*Rt,k*Nt:(k+1)*Nt], 
                                                        np.matrix(lr_F0).T), np.matrix(H[k*Nr:(k+1)*Nr,:]).H), np.matrix(U1[k*Rr:(k+1)*Rr,:]).H), 
                         w1[k*Rr:(k+1)*Rr,d:d+1].conjugate()*b[k*N+d,i])).T
                         
                         - (1)*np.multiply(np.dot(np.dot(np.matrix(sum10[k*Rt:(k+1)*Rt,i:(i+1)]).H, np.multiply(G_F_conj_imag_index[i*N*Rt+d*Rt:i*N*Rt+(d+1)*Rt,k*Nt:(k+1)*Nt], 
                                                        np.matrix(lr_F0).T)), np.matrix(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, 
                                                                  U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:])).H)*cof20110[k*N+d,i]
                         , np.dot(np.dot(np.dot(np.matrix(w1[k*Rr:(k+1)*Rr,d:d+1]).H, U1[k*Rr:(k+1)*Rr,:]), H[k*Nr:(k+1)*Nr,:]), F0*b[k*N+d,i])))

            
    G_p_imag_index = np.mean([G_p_imag_index[i:i+N,:] for i in range(0,N*batch_size,N)], axis=0, keepdims=True)
    G_p_imag_index = np.reshape(G_p_imag_index, (N,K*Rt))
    
    G_w_imag_index = np.mean([G_w_imag_index[i:i+N,:] for i in range(0,N*batch_size,N)], axis=0, keepdims=True)
    G_w_imag_index = np.reshape(G_w_imag_index, (N,K*Rr))
    
    G_U_imag_index = np.mean([G_U_imag_index[i:i+Nr,:] for i in range(0,Nr*N*batch_size,Nr)], axis=0, keepdims=True)
    G_U_imag_index = np.reshape(G_U_imag_index, (Nr,K*Rr))
    
    G_U_conj_imag_index = np.mean([G_U_conj_imag_index[i:i+Nr,:] for i in range(0,Nr*N*batch_size,Nr)], axis=0, keepdims=True)
    G_U_conj_imag_index = np.reshape(G_U_conj_imag_index, (Nr,K*Rr))
    
    G_F_imag_index = np.mean([G_F_imag_index[i:i+Rt,:] for i in range(0,Rt*N*batch_size,Rt)], axis=0, keepdims=True)
    G_F_imag_index = np.reshape(G_F_imag_index, (Rt,K*Nt))
    
    G_F_imag_index = np.mean([G_F_imag_index[:,i:i+Nt] for i in range(0,K*Nt,Nt)], axis=0, keepdims=True)
    G_F_imag_index = np.reshape(G_F_imag_index, (Rt,Nt))
    
    G_F_conj_imag_index = np.mean([G_F_conj_imag_index[i:i+Rt,:] for i in range(0,Rt*N*batch_size,Rt)], axis=0, keepdims=True)
    G_F_conj_imag_index = np.reshape(G_F_conj_imag_index, (Rt,K*Nt))
    
    G_F_conj_imag_index = np.mean([G_F_conj_imag_index[:,i:i+Nt] for i in range(0,K*Nt,Nt)], axis=0, keepdims=True)
    G_F_conj_imag_index = np.reshape(G_F_conj_imag_index, (Rt,Nt))
    
    G_p_index = G_p_real_index + G_p_imag_index
    G_w_index = G_w_real_index + G_w_imag_index
    G_U_index = G_U_real_index + G_U_imag_index
    G_F_index = G_F_real_index + G_F_imag_index
    G_U_conj_index = G_U_conj_real_index + G_U_conj_imag_index
    G_F_conj_index = G_F_conj_real_index + G_F_conj_imag_index
    
    #G_U_conj_real_index = G_U_real_index.conjugate()
    #G_U_conj_imag_index = G_U_imag_index.conjugate()
    G_theu_index = (np.multiply(G_U_real_index+G_U_imag_index, 1j*np.matrix(U1).T) 
                    - np.multiply(G_U_conj_real_index+G_U_conj_imag_index, 1j*np.matrix(U1).H))
    
    #G_F_conj_real_index = G_F_real_index.conjugate()
    #G_F_conj_imag_index = G_F_imag_index.conjugate()
    G_thef_index = (np.multiply(G_F_real_index+G_F_imag_index, 1j*np.matrix(F1).T) 
                    - np.multiply(G_F_conj_real_index+G_F_conj_imag_index, 1j*np.matrix(F1).H))
    
    return G_p_index, G_w_index, G_U_index, G_F_index, G_U_conj_index, G_F_conj_index, G_theu_index, G_thef_index


delta_ap = np.zeros((K*Rt,N*layer))+1j*np.zeros((K*Rt,N*layer))
delta_aw = np.zeros((K*Rr,N*layer))+1j*np.zeros((K*Rr,N*layer))
delta_atheu = np.zeros((K*Rr,Nr*layer))+1j*np.zeros((K*Rr,Nr*layer))
delta_athef = np.zeros((Nt,Rt*layer))+1j*np.zeros((Nt,Rt*layer))
#delta_p_avr = np.zeros((K*Rt,1))+1j*np.zeros(((K*Rt,1)))
#delta_w_avr = np.zeros((K*Rr,1))+1j*np.zeros(((K*Rr,1)))
delta_p_layer = np.zeros((K*Rt,N*layer))+1j*np.zeros(((K*Rt,N*layer)))
delta_w_layer = np.zeros((K*Rr,N*layer))+1j*np.zeros(((K*Rr,N*layer)))
delta_theu_layer = np.zeros((K*Rr,Nr*layer))+1j*np.zeros((K*Rr,Nr*layer))
delta_thef_layer = np.zeros((Nt,Rt*layer))+1j*np.zeros((Nt,Rt*layer))
delta_U_layer = np.zeros((K*Rr,Nr*layer))+1j*np.zeros((K*Rr,Nr*layer))
delta_F_layer = np.zeros((Nt,Rt*layer))+1j*np.zeros((Nt,Rt*layer))
delta_U_conj_layer = np.zeros((K*Rr,Nr*layer))+1j*np.zeros((K*Rr,Nr*layer))
delta_F_conj_layer = np.zeros((Nt,Rt*layer))+1j*np.zeros((Nt,Rt*layer))

p_layer = np.zeros((K*Rt,N*layer))+1j*np.zeros(((K*Rt,N*layer)))
w_layer = np.zeros((K*Rr,N*layer))+1j*np.zeros(((K*Rr,N*layer)))
theu_layer = np.zeros((K*Rr,Nr*layer))+1j*np.zeros((K*Rr,Nr*layer))
thef_layer = np.zeros((Nt,Rt*layer))+1j*np.zeros((Nt,Rt*layer))
U_layer = np.zeros((K*Rr,Nr*layer))+1j*np.zeros((K*Rr,Nr*layer))
F_layer = np.zeros((Nt,Rt*layer))+1j*np.zeros((Nt,Rt*layer))

p_layer[:,0:N] = p
w_layer[:,0:N] = w
theu_layer[:,0:Nr] = theu
thef_layer[:,0:Rt] = thef
U_layer[:,0:Nr] = U
F_layer[:,0:Rt] = F

lr_p_layer = np.concatenate([lr_p for i in range(layer)], axis=1)
lr_w_layer = np.concatenate([lr_w for i in range(layer)], axis=1)
lr_theu_layer = np.concatenate([lr_theu for i in range(layer)], axis=1)
lr_thef_layer = np.concatenate([lr_thef for i in range(layer)], axis=1)

G_p = np.zeros((N*layer,K*Rt))+1j*np.zeros(((N*layer,K*Rt)))
G_w = np.zeros((N*layer,K*Rr))+1j*np.zeros(((N*layer,K*Rr)))
G_theu = np.zeros((Nr*layer,K*Rr))+1j*np.zeros((Nr*layer,K*Rr))
G_thef = np.zeros((Rt*layer,Nt))+1j*np.zeros((Rt*layer,Nt))
G_U = np.zeros((Nr*layer,K*Rr))+1j*np.zeros((Nr*layer,K*Rr))
G_F = np.zeros((Rt*layer,Nt))+1j*np.zeros((Rt*layer,Nt))
G_U_conj = np.zeros((Nr*layer,K*Rr))+1j*np.zeros((Nr*layer,K*Rr))
G_F_conj = np.zeros((Rt*layer,Nt))+1j*np.zeros((Rt*layer,Nt))

G_p_nor = np.zeros((N*layer,K*Rt))+1j*np.zeros(((N*layer,K*Rt)))
G_w_nor = np.zeros((N*layer,K*Rr))+1j*np.zeros(((N*layer,K*Rr)))
G_theu_nor = np.zeros((Nr*layer,K*Rr))+1j*np.zeros((Nr*layer,K*Rr))
G_thef_nor = np.zeros((Rt*layer,Nt))+1j*np.zeros((Rt*layer,Nt))

G_p_L = np.zeros((N,K*Rt))+1j*np.zeros(((N,K*Rt)))
G_w_L = np.zeros((N,K*Rr))+1j*np.zeros(((N,K*Rr)))
G_theu_L = np.zeros((Nr,K*Rr))+1j*np.zeros((Nr,K*Rr))
G_thef_L = np.zeros((Rt,Nt))+1j*np.zeros((Rt,Nt))
G_U_L = np.zeros((Nr,K*Rr))+1j*np.zeros((Nr,K*Rr))
G_F_L = np.zeros((Rt,Nt))+1j*np.zeros((Rt,Nt))
G_U_conj_L = np.zeros((Nr,K*Rr))+1j*np.zeros((Nr,K*Rr))
G_F_conj_L = np.zeros((Rt,Nt))+1j*np.zeros((Rt,Nt))
#G_p_L = np.zeros((batch_size,K*Rt))+1j*np.zeros(((batch_size,K*Rt)))

for index in range(0,I_max):
    learning_rate = 0.01*math.pow(0.95, np.ceil(index/50))
    #learning_rate_p = 0.005*math.pow(0.95, np.ceil(index/50))
    #learning_rate_w = 0.005*math.pow(0.95, np.ceil(index/50))
    p_layer[:,0:N] = p
    w_layer[:,0:N] = w
    theu_layer[:,0:Nr] = theu
    thef_layer[:,0:Rt] = thef
    U_layer[:,0:Nr] = U
    F_layer[:,0:Rt] = F
    #lr_p_layer = np.concatenate([lr_p for i in range(layer)], axis=1)
    #lr_w_layer = np.concatenate([lr_w for i in range(layer)], axis=1)
    for l in range(0,layer-1):
        '''
        if l==0 and index>0:
            p_layer[:,(l+1):(l+2)], w_layer[:,(l+1):(l+2)], delta_p_layer[:,l:l+1], delta_w_layer[:,l:l+1] = forward(
                p_layer[:,layer-1:],w_layer[:,layer-1:],lr_p_layer[:,l:l+1],lr_w_layer[:,l:l+1])

        else:
            p_layer[:,(l+1):(l+2)], w_layer[:,(l+1):(l+2)], delta_p_layer[:,l:l+1], delta_w_layer[:,l:l+1] = forward(
                p_layer[:,l:(l+1)],w_layer[:,l:(l+1)],lr_p_layer[:,l:(l+1)],lr_w_layer[:,l:(l+1)])
        '''

        (p_layer[:,(l+1)*N:(l+2)*N], w_layer[:,(l+1)*N:(l+2)*N],  theu_layer[:,(l+1)*Nr:(l+2)*Nr], thef_layer[:,(l+1)*Rt:(l+2)*Rt], 
                 delta_p_layer[:,l*N:(l+1)*N], delta_w_layer[:,l*N:(l+1)*N], delta_U_layer[:,l*Nr:(l+1)*Nr], delta_F_layer[:,l*Rt:(l+1)*Rt], 
                 delta_U_conj_layer[:,l*Nr:(l+1)*Nr], delta_F_conj_layer[:,l*Rt:(l+1)*Rt], delta_theu_layer[:,l*Nr:(l+1)*Nr], 
                 delta_thef_layer[:,l*Rt:(l+1)*Rt]) = forward(p_layer[:,l*N:(l+1)*N], w_layer[:,l*N:(l+1)*N], 
                         U_layer[:,l*Nr:(l+1)*Nr], F_layer[:,l*Rt:(l+1)*Rt], theu_layer[:,l*Nr:(l+1)*Nr], thef_layer[:,l*Rt:(l+1)*Rt], 
                         lr_p_layer[:,l*N:(l+1)*N], lr_w_layer[:,l*N:(l+1)*N], lr_theu_layer[:,l*Nr:(l+1)*Nr], lr_thef_layer[:,l*Rt:(l+1)*Rt])
        
        U_layer[:,(l+1)*Nr:(l+2)*Nr] = np.exp(1j*theu_layer[:,(l+1)*Nr:(l+2)*Nr])
        F_layer[:,(l+1)*Rt:(l+2)*Rt] = np.exp(1j*thef_layer[:,(l+1)*Rt:(l+2)*Rt])
        
        delta_ap[:,l*N:(l+1)*N] = -delta_p_layer[:,l*N:(l+1)*N].conjugate()
        delta_aw[:,l*N:(l+1)*N] = -delta_w_layer[:,l*N:(l+1)*N].conjugate()
        delta_atheu[:,l*Nr:(l+1)*Nr] = -delta_theu_layer[:,l*Nr:(l+1)*Nr].conjugate()
        delta_athef[:,l*Rt:(l+1)*Rt] = -delta_thef_layer[:,l*Rt:(l+1)*Rt].conjugate()
        

        sum1 = np.zeros((K*Rt,test_num))+1j*np.zeros((K*Rt,test_num))
        b_hat = np.zeros((K*N,test_num))+1j*np.zeros((K*N,test_num))
        y = np.zeros((K*Nr,test_num))+1j*np.zeros((K*Nr,test_num))
        for i in range(0,test_num):
            for k in range(0,K):
                for d in range(0,N):
                    sum1[k*Rt:(k+1)*Rt,i:(i+1)] = sum1[k*Rt:(k+1)*Rt,i:(i+1)]+np.multiply(p_layer[k*Rt:(k+1)*Rt,l*N+d:l*N+d+1], test_b[k*N+d:k*N+d+1,i:(i+1)])
        #print(sum1)
        
        for i in range(0,test_num):
            for k in range(0,K):
                for d in range(0,N):
                    y[k*Nr:(k+1)*Nr,i:i+1] = np.add(np.dot(np.dot(H[k*Nr:(k+1)*Nr,:], F_layer[:,l*Rt:(l+1)*Rt]), sum1[k*Rt:(k+1)*Rt,i:(i+1)]), n[:,k:k+1])
                    b_hat[k*N+d:k*N+d+1,i:(i+1)] = np.dot(np.dot(np.matrix(w_layer[k*Rr:(k+1)*Rr,l*N+d:l*N+d+1]).H, 
                         U_layer[k*Rr:(k+1)*Rr,l*Nr:(l+1)*Nr]), y[k*Nr:(k+1)*Nr,i:i+1])
        
        
        s = np.zeros((K*N,test_num))+1j*np.zeros((K*N,test_num))
        bit_error = np.zeros((1,test_num))
        for i in range(0,test_num):
            for k in range(0,K):
                for d in range(0,N):
                    s[k*N+d:(k*N+d+1), i:(i+1)] = np.sign(b_hat[k*N+d,i].real) + 1j*np.sign(b_hat[k*N+d,i].imag)
    
                    if test_b[k*N+d,i] != s[k*N+d,i]:
                        bit_error[:,i:(i+1)] += 1

        error.append(np.mean(bit_error))
    #if index%5 == 0:
        #print(error[0])
    print('{0:0.6f}'.format(error[-1]))
    
    k1 = np.zeros((K*Rt,N))+1j*np.zeros((K*Rt,N))
    k2 = np.zeros((K*Rr,N))+1j*np.zeros((K*Rr,N))
    k3 = np.zeros((K*Rr,Nr))+1j*np.zeros((K*Rr,Nr))
    k4 = np.zeros((Nt,Rt))+1j*np.zeros((Nt,Rt))
    (k1, k2, k3, k4, delta_p_layer[:,(layer-1)*N:], delta_w_layer[:,(layer-1)*N:], delta_U_layer[:,(layer-1)*Nr:], 
                     delta_F_layer[:,(layer-1)*Rt:], delta_U_conj_layer[:,(layer-1)*Nr:], delta_F_conj_layer[:,(layer-1)*Rt:], 
                     delta_theu_layer[:,(layer-1)*Nr:], delta_thef_layer[:,(layer-1)*Rt:]) = forward(
                             p_layer[:,(layer-1)*N:], w_layer[:,(layer-1)*N:], U_layer[:,(layer-1)*Nr:], F_layer[:,(layer-1)*Rt:], 
                             theu_layer[:,(layer-1)*Nr:], thef_layer[:,(layer-1)*Rt:], lr_p_layer[:,(layer-1)*N:], 
                             lr_w_layer[:,(layer-1)*N:], lr_theu_layer[:,(layer-1)*Nr:], lr_thef_layer[:,(layer-1)*Rt:])
    #delta_ap[:,layer-1:] = -delta_ap[:,layer-1:]
    #delta_aw[:,layer-1:] = -delta_w_layer[:,layer-1:]

    G_w_L = np.matrix(delta_w_layer[:,(layer-1)*N:]).H
    '''
    sum1 = np.zeros((Rt,batch_size))+1j*np.zeros((Rt,batch_size))
    b_hat = np.zeros((K, batch_size)) + 1j * np.zeros(((K, batch_size)))
    for i in range(0,batch_size):
        for k in range(0,K):
            sum1[:,i:(i+1)] = sum1[:,i:(i+1)]+np.dot(p_layer[k*Rt:(k+1)*Rt,layer-1:], b[k:(k+1),i:(i+1)])
        for k in range(0,K):
            b_hat[k*N:(k+1)*N,i:(i+1)] = np.dot(np.matrix(w_layer[k*Rr:(k+1)*Rr,layer-2:layer-1]).H, np.add(np.dot(H[k*Nr:(k+1)*Nr,:], sum1[:,i:(i+1)]), n[k*Rr:(k+1)*Rr,:]))
        for k in range(0,K):
            G_p_L[i:i+1,k*Rt:(k+1)*Rt] = (-np.multiply(G_w_L[:,k*Rr:(k+1)*Rr], np.matrix(lr_w_layer[k*Rr:(k+1)*Rr,layer-2:layer-1]).H)*b[k*N+d,i]/(2*math.sqrt(2*3.14)*rol[k,d].real)
                                        *math.exp(-math.pow(b_hat[k*N+d,i].real,2)/(2*(rol[k,d]*rol[k,d].conjugate()).real))
                                        *(np.dot(H[k*Nr:(k+1)*Nr,:], sum1[:,i:i+1])*(b_hat[k*N+d,i]).real/(rol[k,d]*rol[k,d].conjugate()).real*b[k*N+d,i]
                                          *np.dot(np.matrix(w_layer[k*Rr:(k+1)*Rr,layer-2:layer-1]).H, H[k*Nr:(k+1)*Nr,:])- b[k:k+1,i:i+1]*H[k*Nr:(k+1)*Nr,:]))
    G_p_L = np.mean(G_p_L, axis=0, keepdims=True)
    '''
    
    G_p_L = np.matrix(delta_p_layer[:,(layer-1)*N:]).H
    G_theu_L = np.matrix(delta_theu_layer[:,(layer-1)*Nr:]).H
    G_thef_L = np.matrix(delta_thef_layer[:,(layer-1)*Rt:]).H
    G_U_L = np.matrix(delta_U_layer[:,(layer-1)*Nr:]).H
    G_F_L = np.matrix(delta_F_layer[:,(layer-1)*Rt:]).H
    G_U_conj_L = np.matrix(delta_U_conj_layer[:,(layer-1)*Nr:]).H
    G_F_conj_L = np.matrix(delta_F_conj_layer[:,(layer-1)*Rt:]).H
    #delta_p_layer[:,layer-1:] = np.matrix(G_p_L).H
    #delta_ap[:,layer-1:] = -delta_p_layer[:,layer-1:]

    G_p[(layer-1)*N:,:] = G_p_L
    G_w[(layer-1)*N:,:] = G_w_L
    G_theu[(layer-1)*Nr:,:] = G_theu_L
    G_thef[(layer-1)*Rt:,:] = G_thef_L
    G_U[(layer-1)*Nr:,:] = G_U_L
    G_F[(layer-1)*Rt:,:] = G_F_L
    G_U_conj[(layer-1)*Nr:,:] = G_U_conj_L
    G_F_conj[(layer-1)*Rt:,:] = G_F_conj_L
    
    
    for d in range(0,N):
        for k in range(0,K):
            G_p_nor[(layer-1)*N+d:(layer-1)*N+d+1,k*Rt:(k+1)*Rt] = G_p_L[d:d+1,k*Rt:(k+1)*Rt]/np.mean(np.abs(G_p_L[d:d+1,k*Rt:(k+1)*Rt]))
            G_w_nor[(layer-1)*N+d:(layer-1)*N+d+1,k*Rr:(k+1)*Rr] = G_w_L[d:d+1,k*Rr:(k+1)*Rr]/np.mean(np.abs(G_w_L[d:d+1,k*Rr:(k+1)*Rr]))
    
    for d in range(0,Nr):
        for k in range(0,K):
            G_theu_nor[(layer-1)*Nr+d:(layer-1)*Nr+d+1,k*Rr:(k+1)*Rr] = G_theu_L[d:d+1,k*Rr:(k+1)*Rr]/np.mean(np.abs(G_theu_L[d:d+1,k*Rr:(k+1)*Rr]))
    
    for d in range(0,Rt):
        G_thef_nor[(layer-1)*Rt+d:(layer-1)*Rt+d+1,:] = G_thef_L[d:d+1,:]/np.mean(np.abs(G_thef_L[d:d+1,:]))
    
    
    delta_ap[:,(layer-2)*N:(layer-1)*N] = np.multiply(np.matrix(G_p_nor[(layer-1)*N:,:]).H, delta_ap[:,(layer-2)*N:(layer-1)*N])
    #delta_ap_mean = np.mean(delta_ap, axis=0, keepdims=True)
    #delta_ap_nor = delta_ap/delta_ap_mean
    
    delta_aw[:,(layer-2)*N:(layer-1)*N] = np.multiply(np.matrix(G_w_nor[(layer-1)*N:,:]).H, delta_aw[:,(layer-2)*N:(layer-1)*N])
    #delta_aw_mean = np.mean(delta_aw, axis=0, keepdims=True)
    #delta_aw_nor = delta_aw/delta_aw_mean
    delta_atheu[:,(layer-2)*Nr:(layer-1)*Nr] = np.multiply(np.matrix(G_theu_nor[(layer-1)*Nr:,:]).H, delta_atheu[:,(layer-2)*Nr:(layer-1)*Nr])
    delta_athef[:,(layer-2)*Rt:(layer-1)*Rt] = np.multiply(np.matrix(G_thef_nor[(layer-1)*Rt:,:]).H, delta_athef[:,(layer-2)*Rt:(layer-1)*Rt])
    
    
    #p0, p1, w0, w1, w2, U0, U1, U2, F0, F1, G_p, G_w, G_U, G_F, lr_p, lr_w0, lr_w1, lr_U0, lr_U1, lr_F0, lr_F1
    for l in range(layer-2,0,-1):
        (G_p[l*N:(l+1)*N,:], G_w[l*N:(l+1)*N,:], G_U[l*Nr:(l+1)*Nr,:], G_F[l*Rt:(l+1)*Rt,:], G_U_conj[l*Nr:(l+1)*Nr,:], G_F_conj[l*Rt:(l+1)*Rt,:], 
             G_theu[l*Nr:(l+1)*Nr,:], G_thef[l*Rt:(l+1)*Rt,:]) = back(p_layer[:,l*N:(l+1)*N], p_layer[:,(l+1)*N:(l+2)*N], w_layer[:,(l-1)*N:l*N], 
                w_layer[:,l*N:(l+1)*N], w_layer[:,(l+1)*N:(l+2)*N], U_layer[:,(l-1)*Nr:l*Nr], U_layer[:,l*Nr:(l+1)*Nr], 
                U_layer[:,(l+1)*Nr:(l+2)*Nr], F_layer[:,(l-1)*Rt:l*Rt], F_layer[:,l*Rt:(l+1)*Rt], G_p[(l+1)*N:(l+2)*N,:], G_w[(l+1)*N:(l+2)*N,:], 
                G_U[(l+1)*Nr:(l+2)*Nr,:], G_F[(l+1)*Rt:(l+2)*Rt,:], G_U_conj[(l+1)*Nr:(l+2)*Nr,:], G_F_conj[(l+1)*Rt:(l+2)*Rt,:], 
                lr_p_layer[:,l*N:(l+1)*N], lr_w_layer[:,(l-1)*N:l*N], lr_w_layer[:,l*N:(l+1)*N], lr_theu_layer[:,(l-1)*Nr:l*Nr], 
                lr_theu_layer[:,l*Nr:(l+1)*Nr], lr_thef_layer[:,(l-1)*Rt:l*Rt], lr_thef_layer[:,l*Rt:(l+1)*Rt])
           
        
        #G_p[l*N:(l+1)*N,:] = G_p[l*N:(l+1)*N,:]/np.abs(np.mean(G_p[l*N:(l+1)*N,:]))
        #G_w[l*N:(l+1)*N,:] = G_w[l*N:(l+1)*N,:]/np.abs(np.mean(G_w[l*N:(l+1)*N,:]))
        for d in range(0,N):
            for k in range(0,K):
                G_p_nor[l*N+d:l*N+(d+1),k*Rt:(k+1)*Rt] = G_p[l*N+d:l*N+(d+1),k*Rt:(k+1)*Rt]/np.mean(np.abs(G_p[l*N+d:l*N+(d+1),k*Rt:(k+1)*Rt]))
                G_w_nor[l*N+d:l*N+(d+1),k*Rr:(k+1)*Rr] = G_w[l*N+d:l*N+(d+1),k*Rr:(k+1)*Rr]/np.mean(np.abs(G_w[l*N+d:l*N+(d+1),k*Rr:(k+1)*Rr]))
       
        for d in range(0,Nr):
            for k in range(0,K):
                G_theu_nor[l*Nr+d:l*Nr+d+1,k*Rr:(k+1)*Rr] = G_theu[l*Nr+d:l*Nr+d+1,k*Rr:(k+1)*Rr]/np.mean(np.abs(G_theu[l*Nr+d:l*Nr+d+1,k*Rr:(k+1)*Rr]))
        
        for d in range(0,Rt):
            G_thef_nor[l*Rt+d:l*Rt+d+1,:] = G_thef[l*Rt+d:l*Rt+d+1,:]/np.mean(np.abs(G_thef[l*Rt+d:l*Rt+d+1,:]))
        
        
        delta_ap[:,(l-1)*N:l*N] = np.multiply(np.matrix(G_p_nor[l*N:(l+1)*N,:]).H, delta_ap[:,(l-1)*N:l*N])
        delta_aw[:,(l-1)*N:l*N] = np.multiply(np.matrix(G_w_nor[l*N:(l+1)*N,:]).H, delta_aw[:,(l-1)*N:l*N])
        delta_atheu[:,(l-1)*Nr:l*Nr] = np.multiply(np.matrix(G_theu_nor[l*Nr:(l+1)*Nr,:]).H, delta_atheu[:,(l-1)*Nr:l*Nr])
        delta_athef[:,(l-1)*Rt:l*Rt] = np.multiply(np.matrix(G_thef_nor[l*Rt:(l+1)*Rt,:]).H, delta_athef[:,(l-1)*Rt:l*Rt])
        

    '''
    for l in range(0,layer-1):
        delta_ap[:,l*N:(l+1)*N] = delta_ap[:,l*N:(l+1)*N]/np.abs(np.mean(delta_ap[:,l*N:(l+1)*N]))
        delta_aw[:,l*N:(l+1)*N] = delta_aw[:,l*N:(l+1)*N]/np.abs(np.mean(delta_aw[:,l*N:(l+1)*N]))
    '''
    
    lr_p_layer = lr_p_layer - learning_rate*delta_ap
    lr_w_layer = lr_w_layer - learning_rate*delta_aw
    lr_theu_layer = lr_theu_layer - learning_rate*delta_atheu
    lr_thef_layer = lr_thef_layer - learning_rate*delta_athef

    #delta_w_layer = np.matrix(G_w).H
    #delta_p_layer = np.matrix(G_p).H
#sio.savemat("training_bit_error-10^4_4.mat", {"bit_error5": error})






