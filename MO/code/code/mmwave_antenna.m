tic;
clear all;
clc;close all;
T = 2;
R = 16;
RF_matrix = 2;
Ks_matrix = 2:2:16;
K=2;
epsilon = 1e-3;
frequency = 3e10;
azimuth_receive = 2*pi;
azimuth_transmit = 2*pi;
L=20;
sigma = 1;
P=1;
SNR_matrix = -10;
loop_time = 100;
max_iter = 200;
rate_SNR = zeros(1,length(SNR_matrix));
rate_Ks = zeros(1,length(Ks_matrix));
rate_RFt = zeros(1,length(RF_matrix));
for RF_index = 1:length(RF_matrix)
    RF = RF_matrix(RF_index);
%     rate_sum = 0;
    for Ks_index = 1:length(Ks_matrix)
        Ks = Ks_matrix(Ks_index);
        for SNR_index = 1:length(SNR_matrix)
            SNR = SNR_matrix(SNR_index);
            row = sigma*10^(SNR/10);
            rate_sum = 0;
            for loop = 1:loop_time
                [H,At,Ar] = mmWavechannel_gen_ULA(R,T,K,frequency,L,azimuth_receive,azimuth_transmit);
                Ubb = complex(randn(RF,K),randn(RF,K));
                Urf = complex(randn(R,RF),randn(R,RF));
                Urf = Urf./abs(Urf);
                f = complex(randn(T,K),randn(T,K));
                for ii = 1:K
                    Power = real(trace(f(:,ii)*f(:,ii)'));
                    coefficient = 1/Power;
                    f(:,ii) = f(:,ii)*sqrt(coefficient);
                end 
                delta = diag(ones(1,R)*Ks/R);
%                 delta= randselect(Ks,R);
                rate_before = 0;
                rate_matrix = [];
                V = zeros(R,K);
                for ii = 1:K
                    V(:,ii) = H(:,:,ii)*f(:,ii);
                end
                for iter = 1:max_iter
                    delta = FrankW_cccp(delta,Ks,R,K,row,V,sigma);
                    [f,U] = wmmse_mmwave(f,V,delta,H,K,row,sigma,R,T,epsilon,P);
                    for ii = 1:K
                        V(:,ii) = H(:,:,ii)*f(:,ii);
                    end
                    A = delta*V*V'*delta;
                    rate = real(log(det(eye(K)+row/K/sigma*Ubb'*Urf'*A*Urf*Ubb/(Ubb'*Urf'*Urf*Ubb))));
%                     rate = real(log(det(eye(K)+row/K/sigma*U'*A*U/(U'*U))));
                    if(abs(rate-rate_before)<=epsilon)
                        break;
                    else
                        rate_before = rate;
                        rate_matrix = [rate_matrix rate];
                    end
                end
%                 plot(rate_matrix);
                rate_matrix = [];
                for iter = 1:max_iter
                    Ubb = pinv(Urf)*U;
                    [Urf,Ubb]  = Manifold_Urf(Urf,Ubb,A,row,K,sigma,R,RF,epsilon);
                    rate = real(log(det(eye(K)+row/K/sigma*Ubb'*Urf'*A*Urf*Ubb/(Ubb'*Urf'*Urf*Ubb))));
                    if(abs(rate-rate_before)<=epsilon)
                        break;
                    else
                        rate_before = rate;
                        rate_matrix = [rate_matrix rate];
                    end
                end
%                 plot(rate_matrix);
                rate_sum = rate_sum + rate;
                loop
            end
            SNR_index
            rate_SNR(SNR_index) = rate_sum/loop_time;
        end
        rate_Ks(Ks_index) = rate_sum/loop_time;
    end
    rate_RFt(RF_index) = rate_sum/loop_time;
end
toc;
% aa = 200+120*RF_matrix+Ks_matrix*(20+20*RF_matrix);
% EE = rate_Ks./aa;
% plot(EE);