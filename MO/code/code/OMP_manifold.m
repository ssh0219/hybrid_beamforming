clear all;
clc;close all;
T = 2;
R = 64;
RFt_matrix = 8;
Ns = 4;
Ks_matrix = 16;
K=1;
epsilon = 1e-3;
frequency = 3e10;
azimuth_receive = pi/3;
azimuth_transmit = pi/3;
L=6;
sigma = 1;
SNR_matrix = 0;
loop_time = 100;
max_iter = 200;
rate_SNR = zeros(1,length(SNR_matrix));
rate_Ks = zeros(1,length(Ks_matrix));
rate_RFt = zeros(1,length(RFt_matrix));
for RFt_index = 1:length(RFt_matrix)
    RFt = RFt_matrix(RFt_index);
    rate_sum = 0;
    for Ks_index = 1:length(Ks_matrix)
        Ks = Ks_matrix(Ks_index);
        for SNR_index = 1:length(SNR_matrix)
            SNR = SNR_matrix(SNR_index);
            row = Ns*sigma*10^(SNR/10);
%             rate_sum = 0;
            rate_before = 0;
            for loop = 1:loop_time
                [H,At,Ar] = mmWavechannel_gen_ULA(R,T,K,frequency,L,azimuth_receive,azimuth_transmit);
                Fbb = complex(randn(RFt,Ns),randn(RFt,Ns));
                Frf = complex(randn(T,RFt),randn(T,RFt));
                Frf = Frf./abs(Frf);
                factor = Ns/real(trace(Frf*Fbb*Fbb'*Frf'));
                Fbb = Fbb*sqrt(factor);
                [U,S,V] = svd(H);
                Fopt = V(:,1:Ns);
                Uopt = U(:,1:Ns);
                for iter = 1:max_iter
                    [Frf,Fbb]  = Manifold_for_OMP(Fopt,Frf,Fbb,Ns,T,RFt,epsilon);
                    F = Frf*Fbb;
                    delta = diag(ones(1,R)*Ks/R);
                    delta = FrankW_cccp(delta,Ks,R,F,H,Ns,sigma,row);
                    rate = real(log(det(eye(Ns)+row/Ns/sigma*Uopt'*delta*H*Frf*Fbb*Fbb'*Frf'*H'*delta*Uopt*(Uopt'*Uopt))));
                    if(abs(rate-rate_before)<=1e-3)
                        break;
                    else
                        rate_before = rate;
                    end
                end
                rate_sum = rate_sum + rate;
                loop
            end
            SNR_index
            rate_SNR(SNR_index) = rate_sum/loop_time;
            rate_Ks(Ks_index) = rate_sum/loop_time;
        end
    end
    rate_RFt(RFt_index) = rate_sum/loop_time;
end