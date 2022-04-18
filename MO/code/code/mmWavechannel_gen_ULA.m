function [H,At,Ar] = mmWavechannel_gen_ULA(R,T,K,frequecy,L,azimuth_receive,azimuth_transmit)
H=zeros(R,T,K);
lammda = 3e8/frequecy;
d = lammda/2;
al = complex(randn(1,L,K),randn(1,L,K))/sqrt(2)*2;
azimuth_angle_receive = unifrnd(-azimuth_receive,azimuth_receive,1,L,K);
azimuth_angle_transmit = unifrnd(-azimuth_transmit,azimuth_transmit,1,L,K);
a_receive = zeros(R,1,K);
a_transmit = zeros(T,1,K);
At = zeros(T,L,K);
Ar = zeros(R,L,K);
for user = 1:K
    for ii = 1:L
        for jj = 1:R
            a_receive(jj,1,user) = exp(1i*(jj-1)*2*pi/lammda*d*sin(azimuth_angle_receive(1,ii,user)));
        end
        for jj = 1:T
            a_transmit(jj,1,user) = exp(1i*(jj-1)*2*pi/lammda*d*sin(azimuth_angle_transmit(1,ii,user)));
        end
        At(:,ii,user) = a_transmit(:,1,user)/sqrt(T);
        Ar(:,ii,user) = a_receive(:,1,user)/sqrt(R);
        H(:,:,user) = H(:,:,user) + sqrt(R*T/L)*al(1,ii,user)*a_receive(:,:,user)/sqrt(R)*a_transmit(:,:,user)'/sqrt(T);
    end
end

end

