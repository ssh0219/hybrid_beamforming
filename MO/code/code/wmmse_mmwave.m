function [f,U] = wmmse_mmwave(f,V,delta,H,K,row,sigma,R,T,epsilon,P)
rate_before = 0;
% while(1)
    U = (row/K*delta*V*V'*delta+sigma*eye(R))\delta*V*sqrt(row)/K;
    W = inv(eye(K)/K-sqrt(row)/K*V'*delta*U);
    miu_left = zeros(1,K);
    miu_right = ones(1,K);
    step = 1;
    B = delta*U*W;
    for ii = 1:K
        while(1)
            J = (row/K*H(:,:,ii)'*delta*U*W*U'*delta*H(:,:,ii)+miu_right(ii)*eye(T))\H(:,:,ii)'*B(:,ii)*sqrt(row)/K;
            Power = real(trace(J*J'));
            if(Power<P)
                break;
            else
                miu_right(ii) = miu_right(ii) + step;
            end
        end
    end
    for ii = 1:K
        miu_mid = 1;
        while(miu_mid>=1e-5)
            miu_mid = (miu_left(ii)+miu_right(ii))/2;
            J = (row/K*H(:,:,ii)'*delta*U*W*U'*delta*H(:,:,ii)+miu_mid*eye(T))\H(:,:,ii)'*B(:,ii)*sqrt(row)/K;
            Power = real(trace(J*J'));
            if(Power<P)
                miu_right(ii) = miu_mid;
            elseif(Power>P)
                miu_left(ii) = miu_mid;
            else
                break;
            end
            if(abs(miu_left(ii)-miu_right(ii))<=epsilon)
                break;
            end
        end
        f(:,ii)=J;
    end
%     for ii = 1:K
%         V(:,ii) = H(:,:,ii)*f(:,ii);
%     end
%     rate = real(log(det(eye(R)+row/K/sigma*delta*V*V'*delta)));
%     if(abs(rate-rate_before)<=epsilon)
%         break;
%     else
%         rate_before = rate;
%     end
% end
end

