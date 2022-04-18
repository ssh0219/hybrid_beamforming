function alpha = Armijo_Urf(grad,delta_F,R,K,RF,x,rate_before,sigma,row,Ubb,A)
tuo = 0.5;
c = 0.2;
m = grad'*delta_F;
t = -c*m;
alpha = 1;
iter = 1;
while(iter<= 1000)
    iter = iter+1;
    x_new = (x+alpha*grad)./abs(x+alpha*grad);
    Urf = reshape(x_new,[R RF]);
%     Ubb = pinv(Urf)*U;
    rate = -real(log(det(eye(K)+row/K/sigma*Ubb'*Urf'*A*Urf*Ubb/(Ubb'*Urf'*Urf*Ubb))));
    
    if(rate<=rate_before+alpha*t)
        break;
    else
        alpha = tuo*alpha;
    end   
end
end

