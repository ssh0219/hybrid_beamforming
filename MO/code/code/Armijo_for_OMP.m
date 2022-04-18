function alpha = Armijo_for_OMP(Fopt,grad,delta_F,T,RFt,x,Ubb,rate_before)
tuo = 0.5;
c = 0.2;
m = grad'*delta_F;
t = -c*m;
alpha = 1;
iter = 1;
while(iter<=1000)
    iter = iter+1;
    x_new = (x+alpha*grad)./abs(x+alpha*grad);
    Frf = reshape(x_new,[T RFt]);
    rate = norm(Fopt-Frf*Ubb,'fro')^2;
    
    if(rate<=rate_before+alpha*t)
        break;
    else
        alpha = tuo*alpha;
    end   
end
end

