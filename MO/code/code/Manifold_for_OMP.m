function  [Urf]  = Manifold_for_OMP(U,Urf,Ubb,K,R,RFt,epsilon)
xopt = reshape(U,[R*K,1]);
x = reshape(Urf,[R*RFt,1]);
triangle_f = -2*kron(conj(Ubb),eye(R))*(xopt-kron(Ubb.',eye(R))*x);
grad_F = triangle_f-real(triangle_f.*conj(x)).*x;
delta=-grad_F;
rate_before = norm(U-Urf*Ubb,'fro')^2;
rate_matrix = [];
while(1)
    alpha = Armijo_for_OMP(U,delta,triangle_f,R,RFt,x,Ubb,rate_before);
    x_new = (x+alpha*delta)./abs(x+alpha*delta);
    Frf_new = reshape(x_new,[R RFt]);
    triangle_f_new = triangle_f-real(triangle_f.*conj(x_new)).*x_new;
    grad_f_new = triangle_f_new-real(triangle_f_new.*conj(x_new)).*x_new;
    grad_transp = grad_f_new-real(grad_f_new.*conj(x_new)).*x_new;
    delta_transp = delta-real(delta.*conj(x_new)).*x_new;
    beta = triangle_f_new'*(triangle_f_new-triangle_f)/(triangle_f'*triangle_f);
    delta = -grad_f_new+beta*delta_transp;
    Urf = Frf_new;
    if(real(trace(Urf*Ubb*Ubb'*Urf'))>K)
        factor = K/real(trace(Urf*Ubb*Ubb'*Urf'));
        Ubb = Ubb*sqrt(factor);
    end
    x = reshape(Urf,[R*RFt,1]);
    rate = norm(U-Urf*Ubb,'fro')^2;
    if(abs(rate-rate_before)<=epsilon)
        break;
    else
        rate_matrix = [rate_matrix -rate_before];
        rate_before = rate;
        triangle_f = triangle_f_new;
    end
end
end

