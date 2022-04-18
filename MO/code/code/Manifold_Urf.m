function  [Urf,Ubb]  = Manifold_Urf(Urf,Ubb,A,row,K,sigma,R,RF,epsilon)
x = reshape(Urf,[R*RF,1]);
B = Ubb'*Urf'*Urf*Ubb;
C = eye(K)+row/K/sigma*Ubb'*Urf'*A*Urf*Ubb/B;
triangle_f = -2*row/K/sigma*A*Urf*Ubb/B'/C'*Ubb'+2*row/K/sigma*Urf*Ubb/B'*Ubb'*Urf'*A'*Urf*Ubb/C'/B'*Ubb';
triangle_f = reshape(triangle_f,[R*RF,1]);
grad_F = triangle_f-real(triangle_f.*conj(x)).*x;
delta=-grad_F;
rate_before = -real(log(det(eye(K)+row/K/sigma*Ubb'*Urf'*A*Urf*Ubb/B)));
rate_matrix = [];
iter = 1;
while(iter<=1000)
    alpha = Armijo_Urf(delta,triangle_f,R,K,RF,x,rate_before,sigma,row,Ubb,A);
    x_new = (x+alpha*delta)./abs(x+alpha*delta);
    Urf_new = reshape(x_new,[R RF]);
    B = Ubb'*Urf_new'*Urf_new*Ubb;
    C = eye(K)+row/K/sigma*Ubb'*Urf_new'*A*Urf_new*Ubb/B;
    triangle_f_new = -2*row/K/sigma*A*Urf_new*Ubb/B'/C'*Ubb'+2*row/K/sigma*Urf_new*Ubb/B'*Ubb'*Urf_new'*A'*Urf_new*Ubb/C'/B'*Ubb';
    triangle_f_new = reshape(triangle_f_new,[R*RF,1]);
    grad_f_new = triangle_f_new-real(triangle_f_new.*conj(x_new)).*x_new;
    grad_transp = grad_f_new-real(grad_f_new.*conj(x_new)).*x_new;
    delta_transp = delta-real(delta.*conj(x_new)).*x_new;
    beta = triangle_f_new'*(triangle_f_new-triangle_f)/(triangle_f'*triangle_f);
    delta = -grad_f_new+beta*delta_transp;
    Urf = Urf_new;
    x = reshape(Urf,[R*RF,1]);
    rate = -real(log(det(eye(K)+row/K/sigma*Ubb'*Urf'*A*Urf*Ubb/B)));
    if(abs(rate-rate_before)<=epsilon)
        break;
    else
        rate_matrix = [rate_matrix -rate_before];
        rate_before = rate;
        triangle_f = triangle_f_new;
    end
    iter = iter + 1;
end
end

