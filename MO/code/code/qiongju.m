function delta = qiongju(Urf,Ubb,V,Ks,R,sigma,rho,K)
A=combntns(1:R,Ks);
[row,column] = size(A);
result = zeros(1,row);
for ii = 1:row
    diag_delta = zeros(1,R);
    diag_delta(A(ii,:)) = 1;
    delta = diag(diag_delta);
    result(ii) = real(log(det(eye(K)+rho/K/sigma*Ubb'*Urf'*delta*V*V'*delta*Urf*Ubb/(Ubb'*Urf'*Urf*Ubb))));
end
aa = find(result==max(result));
diag_delta = zeros(1,R);
diag_delta(A(aa,:)) = 1;
delta = diag(diag_delta);
end

