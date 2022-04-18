function delta = FrankW_cccp(delta,Ks,R,K,row,V,sigma)
upbound = zeros(1,length(diag(delta)));
for ii = 1:R
    J = zeros(R,R);
    J(ii,ii) = 1;
    upbound(ii) = -trace(row^2/K^2/sigma^2*V'*J*V/(eye(K)+row/K/sigma*V'*delta*V)*V'*J*V/(eye(K)+row/K/sigma*V'*delta*V));
end
k=0;
while(1)
    if(2^k>abs(real(sum(upbound))/2))
        beta = abs(real(sum(upbound))/2);
    else
        beta = 2^k;
    end
    rate = real(log(det(eye(K)+row/K/sigma*V'*delta*V))) + beta*(sum(diag(delta).^2)-Ks);
    coefficient = real(diag(row/K/sigma*V/(eye(K)+row/K/sigma*V'*delta*V)*V'))+2*beta*diag(delta);
    [a,index]=sort(coefficient,'descend');
    diag_after = zeros(length(diag(delta)),1);
    diag_after(index(1:Ks))=1;
    delta_after = diag(diag_after);
    rate_after = real(log(det(eye(K)+row/K/sigma*V'*delta*V)))+beta*(sum(diag(delta_after).^2)-Ks);
    if(rate_after>rate || beta>=abs(real(sum(upbound))/2))
        delta = delta_after;
        break;
    else
        k = k+1;
    end
end
end

