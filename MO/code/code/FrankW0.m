function delta = FrankW0(delta,F,H,Ks,Ns,sigma,row)
diag_before = diag(delta);
k = 0;
rate = real(log(det(eye(Ns)+row/Ns/sigma*F'*H'*delta*H*F)));
while(1)
	%求导
    coefficient = real(diag(row/Ns/sigma*H*F/(eye(Ns)+row/Ns/sigma*F'*H'*delta*H*F)*F'*H'));
    [a,index]=sort(coefficient,'descend');
    diag_after = zeros(length(diag(delta)),1);
    diag_after(index(1:Ks))=1;
    detection = sum(coefficient.*(diag_after-diag_before));
    if(detection<=0.01)
        break;
    else
        diag_before = diag_before + 2/(k+2)*(diag_after-diag_before);
        delta = diag(diag_before);
    end
    k = k+1;
    rate = [rate real(log(det(eye(Ns)+row/Ns/sigma*F'*H'*delta*H*F)))];
end
% figure;plot(rate);
end

