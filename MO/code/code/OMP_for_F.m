function [ Frf,Fbb ] = OMP_for_F(Fopt,Nrf,At,Ns)
Frf = [];
Fres = Fopt;
for ii = 1:Nrf
    phi = At'*Fres;
    diag_phi = real(diag(phi*phi'));
    index = diag_phi==max(diag_phi);
    Frf = [Frf At(:,index)];
    Fbb = pinv(Frf)*Fopt;
    aa = Fopt-Frf*Fbb;
    Fres = aa/norm(aa,'fro');
end
Fbb = sqrt(Ns)*Fbb/norm(Frf*Fbb,'fro');
end