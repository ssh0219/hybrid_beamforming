function [ Urf,Ubb ] = OMP_for_U(Uopt,Nrf,Ar)
Urf = [];
Fres = Uopt;
phi = Ar'*Fres;
diag_phi = real(diag(phi*phi'));
index = diag_phi==max(diag_phi);
Urf = [Urf Ar(:,index)];
Urf = Urf./abs(Urf);
Ubb = pinv(Urf)*Uopt;
aa = Uopt-Urf*Ubb;
Fres = aa/norm(aa,'fro');
end