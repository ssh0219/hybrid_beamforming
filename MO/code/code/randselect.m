function delta= randselect(Ks,R)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
delta_diag = randperm(R);
delta_before = zeros(1,R);
delta_before(delta_diag(1:Ks))=1;
delta = diag(delta_before);
end

