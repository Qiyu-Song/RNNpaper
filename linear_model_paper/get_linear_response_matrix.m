load model/sys_msinefx4_0_2_64_free_prediction_estx0_nomean_tabs_addp_cov_freq2t_CVA_96_128_128.mat sys
A=sys.A;B=sys.B;C=sys.C(1:40,:);
M=-inv(C*((eye(64)-A)\B));
writematrix(M, 'linear_response_matrix.txt', 'Delimiter', 'tab');

[V,D] = eig(M);
scatter(real(diag(D)), imag(diag(D))); hold on

% Flip the sign of the only positive eigenvalue
for i = 1:length(D)
    if real(D(i,i)) > 0
        D(i,i)
        D(i,i) = -real(D(i,i))+1j*imag(D(i,i));
    end
end
M_star = real(V*D/V);

[V,D] = eig(M_star);
scatter(real(diag(D)), imag(diag(D)));

writematrix(M_star, 'linear_response_matrix_star.txt', 'Delimiter', 'tab');