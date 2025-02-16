function [stress] = lin2016(incidence_mat, nominal_config)
% nominal config: DxN
    [N,M] = size(incidence_mat);
    [D,~] = size(nominal_config);

    augmented_config = [nominal_config',ones(N,1)];
    [U,~,~] = svd(augmented_config);
    Q = U(:,D+2:end);
    
    % this is the original formulation
    % cvx_begin sdp
    % 
    %     variables weights(M) lambda stress(N,N)
    %     minimize (-lambda)
    %     subject to
    %         lambda >= 0;
    %         lambda <= 5; 
    %         stress == incidence_mat*diag(weights)*incidence_mat';
    %         Q'*stress*Q >= lambda*eye(N-D-1);
    %         stress*nominal_config'== 0;
    % 
    % cvx_end

    % this is an simplified formulation
    cvx_begin sdp
        
        variables weights(M) lambda stress(N,N)       
            stress == incidence_mat*diag(weights)*incidence_mat';
            Q'*stress*Q >= 1e-3*eye(N-D-1);
            stress*nominal_config'== 0;

    cvx_end
    % weights
    weights = weights./norm(weights);
    stress = incidence_mat*diag(weights)*incidence_mat';

    % stress = stress./norm(stress);
end