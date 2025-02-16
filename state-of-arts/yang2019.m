function [stress] = yang2019(nominal_config)
    % nominal_config: dxn

    [d,n] = size(nominal_config);
    nominal_config_aug = [nominal_config', ones(n,1)]';

    % Validate inputs
    if d > n - 2
        error('n must be at least d+2 for a valid matrix.');
    end
    
    % get D matrix in a vectorized version
    % Define matrix dimensions
    rows = n;
    cols = n - d - 1;
    
    % Initialize matrix with zeros
    M = zeros(rows, cols);
    
    % Populate the matrix with random values where conditions are met
    for j = 1:cols
        % The nonzero elements should start from the 1st to (d+2)th rows and
        % end at the (n-(d+1))th to nth columns for each column
        for i = j:j+d+1
            M(i, j) = rand();  % Assign random numbers to non-zero elements
        end
    end
    
    % Vectorize the matrix (convert it to a column vector)
    M_vectorized = M(:);
    
    % Get indices of zero elements after vectorization
    zero_idx = find(M_vectorized == 0);

    % prepare Q to be solved
    Q_reduced = kron(eye(n-d-1), nominal_config_aug);
    Q_reduced(:,zero_idx) = [];
    
    % solve the zero-free underdetermined system
    [U,S,V] = svd(Q_reduced);
    D_vec_short = V(:,end);
    
    % put the zero values back
    D_vec_full = zeros(n*(n-d-1), 1);
    % Assign the nonzero values from the shortened vector to the remaining indices
    nonzero_indices = setdiff(1:n*(n-d-1), zero_idx); % Find indices that are not zero
    D_vec_full(nonzero_indices) = D_vec_short;
    
    D = reshape(D_vec_full,[n,n-d-1]);
    disp(D);
    stress = D*D';

    % normalize the values for fair comparison with other algorithms
    % stress = stress./norm(stress);


end