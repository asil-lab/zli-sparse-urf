function I = incidence_matrix_fully_connected(n)
    % I is the NxM incidence matrix
    % n is the the number of nodes in the graph

    % Calculate the number of edges
    m = n * (n - 1) / 2;
    
    % Initialize the incidence matrix with zeros
    I = zeros(n, m);
    
    % Initialize edge counter
    edge_counter = 1;
    
    % Fill in the incidence matrix
    for i = 1:n-1
        for j = i+1:n
            I(i, edge_counter) = 1;
            I(j, edge_counter) = -1;
            edge_counter = edge_counter + 1;
        end
    end
end