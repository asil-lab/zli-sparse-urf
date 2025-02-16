function positions = circular_config(r, number_of_sensors)

    % Create an array of angles for the sensors on the circle, excluding 2π
    angles = linspace(0, 2 * pi, number_of_sensors + 1);  % Include an extra point
    angles(end) = [];  % Remove the last angle (2π) to avoid overlap
    
    % Calculate the x and y coordinates for each sensor on the circle
    x_positions = r * cos(angles);
    y_positions = r * sin(angles);
    
    % Stack the x and y positions into a 2xN array
    positions = [x_positions; y_positions];
    

end
