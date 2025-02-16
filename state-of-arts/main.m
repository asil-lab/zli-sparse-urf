clear;clc;close all;

% circular config
n_nodes = 10;
nominal_config = circular_config(2,n_nodes);

stress_mat_yang2019 = yang2019(nominal_config);
eig(stress_mat_yang2019)

% save('stress_yang2019_circular',"stress_mat_yang2019");


% Grunbaum polygon + lin2016 
B = [1,-1,0,0,0,0,0,0,0,0;
     1,0,-1,0,0,0,0,0,0,0;
     1,0,0,-1,0,0,0,0,0,0;
     1,0,0,0,-1,0,0,0,0,0;
     1,0,0,0,0,-1,0,0,0,0;
     1,0,0,0,0,0,-1,0,0,0;
     1,0,0,0,0,0,0,-1,0,0;
     1,0,0,0,0,0,0,0,-1,0;
     1,0,0,0,0,0,0,0,0,-1;
     0,1,-1,0,0,0,0,0,0,0;
     0,1,0,0,0,0,0,0,0,-1;
     0,0,1,-1,0,0,0,0,0,0;
     0,0,0,1,-1,0,0,0,0,0;
     0,0,0,0,1,-1,0,0,0,0;
     0,0,0,0,0,1,-1,0,0,0;
     0,0,0,0,0,0,1,-1,0,0;
     0,0,0,0,0,0,0,1,-1,0;
     0,0,0,0,0,0,0,0,1,-1]';

stress_mat_lin2016_Grunbaum = lin2016(B,nominal_config);
save('stress_lin2016_Grunbaum_circular',"stress_mat_lin2016_Grunbaum",'B');

