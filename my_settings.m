function my_settings()
global activation_fn
global cost_type
global eta
global lmbda
global monitor_training_cost
global monitor_training_accuracy
global monitor_evaluation_cost
global monitor_evaluation_accuracy
global sizes
global num_layers
% define each layer of fully connected network
sizes=[784, 30, 10];
num_layers=length(sizes);

% activation_fn is used to choose activation function
% activation_fn==1    sigmoid
% activation_fn==2    hidden ayers: soft max; 
%                     output layer: soft max;
% activation_fn==3    hidden ayers: sigmoid; 
%                     output layer: soft max;
activation_fn=3;

% cost_type is used to choose cost function
% cost_type=1       Quadratic Cost for sigmoid 
% cost_type=2       Cross Entropy Cost for sigmoid
% cost_type=3       log-likelihood Cost for soft max (unfinished)
cost_type=3;

% learning rate
eta=0.5;

% regulation
lmbda=0;

% display settings, 1 for open, 0 for close
monitor_training_cost=0;
monitor_training_accuracy=1;
monitor_evaluation_cost=0;
monitor_evaluation_accuracy=1;
