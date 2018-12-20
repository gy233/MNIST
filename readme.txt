this program is based on Michael A. Nielsen's program which is written in python, but this program provides more choises.
http://neuralnetworksanddeeplearning.com/

new choises:
% activation_fn==2    hidden ayers: soft max; 
%                     output layer: soft max;
% activation_fn==3    hidden ayers: sigmoid; 
%                     output layer: soft max;

% cost_type=3       log-likelihood Cost for soft max

for example in this program, activation_fn==3 works the best while activation_fn==2 works the worst.

main function: network2.m
	after run network2.m, you can get training accuracy, evaluation accuracy, training cost and evaluation cost based on your settings. you can change number of epochs by changing the value of the variable: epochs (default value is 30). you can also change the number of datas in a batch by defining the variable: mini_batch_size (default value is 10).

my_settings.m
	you can design your network by define variable: sizes. the length of it represents the number of layers including input layer, hidden layer and output layer.





2018/12/20
at Huangdu Science and Technology college
