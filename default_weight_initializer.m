function default_weight_initializer(sizes)
global weights
global biases
global num_layers

biases=cell(0);
weights=cell(0);

for i=1:num_layers-1
    biases(i)={randn(sizes(i+1),1)};
    weights(i)={normrnd(0,1/sqrt(sizes(i)),sizes(i+1),sizes(i))};
end