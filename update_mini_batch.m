function update_mini_batch(mini_batch, eta, lmbda, n)
% reference network2.py

global weights
global biases
global num_layers

nabla_b=cell(size(biases));
for i=1:length(biases)
    nabla_b(i)={zeros(size(biases{i}))};
end
nabla_w=cell(size(weights));
for i=1:length(weights)
    nabla_w(i)={zeros(size(weights{i}))};
end
for i=1:size(mini_batch,1)
    [delta_nabla_b, delta_nabla_w]=backprop(mini_batch{i,1},mini_batch{i,2});
    for j=1:num_layers-1
        nabla_b{j}=nabla_b{j}+delta_nabla_b{j};
        nabla_w{j}=nabla_w{j}+delta_nabla_w{j};
    end
end
for j=1:num_layers-1
    biases{j}=biases{j}-(eta/size(mini_batch,1))*nabla_b{j};
    weights{j}=(1-eta*(lmbda/n))*weights{j}-(eta/size(mini_batch,1))*nabla_w{j}; % regulation
end
