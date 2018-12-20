function [nabla_b,nabla_w]=backprop(x,y)
% for fully connected layer
% reference network2.py

global cost_type
global num_layers
global weights
global biases
global activation_fn

%% activation_fn==1    sigmoid
if activation_fn==1
    %% cost_type=1       Quadratic Cost
    if cost_type==1
        nabla_b=cell(size(biases));
        nabla_w=cell(size(weights));
        % feedforward
        activation = x;
        activations=cell(0);    % list to store all the activations, layer by layer
        activations(1)={x};
        zs=cell(0);
        for i=1:length(biases)
            z=weights{i}*activation+biases{i};
            zs(end+1)={z};
            activation=sigmoid(z);
            activations(end+1)={activation};
        end
        % backward pass
        delta = QuadraticCost_delta(zs{end}, activations{end}, y);
        nabla_b(end) = {delta};
        nabla_w(end) = {delta*activations{end-1}'};
        for i=1:num_layers-2
            z=zs{end-i};
            sp = sigmoid_prime(z);
            delta=weights{end-i+1}'*delta.*sp;
            nabla_b(end-i)={delta};
            nabla_w(end-i)={delta*activations{end-i-1}'};
        end
        
        %% cost_type=2       Cross Entropy Cost
    else
        nabla_b=cell(size(biases));
        nabla_w=cell(size(weights));
        % feedforward
        activation = x;
        activations=cell(0);    % list to store all the activations, layer by layer
        activations(1)={x};
        zs=cell(0);
        for i=1:length(biases)
            z=weights{i}*activation+biases{i};
            zs(end+1)={z};
            activation=sigmoid(z);
            activations(end+1)={activation};
        end
        % backward pass
        delta = CrossEntropyCost_delta(zs{end}, activations{end}, y);
        nabla_b(end) = {delta};
        nabla_w(end) = {delta*activations{end-1}'};
        for i=1:num_layers-2
            z=zs{end-i};
            sp = sigmoid_prime(z);
            delta=weights{end-i+1}'*delta.*sp;
            nabla_b(end-i)={delta};
            nabla_w(end-i)={delta*activations{end-i-1}'};
        end
    end
    %% activation_fn==2    hidden layers: soft max
elseif activation_fn==2
    %% cost_type=3       log-likelihood Cost for soft max
    if cost_type==3
        nabla_b=cell(size(biases));
        nabla_w=cell(size(weights));
        % feedforward
        activation = x;
        activations=cell(0);    % list to store all the activations, layer by layer
        activations(1)={x};
        zs=cell(0);
        for i=1:length(biases)
            z=weights{i}*activation+biases{i};
            zs(end+1)={z};
            activation=softmax(z);
            activations(end+1)={activation};
        end
        % backward pass
        delta = activations{end}-y;
        nabla_b(end) = {delta};
        nabla_w(end) = {delta*activations{end-1}'};
        for i=1:num_layers-2
            z=zs{end-i};
            sp = softmax_prime(z);
            delta=sp*(weights{end-i+1}'*delta);
            nabla_b(end-i)={delta};
            nabla_w(end-i)={delta*activations{end-i-1}'};
        end
    else
        error('when activation function is soft max, cost function must be log-likelihood cost')
    end
    %% activation_fn==3     hidden layers: sigmoid
elseif activation_fn==3
    %% cost_type=3       log-likelihood Cost for soft max
    if cost_type==3
        nabla_b=cell(size(biases));
        nabla_w=cell(size(weights));
        % feedforward
        activation = x;
        activations=cell(0);    % list to store all the activations, layer by layer
        activations(1)={x};
        zs=cell(0);
        for i=1:length(biases)-1
            z=weights{i}*activation+biases{i};
            zs(end+1)={z};
            activation=sigmoid(z);
            activations(end+1)={activation};
        end
        z=weights{length(biases)}*activation+biases{length(biases)};
        zs(end+1)={z};
        activation=softmax(z);
        activations(end+1)={activation};
        % backward pass
        delta = activations{end}-y;
        nabla_b(end) = {delta};
        nabla_w(end) = {delta*activations{end-1}'};
        for i=1:num_layers-2
            z=zs{end-i};
            sp = sigmoid_prime(z);
            delta=weights{end-i+1}'*delta.*sp;
            nabla_b(end-i)={delta};
            nabla_w(end-i)={delta*activations{end-i-1}'};
        end
    else
        error('when activation function is soft max, cost function must be log-likelihood cost')
    end
end