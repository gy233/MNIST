function SGD(training_data_raw, epochs, mini_batch_size, eta, lmbda, evaluation_data)
% stochastic gradient descent

global monitor_training_cost
global monitor_training_accuracy
global monitor_evaluation_cost
global monitor_evaluation_accuracy
global evaluation_cost
global evaluation_accuracy
global training_cost
global training_accuracy
% global weights
% global biases
global n
global n_data

n_data=size(evaluation_data,1);
n = size(training_data_raw,1);

training_data=training_data_raw;
for i=1:n
    training_data(i,2)={vectorized_result(training_data_raw{i,2})};
end

evaluation_cost = [];
evaluation_accuracy = [];
training_cost = [];
training_accuracy = [];

for i=1:epochs
    randIndex=randperm(size(training_data,1));
    training_data=training_data(randIndex,:);
    mini_batches=cell(0);
    for k=1:mini_batch_size:n-1
        mini_batches(end+1)={training_data(k:k+mini_batch_size-1,:)};
    end
    for k=1:length(mini_batches)
        update_mini_batch(mini_batches{k}, eta, lmbda, n);
    end
    display(strcat('Epoch ',num2str(i),' training complete'));
    if monitor_training_cost==1
        cost=total_cost(training_data_raw, lmbda);
        training_cost=[training_cost,cost];
        disp('Cost on training data:');
        disp(cost);
    end
    if monitor_training_accuracy==1
        accuracy=accuracy_fun(training_data_raw);
        training_accuracy=[training_accuracy,accuracy];
        disp('Accuracy on training data:')
        disp(strcat(num2str(accuracy),'/',num2str(n)));
    end
    if monitor_evaluation_cost==1
        cost=total_cost(evaluation_data, lmbda);
        evaluation_cost=[evaluation_cost,cost];
        disp('Cost on evaluation data:');
        disp(cost);
    end
    if monitor_evaluation_accuracy==1
        accuracy=accuracy_fun(evaluation_data);
        evaluation_accuracy=[evaluation_accuracy,accuracy];
        disp('Accuracy on evaluation data:')
        disp(strcat(num2str(accuracy),'/',num2str(n_data)));
    end
    disp(' ')
end

%% accuracy
function acc=accuracy_fun(data)
[temp,result]=max(feedforward(data(:,1)));
result_hope=[data{:,2}]+1;
if size(result)==size(result_hope)
    acc=sum(result==result_hope);
else
    error('error in accuracy calculation')
end

%% cost
function cost=total_cost(data, lmbda)
global weights
global cost_type
global num_layers
cost = 0;
a=feedforward(data(:,1));
y=[];
for i=1:size(data,1)
    y = [y,vectorized_result(data{i,2})];
end
if cost_type==1         % cost_type=1       Quadratic Cost for sigmoid 
    cost=mean(QuadraticCost_fn(a,y));
elseif cost_type==2     % cost_type=2       Cross Entropy Cost for sigmoid
    cost=mean(CrossEntropyCost_fn(a,y));
end
w=0;
for i=1:num_layers-1
    w=w+norm(weights{i});
end
cost=cost+0.5*(lmbda/size(data,1))*w;

%% feedforward
function y=feedforward(data)
global num_layers
global weights
global biases
global activation_fn
y=[];
for i=1:size(data,1)
    x=data{i};
    for j=1:num_layers-1
        if activation_fn==1
            x=sigmoid(weights{j}*x+biases{j});
        elseif activation_fn==2
            x=softmax(weights{j}*x+biases{j});
        elseif activation_fn==3
            if j==num_layers-1
                x=softmax(weights{j}*x+biases{j});
            else
                x=sigmoid(weights{j}*x+biases{j});
            end
        end
    end
    y=[y,x];
end



