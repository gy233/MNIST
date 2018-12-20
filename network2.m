clc;clear;close all;
global eta
global lmbda
global monitor_training_cost
global monitor_training_accuracy
global monitor_evaluation_cost
global monitor_evaluation_accuracy
global evaluation_cost
global evaluation_accuracy
global training_cost
global training_accuracy
global n
global n_data
global sizes
global num_layers
global weights
global biases
global activation_fn

% settings
my_settings();

% load MNIST data set
load('MNIST_data.mat');
num_training_data=length(train_label);
training_data=cell(num_training_data,2);
for i=1:num_training_data
    training_data(i,1)={reshape(train_image(:,:,i)/255,[sizes(1),1])};
    training_data(i,2)={train_label(i)};
end
num_test_data=length(test_label);
evaluation_data=cell(num_test_data,2);
for i=1:num_test_data
    evaluation_data(i,1)={reshape(test_image(:,:,i)/255,[sizes(1),1])};
    evaluation_data(i,2)={test_label(i)};
end

% initialize network
default_weight_initializer(sizes);

% start training
epochs=30;
mini_batch_size=10;
SGD(training_data, epochs, mini_batch_size, eta, lmbda, evaluation_data);

%% save
save('weight_and_bias.mat','weights','biases','num_layers','activation_fn')

%% cost figure
figure
hold on
if monitor_evaluation_cost==1
    disp('evaluation_cost')
    disp(evaluation_cost)
    h1=plot(evaluation_cost,'b');
end
if monitor_training_cost==1
    disp('training_cost')
    disp(training_cost)
    h2=plot(training_cost,'r');
end
title('cost')
if monitor_evaluation_cost==1 && monitor_training_cost==1
    legend([h1,h2],'evaluation cost','training cost')
end
grid on

%% accuracy figure
figure 
hold on
if monitor_evaluation_accuracy==1
    disp('evaluation_accuracy')
    disp(evaluation_accuracy)
    h1=plot(evaluation_accuracy/n_data,'b');
end

if monitor_training_accuracy==1
    disp('training_accuracy')
    disp(training_accuracy)
    h2=plot(training_accuracy/n,'r');
end
title('accuracy')
if monitor_evaluation_accuracy==1 && monitor_training_accuracy==1
    legend([h1,h2],'evaluation accuracy','training accuracy')
end
grid on

