function y=sigmoid_prime(x)
y=sigmoid(x).*(1-sigmoid(x));