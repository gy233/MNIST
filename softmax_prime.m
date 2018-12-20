function y=softmax_prime(x)
y=diag(softmax(x))-softmax(x)*softmax(x)';