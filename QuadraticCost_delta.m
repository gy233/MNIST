function delta=QuadraticCost_delta(z, a, y)
% Return the error delta from the output layer.
% a: output
% y: desired output
delta=(a-y) .* sigmoid_prime(z);