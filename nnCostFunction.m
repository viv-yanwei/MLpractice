function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m


A1 = [ones(m, 1) X];

Z2 = A1 * Theta1';

A2 = 1./(1+exp(-Z2));

A2 = [ones(size(A2,1), 1)  A2];

Z3 = A2 * Theta2';

A3 = 1./(1+exp(-Z3));

h = A3;

jLab = zeros(size(h,1),1);

for i = 1:size(h,1),

    y_conv = zeros(size(h,2),1);
    y_conv(y(i)) = 1;

    jLab(i) = 1/m * (-log(h(i,:)) * y_conv -  log(1-h(i,:))* (1-y_conv));
end;

%J=sum(jLab);

temp1 = Theta1;
temp2 = Theta2;

temp1(:,1) = 0;
temp2(:,1) = 0;


J = sum(jLab) + lambda/(2*m) * (sum(sum(temp1 .^2) )+ sum(sum(temp2 .^2)));


%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.


Delta = 0;
y_conv = (1:num_labels)==y; % m x num_labels == 5000 x 10
                       
delta3 = h - y_conv;
                       
Z2_conv = [ones(size(Z2,1),1)  sigmoidGradient(Z2)];

delta2 =  (delta3 * Theta2)  .* Z2_conv;

delta2 = delta2(:,2:end);

Delta2 = Delta + delta3' * A2
Theta2_grad = 1/m * Delta2; % 10 x 26


Delta1 = Delta + delta2' * A1
Theta1_grad = 1/m * Delta1;     % 25 x 401


%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


Theta1_grad = Theta1_grad + 1/m * (lambda * temp1) ;


Theta2_grad = Theta2_grad + 1/m * (lambda * temp2) ;














% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
