function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X*theta);

% -1/m [SUM(H(x)-y)^2] + lamda * sum(theta_reg^2)
% vectorize implementaion
% J = (1/m) * ((-y' * log(g(X*theta) - (1-y)' * (log(1-g(X*theta)) )) + (lambda/2m)*theta

theta_reg = [0 ; theta(2:size(theta)),:];
J = (1/m) * ((-y' * log(h)) - ((1-y)' * log(1 - h))) + (lambda/(2*m)) * theta_reg' * theta_reg;

% gradient = (1/m) * SUM((h(xi)-yi) * xi) + lambda/m * theta_reg
% vectorized implementation
% gradient = (1/m) * Xtranspose * (g(X*theta) - y) + lambda/m * theta_reg

grad = 1/m * X' * (h-y) + lambda/m * theta_reg;




% =============================================================

end
