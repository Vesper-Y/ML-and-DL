%加载数据
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

%特征归一化
function [X_norm, mu, sigma] = featureNormalize(X)
	X_norm = X;
	mu = zeros(1, size(X, 2));
	sigma = zeros(1, size(X, 2));

	mu = mean(X_norm);
	sigma = std(X_norm);
	for i = 1 : size(X, 2)
		X_norm(:, i) = X_norm(:, i) - mu(i);
		X_norm(:, i) = X_norm(:, i) ./ sigma(i);
	end
end

[X, mu, sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];

%代价函数
function J = computeCostMulti(X, y, theta)
	m = length(y);
	J = 0;
	J = (1 / (2 * m)) * (X * theta - y)' * (X * theta - y);
end;

%梯度下降
function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
	m = length(y);
	n = length(theta);
	J_history = zeors(num_iters, 1);
	temp = zeros(n, num_iters);

	for iter = 1 : num_iters
		temp(:, iter) = theta - alpha * 1 / m * X' * (X * theta - y);
		theta = temp(:, iter);
		J_history(iter) = computeCostMulti(X, y, theta);
	end
end

% Run gradient descent
% Choose some alpha value
alpha = 0.1;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, ~] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Display gradient descent's result
fprintf('Theta computed from gradient descent:\n%f\n%f\n%f',theta(1),theta(2),theta(3))

% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================

price = [theta(1) + theta(2) * 1650 + theta(3) * 3]; % Enter your price formula here

% ============================================================

fprintf('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f', price);

%选择learning rate
% Run gradient descent:
% Choose some alpha value
alpha = 1;
num_iters = 50;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[~, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
plot(1:num_iters, J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

%正规方程
function [theta] = normalEqn(X, y)
	theta = zeros(size(X, 2), 1);
	theta = pinv(X' * X) * X' * y;
end

% Solve with normal equations:
% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations:\n%f\n%f\n%f', theta(1),theta(2),theta(3));

% Estimate the price of a 1650 sq-ft, 3 br house. 
% ====================== YOUR CODE HERE ======================

price = [theta(1) + theta(2) * 1650 + theta(3) * 3]; % Enter your price forumla here

% ============================================================

fprintf('Predicted price of a 1650 sq-ft, 3 br house (using normal equations):\n $%f', price);     























































































