data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);

%用数据作图
function plotData(x, y)
	figure;
	plot(x, y, 'rx', 'MarkerSize', 10);
	ylabel('Profit in $10,000s');
	xlabel('Population of City in 10,000s');
end
plotData(X, y);

%将theta0项添加进X矩阵，设置learning rate
m = length(X);
X = [ones(m, 1), data(:, 1)];
theta = zeros(2, 1);
iterations = 1500;
alpha = 0.01;

%代价函数
function J = computeCost(X, y, theta)
	m = length(y);
	J = 0;
	J = sum((X * theta - y) .^ 2) / (2 * m);
end

%梯度下降
function [theta, J_history] = gradientDescent(X, y, theta, num_iters)
	m = length(y);
	J_history = zeros(num_iters, 1);

	for iter = 1 : num_iters
		theta = theta - alpha * 1 / m * X' * (X * theta  - y);
		J_history(iter) = computeCost(X, y, theta);
	end
end


%运行
theta = gradientDescent(X, y, theta, alpha, iterations);
fprintf('Theta computed from gradient descent:\n%f,\n%f',theta(1),theta(2))


%作图
hold on;
plot(X(:, 2), X * theta, '-')
legend('Traing data', 'Linear regression')
hold off

%可视化J of theta

% Visualizing J(theta_0, theta_1):
% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];    
	  J_vals(i,j) = computeCost(X, y, t);
    end
end

% Because of the way meshgrids work in the surf command, we need to 
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';

% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
hold off;



























