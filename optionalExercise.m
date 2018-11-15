%%%Optional Grading exercises
clear;clc;

load ('ex5data1.mat'); %Loading data

lambda = 3 %chosen from the learning curve obtained in the last exercise
n = length(Xtest);

p = 4;

% Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p);
[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
X_poly = [ones(size(X_poly,1), 1), X_poly];                   % Add Ones

% Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p);
X_poly_test = bsxfun(@minus, X_poly_test, mu);
X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];         % Add Ones

% Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p);
X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           % Add Ones


[theta] = trainLinearReg(X_poly_test, ytest, lambda)% uses fmincg to optimise theta value
[J grad] = linearRegCostFunction(X_poly_test, ytest, theta, 0); %Error in training setdisp(J);

x = size(X_poly_val);
error_train = zeros(size(X,1),1);
error_val = zeros(size(X,1),1);
for i = 1:size(X,1)
	for m = 1:50
		j = randperm(i);
		k = randperm(x);
		[theta] = trainLinearReg(X_poly(j,:), y(j), 0.01); % uses fmincg to optimise theta value
		error_train(i) = error_train(i) + linearRegCostFunction(X_poly(j,:), y(j), theta, 0); %Error in training set
		error_val(i) = error_val(i) + linearRegCostFunction(X_poly_val(k,:), yval(k), theta, 0); %Error in CV set
	end
	
	error_train(i) = error_train(i)/50;
	error_val(i) = error_val(i)/50;
end

plot(1:size(X,1), error_train(1:size(X,1)), 1:size(X,1), error_val(1:size(X,1)));
title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', 0.01));
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 100])
		
