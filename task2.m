% close all
% clear all
clc

% main code
file_path = 'california.txt'; % path to the file containing data
global data_fraction; % fraction of data that's being read
data_fraction = 0.01;
frac1 = 0.7; % fraction of trainval
frac2 = 0.8; % fraction of train in trainval

num_of_reps = 100; % number of repetitions per frac1 when generating statistics
lambdas = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]; % lambdas to be tested
frac1_init = 0.1; % initial frac1
frac1_end = 0.99; % final frac1
frac1_step = 0.01; % frac1 step

D = retrieve_data(file_path); % retrieve data from database
stats = generate_stats(D, lambdas, num_of_reps, frac1_init, frac1_end, frac1_step, frac2); % generate statistics -> [frac1, mean, sigma] columns
plot_data(stats); % plot statistics

% [trainval_D, test_D] = random_split(D, frac1); % split entire data set into trainval and test
% [train_D, val_D] = random_split(trainval_D, frac2); % split trainval into train and val
% lambdas = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]; % lambdas to be tested
% best_lambda = find_best_lambda(train_D, val_D, lambdas); % find lambda, for which the error is minimum
% w = ridge_regression(trainval_D, best_lambda); % find best estimate, using the best lambda value
% error = compute_mean_squared_error(test_D, w); % find error over test data

% functions
% functions
% retrieve_data - retrieves data from .txt file
function data = retrieve_data(file_path)
global data_fraction;
fileID = fopen(file_path, 'r'); % open the file
file_line = fgetl(fileID); % get the 1st line
data = str2num(file_line); % write the 1st line
while ischar(file_line) % loop while the line read from the file is a list of characters (i.e. contains smth)
    file_line = fgetl(fileID); % get new line
    if file_line ~= -1 % if the line contains numbers
        new_row = str2num(file_line); % turn the string row into an array of doubles
        data = [data; new_row]; % append the new record to the existing matrix
    end
end
fclose(fileID); % close the file
data = data(1:(round(length(data) * data_fraction)), :);
%need to normalize the data (parameters, not the output value)
i = 1; % counter
while i <= length(data(1,:))  % go through all columns except for the last one
    column = data(:, i); % select a column
    mean = sum(column) / length(column); % find the mean
    column = column - mean; % subtract mean
    sigma = 0; % standard deviation
    j = 1; % counter
    while j <= length(column)
        sigma = sigma + column(j) ^ 2;
        j = j + 1;
    end
    sigma = sqrt(sigma / length(column)); % compute the standard deviation
    column = column / sigma; % divide by standard deviation
    data(:, i) = column; % replace the original parameter
    
    i = i + 1;
end
end

% random_split - splits entire data into training and test datasets
function [train_set, test_set] = random_split(data_set, frac)
% variables
data_set_n = length(data_set); % length of the data set
train_D_n = round(data_set_n * frac); % length of the training data set
train_set = []; % training data
test_set = []; % testing data

% main code
train_set_indices = randperm(data_set_n, train_D_n); % generate a random set of indices
data_set_indices = 1:length(data_set); % all indices

i = 1; % counter
while i <= length(train_set_indices) % negating train_set indices
    data_set_indices(train_set_indices(i)) = -1 * data_set_indices(train_set_indices(i)); % negate all indices that describe test data
    i = i + 1;
end

i = 1; % counter
while i <= length(data_set_indices)
    if abs(data_set_indices(i)) == data_set_indices(i) % if it's a positive index, then append to test_set
        test_set = [test_set; data_set(data_set_indices(i),:)]; % appending to test_set
    else % if it's a negative index, then append to train_set
        train_set = [train_set; data_set(abs(data_set_indices(i)),:)]; % appending to train_set
    end
    i = i + 1;
end

end

% ridge_regression - returns the best estimate of w based on ridge
% regression
function w = ridge_regression(train_set, lambda)
col_num = length(train_set(1,:)); % number of columns
A = train_set(:, 1:(col_num - 1)); % form matrix A
b = train_set(:, col_num); % form vector b
n = col_num - 1;

% Define the loss function as: J(w) = lambda/2*w'*w + 1/n*(Aw-y)'*(Aw-y),
% so w = 2/n * inv(lambda * I + 2/n * A' * A) * (A' * y)
w = 2 / n * inv(lambda * eye(n) + 2 / n * (A' * A)) * A' * b;
%w = inv(lambda * eye(n) + A' * A) * A' * b; % for n = 1, lambda->2lambda
end

% compute_mean_squared_error - computes mean squared error of the solution
function mse = compute_mean_squared_error(test_set, w)
col_num = length(test_set(1,:)); % number of columns

A = test_set(:, 1:(col_num - 1)); % form matrix A
b = test_set(:, col_num); % form vector b (real values)

b_w = A * w; % estimated values
errors = b_w - b; % a vector of estimated_value - real_value
% mse = sqrt(sum_of_squares / n)
mse = 0; % mean squared error
i = 1; % counter
while  i <= length(errors)
    mse = mse + errors(i) ^ 2;
    i = i + 1;
end
mse = mse / length(errors);
end

% find_best_lambda - returns lambda, for which the data set has
% statistically least error
function best_lambda = find_best_lambda(train_set, val_set, lambdas)
num_of_lambdas = length(lambdas); % number of lambdas to be tested
errors = zeros(1, num_of_lambdas); % array of average errors for each lambda. Parallel to lambdas

i = 1; % counter
while i <= num_of_lambdas
    lambda = lambdas(i); % pick a value for lambda from lambdas
    w = ridge_regression(train_set, lambda); % find the best estimate
    error = compute_mean_squared_error(val_set, w); % compute error for this estimate
    errors(i) = error; % write down to error
    i = i + 1;
end

[~, best_lambda_index] = min(errors);
best_lambda = lambdas(best_lambda_index);

end

% make_stats - finds the mean and standard deviation
function [mean, sigma] = make_stats(array)
n = length(array);
% find mean
mean = sum(array) / n;
% find standard deviation (sigma)
sigma = 0;
array = array - mean; % subtract mean from all elements in the array
i = 1; % counter
while i <= n % sum of squares
    sigma = sigma + array(i) ^ 2;
    i = i + 1;
end
sigma = sigma / n; % divided by number of elements
sigma = sqrt(sigma); % square root of it
end

% generate_stats - generate statistics for given range of frac1 (frac2 is
% fixed)
function stats = generate_stats(data_set, lambdas, num_of_reps, frac1_init, frac1_end, frac1_step, frac2)
FRAC1 = frac1_init:frac1_step:frac1_end; % generate an array of frac1-s
stats = []; % mxn, where m - number of frac1's, n=3 (for [frac1 mean sigma])

i = 1; % counter
while i <= length(FRAC1)
    % for each frac1
    frac1 = FRAC1(i) % current frac1
    errors = zeros(1, num_of_reps); % errors array: contains errors for each repetition
    
    j = 1; % counter
    while j <= num_of_reps
        [trainval_set, test_set] = random_split(data_set, frac1); % split entire data set into trainval and test
        [train_set, val_set] = random_split(trainval_set, frac2); % split trainval into train and val
        best_lambda = find_best_lambda(train_set, val_set, lambdas); % find lambda, for which the error is minimum
        w = ridge_regression(trainval_set, best_lambda); % find best estimate, using the best lambda value
        error = compute_mean_squared_error(test_set, w); % find error over test data
        errors = [errors, error]; % append new error to errors
        
        j = j + 1;
    end
    
    [mean, sigma] = make_stats(errors); % compute mean and standard deviation for this set of errors
    stats = [stats; [frac1 mean sigma]]; % append statistics for this frac1 to stats
    
    i = i + 1;
end

end

% plot_data - plots mean and standard deviation vs frac. Receives matrix
% with columns [frac, mean, sigma]
function plot_data(data)
% figure('Name', 'Ridge Regression')
subplot(2, 1, 1);
hold on
plot(data(:, 1), data(:, 2), 'b'); % plot mean depending on frac
title('Mean vs frac');
xlabel('frac')
ylabel('mean')
grid on
hold off
subplot(2, 1, 2);
hold on
plot(data(:, 1), data(:, 3), 'b'); % plot standard deviation depending on frac
title('Standard deviation vs frac');
xlabel('frac')
ylabel('sigma')
grid on
hold off
end