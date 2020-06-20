close all
clear all
clc

% main code
clear % clear all variables
file_path = 'california.txt'; % path to the file
global data_fraction; % fraction of data that's being read
data_fraction = 0.01;
D = retrieve_data(file_path); % retrieve data from the .txt file
frac_init = 0.1;
frac_end = 0.99;
frac_step = 0.01;
iterations_per_frac = 100;
statistical_data = generate_data(D, frac_init, frac_end, frac_step, iterations_per_frac); % generate statistical data with [frac, mean, standard_deviation] columns

plot_data(statistical_data); % plot the data

% frac = 0.8; % fraction of data that becomes training data
% [train_D, test_D] = random_split(D, frac); % split the data into training and testing data sets
% w = lsq_regression(train_D) % find least squares approximation
% error = compute_mean_squared_error(test_D, w) % compute mean squared error

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

% lsq_regression - gives the best approximate solution to the training data
function w = lsq_regression(train_set)
col_num = length(train_set(1,:)); % number of columns

A = train_set(:, 1:(col_num - 1)); % form matrix A
b = train_set(:, col_num); % form vector b

w = inv(A' * A) * A' * b; % finding w using least squares
end

% compute_mean_squared_error - computes mean squared error of the solution
function mse = compute_mean_squared_error(test_set, w)
col_num = length(test_set(1,:)); % number of columns

A = test_set(:, 1:(col_num - 1)); % form matrix A
b = test_set(:, col_num); % form vector b (real values)
n = length(b); % number of records

b_w = A * w; % estimated values
errors = b_w - b; % a vector of estimated_value - real_value
mse = 0; % mean squared error
i = 1; % counter
while  i <= length(errors)
    mse = mse + errors(i) ^ 2;
    i = i + 1;
end
mse = mse / length(errors);
end

% generate_data - generates statistical data about mean and standard
% deviation for different values of frac
% returns nx3 matrix, where each row is [frac, mean, standard deviation]
function stat_data = generate_data(data, frac_init, frac_end, frac_step, repetitions_per_frac)
fractions = frac_init:frac_step:frac_end; % generate an array of different value of frac
stat_data = []; % create empty matrix of statistical values data

i = 1; % counter
while i <= length(fractions)
    frac = fractions(i) % choose frac
    errors = []; % array of errors for this frac
    
    j = 1;
    while j <= repetitions_per_frac % repeat repetitions_per_frac times for this frac
        [train_data, test_data] = random_split(data, frac); % split the data randomly
        w = lsq_regression(train_data); % find least squares approximation
        error = compute_mean_squared_error(test_data, w); % compute mean squared error
        errors = [errors, error]; % append the error to errors for this frac
        
        j = j + 1;
    end
    
    % compute mean and standard deviation for this frac
    stand_dev = 0; % declare standard deviation for this frac
    mean = sum(errors) / length(errors); % compute mean
    errors = errors - mean; % subtract mean from errors
    k = 1;
    while k <= length(errors)
        stand_dev = stand_dev + errors(k) ^ 2; % sum of squares
        k = k + 1;
    end
    stand_dev = stand_dev / length(errors); % compute standard deviation
    
    stat_data = [stat_data; [frac, mean, stand_dev]];
    
    i = i + 1;
end
end

% plot_data - plots mean and standard deviation vs frac. Receives matrix
% with columns [frac, mean, sigma]
function plot_data(data)
figure('Name', 'Least Squares Regression vs Ridge Regression');
subplot(2, 1, 1);
plot(data(:, 1), data(:, 2), 'r'); % plot mean depending on frac
title('Mean vs frac');
xlabel('frac')
ylabel('mean')
grid on
subplot(2, 1, 2);
plot(data(:, 1), data(:, 3), 'r'); % plot standard deviation depending on frac
title('Standard deviation vs frac');
xlabel('frac')
ylabel('sigma')
grid on
end