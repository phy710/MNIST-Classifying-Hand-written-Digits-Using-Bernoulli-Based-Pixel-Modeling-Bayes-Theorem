% Classifying Hand-written Digits Using Bernoulli Based Pixel Modeling & Bayes Theorem
% Zephyr
% 02/03/2018
clear;
close all;
clc;
tic;
trainData = loadMNISTImages('train-images.idx3-ubyte')';
trainLabels = loadMNISTLabels('train-labels.idx1-ubyte');
testData = loadMNISTImages('t10k-images.idx3-ubyte')';
testLabels = loadMNISTLabels('t10k-labels.idx1-ubyte');
trainImg = reshape(trainData', 28, 28, 60000);
testImg = reshape(testData', 28, 28, 10000);
% Bernoulli model requires binary images
threshold = 0.2;
bwTrainData = zeros(60000, 784);
bwTestData = zeros(10000, 784);
bwTrainData(trainData>=threshold) = 1;
bwTestData(testData>=threshold) = 1;
% Calculate prossibility distribution function of each pixel of 10 nums
pdf = zeros(10, 784);
num = zeros(10, 1);
for a = 1 : 60000
    for b = 1 : 784
%  If the pixel is black, its count++
        if bwTrainData(a, b)
            pdf(trainLabels(a)+1, b) = pdf(trainLabels(a)+1, b) + 1;            
        end
    end
    num(trainLabels(a)+1) =  num(trainLabels(a)+1) + 1;
end
% Using Laplace smoothing
% pdf = (number of black pixels for each pixel of each num + 1) / (number of this num + 2)
for a = 1 : 10
    pdf(a, :) = (pdf(a, :)+1) / (num(a)+2);
end
% Calculate accuracy of training data
trainAccuracy = 0;
estimatedTrainLabels = zeros(60000, 1);
for a = 1 : 60000
% P.M.F model P(X) = Pi^xi * (1-Pi)^(1-xi)
%                  = Pi, when xi = 1
%                    1-Pi, when xi = 0
% xi represents value (0 or 1) of pixel i
% P(X|Y=j) = product of {Pij^xi * (1-Pij)^(1-xi)} where i is from 1 to 784
    if bwTrainData(a, 1)
        p = pdf(:, 1);
    else
        p = 1 - pdf(:, 1);
    end
    for b = 2 : 784
        if bwTrainData(a, b)
            p = p .* pdf(:, b);
        else
            p = p .* (1-pdf(:, b));
        end
    end
    p = p .* num / 60000;
% Estamated label j = argmax pi(j)*P(X|Y=j)
    estimatedTrainLabels(a) = find(p==max(p)) - 1;
    if estimatedTrainLabels(a) == trainLabels(a)
        trainAccuracy = trainAccuracy + 1;
    end
end
trainAccuracy = trainAccuracy / 60000;
% Calculate accuracy of test data
testAccuracy = 0;
estimatedTestLabels = zeros(10000, 1);
for a = 1 : 10000
    if bwTestData(a, 1)
        p = pdf(:, 1);
    else
        p = 1 - pdf(:, 1);
    end
    for b = 2 : 784
        if bwTestData(a, b)
            p = p .* pdf(:, b);
        else
            p = p .* (1-pdf(:, b));
        end
    end
    p = p .* num / 60000;
    estimatedTestLabels(a) = find(p==max(p), 1) - 1;
    if estimatedTestLabels(a) == testLabels(a)
        testAccuracy = testAccuracy + 1;
    end
end
testAccuracy = testAccuracy / 10000;
toc;
% Display accuracy
disp(['Train data accuracy: ', num2str(trainAccuracy*100), '%']);
disp(['Test data accuracy: ', num2str(testAccuracy*100), '%']);
% Display demo recognition resutlt
figure('NumberTitle', 'off', 'Name', 'Right Labels');
for a = 1 : 25
    subplot(5, 5, a);
    imshow(testImg(:, :, a));
    title(num2str(testLabels(a)));
end
figure('NumberTitle', 'off', 'Name', 'Estimated Labels');
for a = 1 : 25
    subplot(5, 5, a);
    imshow(testImg(:, :, a));
    title(num2str(estimatedTestLabels(a)));
end