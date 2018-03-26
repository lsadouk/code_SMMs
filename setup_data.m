%% function imdb = setup_data(trainData,trainLabel,testData,testLabel,nb_accelerometers)
%  This function gets training and testing data with their corresponding
%  labels and:
%    1. formats all data into a single struct "imdb" (see description below)
%    2. normalizes the data
%
% Inputs:
%       - trainData: training data
%       - trainLabel : labels of training data
%       - testData : testing data
%       - testLabel : labels of testing data

% Outputs:
%       imdb :struct containing the following:
%       - imdb.images.data : data
%       - imdb.images.labels : 1xN vector containing labels where N is the 
%       total number of instances
%       - imdb.images.set : 1xN vector indicating the set of each instance
%       (set=1 for training data / set=2 for testing data)

function imdb = setup_data(trainData,trainLabel,testData,testLabel,nb_accelerometers)

%% 1. format all data into a single struct "imdb" 
d_size = size(trainData,2);
%nb_labels = max(trainLabel); % that is 2 labels
nb_train = size(trainLabel,1);
nb_test = size(testLabel,1);
trainData =trainData'; %
testData = testData';
nb_total = nb_train + nb_test;

nb_directions = 3; % x, y, z
%nb_accelerometers = 3; % right, left, torso
image_size = [d_size/(nb_directions*nb_accelerometers) nb_directions*nb_accelerometers]; 

imdb.images.data   = zeros(image_size(1), 1, image_size(2), nb_total, 'single');
%here, the number of channels = x,y,z directions * 3 accelerometers = 9
imdb.images.labels = zeros(1, nb_total, 'single'); % 1*n
imdb.images.set    = zeros(1, nb_total, 'uint8');

imdb.images.labels(1, 1:nb_train) = single(trainLabel');
imdb.images.labels(1, nb_train+1:nb_train+nb_test) = single(testLabel');

imdb.images.set(1, 1:nb_train) = 1;
imdb.images.set(:, nb_train+1:nb_train+nb_test) = 2;

imdb.images.data(:,:,:,1:nb_train) = reshape(trainData,image_size(1),1,image_size(2),nb_train);  
imdb.images.data(:,:,:,nb_train+1:nb_train+nb_test) = reshape(testData,image_size(1),1,image_size(2),nb_test);  

%% 2. normalize the data
imdb.images.data = dimensionNormalize(imdb.images.data);

end
