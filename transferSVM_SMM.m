%% The code below takes training and testing data, uses low and mid-level 
%  weights of a pretrained CNN and places an SVM on top, then trains the SVM
% Output: F1-score

%run(fullfile('..','..','..','matconvnet-1.0-beta16', 'matlab', 'vl_setupnn.m')) ;
run ../../../matconvnet-1.0-beta16/matlab/vl_setupnn ;

% --------------------------------------------------------------------
%                              1.  Load pretrained CNN (SMM CNN or HAR CNN)
% --------------------------------------------------------------------
finetuningCNN = logical(false);
net = cnn_init_transferLearning_SMM(finetuningCNN, domain_type,training_type,subjectID, studyType); % domain_type= 1(freq)/ 2(time)
% training_type =2(train using pre-trained SMM network)/3(train using
% pre-trained HAR network)

% --------------------------------------------------------------------
%                       2. Format and normalize training and testing data
% --------------------------------------------------------------------
nb_accelerometers = 1;
imdb = setup_data(trainData,trainLabel,testData,testLabel,nb_accelerometers);

%% --------------------------------------------------------------------
%                           3. With training data, get CNN features + train SVM
% --------------------------------------------------------------------
training= logical(false);
if training, dzdy = one; else, dzdy = [] ; end
%opts.batchSize = 150; % was 150  or size(testLabel,1) or floor(size(imdb.images.data,4)/2) i.e. 13000
opts.conserveMemory = false ;
opts.backPropDepth = +inf ;
opts.sync = false ;
opts.cudnn = true ;
res = [] ;

training_batch = find(imdb.images.set == 1);
training_batch = training_batch(1:2000);
[im, labels] = getBatch(imdb, training_batch) ;
res = vl_simplenn(net, im, dzdy, res, ...
                  'accumulate', ~training, ... %s ~= 1
                  'disableDropout', ~training, ...
                  'conserveMemory', opts.conserveMemory, ...
                  'backPropDepth', opts.backPropDepth, ...
                  'sync', opts.sync, ...
                  'cudnn', opts.cudnn) ;
data = res(end).x;% was data = res(10).x;
data = reshape(data, size(data,3), size(data,4)); % convert from 1x1xdxn to dxn
SVMModel = fitcsvm(data',labels', 'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');  



%% --------------------------------------------------------------------
%                        4. With Testing data, get CNN features + test SVM
% --------------------------------------------------------------------
testing_batch = find(imdb.images.set == 2);

[im, labels] = getBatch(imdb, testing_batch) ;
res = [] ;
res = vl_simplenn(net, im, dzdy, res, ...
                  'accumulate', ~training, ... %s ~= 1
                  'disableDropout', ~training, ...
                  'conserveMemory', opts.conserveMemory, ...
                  'backPropDepth', opts.backPropDepth, ...
                  'sync', opts.sync, ...
                  'cudnn', opts.cudnn) ;
data = res(end).x;% was data = res(10).x;
data = reshape(data, size(data,3), size(data,4)); % convert from 1x1xdxn to dxn
[predicted_label,score] = predict(SVMModel,data');

try
nb_labels = 2;
batch_test_size = size(testing_batch,2);
%best_p = reshape(predictions,nb_labels,batch_size_test);
Y = bsxfun(@eq, (1:nb_labels)' * ones(1, batch_test_size), predicted_label'); %added
Ylabel = bsxfun(@eq, (1:nb_labels)' * ones(1, batch_test_size), labels); %added
confusion = double(Ylabel) * double(Y'); %added
fprintf('Confusion matrix: \n');
disp(confusion); %added
tp = confusion(2,2);
fp = confusion(1,2);
tn = confusion(1,1);
fn = confusion(2,1);
% Precision = true_positive / (true_positive + false_positive)
precision=tp/(tp+fp);
fprintf('Precision %f\n',precision);
% Recall = true_positive / (true_positive + false_negative)
recall=tp/(tp+fn);
fprintf('Recall %f\n',recall);
F1_score = 2* (precision*recall)/(precision+recall);
fprintf('F1 score %f\n',F1_score);
accuracy = (tp+tn)/batch_test_size;
fprintf('Accuracy %f\n',accuracy);
catch
end

