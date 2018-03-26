
run ../../../matconvnet-1.0-beta16/matlab/vl_setupnn ;
%run('..', '..', '..', 'matconvnet-1.0-beta16', 'matlab', 'vl_setupnn.m') ;

%opts.expDir is where trained networks and plots are saved.
opts.expDir = fullfile('..', strcat('data_CNN_RockOnly',int2str(training_data_type) ,...
                        '_FreqOrTime',int2str(domain_type),'_study',int2str(studyType),...
                        '_subjectID',subjectID,'_NBsessions',int2str(selectedSessionNb)) ,'results') ;

% to get F1-score of a trained CNN, set the following
% 1. opts.batchSize = opts.size_testLabel;
% 2. opts.numEpochs = 36; % number of epochs of training +1
% 3. opts.continue = true ;
opts.size_testLabel = size(testLabel,1);
opts.batchSize = 50; % set to size(testLabel,1) for testing
opts.numEpochs = 35; % set to 36 for testing
opts.continue = false ; % set to true for testing
opts.learningRate = 0.01;% 0.01 for the first 10 epochs % and 0.001 after 10 epochs

%GPU support is off by default.
% opts.gpus = [] ;

%% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------
% The cnn_init function specifies the network architecture. You will be
% modifying the function.
if training_data_type == 0 % all  (450 d data)
    nb_accelerometers = 3;
else % Rock only (150 d data)
    nb_accelerometers = 1;
end

if domain_type == 1 % frequency domain
    net = proj6_part1_cnn_init_frequencyD(nb_accelerometers);
else % domain_type == 2  , i.e. time domain
    net = proj6_part1_cnn_init_st9_d90(nb_accelerometers); % was proj6_part1_cnn_init_timeD(nb_accelerometers);
end

%% setup data ==> imdb
imdb = setup_data(trainData,trainLabel,testData,testLabel,nb_accelerometers); % orginal
%imdb = setup_data(trainData(1:1500,:),trainLabel(1:1500),testData,testLabel,nb_accelerometers); % orginal

%% -------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------
%opts.train = [] ; % to be deleted later

[net, info] = cnn_train(net, imdb, @getBatch, ...
    opts,...
    'val', find(imdb.images.set == 2)) ;

fprintf('Lowest validation error is %f\n',min(info.val.error(1,:)))

% --------------------------------------------------------------------
