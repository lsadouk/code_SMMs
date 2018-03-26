function net = cnn_init_transferLearning_SMM(finetuning, domain_type, training_type,subjectID, studyType)

if domain_type == 1 % frequency domain network
    domain = 'freq';
    nb_epochs = 30;
else  % time domain network
    domain = 'time';
    nb_epochs = 15;
end

if training_type == 2 % train using pre-trained SMM network
    net = load(fullfile(strcat('../../transfer_learning_nets/data_CNN_RockSMM_', domain ,'_study',int2str(studyType),'wout',subjectID,'/results/net-epoch-',int2str(nb_epochs) ,'.mat')));
else %training_type == 3 % train using pre-trained HAR network
    net = load(fullfile(strcat('../../transfer_learning_nets/data_CNN_HAR_', domain ,'/results/net-epoch-30.mat')));
end

    net = net.net;

f=1/100; 
net.layers = net.layers(1:end-2);
if finetuning  % if we are finetuning with another CNN training (if not ==> SVM)
    net.layers{end + 1} = struct('type', 'conv', ...
                               'weights', {{f*randn(1,1,500,2, 'single'), zeros(1, 2, 'single')}}, ...
                               'size', [1 1 500 2], ...
                               'stride', 1, ...
                               'pad', 0, ...
                               'name', 'fc2') ;
    net.layers{end+1} = struct('type', 'softmaxloss', ...
                               'stride', 1, ...
                               'pad', 0) ;  
end
%vl_simplenn_display(net, 'inputSize', [50 1 3 50])  % was [90 1 3 50]
disp(size(net.layers));



end
