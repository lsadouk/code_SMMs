%% stereotypyMain
% stereotypyMain.m is an script which is able to do following tasks if the
% parameter related to that task is enabled('1') in the stereotypyParameters.m
% file.
%
%
% Steps:
% 1) Get data:
%       Retrieve labeled data used for experiment according to the user's choice: 
%       1)Subject ID, 2)Study type, 3)Record sessions for that Subject ID 
%       and study, 4)Time or frequency domain
%
%
% 2) Extract features (featureExtraction.m for frequency-domain) (featureExtractionBaseline.m for time-domain)
%    This function uses the preprocessed file "preprocessedDataAndLabels.mat"
%    to produce feature vectores that will further be used for
%    training. The Output "featureVectorAndLabels.mat" is saved in the same folder as
%    the chosen dataset.
%
% 3) Preprocess features before training (Classification2Labels.m)
%    this function gets features "featureVectorAndLabels.mat" and 
%     a. classifies vectors into SMM/non-SMM labels
%     b. splits data into train and test set according to k-fold cross validation, 
%     c. balances training data 
% 4)  Train one of the 2 frameworks (CNN or CNN+SVM)
%
%%

clear all;
setRoot;

%% 1) Retrieve data used for experiment including 1)Subject ID, 2) Study type, 
% Record sessions for that Subject ID and study, 4)Time or frequency
% domain

subjectID=input('Please select subject ID (1-6):','s');
studyType=str2double(input('Please select study type (1-2):','s'));
labelingParameters.studyType=num2str(studyType);
if studyType==1
    studyXDataPath=strcat(rootPath,'data', filesep, 'Study1', filesep);
    col4id=7;
else
    studyXDataPath=strcat(rootPath,'data', filesep, 'Study2', filesep);
    col4id=3;
end
sessionList0=ls(studyXDataPath);
sessionListChar=sessionList0(3:end,:);
sessionListCell=cellstr(sessionListChar);

selectedSubjectSessionList=sessionListChar(strcmp(cellstr(sessionListChar(:,col4id)),subjectID),:);
for i=1:size(selectedSubjectSessionList,1)
    disp(strcat('[',num2str(i),']: ',selectedSubjectSessionList(i,:)));
end
%selectedSessionNdx=str2double(input('Please select the desired session to process from the printed list:','s'));
selectedSessionNb=str2double(input('Please select the number of sessions to process from the printed list:','s'));
selectedSession=selectedSubjectSessionList(1:selectedSessionNb,:);


domain_type = input('Please select: (1)frequency domain / (2)time domain:');
training_type=input...
    ('Please select: (1)train network from scratch / (2)train using pre-training SMM network / (3)train using pre-trained HAR network: ');
    % either train network from scratch (1) or train using fine-tuned HAR network (2)
featureExtractionFlag = input('Please select: (1)do feature extraction / (2)do not extract features (already extracted) ');

fvMultSessions =[];
videoLabelvecMultSessions =[];

for i=1:selectedSessionNb
    selectedSessionPath=strcat(studyXDataPath,selectedSession(i,:));
    cd('../');
    cd(selectedSessionPath);

    %% 1. Get preprocessed data and their labels
    load preprocessedDataAndLabels;
    load Hd;

    %% 2. Extract time or frequency domain features from data
    if featureExtractionFlag == 1
        preprocessedData=preprocessedDataAndLabels{1};
        preprocessedLabels=preprocessedDataAndLabels{2};
        if domain_type == 2 % time domain
            featureVectorAndLabels=featureExtractionBaseline(preprocessedData,preprocessedLabels,Hd, training_type);
        else % frequency domain
            featureVectorAndLabels=featureExtraction(preprocessedData,preprocessedLabels,Hd);
        end
    else
        if domain_type == 2 % time domain
            if training_type ==1 || training_type ==2  % train from scratch or transfer learning w/SMMs
                load featureVectorAndLabelsBaseline; % for randomly initialized weight (90 points/axis in time-series)
            else % training_type==3 (resample from 90 to 50 becoz HAR network is trained on 50)
                load featureVectorAndLabelsBaseline90sampled50; % for TransferLearning w/ HAR (50 points/axis in time-series)
            end
        else % frequency domain
            load featureVectorAndLabels; %featureVectorAndLabels (50 freq points)
        end
    end
    fvMultSessions = [fvMultSessions; featureVectorAndLabels.fv];
    videoLabelvecMultSessions = [videoLabelvecMultSessions ; featureVectorAndLabels.videoLabelvec];
    
end

featureVectorAndLabels.fv = fvMultSessions;
featureVectorAndLabels.videoLabelvec = videoLabelvecMultSessions;
featureVectorAndLabels.Sessionnb = selectedSessionNb;

%% 3. Classify data into 2 labels: SMM/no-SMM
training_data_type=str2double(input('Please select: (0)train on all data (non-smm, rock, flap-rock, flap) / (1)train on rock data only ((non-smm, rock, flap-rock):','s'));
[trainData,trainLabel,testData,testLabel]=classification2Labels(featureVectorAndLabels, domain_type, training_data_type, training_type);

%% 4. Train and test one of the 2 frameworks: (1)CNN/(2)CNN+SVM 
%training type: (1)train from scratch or (2) transfer learning + train SVM
if training_type==1  % train from scratch using data of selected subject
    ASD_movement_CNN;
else        % if training_type==2 % transfer learning+ train SVM using data of alls subjects except the selected one
            % if training_type==3 % transfer learning+ train SVM using data of basic human movements (HAR)
    transferSVM_SMM; 
end
    


