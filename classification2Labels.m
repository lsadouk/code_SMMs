%% function [...]=classification2Labels(featureVectorAndLabels, domain_type, training_data_type)
%  This function gets feature vector and labels and:
%     1. classifies vectors into SMM/non-SMM labels
%     2. splits data into train and test set according to k-fold cross validation, 
%     3. balances training data 
%
% Inputs:
%       - featureVectorAndLabels
%       - domain_type : 1 time domain and 2 for frequency domain
%       - training_data_type
%       if training_data_type==0 % ALL SMM movements including flapp --> we
%       are dealing w/ 450 or 810 d (features) =  (50 or 90)*3accelerometers(torso,right,left)*3directions
%       else  we are dealing w/ 270 d (features) = (50 or 90)*3directions

% Outputs:
%       - trainData : training data
%       - trainLabel : labels of training data
%       - testData : testing data
%       - testLabel : labels of testing data
%
%%
function [trainData,trainLabel,testData,testLabel]=classification2Labels(featureVectorAndLabels, domain_type, training_data_type, training_type)
fv=featureVectorAndLabels.fv;
Labelvec=featureVectorAndLabels.videoLabelvec;

%% 1. Classify vectors into SMM/non-SMM labels
% Counting number of data points in each class before resampling for balancing
uknownClassNdx=(find(Labelvec==1));    %'uknown' NON SMM
Labelvec(uknownClassNdx)=1;

rockClassNdx =(find(Labelvec==2));   %'rock'
Labelvec(rockClassNdx) = 2; % SMM

rockflapClassNdx=(find(Labelvec==3));   %'rock-flap'
Labelvec(rockflapClassNdx) = 2; % SMM

if training_data_type == 0 % if all movements (rock and flap), include flap movements
    flapClassNdx=(find(Labelvec==4));   %'flap'
    Labelvec(flapClassNdx) = 2; % SMM
    smmClassNdx=(find(Labelvec==2));  %'SMM'
else % else, only rock movements included + choose only torso acceleration signals (non SMM and rock SMM behaviors (exclude flapping))
    smmClassNdx=(find(Labelvec==2));  %'SMM'
    Labelvec = [Labelvec(smmClassNdx); Labelvec(uknownClassNdx)];
    if domain_type ==1 % for frequency domain
        fv = [fv(smmClassNdx,301:450); fv(uknownClassNdx,301:450)]; 
    elseif domain_type ==2 && training_type==3 % for frequency domain
        fv = [fv(smmClassNdx,301:450); fv(uknownClassNdx,301:450)]; 
    else % for time domain
        fv = [fv(smmClassNdx,541:810); fv(uknownClassNdx,541:810)]; 
    end
    
end

% random premutation of data
randNdx=randperm(length(Labelvec));
fv=fv(randNdx,:);
Labelvec=Labelvec(randNdx);
dataPointsInEachClass.uknownClassNumPoints=length(uknownClassNdx);    %'uknown' non SMM
dataPointsInEachClass.smmClassPoints = length(smmClassNdx); % SMM


%% 2. Split data into train and test set according to k-fold cross validation
% if nb of sessions = 1, then perform 10 fold
% else if nb of sessions = k (>1) , then perform k fold
if (featureVectorAndLabels.Sessionnb == 1)
    kfold = 10;
else % if =k>1
    kfold = featureVectorAndLabels.Sessionnb;
end
sizekmul =size(fv,1)-mod(size(fv,1),kfold);  % for 10-fold cross validation
trainData=fv(1:sizekmul/kfold*(kfold-1),:); %8/10 samples are for training
trainLabel=Labelvec(1:sizekmul/kfold*(kfold-1),:); %8/10 samples are for training
testData=fv(sizekmul/kfold*(kfold-1)+1:sizekmul,:);%2/10 samples are for training
testLabel=Labelvec(sizekmul/kfold*(kfold-1)+1:sizekmul,:);%2/10 samples are for training

%% 3. balances training data 
% 4.1. Counting number of data points in each class before resampling for balancing
rockClassNdx=(find(trainLabel==2));  %'SMM'
uknownClassNdx=(find(trainLabel==1));    %'uknown' non SMM
targetNumOfSamplesInEachClass=floor(length(trainLabel)/length(unique(trainLabel)));

% 4.2. randomly resampling two other class with less number of data
% points to targetNumOfSamplesInEachClass
if 0<length(rockClassNdx)&& length(rockClassNdx)<targetNumOfSamplesInEachClass
    requiredSamplesLength=targetNumOfSamplesInEachClass-length(rockClassNdx);
    if requiredSamplesLength <= length(rockClassNdx)
        requiredSamples = randsample(rockClassNdx,requiredSamplesLength); %returns  k='requiredSamplesLength'  values sampled uniformly at random, without replacement, from the values in the vector rockClassNdx
    else
        requiredSamples = datasample(rockClassNdx,requiredSamplesLength); %returns k='requiredSamplesLength' observations sampled uniformly at random, with replacement, from the data in rockClassNdx.
    end
    balancedFvRock=[trainData(rockClassNdx,:);trainData(requiredSamples,:)];
elseif 0<length(rockClassNdx)&& length(rockClassNdx)>targetNumOfSamplesInEachClass
    rockSubsampleNdx = randsample(rockClassNdx,targetNumOfSamplesInEachClass);
    balancedFvRock=trainData(rockSubsampleNdx,:);
else
    balancedFvRock=trainData(rockClassNdx,:);
end

% 4.3. randomly subsampling of unknown class
%             if length(uknownClassNdx)>targetNumOfSamplesInEachClass
%                 UknownSubsampleNdx = randsample(uknownClassNdx,targetNumOfSamplesInEachClass);
%             else
%                 UknownSubsampleNdx = datasample(uknownClassNdx,targetNumOfSamplesInEachClass);
%             end
%             balancedFvUnknown=fv(UknownSubsampleNdx,:);
if 0<length(uknownClassNdx) && length(uknownClassNdx)<targetNumOfSamplesInEachClass
    requiredSamplesLength=targetNumOfSamplesInEachClass-length(uknownClassNdx);
    if requiredSamplesLength <= length(uknownClassNdx)
        requiredSamples = randsample(uknownClassNdx,requiredSamplesLength);
    else
        requiredSamples = datasample(uknownClassNdx,requiredSamplesLength);
    end
    balancedFvUnknown=[trainData(uknownClassNdx,:);trainData(requiredSamples,:)];
elseif 0<length(uknownClassNdx)&& length(uknownClassNdx)>targetNumOfSamplesInEachClass
    UknownSubsampleNdx = randsample(uknownClassNdx,targetNumOfSamplesInEachClass);
    balancedFvUnknown=trainData(UknownSubsampleNdx,:);
else
    balancedFvUnknown=trainData(uknownClassNdx,:);
end

fvBalanced=[balancedFvRock;balancedFvUnknown];
labelVecBalanced=[2*ones(1,size(balancedFvRock,1)) ones(1,size(balancedFvUnknown,1))];

randNdx=randperm(length(labelVecBalanced));
fvBalanced=fvBalanced(randNdx,:);
labelVecBalanced=labelVecBalanced(randNdx);

trainData=fvBalanced;
trainLabel=labelVecBalanced';


