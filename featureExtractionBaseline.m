%% function featureVectorAndLabels=featureExtraction(featureExtractionParameters,preprocessedData,preprocessedLabels,Hd)
%  featureExtraction(featureExtractionParameters,preprocessedData,preprocessedLabels,Hd)
%  gets data and labels and parameters to extract the desired features.... 
% 
%   The inputs of the function:
%      featureExtractionParameters -
%      preprocessedData - 
%      preprocessedLabels -
%      Hd -
%
%   The outputs of the function:
%       featureVectorAndLabels - 
%
%%
function featureVectorAndLabels=featureExtractionBaseline(preprocessedData,preprocessedLabels,Hd, training_type)
% Steps:
% 1. filter data with hight-pass filter with 0.2 cut off frequency
% 2. extracting feature vector for each sample (1-sec window length)
%       (1) Stockwell transform
%       (2) ...?
%% Load Data & Labels
% data:
rightWristXYZ=preprocessedData{1};% =rightWristXYZ; 136423*4
leftWristXYZ=preprocessedData{2};% =leftWristXYZ;  136423*4
torsoXYZ=preprocessedData{3};  % =torsoXYZ;  136423*4

% video labels:
rightLabel=preprocessedLabels{1}{1}(:,1);% load rightLabel;
leftLabel=preprocessedLabels{1}{2}(:,1);% load leftLabel;
torsoLabel=preprocessedLabels{1}{3}(:,1);% load torsoLabel;

% phone labels:
rightLabelPhone=preprocessedLabels{2}{1}(:,1);% load rightLabel;

% times
tr=rightWristXYZ(:,1);% load tr;
tl=leftWristXYZ(:,1);% load tl;
tt=torsoXYZ(:,1);% load tt;
% load('Hd.mat');

%% Filter data (high-pass IIR,fc= 0.1)
w0=filter(Hd,rightWristXYZ); % 136423*4
w1=filter(Hd,leftWristXYZ); % 136423*4
w2=filter(Hd,torsoXYZ); % 136423*4

fs = 90;
%rightLabel = rightLabel(length(rightLabel));
%leftLabel = leftLabel(45: length(leftLabel)-45);
%torsoLabel= torsoLabel(45: length(torsoLabel)-45);

% downsampling of labelVectors and substracting the last 90 samples
tr1=downsample(tr,10,0);
tl1=downsample(tl,10,0);
tt1=downsample(tt,10,0);

tr1=tr1(1:size(tr1,1)-fs/10); % to get same nb of samples as sampled data (-9 samples at the end) see below for details
tl1=tl1(1:size(tl1,1)-fs/10); % to get same nb of samples as sampled data (-9 samples at the end) 
tt1=tt1(1:size(tt1,1)-fs/10); % to get same nb of samples as sampled data (-9 samples at the end) 



%% get 90 time-point samples using a sliding window that moves with 10 time steps
% METHOD 1
% right--------------------------------------------------------------------
for i=2:4
    wi1=w0(:,i);
    nbsamples = floor((size(wi1,1)-fs)/10);
    fv1 = zeros(nbsamples, fs); % zeros(size(wi0,1)-fs, fs); %13400-90 by 90
    rightLabel1 = zeros(nbsamples,1); 
    count = 1;
   for j=fs/2:10:size(wi1)-fs/2
        fv1(count,:) = (wi1(j-fs/2+1:j+fs/2))';
        obtLabel = rightLabel(j-fs/2+1:j+fs/2);
        rightLabel1(count,1)  = mode(obtLabel); % to obtain the most frequent value
        count = count +1;
    end
    clear st;
    if    i==2
        fvw0x=[fv1];
    elseif i==3
        fvw0y=[fv1];
    elseif i==4
        fvw0z=[fv1];
    end
end
clear fv1;
%     lf=length(f)
%     b=f
% left--------------------------------------------------------------------
for i=2:4
    wi1=w1(:,i);
    nbsamples = floor((size(wi1,1)-fs)/10);
    fv1 = zeros(nbsamples, fs); % zeros(size(wi0,1)-fs, fs); %13400-90 by 90
    count = 1;
   for j=fs/2:10:size(wi1)-fs/2
        fv1(count,:) = (wi1(j-fs/2+1:j+fs/2))';
        count = count +1;
    end
    clear st;
    if    i==2
        fvw1x=[fv1];
    elseif i==3
        fvw1y=[fv1];
    elseif i==4
        fvw1z=[fv1];
    end
end
clear fv1;
% torso--------------------------------------------------------------------
for i=2:4
    wi2=w2(:,i);
    nbsamples = floor((size(wi2,1)-fs)/10);
    fv1 = zeros(nbsamples, fs); % zeros(size(wi0,1)-fs, fs); %13400-90 by 90
    count = 1;
   for j=fs/2:10:size(wi2)-fs/2
        fv1(count,:) = (wi2(j-fs/2+1:j+fs/2))';
        count = count +1;
    end
    clear st;
    if    i==2
        fvw2x=[fv1];
    elseif i==3
        fvw2y=[fv1];
    elseif i==4
        fvw2z=[fv1];
    end

end

if training_type ==1 || training_type ==2 % train from scratch
    fv=[fvw0x';fvw0y';fvw0z';fvw1x';fvw1y';fvw1z';fvw2x';fvw2y';fvw2z']';
else % transfer learning => resample the 90 points signals into 50 point signal
    fvw0x_new =resample(fvw0x',5,9);% d x n = (50 x n)
    fvw0y_new =resample(fvw0y',5,9);% d x n = (50 x n)
    fvw0z_new =resample(fvw0z',5,9);% d x n = (50 x n)
    fvw1x_new =resample(fvw1x',5,9);% d x n = (50 x n)
    fvw1y_new =resample(fvw1y',5,9);% d x n = (50 x n)
    fvw1z_new =resample(fvw1z',5,9);% d x n = (50 x n)
    fvw2x_new =resample(fvw2x',5,9);% d x n = (50 x n)
    fvw2y_new =resample(fvw2y',5,9);% d x n = (50 x n)
    fvw2z_new =resample(fvw2z',5,9);% d x n = (50 x n)
    fv=[fvw0x_new;fvw0y_new;fvw0z_new;fvw1x_new;fvw1y_new;fvw1z_new;fvw2x_new;fvw2y_new;fvw2z_new]'; % n x 3d
end
%% aggregate all labeling vectors by accelerometer (torso, left, right sensor)
featureVectorAndLabels.fv=fv;
rightLabelvec(find(rightLabel1==0))=1;rightLabelvec(find(rightLabel1==400))=2;rightLabelvec(find(rightLabel1==600))=3;rightLabelvec(find(rightLabel1==800))=4;
Labelvec=[rightLabelvec]';
featureVectorAndLabels.videoLabelvec=Labelvec;

if training_type ==1 || training_type ==2 % train from scratch or transfer learning with a CNN pretrained on SMMs
    save('featureVectorAndLabelsBaseline','featureVectorAndLabels');
    disp('saved featureVectorAndLabelsBaseline')    
else % training_type == 3 % transfer learning with a CNN pretrained on HAR
    save('featureVectorAndLabelsBaseline90sampled50','featureVectorAndLabels');
    disp('saved featureVectorAndLabelsBaseline90sampled50')
end

end