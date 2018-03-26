%% function featureVectorAndLabels=featureExtraction(featureExtractionParameters,preprocessedData,preprocessedLabels,Hd)
%  This function gets the dataset and their corresponding labels, then 
%  extract the Stockwell features from it
% 
%   The inputs of the function:
%      preprocessedData - 
%      preprocessedLabels -
%      Hd -
%
%   The outputs of the function:
%       featureVectorAndLabels - 
%

function featureVectorAndLabels=featureExtraction(preprocessedData,preprocessedLabels,Hd)
% Steps:
% 1. filter data with hight-pass filter with 0.2 cut off frequency
% 2. extracting feature vector for each sample (1-sec window length)
%       (1) Stockwell transform
%       (2) ...?
%% Load the data
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
%% keep btw two sync only


% downsampling of labelVectors
rightLabel1=downsample(rightLabel,10,5);%Decrease sampling rate of the sequence by 10 and add a phase offset of 5.
leftLabel1=downsample(leftLabel,10,5);
torsoLabel1=downsample(torsoLabel,10,5);
rightLabelPhone1=downsample(rightLabelPhone,10,5);
tr1=downsample(tr,10,5);
tl1=downsample(tl,10,5);
tt1=downsample(tt,10,5);

%% Feature Extraction with Stockwell transform

fs = 90;
Nyquist = fs;
maxfreq_org =3; %!!!! was 3 % 5;
minfreq = 0;
freqSteps = 51; % !!!! was 51
endst=51; % !!!! was 51
maxfreq_old = fix(size(w0,1)/2); %  %49027
maxfreq = fix(maxfreq_old * maxfreq_org / (Nyquist / 2)); %3268
freqsamplingrate = round((maxfreq - minfreq+1)/freqSteps); %64 %Sampling Rate (freq.  domain)

% right--------------------------------------------------------------------
for i=2:4
    wi0=w0(:,i);
    [st,t,f] = stockwell(wi0,-1,maxfreq,0.011,freqsamplingrate); %maxfreq=4547  %0.011=1/90 time points Sampling Rate (time   domain) 
    %     figure,subplot(2,1,1);imagesc(log10(abs(stockwell(1:10,2e4:6e4))));axis tight;
    %     subplot(2,1,2);plot(rightLabel(2e4:6e4),'r');axis tight;
    fv1=abs(st(2:endst,:));
    fv1=(downsample(fv1',10,5))';
    %     st2=sort(abs(st),1,'descend');
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
    [st,t,f] = stockwell(wi1,-1,maxfreq,0.011,freqsamplingrate); %maxfreq=4547  %0.011=1/90 time points
    fv1=abs(st(2:endst,:));
    fv1=(downsample(fv1',10,5))';
    %     st2=sort(abs(st),1,'descend');
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
    [st,t,f] = stockwell(wi2,-1,maxfreq,0.011,freqsamplingrate); %maxfreq=4547  %0.011=1/90 time points
    fv1=abs(st(2:endst,:));
    fv1=(downsample(fv1',10,5))';
    clear st;
    if    i==2
        fvw2x=[fv1];
    elseif i==3
        fvw2y=[fv1];
    elseif i==4
        fvw2z=[fv1];
    end
end
clear fv1;
fv=[fvw0x;fvw0y;fvw0z;fvw1x;fvw1y;fvw1z;fvw2x;fvw2y;fvw2z]';
% fv=[fvw0x;fvw0y;fvw0z;fvw2x;fvw2y;fvw2z]';

%% aggregate all labeling vectors by accelerometer (torso, left, right sensor)
featureVectorAndLabels.fv=fv;
rightLabelvec(find(rightLabel1==0))=1;rightLabelvec(find(rightLabel1==400))=2;rightLabelvec(find(rightLabel1==600))=3;rightLabelvec(find(rightLabel1==800))=4;
Labelvec=[rightLabelvec]';
featureVectorAndLabels.videoLabelvec=Labelvec;

save('featureVectorAndLabels','featureVectorAndLabels');
disp('saved featureVectorAndLabels')
end

