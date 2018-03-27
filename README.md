# code_CNN_SMM
Deep Learning Approach for Recognizing Stereotypical Motor Movements (SMMs) within and across subjects on the Autism Spectrum Disorder
 
Welcome. This repository contains the data and scripts comprising the 'Deep Learning Networks for the Recognition of Stereotypical Motor Movements (SMMs) within and across subjects on the Autism Spectrum Disorder'. This work is a real-time automatic system for predicting SMMs within and across atypical subjects (on the autism spectrum).
 
Included are the tools to allow you to easily run the code.
 
This readme is a brief overview and contains details for setting up and running the code. Please refer to the following:
 
<h1>Running the code</h1><br/>
<h2>Initial requirements</h2>
 
1. To the code, the environment needed is Matlab. So you need to install Matlab.
2. To run the code, the MatConvNet Toolbox has already been downloaded and compiled for you. So, you don't need to install and compile MatConvNet. But, if you have your own version of MatConvNet, you can do so by replacing the MatConvNet folder within the main repository with your own.
 
<h2>Usage</h2>
First, you need to change the root folder in the file “setRoot.m” by your root folder (of your local computer).

Next, you select one of the following use cases:
 
<b>1.  SMM detection within each atypical subject (with a randomly initialized deep learning network): </b>  
In this study, randomly initialized Convolutional Neural Networks (CNNs) are trained. You can choose to train the CNN:
- in time-domain or frequency-domain,  
- on 1 or multiple record sessions within the chosen atypical subject,
- on one of the 6 atypical subjects of one of the 2 studies. 
- on (0) all movements of the selected atypical subject (flap, rock, flap-rock and non-SMMs) using the 3 accelerometers (torso, right and left sensors) or only on rock-movements (rock, flap-rock and non-SMMs) taken from only 1 accelerometer (torso sensor).

You can train and test the time-domain CNN by running the file ‘stereotypyMainMultSessions.m' and following steps of Example 1 (see below).

PS: If you choose 1 record session, the data will be divided in a 10-fold cross validation manner. However, if you choose more than 1 record session (2 or 3), 2-fold cross validation will be used (Half record sessions are chosen for training and the other half for testing).


<b>2.  SMM detection across atypical subjects via knowledge transfer framework: </b>  
In this study, you can choose to train the CNN:
- in time-domain or frequency-domain  
- for one of the 6 atypical subjects of one of the 2 studies  
- based on “within-domain transfer learning approach” (knowledge transfer within the same domain)  or “across-domains transfer learning approach” (knowledge transfer across domains).  

You can train and test this framework by running the file ‘stereotypyMainMultSessions.m' and following steps of Example 2 for “knowledge transfer within the same domain” and steps of Example 4 for “knowledge transfer within the same domain”   (see below).

<h2>Examples for training and/or testing our models : </h2>

<h3>Example 1: SMM detection within a subject with randomly initialized CNNs in time-domain</h3>
<b>Goal:</b>Train a randomly initialized CNN on record sessions within a subject and a study (subject 1 of study 1).  
<b>Details:</b> Train a randomly initialized CNN on SMM data which consists of 2 record sessions of Subject 1 (Study 1). The SMM data taken into account are only SMMs containing rock movements, namely rock and flap-rock activities (flap movements not included). 


<b>Steps:</b> Run the code “stereotypyMainMultSessions.m” and enter values below:

- Please select subject ID (1-6):1
- Please select study type (1-2):1
	[1]:URI-001-01-18-08
	[2]:URI-001-01-25-08
- Please select the number of sessions to process from the printed list: 2
- Please select: (1)frequency domain / (2)time domain: 2
- Please select: (1)train network from scratch / (2)train using pre-training SMM network / (3)train using pre-trained HAR network: 1
- Please select: (1)do feature extraction / (2)do not extract features (already extracted) 1  
- Please select: (0)train on all data (non-smm, rock, flap-rock, flap) / (1)train on rock data only (non-smm, rock, flap-rock): 0 

PS: in the options, we choose to extract features from raw data. This step uses the file “preprocessedDataAndLabels.mat” (labeled data) and creates the file “featureVectorAndLabels.mat” (extracted features) for each record sessions of the selected subject(s) within the selected study. In this example, a .mat file is created within each record session of Subject 1 in the path ‘data/Study1/URI-001-XX-XX-XX’ folder). Furthermore, for future experiments, once you create the file “featureVectorAndLabels.mat”  for a subject within a study, you don't need to create it again for upcoming experiments; so you will select "(2)do not extract features" in the option menu.

The displayed result (at test time) in terms of error:

Lowest validation error is 0.038870

<b>How to get the F1-score:</b>

To display the result at that epoch (epoch 35) in terms of the F1-score measure (during training and during testing):
1)	go to the file “ASD_movement_CNN.m”  
2)	change the value of the variable  opts.batchSize” from “50” to  “size(testLabel,1)”   
3)	change the value of the variable  “opts.numEpochs” from “35” to “36”  
4)	change the value of the variable “opts.continue” from “false” to “true”   
5)	and run the file  

The dislayed result is:  
	training: epoch 36: batch   1/  1: Confusion matrix: 
	
		6670         109
		83        6696

	Precision 0.983982\n
	Recall 0.987756\n
	F1 score 0.985866\n
	Accuracy 0.985839
	738.4 Hz obj:0.0397 top1e:0.0142 top5e:0 [13558/13558]
	validation: epoch 36: batch   1/  1: Confusion matrix: 
		   10297         394
			 195        2672
	Precision 0.871494
	Recall 0.931985
	F1 score 0.900725
	Accuracy 0.956557
	2159.9 Hz obj:0.151 top1e:0.0434 top5e:0 [13558/13558]

<h3>Example 2: SMM detection across subjects via knowledge transfer (within the same domain) in frequency-domain</h3>
<b>Goal:</b>Train a framework for the recognition of SMMs of Subject 2 (Study 1) using a CNN network already pretrained on SMMs of subjects other than Subject 2 (i.e., Subjects 1, 3, 4, 5, 6 of Study 2).  

<b>Details:</b>   
1) Use low and mid-level features of a CNN which was already pretrained on SMMs of subjects other than Subject 2 (e.g., records sessions of Subjects 2, 3, 4, 5, 6 of the selected study (Study 2)). The record sessions exclude flap movement activities and keep non-SMM, rock-SMM and flap-rock SMM activities.    
2) Apply them on top of an SVM to form a framework and train the SVM via this framework using 2000 instances randomly selected from the training dataset of Subject 2 (Study 1).  

<b>Steps:</b> Run the code “stereotypyMainMultSessions.m” and enter values below:  
- Please select subject ID (1-6):2
- Please select study type (1-2):1
    [1]:URI-002-01-18-08
    [2]:URI-002-01-24-08
- Please select the number of sessions to process from the printed list:2
- Please select: (1)frequency domain / (2)time domain:1
- Please select: (1)train network from scratch / (2)train using pre-training SMM network / (3)train using pre-trained HAR network: 2
- Please select: (1)do feature extraction / (2)do not extract features (already extracted) 1 
- Please select: (0)train on all data (non-smm, rock, flap-rock, flap) / (1)train on rock data only ((non-smm, rock, flap-rock):1  

Displayed result:  

	Confusion matrix: 
				6429           2
			   0          39
	Precision 0.951220
	Recall 1.000000
	F1 score 0.975000
	Accuracy 0.999691


<h3>Example 3: SMM detection across subjects via knowledge transfer (across domains) in frequency-domain</h3>  
<b>Goal:</b> Train a framework for the recognition of SMMs of Subject 1 (Study 2) using a CNN network already pre-trained on human movements/activities of everyday life of typical subjects.  

<b>Details:</b>
1) Use low and mid-level features of a CNN which was already pre-trained on human activities (for human activity recognition).   
2) Apply them on top of an SVM to form a framework and train the SVM via this framework using 2000 instances randomly selected from the training dataset of Subject 2 (Study 1).   

<b>Steps:</b> Run the code “stereotypyMainMultSessions.m” and enter values below:  
- Please select subject ID (1-6):2
- Please select study type (1-2):1
[1]:URI-002-01-18-08  
[2]:URI-002-01-24-08  
- Please select the number of sessions to process from the printed list:2
- Please select: (1)frequency domain / (2)time domain:1  
- Please select: (1)train network from scratch / (2)train using pre-training SMM network / (3)train using pre-trained HAR network: 3
- Please select: (1)do feature extraction / (2)do not extract features (already extracted) 2
- Please select: (0)train on all data (non-smm, rock, flap-rock, flap) / (1)train on rock data only ((non-smm, rock, flap-rock):1  

Displayed result:  

		Confusion matrix: 
				6424           6
				   0          40
		Precision 0.869565
		Recall 1.000000
		F1 score 0.930233
		Accuracy 0.999073
