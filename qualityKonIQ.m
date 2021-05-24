clear all
close all

load KonIQ10k.mat    % This mat file contains information about the database (KonIQ-10k)

Directory = 'C:\Users\Public\QualityAssessment\KonIQ-10k\1024x768';  % path to KonIQ-10k database 
Waterloo  = 'C:\Users\Public\QualityAssessment\exploration_database_and_code'; % path to Waterloo database
numberOfImages = size(mos,1);   % number of images in KonIQ-10k database
numberOfTrainImages = round( 0.8*numberOfImages );   % appx. 80% of images is used for training
numberOfSplits = 100;

Constants.net = vgg16;
Constants.L   = Constants.net.Layers(1,1).InputSize(1);
Constants.layer = 'fc6';
Constants.N1 = 15;
Constants.N2 = 20;
Constants.pooling = 'avg';
numLayers = size(Constants.net.Layers, 1);
for i=1:numLayers
    name = Constants.net.Layers(i,1).Name;
    if(strcmp(name,Constants.layer))
        Constants.length = size(Constants.net.Layers(i,1).Bias, 1);
        break;
    end
end
Constants.TransferLearning = true; % false
Constants.Regression = 'rqgpr';

GlobalFeatures = zeros(numberOfImages, Constants.length);
LocalFeatures1 = zeros(numberOfImages, Constants.length*3);
LocalFeatures2 = zeros(numberOfImages, Constants.length*3);

disp('Feature Extraction');
for i=1:numberOfImages
    %if(mod(i,1000)==0)
        disp(i);
    %end
    img           = imread( strcat(Directory, filesep, names{i}) );
    tmp           = getFeatures(img, Constants);
    GlobalFeatures(i,:) = tmp.global;
    LocalFeatures1(i,:) = tmp.local1;
    LocalFeatures2(i,:) = tmp.local2;
end

disp('Training and Testing');
for i=1:numberOfSplits
    rng(i);
    if(mod(i,10)==0)
        disp(i);
    end
    p = randperm(numberOfImages);
    
    Data_1 = GlobalFeatures(p,:);
    Data_2 = LocalFeatures1(p,:);
    Data_3 = LocalFeatures2(p,:);
    Target = mos(p);
    
    Train_1 = Data_1(1:round(numberOfImages*0.8),:);
    Train_2 = Data_2(1:round(numberOfImages*0.8),:);
    Train_3 = Data_3(1:round(numberOfImages*0.8),:);
    TrainLabel = Target(1:round(numberOfImages*0.8));
    
    Test_1  = Data_1(round(numberOfImages*0.8)+1:end,:);
    Test_2  = Data_2(round(numberOfImages*0.8)+1:end,:);
    Test_3  = Data_3(round(numberOfImages*0.8)+1:end,:);
    TestLabel = Target(round(numberOfImages*0.8)+1:end);
    
    if( strcmp(Constants.Regression, 'rbfsvr') )
        Mdl_1 = fitrsvm(Train_1, TrainLabel', 'KernelFunction', 'gaussian', 'KernelScale', 'auto', 'Standardize', true);
        Mdl_2 = fitrsvm(Train_2, TrainLabel', 'KernelFunction', 'gaussian', 'KernelScale', 'auto', 'Standardize', true);
        Mdl_3 = fitrsvm(Train_3, TrainLabel', 'KernelFunction', 'gaussian', 'KernelScale', 'auto', 'Standardize', true);
    elseif( strcmp(Constants.Regression, 'linsvr') )
        Mdl_1 = fitrsvm(Train_1, TrainLabel', 'KernelFunction', 'linear', 'Standardize', true);
        Mdl_2 = fitrsvm(Train_2, TrainLabel', 'KernelFunction', 'linear', 'Standardize', true);
        Mdl_3 = fitrsvm(Train_3, TrainLabel', 'KernelFunction', 'linear', 'Standardize', true);
    elseif( strcmp(Constants.Regression, 'rqgpr') )
        Mdl_1 = fitrgp(Train_1, TrainLabel', 'KernelFunction', 'rationalquadratic', 'Standardize', true);
        Mdl_2 = fitrgp(Train_2, TrainLabel', 'KernelFunction', 'rationalquadratic', 'Standardize', true);
        Mdl_3 = fitrgp(Train_3, TrainLabel', 'KernelFunction', 'rationalquadratic', 'Standardize', true);
    else
        error('Not defined regression algorithm');
    end
    
    Pred_1 = predict(Mdl_1, Test_1);
    Pred_2 = predict(Mdl_2, Test_2);
    Pred_3 = predict(Mdl_3, Test_3);
    
    Pred = (Pred_1 + Pred_2 + Pred_3)/3;
    
    beta(1) = max(TestLabel); 
    beta(2) = min(TestLabel); 
    beta(3) = mean(TestLabel);
    beta(4) = 0.5;
    beta(5) = 0.1;
    
    [bayta,ehat,J] = nlinfit(Pred,TestLabel,@logistic,beta);
    [pred_test_mos_align, ~] = nlpredci(@logistic,Pred,bayta,ehat,J);
    
    PLCC(i) = corr(pred_test_mos_align,TestLabel);
    SROCC(i)= corr(Pred,TestLabel,'Type','Spearman');
end

disp('----------------------------------');
X = ['Average PLCC after ', num2str(numberOfSplits), ' random train-test splits: ', num2str(round(mean(PLCC(:)),3))];
disp(X);
X = ['Average SROCC after ', num2str(numberOfSplits),' random train-test splits: ', num2str(round(mean(SROCC(:)),3))];
disp(X);

%PLCC(i) = corr(pred_test_mos_align,YTest,'type','Pearson');
%RMSE(i) = sqrt(mean((YTest - pred_test_mos_align).^2));