% Ensemble Consensus clustering for time window determination of grand average ERP

% Copyright info

% First version of Concenuc clustering (May 2017)
% Updated at Aug 2019 Stabilization (Mahini et al. 2020)
% Code by Reza Mahini r.mahini@gmail.com, remahini@jyu.fi

% Plese cite this work as :

% Mahini, R., Li, Y., Ding, W., Fu, R., Ristaniemi, T., Nandi, A.K., et al. (2020).
% Determination of the Time Window of Event-Related Potential Using Multiple-Set Consensus Clustering.
% Frontiers in Neuroscience 14(1047). doi: 10.3389/fnins.2020.521595.

% Mahini, R., Xu, P., Chen, G., Li, Y., Ding, W., Zhang, L., Qureshi, N. K., Hämäläinen, T., Nandi, A. K., & Cong, F. (2022).
% Optimal Number of Clusters by Measuring Similarity Among Topographies for Spatio-Temporal ERP Analysis.
% Brain Topography. https://doi.org/10.1007/s10548-022-00903-2

% Mahini, R., Xu, P., Chen, G., Li, Y., Ding, W., Zhang, L., et al. (2019).
% Optimal Number of Clusters by Measuring Similarity among Topographies for Spatio-temporal ERP Analysis.
% arXiv preprint arXiv:1911.09415.

% ------------------------- Before you run this Code -----------------------
%% We need to run your individual DNNs to get these datasets 

%  Therefore

%  Prepare the below files before :
%    1- dataFeature_MLP.mat
%    2- dataFeature_CNN.mat
%    3- dataFeature_LSTM.mat
%    4- dataFeature_AE
%    5-dataFeature_VAE % currentPrediction
%    6- DEC_lb.mat % 

% So, Open Rstodio (we suppose you already have installed Python up v3.7 and the required packages in R,
% especially: tensorflow,keras, ggplot2, ... 
% (see more in our R codes and follow the 'Readme' fiel)


%% ----------------------------Beging -------------------------------------

clc;
clear;
close all;
delete K_lg_eig_M2.mat % to make sure the dataset is replaced with new

% Adding a path for this code ------------------------------------------

% make sure you have this folder in the current work matla 'Workplace'
% address

addpath("Sim_MS3_data\")

% -------------------------Input ERP (all)----------------------------

load chanlocs; % coordination file for electrodes
load SimData_all; % the simulated data (Demo)
load StEd_TW_area.mat; % the ground-truth TW properties


chanLoc=chanlocs;

% noise levels : 1=-10dB, 2=-5dB, 3=0dB, 4=5dB, 5=10dB, 6=20dB, 7=50dB

nsl=input('Noise level (1=-10dB, 2=-5dB, 3=0dB, 4=5dB, 5=10dB, 6=20dB, 7=50dB) = ? ');
Data=squeeze(SimData_all(:,:,:,:,nsl));

size(Data) % chan x sam x subj x cond

% ------------------------------Initializing -----------------------------

% Please be carefull about setting the parameters !!!

G=1; % group
St=2; % condition
Sa=300; % samples (upsampled from 150 to 300)
Subj=20; % subjects
startEph=-100; % starting epoch (ms)
endEph=600; % end epoch (ms)
Comp=1; % ERP component
Chan=65; % channels
nSam=St*Sa; % special parmeter

% Processing parameters -------------------------------------------------

count=1; % repetitions if needed
minSamThr=10;
InSim_Thr=input('The inner similarity threshold 0 <InSim_Thr <1 (def. = 0.90)?');
if isempty(InSim_Thr)
   InSim_Thr=0.90; %  Inner-similarity threshold,
end

% More options
stb=0; % stabilization
plotonoff=1; % plot permission

% Experimental interval (a rough estimation for focusing on target component)
twStart=261; % (ms)
twEnd=370; % (ms)

stimSet={'Cond1','Cond2'};
compSet={'P3'};

CSPA_f_result=[];
CSPA_Cl_pow_res=[];

% ------------------------------Data preparing------------------------------

inData=DimPrep(Data,Chan,Sa,St,Subj,G);
% Channel x Sample x Stim x Subject x Group

[GA_ERP]=grand_ERP(inData,G);

% Temporal concatenating the ERP data (be careful about the parameters)
[ERP_Subj,inDaGA_M1]=Data_Preparing(inData,Subj,St,Sa,G); % subjects data and grand average data modeled data

% save inDaGA.mat inDaGA_M1 % optional

x=inDaGA_M1; % the grand average data ""samples x channels x group""

% -------------------- Clustering with multiple-method --------------------

tic

stb=input('Do we need stablilization? Enter (1=yes, Def(0=No)?');
if isempty(stb)
   stb=0;
end

% % IndAnlz=input('Do you need individual subject analysis results (No =0, Yes=1) ? ');
% % plotonoff=input(' Do you need the component plot for each subject (No =0, Yes=1) ?');

clmethod={'K-means','Hierachial','FCM','SOM ','DFS', 'MKMS', 'AAHC', 'SPC', 'KMD', 'GMM'};
% 1=K-means', 2= 'Hierachial', 3= 'FCM',4 ='SOM ','5 = DFS', 6= 'MKMS',...
% 7= 'AAHC', 8= 'SPC', 9= 'KMD', 10='GMM'

DCls={'MLP','1DCNN','LSTM','AE','VAE','DEC','EnsDCs'};

% Selecting the methods for CC
M_list=[1 2 10];
Stb_list=[1 5 10]; % declare which methods needs stabilization
rep=[5 5 5]; % number of repeat

% K=input('Enter number of clusters: ');
k=6; % number of clusters obtained form (Mahini et al., 2022)


% MethLabs_M1=Labeling_AllCls_M1(inDaGA_M1,k,1,rep,stb,M_list,Stb_list);

%     for g=1:G
f_result_K=[];
Cl_pow_res=[];
Rank_Res=[];
STD_value=[];
cluster_N=[];
DC_LBs=[];


%% ------------------------  Cluster analysis  ----------------------------

tag1=input('Which Method results you want to analysis? (Type 0=CC (def), 1=DC) ');
if isempty(tag1)
   tag1=0; % ****
end

if tag1==0

   tg1=input('Doyou need new labeling from the begining? (0 = No (def) m 1 = Yes)');
   if isempty(tg1)
      tg1=0; % ****
   end

   if tg1==1

      % Generatio phase of CC
      MethLabs_M1(count).labels=Labeling_AllCls_M1(inDaGA_M1,k,rep,stb,M_list,Stb_list);

      methods=[];

      methods=MethLabs_M1(count).labels.data;

      [p, q]=size(methods); % all method results in one single dataset...

      CC_Lab=CSPA(methods,k); % consensus function
      figure('Renderer', 'painters', 'Position', [10 10 650 450])
      imagesc([methods CC_Lab])
      Clu_idx=CC_Lab;
      CC_label=CC_Lab;
      xticks(1:q+1)
      xticklabels([clmethod{M_list} "CC"])% "cl" includes involved clustering methods
      save CC_label.mat CC_label;
   else
      load CC_label
      Clu_idx=CC_label;
   end

else

   % Deep clustering results loading -----------------------------------

   tag2=input('which DC do you need ? (type 1 = MLP, 2 = 1DCNN, 3=LSTM , 4=AE, 5=VAE, 6= DEC, (EDC= def)),');
   if isempty(tag2)
      tag2=8; % ****
   end
   
   % We need to run your individual DNNs to get these datasets before executing this come 

   load dataFeature_MLP.mat
   load dataFeature_CNN.mat
   load dataFeature_LSTM.mat
   load dataFeature_AE
   load dataFeature_VAE % currentPrediction
   load DEC_lb.mat % DEC_idx.mat

   DC_LBs(:,1)=kmeans(dataFeature_MLP,k); %CSPA([kmeans(dataFeature_MLP,k) kmeans(dataFeature_MLP,k) kmeans(dataFeature_MLP,k)],k); %kmeans(dataFeature_MLP,k);
   DC_LBs(:,2)=kmeans(dataFeature_CNN,k); %CSPA([kmeans(dataFeature_CNN,k) kmeans(dataFeature_CNN,k) kmeans(dataFeature_CNN,k)],k);
   DC_LBs(:,3)=kmeans(dataFeature_LSTM,k); %CSPA([kmeans(dataFeature_LSTM,k) kmeans(dataFeature_LSTM,k) kmeans(dataFeature_LSTM,k)],k); % kmeans(dataFeature_LSTM,k);
   DC_LBs(:,4)=CSPA([kmeans(dataFeature_AE,k) kmeans(dataFeature_AE,k) kmeans(dataFeature_AE,k)],k);
   %DC_LBs(:,5)= double(currentPrediction);
   DC_LBs(:,5)= kmeans(dataFeature_VAE,k); %CSPA([kmeans(dataFeature_VAE,k) kmeans(dataFeature_VAE,k) kmeans(dataFeature_VAE,k)],k); %clusterdata(dataFeature_VAE,'linkage','complete','distance','euclidean','maxclust',k);
   DC_LBs(:,6)=yPredicted; % y_pred;
   [p,q]=size(DC_LBs);

   switch tag2
      case 1
         DC_LB=DC_LBs(:,1);
      case 2
         DC_LB=DC_LBs(:,2);
      case 3
         DC_LB=DC_LBs(:,3);
      case 4
         DC_LB=DC_LBs(:,4);
      case 5
         DC_LB=DC_LBs(:,5);
      case 6
         DC_LB=DC_LBs(:,6);
      otherwise
         DC_LB=CSPA(DC_LBs(:,1:6),k);
         DC_LBs(:,q+1)=DC_LB;
   end

   %             DC_LB=kmeans(cd_LS,k);
   %             rand_index(Clu_idx,DC_mlp)
   Clu_idx=DC_LB;

   if q==6 % all clusterings available
      save DC_LBs.mat DC_LBs;
      figure
      imagesc(DC_LBs);
      xticks(1:q+1)
      xticklabels([DCls])% "cl" includes involved clustering methods
   end
end
if ~isempty(DC_LBs)
   %       load DC_LBs
   [p1,q1]=size(DC_LBs);

   if q1==7
      load CC_label
      figure('Renderer', 'painters', 'Position', [10 10 650 450])
      imagesc([DC_LBs CC_label])
      xticks(1:q1+1)
      xticklabels([DCls "CC"])% "cl" includes involved clustering methods
   end
end


%% Analysis of the clustering results ----------------------------
for com=1:Comp
% Convertor (to the sample)
   [v,w]=time_conv_ts(Sa,twStart(com),twEnd(com),startEph,endEph);
   
   %         v=twStart(com);
   %         w=twEnd(com);

   % Channels' address for processing
   selChan={'CP2','CPz','Cz'};
   ch_loc=[42    58    65];
   
   % channel for plot
   selChan1={'Cz'};
   ch_loc1=[65];

   for g=1:G

      x1=squeeze(x(:,:,g));

      [CSPA_f_result,comp_pow,innerCorr,winnID,InnSimWcl]=Comp_detect_ERP_CC_Upd(Clu_idx,x1,chanlocs,k,Sa,St,v,w,com,stimSet,compSet,InSim_Thr,minSamThr);

      [selected_TW,TWs_ms,selTWs_ms,sel_innerCorr,InnSim,selPower_amp]=...
         Sel_TW_Upd(CSPA_f_result,innerCorr,v,w,St,g,Sa,startEph,endEph,winnID,InnSimWcl,comp_pow); % TWs selection algorithm

      for st=1:St

         compCorr=[];
         compCorr=sel_innerCorr(st).data;
         n=size(compCorr,1);
         meanRow=sum(sum(compCorr,1))-n;
         innSim(st)=meanRow/(n^2-n);

         meantop_amp(st,:)=selPower_amp(st).data;

      end

      % Plot the component1 --------------------------------------------
      if plotonoff

         index=Indexing_M1(nSam,Clu_idx,k,St,g);

         PlotAmp_M1_SIM(x1,index,Sa,startEph,endEph,St,ch_loc1,selChan1,com,compSet,stimSet);

         for st=1:St

            WI=selected_TW(st,1);
            figure('Renderer', 'painters', 'Position', [10 10 750 350])

            subplot(1,2,1);
            topoplot(squeeze(meantop_amp(st,:)),chanLoc)

            title(['Topography map, ClustNo.', int2str(WI),', ', stimSet{st},]);
            set(gca,'fontsize',12);
            colorbar;
            caxis([-2 2]);

            subplot(1,2,2)
            imagesc(sel_innerCorr(st).data);
            title(['Samples Correlation']);
            xlabel('Sample #');
            ylabel('Sample #');
            set(gca,'fontsize',12);
            colorbar;
            caxis([-1 1]);

         end

      end

   end

   selTWs(com).data=selTWs_ms;
   disp(compSet{com})
   disp(selTWs(com).data)

   compGroup_CC(count).comp(com).innSimm(k).data=innSim; %  access innSim(st,g), st =stimulus, g= group
   % % %         compGroup_CC(count).comp(com).Corr(k).data=sel_innerCorr; % innerCorr
   compGroup_CC(count).comp(com).sel_TW(k).data=selected_TW;
   compGroup_CC(count).comp(com).idx(k).data=Clu_idx;
   compGroup_CC(count).comp(com).sel_TW_ms(k).data=selTWs(com).data; % ms
   compGroup_CC(count).comp(com).meantop(k).data=meantop_amp; % access meantop_amp(st,:,g)

   [SPSS_tab_avg]=ERP_statTable2_100s(selected_TW,inData,ch_loc,Subj,St,G);
   SPSStab_avg(count).comp(com).data=SPSS_tab_avg;

   [ranova_tbl]=ranova_ERP_100(SPSS_tab_avg);
   ranovatbl_all(count).comp(com).data=ranova_tbl;

   stim_pvalue(count,com,:)=ranova_tbl.pValue(3);
   chan_pvalue(count,com,:)=ranova_tbl.pValue(5);
   intStCh_pvalue(count,com,:)=ranova_tbl.pValue(7);

   ranova_tot.count(count).comp(com).data=ranova_tbl;

end

disp('Statistical analysis results --------------------------------')
disp('stim_pvalue,   chan_pvalue,   intStCh_pvalue');
all_pvalue=[stim_pvalue,chan_pvalue,intStCh_pvalue];
disp(all_pvalue);

%% Optional performance of deep learnings
% Perf_Dcs={'tr_inf_MLP', 'tr_inf_CNN','tr_inf_LSTM','tr_inf_AE'}; %,'tr_inf_VAE'};
% 
% for m=1:4
%    load (Perf_Dcs{m});
%    tr_inf(m).fold(1).data=tr_inf_F1;
%    tr_inf(m).fold(2).data=tr_inf_F2;
%    tr_inf(m).fold(3).data=tr_inf_F3;
%    tr_inf(m).fold(4).data=tr_inf_F4;
%    tr_inf(m).fold(5).data=tr_inf_F5;
% 
%    for f=1:5
%       tr_av_accloss(m,f,1)=mean(tr_inf(m).fold(f).data(1,:)); % av_tr_accloss(m,f,1=tr_loss/2=tr_acc/3=tr_valloss/4=tr_valacc)
%       tr_sd_accloss(m,f,1)=std(tr_inf(m).fold(f).data(1,:)); % av_tr_accloss(m,f,1=tr_loss/2=tr_acc/3=tr_valloss/4=tr_valacc)
% 
%       tr_av_accloss(m,f,2)=mean(tr_inf(m).fold(f).data(2,:));
%       tr_sd_accloss(m,f,2)=std(tr_inf(m).fold(f).data(2,:)); % av_tr_accloss(m,f,1=tr_loss/2=tr_acc/3=tr_valloss/4=tr_valacc)
% 
%       tr_av_accloss(m,f,3)=mean(tr_inf(m).fold(f).data(3,:));
%       tr_sd_accloss(m,f,3)=std(tr_inf(m).fold(f).data(3,:)); % av_tr_accloss(m,f,1=tr_loss/2=tr_acc/3=tr_valloss/4=tr_valacc)
% 
%       tr_av_accloss(m,f,4)=mean(tr_inf(m).fold(f).data(4,:));
%       tr_sd_accloss(m,f,4)=std(tr_inf(m).fold(f).data(4,:)); % av_tr_accloss(m,f,1=tr_loss/2=tr_acc/3=tr_valloss/4=tr_valacc)
%    end
% 
% 
%    te_acclossall(1,m)=te_accloss(1); % loss
%    if length(te_accloss)<2
%       te_acclossall(2,m)=0; % acc
%    else
%       te_acclossall(2,m)=te_accloss(2); % acc
%    end
% 
% end

%% Statistical power analysis for within factors (Stim x Chann) -----------


function [ranova_tbl]=ranova_ERP_100(SPSS_tab_avg)

Rdata=SPSS_tab_avg(:,2:7);

Group={'G1';'G1';'G1';'G1';'G1';'G1';'G1';'G1';'G1';'G1';'G1';'G1';'G1';'G1';'G1';'G1';'G1';'G1';'G1';'G1'};

varNames={'Group','St1_ch1','St1_ch2','St1_ch3','St2_ch1','St2_ch2','St2_ch3'};

tbl = table(Group,Rdata(:,1),Rdata(:,2),Rdata(:,3),Rdata(:,4),...
   Rdata(:,5),Rdata(:,6),'VariableNames',varNames);

factNames = {'Stim','Chan'};

within_R = table({'St1';'St1';'St1';'St2';'St2';'St2'},{'Ch1';'Ch2';'Ch3';'Ch1';'Ch2';'Ch3'},'VariableNames',factNames);

rm = fitrm(tbl,'St1_ch1-St2_ch3~1','WithinDesign',within_R);

[ranova_tbl] = ranova(rm, 'WithinModel','Stim*Chan');
end

% ---------------------- End of Cluster analysis --------------------------
