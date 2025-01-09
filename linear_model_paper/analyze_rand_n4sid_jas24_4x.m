%==============================
%Identify the state-space model
%==============================

clear all

nT=26; %number of temperature modes
nQ=14; %number of humidty modes

%define a norm
load IDEAL_dm.mat  %reading mass in each layer
dm=double(dm);
%weight variance by mass. Works best for vertically coherent signal
%in that subdivide a layer into multiple layers that are coherent
%does not change the answer.
mass_weight=diag(sqrt([dm(1:nT);2.5^2*dm(1:nQ)]));
prec_weight=[0.*dm(1:nT);2.5*dm(1:nQ)]; %weighting to compute net precipitation
mse_weight=[dm(1:nT);2.5*dm(1:nQ)];     %weighting to compute column MSE

caseid=1
switch caseid
  case 1
   casename='msinefx4_0_2';
   experiments={'msinefx4_0','msinefx4_1', 'msinefx4_2'};
   num_P=4; %number of periods
   len_P=19200; % length of a period (in unit of 900ss)
   day_P=len_P/96; %length of a period in days
   timestep=900; %in seconds
  otherwise
  display('unknown caseid');
  stop
end

%frequency
omega=((1:len_P)-1)/len_P;
num_Exp=numel(experiments)

for i=1:num_Exp
  %read out the average spectra
  [Y,U,LOGP]=getSpectra(experiments{i},num_P,len_P,nT,nQ);
  
  Y=Y';
  U=U';
  LOGP=LOGP';

  %remove mean
  Y=Y-mean(Y);
  U=U-mean(U);
  LOGP=LOGP-mean(LOGP);

  %turn into energy unit
  Y=Y*mass_weight;
  U=U*mass_weight;
  LOGP=LOGP;

  tmpdata=iddata([Y, LOGP],U,timestep);

  alldata{i}=tmpdata;
end

%merge the experiments together
data=alldata{1};

if(num_Exp>1)
  for i=2:num_Exp
    data=merge(data,alldata{i});
  end
end

%Set the identification options
opt=n4sidOptions('Display','on','Focus','prediction');
opt.InitialState='estimate';
opt.EstCovar=false;

%%
nstate=64 %number of states
disp('starting system identification')

tic
sys=n4sid(data,nstate,'Form','free',opt);
toc

save(['./model/sys_' casename '_' int2str(nstate) ...
      '_free_' opt.Focus '_estx0_nomean_tabs_addp_cov_freq2t_' sys.Report.N4Weight ...
      '_' int2str(sys.Report.N4Horizon(1)) '_' int2str(sys.Report.N4Horizon(2)) '_' int2str(sys.Report.N4Horizon(3)) '.mat'], 'sys');
  
