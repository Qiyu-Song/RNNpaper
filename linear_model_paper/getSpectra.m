function [my,mu,mlogp]=getSpectra(casename,num_P,len_P,nT,nQ)
Y=[];U=[];P=[];

%for i=4:num_P*4+7
for i=0:num_P*4+3
  out=read_rand_forcing2(casename,nT,nQ,i);
  Y=[Y out.Y];
  U=[U out.U];
  P=[P out.P];
end

%Fourier analysis
%skips the first period because of transients
for i=1:num_P
  u(:,:,i)=U(:,len_P*i+(1:len_P));
  y(:,:,i)=Y(:,len_P*i+(1:len_P));
  logp(:,:,i)=log(P(len_P*i+(1:len_P)));
end

%mean
my=mean(y(:,:,1:num_P),3);
mu=mean(u(:,:,1:num_P),3);
mlogp=mean(logp(:,:,1:num_P),3);

nc_spinup='../data/RCE_randmultsine_spinup_10243.nc';
T_spinup=double(ncread(nc_spinup,'TABS'));
Q_spinup=double(ncread(nc_spinup,'QV'));
Y_spinup=[T_spinup(1:nT,:); Q_spinup(1:nQ,:)];
Y_spinup=mean(Y_spinup, 2)
size(my)
size(Y_spinup)
my=my-Y_spinup;
