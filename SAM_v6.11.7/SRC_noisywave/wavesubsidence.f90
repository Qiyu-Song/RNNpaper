!Add vertical advection of q,w due to the large-scale wave and some largescale damping of T, q
subroutine wavesubsidence()

use vars
use params
use microphysics
use module_linearwave
implicit none

integer i,j,k
real ttend_center(nzm), qtend_center(nzm) !cell center tendencies
real rrr,ranf_,persistence,amp,fixTQdampingtime !for random forcing tendencies,
integer number_of_steps
double precision coef, coef1, buffer(nzm), buffer1(nzm)
real pii,factor

if(nstep.gt.nstartlinearwave+nsteplinearwavebg.and.icycle.eq.ncycle) then
   ttend_center=0.
   qtend_center=0.

   !Obtain the tendency from advecting the background/reference T,q fields
   if(doadvectbg) then
      call linearwavesubsidence(           w_wave(1:nzm-1),      &
              t_wavebg(1:nzm-1),                       q_wavebg(1:nzm-1),    &
              z(1:nzm-1),                          ttend_center(1:nzm-1),        &
              qtend_center(1:nzm-1),                           nzm-1 )
   else
      !advect the full T,q fields
      call linearwavesubsidence(           w_wave(1:nzm-1),      &
              t_wave(1:nzm-1),                       q_wave(1:nzm-1),    &
              z(1:nzm-1),                          ttend_center(1:nzm-1),        &
              qtend_center(1:nzm-1),                           nzm-1 )
   endif
   
   if(mod(nstep-nstartlinearwave-nsteplinearwavebg,nsteplinearwave).eq.0) then
      !record the original wave tendencies for output statistics
      ttend_wave0 = ttend_center
      qtend_wave0 = qtend_center
   end if

   ! add damping
   do k=1,nzm
      ttend_wavedamp(k) = -(t_wave(k)-t_wavebg(k))/dble(86400.*wavetqdampingtime)
      qtend_wavedamp(k) = -(q_wave(k)-q_wavebg(k))/dble(86400.*wavetqdampingtime)
   end do

   ! add noise
   if(donoisywave) then
      pii = atan2(0.d0,-1.d0)
      !update the forcing every nsteplinearwave time steps
      !index for the nsteplinearwave*dt chunks
      i=(nstep-nstartlinearwave-nsteplinearwavebg)/nsteplinearwave
      i=mod(i,lenP) !reuse the same set of random numbers after lenP time
      ttend_wavenoise=0.
      qtend_wavenoise=0.
      !include all frequency forcings
      do k=1,lenP/freqfactor
         factor=sqrt(0.5)
         do j=1,nT
            ttend_wavenoise(j)=ttend_wavenoise(j)+cos((k*real(i)/real(lenP)+randcoef(j,k))*2.*pii)/factor
         end do
         do j=1,nQ
            qtend_wavenoise(j)=qtend_wavenoise(j)+cos((k*real(i)/real(lenP)+randcoef(j+nT,k))*2.*pii)/factor
         end do
      end do

      do k=1,nT
         ttend_wavenoise(k) = ttend_center(k) * noiselevel * ttend_wavenoise(k) / sqrt(real(lenP/freqfactor))
         ttend_center(k) = ttend_center(k) + ttend_wavenoise(k)
      end do
      do k=1,nQ
         qtend_wavenoise(k) = qtend_center(k) * noiselevel * qtend_wavenoise(k) / sqrt(real(lenP/freqfactor))
         qtend_center(k) = qtend_center(k) + qtend_wavenoise(k)
      end do
   endif

   do k=1,nzm
      ttend_center(k) = ttend_center(k) + ttend_wavedamp(k)
      qtend_center(k) = qtend_center(k) + qtend_wavedamp(k)
   end do
   
   number_of_steps=1
   do k=1,nzm
      water_fill(k)=0.
      do j=1,ny
         do i=1,nx
            t(i,j,k) = t(i,j,k) + number_of_steps*dt * ttend_center(k)
            !In SAM1MOM, micro_field(:,:,:,index_water_vapor) actually includes nonprecipitating water as well
            micro_field(i,j,k,index_water_vapor)=micro_field(i,j,k,index_water_vapor) + number_of_steps*dt *&
                 qtend_center(k)
            if (micro_field(i,j,k,index_water_vapor).lt.0) then
               water_fill(k)=water_fill(k)-micro_field(i,j,k,index_water_vapor)
               micro_field(i,j,k,index_water_vapor) = 0.
            end if
         end do
      end do
      water_fill(k)=water_fill(k)/dble(nx*ny)
   end do

   if(mod(nstep-nstartlinearwave-nsteplinearwavebg,nsteplinearwave).eq.0) then
      !compute T and q profiles after large-scale advection and before convection
      !because of the removal of negative qv above, this may incur a small error
      number_of_steps=nsteplinearwave
      coef=1./dble(nx*ny)
      do k=1,nzm
         bc_tabs0_local(k)=t_wave_local(k)+number_of_steps*dt*ttend_center(k)
         !there may be slight inaccuracies due to zeroing of negative water field as above
         bc_qv0_local(k)=q_wave_local(k)+water_fill(k)+number_of_steps*dt*qtend_center(k)
      end do
      coef1 = 1./dble(nsubdomains)
      buffer=bc_qv0_local
      call task_sum_real8(buffer,buffer1,nzm)
      bc_qv0=buffer1*coef1
      buffer=bc_tabs0_local
      call task_sum_real8(buffer,buffer1,nzm)
      bc_tabs0=buffer1*coef1
      !record the tendencies for output statistics
      ttend_wave = ttend_center
      qtend_wave = qtend_center
   end if
end if

end subroutine wavesubsidence

