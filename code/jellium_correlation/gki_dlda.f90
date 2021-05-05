
subroutine high_freq(rs,param,finf,bn)

  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)
  real(dp), parameter :: pi = 3.14159265358979323846264338327950288419_dp
  real(dp), parameter :: gam = 1.311028777146059809410871821455657482147216796875_dp
  real(dp), parameter :: cc = 4.81710873550434914847073741839267313480377197265625_dp

  real(dp), intent(in) :: rs
  character(len=4), intent(in) :: param
  real(dp), intent(out) :: finf,bn

  real(dp) :: f0,ec,d_ec_drs,n,finf_x,finf_c,df

  call alda(rs,param,f0)
  call lda_derivs(rs,param,ec,d_ec_drs)

  n = 3._dp/(4*pi*rs**3)

  finf_x = -(3._dp/(pi*n**2))**(1._dp/3._dp)/5._dp
  finf_c = -(22*ec + 26*rs*d_ec_drs)/(15*n)
  finf = finf_x + finf_c

  df = finf - f0
  bn = 0._dp
  if (df > 0._dp) then
    bn = ( gam/cc*df )**(4._dp/3._dp)
  end if

end subroutine high_freq


subroutine gki_dynamic_real_freq(rs,freq,nw,param,revised,fxc)

  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)

  real(dp), parameter :: gam = 1.311028777146059809410871821455657482147216796875_dp
  real(dp), parameter :: cc = 4.81710873550434914847073741839267313480377197265625_dp

  real(dp), parameter :: apar = 0.1756_dp,bpar = 1.0376_dp,cpar = 2.9787_dp
  real(dp), parameter :: powr = 7._dp/(2*cpar), aj = 0.63_dp, h0 = 1._dp/gam


  integer, intent(in) :: nw
  real(dp), intent(in) :: rs
  real(dp), dimension(nw), intent(in) :: freq
  character(len=4), intent(in) :: param
  logical, intent(in) :: revised
  complex(dp), dimension(nw), intent(out) :: fxc

  real(dp) :: bn,finf
  real(dp), dimension(nw) :: u,gx,hx


  call high_freq(rs,param,finf,bn)
  u = bn**(0.5_dp)*freq

  gx = u/(1._dp + u**2)**(5._dp/4._dp)

  if (revised) then
    hx = h0*(1._dp - apar*u**2)/( 1._dp + bpar*u**2 + &
  &    (apar*h0)**(1._dp/powr)*u**cpar )**powr
  else
    hx = h0*(1._dp - aj*u**2)/( 1._dp + (h0*aj)**(4._dp/7._dp)*u**2 )**(7._dp/4._dp)
  end if

  fxc = finf - cc*bn**(3._dp/4._dp)*(hx + gx*cmplx(0._dp,1._dp))

end subroutine gki_dynamic_real_freq
