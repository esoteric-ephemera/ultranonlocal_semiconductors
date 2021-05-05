
subroutine alda(rs,param,fxc)

  ! wrapper routine that selects between different
  ! parameterizations of the ALDA
  ! MCP07 uses PZ81, TC21 and Qian-Vignale kernel use PW92

  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)

  real(dp) :: rs
  character(len=4), intent(in) :: param
  real(dp), intent(out) :: fxc

  if (param == 'PZ81') then
    call pz81_alda(rs,fxc)
  else if (param == 'PW92') then
    call pw92_alda(rs,fxc)
  end if

end subroutine alda

!===========================================================
! PZ81 ALDA:
! From J. P. Perdew and Alex Zunger,
! Phys. Rev. B 23, 5048, 1981
! doi: 10.1103/PhysRevB.23.5048
!===========================================================

subroutine pz81_alda(rs,fxc)

  implicit none

  integer, parameter :: dp = selected_real_kind(15, 307)
  real(dp), parameter :: pi = 3.14159265358979323846264338327950288419_dp

  ! PZ81 parameters
  real(dp), parameter :: au = 0.0311_dp, bu = -0.048_dp, cu = 0.0020_dp
  real(dp), parameter :: du = -0.0116_dp, gu = -0.1423_dp, b1u = 1.0529_dp
  real(dp), parameter :: b2u = 0.3334_dp, gp = -0.0843_dp

  real(dp), intent(in) :: rs
  real(dp), intent(out) :: fxc

  real(dp) :: rsh,n,kf,fx,fc

  rsh = rs**(0.5_dp)
  n = 3._dp/(4*pi*rs**3)
  kf = (9*pi/4._dp)**(1._dp/3._dp)/rs
  fx = -pi/kf**2

  if (rs < 1._dp) then
    fc = -(3*au + 2*cu*rs*log(rs) + (2*du + cu)*rs)/(9*n)
  else
    fc = gu/(36*n)/(1._dp + b1u*rsh + b2u*rs)**3*(5*b1u*rsh &
  &    + (7*b1u**2 + 8*b2u)*rs + 21*b1u*b2u*rsh**3 + (4*b2u*rs)**2 )
  end if

  fxc = fx + fc

end subroutine pz81_alda


!===========================================================
! PW92 ALDA
! From J. P. Perdew and W. Yang
! Phys. Rev. B 45, 13244 (1992).
! doi: 10.1103/PhysRevB.45.13244
!===========================================================

subroutine pw92_alda(rs,fxc)

  implicit none

  integer, parameter :: dp = selected_real_kind(15, 307)
  real(dp), parameter :: pi = 3.14159265358979323846264338327950288419_dp
  real(dp), parameter :: afac = 0.0310906908696549008630505284145328914746642112731933593_dp

  ! PW92 parameters
  real(dp), parameter :: alpha = 0.21370_dp, beta1 = 7.5957_dp
  real(dp), parameter :: beta2 = 3.5876_dp, beta3 = 1.6382_dp, beta4 = 0.49294_dp

  real(dp), intent(in) :: rs
  real(dp), intent(out) :: fxc

  real(dp) :: rsh,n,kf, fx,fc, q, dq, ddq
  real(dp) :: d_ec_d_rs,d_ec_d_rs2

  rsh = rs**(0.5_dp)
  n = 3._dp/(4*pi*rs**3)
  kf = (9*pi/4._dp)**(1._dp/3._dp)/rs
  fx = -pi/kf**2

  q = 2*afac*(beta1*rsh + beta2*rs + beta3*rsh**3 + beta4*rs**2)
  dq = afac*(beta1/rsh + 2*beta2 + 3*beta3*rsh + 4*beta4*rs)
  ddq = afac*(-beta1/2._dp/rsh**3 + 3._dp/2._dp*beta3/rsh + 4*beta4)

  d_ec_d_rs = 2*afac*( -alpha*log(1._dp + 1._dp/q) + (1._dp + alpha*rs)*dq/(q**2 + q) )
  d_ec_d_rs2 = 2*afac/(q**2 + q)*(  2*alpha*dq + (1._dp + alpha*rs) &
&    *( ddq - (2*q + 1._dp)*dq**2/(q**2 + q) )  )

  fc = rs/(9*n)*(rs*d_ec_d_rs2 - 2*d_ec_d_rs)

  fxc = fx + fc

end subroutine pw92_alda



! some helper routines needed for infinite frequency limits
! and other exact constraints

subroutine lda_derivs(rs,param,ec,d_ec_drs)


  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)

  real(dp), intent(in) :: rs
  character(len=4), intent(in) :: param
  real(dp), intent(out) :: ec,d_ec_drs

  if (param == 'PZ81') then
    call pz81_derivs(rs,ec,d_ec_drs)
  else if (param == 'PW92') then
    call pw92_derivs(rs,ec,d_ec_drs)
  end if

end subroutine lda_derivs


subroutine pz81_derivs(rs,ec,d_ec_drs)

  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)

  ! PZ81 parameters
  real(dp), parameter :: au = 0.0311_dp, bu = -0.048_dp, cu = 0.0020_dp
  real(dp), parameter :: du = -0.0116_dp, gu = -0.1423_dp, b1u = 1.0529_dp
  real(dp), parameter :: b2u = 0.3334_dp, gp = -0.0843_dp

  real(dp), intent(in) :: rs
  real(dp), intent(out) :: ec,d_ec_drs

  real(dp) :: rsh

  rsh = rs**(0.5_dp)
  if (rs < 1._dp) then
    ec = au*log(rs) + bu + cu*rs*log(rs) + du*rs
    d_ec_drs = au/rs + cu + cu*log(rs) + du
  else
    ec = gu/(1._dp + b1u*rsh + b2u*rs)
    d_ec_drs = -gu*(0.5*b1u/rsh + b2u)/(1._dp + b1u*rsh + b2u*rs)**2
  end if

end subroutine pz81_derivs


subroutine pw92_derivs(rs,ec,d_ec_drs)

  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)

  ! PW92 parameters
  real(dp), parameter :: afac = 0.0310906908696549008630505284145328914746642112731933593_dp
  real(dp), parameter :: alpha = 0.21370_dp, beta1 = 7.5957_dp
  real(dp), parameter :: beta2 = 3.5876_dp, beta3 = 1.6382_dp, beta4 = 0.49294_dp

  real(dp), intent(in) :: rs
  real(dp), intent(out) :: ec,d_ec_drs

  real(dp) :: rsh,q,dq

  rsh = rs**(0.5_dp)
  q = 2*afac*(beta1*rsh + beta2*rs + beta3*rsh**3 + beta4*rs**2)
  dq = afac*(beta1/rsh + 2*beta2 + 3*beta3*rsh + 4*beta4*rs)

  ec = -2*afac*(1._dp + alpha*rs)*log(1._dp + 1._dp/q)
  d_ec_drs = 2*afac*( -alpha*log(1._dp + 1._dp/q) + &
&     (1._dp + alpha*rs)*dq/(q**2 + q) )

end subroutine pw92_derivs
