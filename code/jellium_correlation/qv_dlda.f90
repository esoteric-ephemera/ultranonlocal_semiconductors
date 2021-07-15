

function s_3_l(kf)

  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)

  real(dp), parameter :: pi = 3.14159265358979323846264338327950288419_dp

  real(dp),intent(in) :: kf
  real(dp) :: s_3_l

  real(dp) :: lam

  lam = (pi*kf)**(0.5_dp)
  s_3_l = -(  5._dp - (lam + 5._dp/lam)*atan(lam) - &
&     2._dp/lam*asin( lam/(1._dp + lam**2)**(0.5_dp) ) &
&    + 2._dp/(lam*(2._dp + lam**2)**(0.5_dp))*(pi/2._dp &
&    - atan( 1._dp/(lam*(2._dp + lam**2)**(0.5_dp))) )  )/(45*pi)

end function s_3_l



function mu_xc_n(rs)

  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)
  real(dp),parameter :: ca = 0.03115158529677855_dp, cb = 0.011985054514894128_dp, cc=2.267455018224077_dp

  real(dp),intent(in) :: rs
  real(dp) :: mu_xc_n

  mu_xc_n  = ca/rs + (cb - ca)*rs/(rs**2 + cc)

end function mu_xc_n

subroutine qv_static(rs,use_mu_xc,f0)

  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)
  real(dp), parameter :: pi = 3.14159265358979323846264338327950288419_dp
  logical,intent(in) :: use_mu_xc
  real(dp),intent(in) :: rs
  real(dp),intent(out) :: f0
  real(dp) :: n,mu_xc_n

  call alda(rs,'PW92',f0)
  n = 3._dp/(4*pi*rs**3)

  if (use_mu_xc) then
    f0 = f0 + 4._dp/3._dp*mu_xc_n(rs)/n
  end if

end subroutine qv_static


subroutine get_qv_pars(rs,use_mu_xc,a3l,b3l,g3l,o3l)

  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)
  real(dp),parameter :: c3l =23._dp/15._dp, pi = 3.14159265358979323846264338327950288419_dp
  ! c3l from just below Eq. 13

  logical, intent(in) :: use_mu_xc
  real(dp), intent(in) :: rs
  real(dp), intent(out) :: a3l,b3l,g3l,o3l

  integer :: nbracks,ibrack
  real(dp), dimension(10,2) :: bracks
  real(dp) :: s3l,kf,n,f0,finf,df,dummy,tg3l,suc
  real(dp) :: s_3_l,mu_xc_n

  kf = (9*pi/4._dp)**(1._dp/3._dp)/rs
  n = 3._dp/(4*pi*rs**3)

  s3l = s_3_l(kf)
  ! eq. 28
  a3l = 2._dp*(2._dp/(3*pi**2))**(1._dp/3._dp)*rs**2*s3l
  ! eq. 29
  b3l = 16._dp*(2._dp**10/(3*pi**8))**(1._dp/15._dp)*rs*(s3l/c3l)**(4._dp/5._dp)

  call high_freq(rs,'PW92',finf,dummy)
  call qv_static(rs,use_mu_xc,f0)
  df = f0 - finf

  g3l = 1.d-14
  call bracket(g3l_res,(/1.d-6,5._dp/),bracks,nbracks)
  if (nbracks > 0) then
    do ibrack = 1,nbracks
      call bisect(g3l_res,bracks(ibrack,:),tg3l,suc)
      if (abs(suc) < 1.5d-7) then
        g3l = max(g3l,tg3l)
      end if
    end do
  end if

  o3l = 1._dp - 1.5_dp*g3l

  contains

    function g3l_res(tmp)

      implicit none
      integer, parameter :: dp = selected_real_kind(15, 307)
      ! [Gamma(1/4)]**2
      real(dp),parameter :: gamma_14_sq = 13.1450472065968728685447786119766533374786376953125_dp

      real(dp),intent(in) :: tmp
      real(dp) :: g3l_res

      real(dp) :: o3l,o3l2,res1,res2

      ! eq. 27
      o3l = 1._dp - 1.5_dp*tmp
      o3l2 = o3l**2
      ! eq. 30
      res1 = 4._dp*(2*pi/b3l)**(0.5_dp)*a3l/gamma_14_sq
      res2 = o3l*tmp*exp(-o3l2/tmp)/pi + 0.5_dp*(tmp/pi)**(0.5_dp)*(tmp + 2*o3l2)&
    &     *(1._dp + erf(o3l/tmp**(0.5_dp)))

    g3l_res = 4._dp*(pi/n)**(0.5_dp)*(res1 + res2) +df

    end function g3l_res

end subroutine get_qv_pars



subroutine im_fxc_qv(omega,rs,nw,ca,cb,cg,co,imfxc)

  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)
  real(dp),parameter :: pi = 3.14159265358979323846264338327950288419_dp

  integer, intent(in) :: nw
  real(dp), dimension(nw), intent(in) :: omega
  real(dp), intent(in) :: rs,ca,cb,cg,co
  real(dp), dimension(nw), intent(out) :: imfxc

  integer :: iw
  real(dp) :: wp0,n,wt

  n = 3._dp/(4*pi*rs**3)
  wp0 = (3._dp/rs**3)**(0.5_dp)

  do iw = 1,nw
    wt = omega(iw)/(2*wp0)
    imfxc(iw) = -omega(iw)/n*( ca/(1._dp + cb*wt**2)**(5._dp/4._dp) &
  &    + wt**2*exp(-(abs(wt)-co)**2/cg) )
  end do

end subroutine im_fxc_qv


subroutine fxc_qv(omega,rs,use_mu_xc,nw,igrid,iwg,ng,fxc)

  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)
  real(dp),parameter :: pi = 3.14159265358979323846264338327950288419_dp

  integer, intent(in) :: nw,ng
  real(dp), dimension(nw), intent(in) :: omega
  real(dp), intent(in) :: rs
  logical, intent(in) :: use_mu_xc
  ! igrid should be a grid from 0 to infinity,
  ! iwg should be the corresponding integration weights
  real(dp), dimension(ng), intent(in) :: igrid,iwg
  complex(dp), dimension(nw), intent(out) :: fxc

  integer :: iw
  real(dp) :: ca,cb,cg,co,finf,dummy,w
  real(dp),dimension(nw) :: imfxc
  real(dp),dimension(ng) :: ugrid,lgrid,imfxc1,imfxc2

  ! first get QV parameters for given rs
  call get_qv_pars(rs,use_mu_xc,ca,cb,cg,co)
  ! then the imaginary part of fxc
  call im_fxc_qv(omega,rs,nw,ca,cb,cg,co,imfxc)
  ! and the infinite frequency limit of fxc
  call high_freq(rs,'PW92',finf,dummy)

  ! construct Re fxc - finf using Kramers-Kronig principal value integral
  do iw = 1,nw

    w = omega(iw)
    lgrid = -igrid + w - 1.d-10
    call im_fxc_qv(lgrid,rs,ng,ca,cb,cg,co,imfxc1)
    ugrid = igrid + w + 1.d-10
    call im_fxc_qv(ugrid,rs,ng,ca,cb,cg,co,imfxc2)

    fxc(iw) = dot_product(iwg,imfxc1/(lgrid - w) + imfxc2/(ugrid - w))

  end do

  fxc = fxc/pi + finf + cmplx(0._dp,1._dp)*imfxc

end subroutine fxc_qv
