
subroutine chi_ks_ifreq(q,u,rs,nw,chi0)

  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)
  real(dp), parameter :: pi = 3.14159265358979323846264338327950288419_dp

  integer, intent(in) :: nw
  real(dp), intent(in) :: q,rs
  real(dp), dimension(nw), intent(in) :: u
  real(dp), dimension(nw), intent(out) :: chi0

  real(dp) :: kf,qq
  real(dp), dimension(nw) :: ut,ut2

  kf = (9*pi/4._dp)**(1._dp/3._dp)/rs
  qq = q/(2*kf)
  ut = u/(q*kf)
  ut2 = ut*ut
  chi0 = (  (qq**2 - ut2 - 1._dp)/(4*qq)*log((ut2 + (qq + 1._dp)**2)/ &
&     (ut2 + (qq - 1._dp)**2)) - 1._dp + ut*atan( (1._dp + qq)/ut ) &
&     + ut*atan( (1._dp - qq)/ut)  )*kf/(2*pi**2)

end subroutine chi_ks_ifreq



subroutine fxc_selector(q,w,rs,nw,wfxc,fxc)

  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)

  integer, intent(in) :: nw
  real(dp) :: q,rs
  real(dp), dimension(nw), intent(in) :: w
  character(len=10) :: wfxc
  complex(dp), dimension(nw), intent(out) :: fxc
  real(dp) :: fxcr

  if (wfxc(1:5)=='MCP07') then
    call mcp07_dynamic(q,w,rs,nw,fxc)

  else if (wfxc(1:7)=='MCP07k0') then
    call mcp07_k0(q,w,rs,nw,fxc)

  else if (wfxc(1:4)=='DLDA') then
    call gki_dynamic_real_freq(rs,w,nw,'PZ81',.false.,fxc)

  else if (wfxc(1:4)=='ALDA') then
    call alda(rs,'PZ81',fxcr)
    fxc = fxcr

  else if (wfxc(1:3)=='RPA') then
    fxc = 0._dp

  end if

end subroutine fxc_selector


subroutine fxc_gki_ifreq(w,rs,nw,digrid,diwg,ng,ifxc)

  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)
  real(dp), parameter :: pi = 3.14159265358979323846264338327950288419_dp

  integer, intent(in) :: nw,ng
  real(dp), intent(in) :: rs
  real(dp), dimension(nw), intent(in) :: w
  real(dp), dimension(ng), intent(in) :: digrid,diwg
  real(dp), dimension(nw), intent(out) :: ifxc

  integer :: iw
  real(dp) :: finf,tmp
  real(dp), dimension(ng) :: integrand,denom
  complex(dp), dimension(ng) :: fxc_tmp

  call high_freq(rs,'PZ81',finf,tmp)

  call gki_dynamic_real_freq(rs,digrid,ng,'PZ81',.false.,fxc_tmp)

  do iw = 1,nw
    denom = digrid**2 + w(iw)**2
    integrand = ( w(iw)*(real(fxc_tmp) - finf) + digrid*aimag(fxc_tmp) )
  !  integrand = integrand + ( -digrid*(real(fxc_tmp) - finf)&
  !&    + w(iw)*aimag(fxc_tmp) )*cmplx(0._dp,1._dp)

    ifxc(iw) = dot_product(diwg, integrand/denom)

  end do

  ifxc = ifxc/(2*pi) + finf

end subroutine fxc_gki_ifreq


subroutine fxc_qv_ifreq(w,rs,use_mu_xc,nw,digrid,diwg,ng,igrid,iwg,ni,ifxc)

  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)
  real(dp), parameter :: pi = 3.14159265358979323846264338327950288419_dp

  integer, intent(in) :: nw,ng,ni
  real(dp),intent(in) :: rs
  logical, intent(in) :: use_mu_xc
  real(dp), dimension(nw), intent(in) :: w
  real(dp), dimension(ng), intent(in) :: digrid,diwg
  real(dp), dimension(ni), intent(in) :: igrid,iwg
  real(dp), dimension(nw), intent(out) :: ifxc

  integer :: iw
  real(dp) :: finf,tmp
  real(dp), dimension(ng) :: denom
  complex(dp), dimension(ng) :: integrand,fxc_tmp

  call high_freq(rs,'PW92',finf,tmp)

  call fxc_qv(digrid,rs,use_mu_xc,ng,igrid,iwg,ni,fxc_tmp)

  do iw = 1,nw
    denom = digrid**2 + w(iw)**2
    integrand = ( w(iw)*(real(fxc_tmp) - finf) + digrid*aimag(fxc_tmp) )
    !integrand = integrand + ( -digrid*(real(fxc_tmp) - finf)&
  !&    + w(iw)*aimag(fxc_tmp) )*cmplx(0._dp,1._dp)

    ifxc(iw) = dot_product(diwg, integrand/denom)
  end do

  ifxc = ifxc/(2*pi) + finf

end subroutine fxc_qv_ifreq
