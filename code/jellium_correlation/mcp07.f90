
subroutine mcp07_static(q,rs,param,fxc,f0,akn)

  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)

  real(dp), parameter :: pi = 3.14159265358979323846264338327950288419_dp

  real(dp), intent(in) :: q,rs
  character(len=4), intent(in) :: param
  real(dp), intent(out) :: fxc,f0,akn

  real(dp) :: rsh,kf,bn,ec,d_ec_drs, d_rs_ec_drs
  real(dp) :: cn,cxc,dn, vc, zp,cfac,n

  rsh = rs**(0.5_dp)
  bn = (1._dp + 2.15_dp*rsh + 0.435_dp*rsh**3)/(3._dp + 1.57_dp*rsh + 0.409_dp*rsh**3)

  call alda(rs,param,f0)
  akn = -f0/(4*pi*bn)

  call lda_derivs(rs,param,ec,d_ec_drs)
  d_rs_ec_drs = ec + rs*d_ec_drs
  kf = (9*pi/4._dp)**(1._dp/3._dp)/rs
  cn = -pi/(2*kf)*d_rs_ec_drs

  n = 3._dp/(4*pi*rs**3)
  cxc = -0.00238_dp + 0.00423_dp*(1._dp + 3.138_dp*rs + 0.3_dp*rs**2)/(1._dp + 3._dp*rs + 0.5334_dp*rs**2)
  dn = 2._dp*cxc/(n**(4._dp/3._dp)*4*pi*bn) - 0.5_dp*akn**2

  vc = 4*pi/q**2
  zp = akn*q**2
  cfac = 4*pi/kf**2
  fxc = vc*bn*(exp(-zp)*(1._dp + dn*q**4) - 1._dp) - cfac*cn/(1._dp + 1._dp/zp**2)

end subroutine mcp07_static


subroutine mcp07_dynamic(q,freq,rs,nw,fxc)

  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)

  integer, intent(in) :: nw
  real(dp), intent(in) :: q,rs
  real(dp), dimension(nw), intent(in) :: freq
  complex(dp), dimension(nw), intent(out) :: fxc

  real(dp) :: fxcq,f0,akn
  complex(dp), dimension(nw) :: fxcw

  call mcp07_static(q,rs,'PZ81',fxcq,f0,akn)
  call gki_dynamic_real_freq(rs,freq,nw,'PZ81',.false.,fxcw)

  fxc = (1._dp + exp(-akn*q**2)*(fxcw/f0 - 1._dp))*fxcq

end subroutine mcp07_dynamic

subroutine mcp07_k0(q,freq,rs,nw,fxc)

  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)

  integer, intent(in) :: nw
  real(dp), intent(in) :: q,rs
  real(dp), dimension(nw), intent(in) :: freq
  complex(dp), dimension(nw), intent(out) :: fxc

  real(dp) :: fxcq,f0,akn
  complex(dp), dimension(nw) :: fxcw

  call mcp07_static(q,rs,'PZ81',fxcq,f0,akn)
  call gki_dynamic_real_freq(rs,freq,nw,'PZ81',.false.,fxcw)

  fxc = fxcw/f0*fxcq

end subroutine mcp07_k0
