program jellium_ec

  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)
  real(dp),parameter :: pi = 3.14159265358979323846264338327950288419_dp

  integer, parameter :: ninf=200, nlam=100
  real(dp), parameter :: rs_min=1._dp,rs_max = 10._dp,drs = 0.1_dp

  integer, parameter :: nrs = ceiling((rs_max-rs_min)/drs)

  real(dp), dimension(0:nrs) :: rsl, ec_rpa, ec_alda, ec_dlda
  real(dp), dimension(0:nrs) :: ec_mcp07_stat,ec_mcp07_k0, ec_mcp07
  real(dp), dimension(0:nrs) :: ec_qv,ec_qv_hyb1,ec_qv_hyb2
  real(dp) :: rs,ca,cb,cg,co
  integer :: irs

  character(len=1000) :: str

  !$OMP PARALLEL DO
  do irs = 0,nrs
    rsl(irs) = rs_min + drs*irs
    call get_eps_c(ninf,nlam,rsl(irs),ec_rpa(irs),ec_alda(irs),ec_dlda(irs),&
      &    ec_mcp07_stat(irs),ec_mcp07_k0(irs),ec_mcp07(irs),ec_qv(irs),&
      &    ec_qv_hyb1(irs),ec_qv_hyb2(irs))
  end do
  !$OMP END PARALLEL DO

  open(unit=2,file='jell_eps_c.csv')
  write(2,'(a)') 'rs, RPA, ALDA, Dyn. LDA, MCP07 static, MCP07 k=0, MCP07, QV, &
  & QV-MCP07 TD, QV-MCP07 TDC'
  ! unfortunately the second do loop is needed because the multithreading removes
  ! the ordering of the rs values
  do irs = 0,nrs
    write(str,*) rsl(irs),',',ec_rpa(irs),',',ec_alda(irs),',',ec_dlda(irs),',',&
    &    ec_mcp07_stat(irs),',',ec_mcp07_k0(irs),',',ec_mcp07(irs),',',ec_qv(irs),&
    &    ',',ec_qv_hyb1(irs),',', ec_qv_hyb2(irs)
    write(2,'(a)') trim(adjustl(str))
  end do

  close(2)

end program jellium_ec
