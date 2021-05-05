program jellium_ec

  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)
  real(dp),parameter :: pi = 3.14159265358979323846264338327950288419_dp

  integer, parameter :: ninf=200, nlam=100
  real(dp), parameter :: rs_min=1._dp,rs_max = 10._dp,drs = 0.1_dp

  real(dp) :: ec_rpa, ec_alda, ec_dlda, ec_mcp07_stat,ec_mcp07_k0, ec_mcp07
  complex(dp) :: ec_qv,ec_qv_hyb1,ec_qv_hyb2
  real(dp) :: rs
  integer :: irs,nrs

  character(len=1000) :: str

  nrs = ceiling((rs_max-rs_min)/drs)

  open(unit=2,file='jell_eps_c.csv')
  write(2,'(a)') 'rs, RPA, ALDA, Dyn. LDA, MCP07 static, MCP07 k=0, MCP07, Re QV, Im QV,&
  & Re QV-MCP07 TD, Im QV-MCP07 TD, Re QV-MCP07 TDC, Im QV-MCP07 TDC'

  do irs = 0,nrs

    rs = rs_min + drs*irs
    call get_eps_c(ninf,nlam,rs,ec_rpa,ec_alda,ec_dlda,&
      &    ec_mcp07_stat,ec_mcp07_k0,ec_mcp07,ec_qv,ec_qv_hyb1,ec_qv_hyb2)

    write(str,*) rs,',',ec_rpa,',',ec_alda,',',ec_dlda,',',ec_mcp07_stat,&
    &    ',',ec_mcp07_k0,',',ec_mcp07,',',real(ec_qv),',',aimag(ec_qv),',',&
    &    real(ec_qv_hyb1),',',aimag(ec_qv_hyb1),',',real(ec_qv_hyb2),',',&
    &    aimag(ec_qv_hyb2)
    write(2,'(a)') trim(adjustl(str))
    
  end do

  close(2)

end program jellium_ec
