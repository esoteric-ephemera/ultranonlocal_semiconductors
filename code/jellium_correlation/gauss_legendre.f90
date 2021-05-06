

subroutine gauss_legendre(npts,grid,wg)

  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)

  integer, intent(in) :: npts
  real(dp), dimension(npts), intent(out) :: grid,wg

  integer :: i
  real(dp), dimension(npts,npts) :: eigvs
  real(dp), dimension(npts-1) :: udiag
  real(dp), dimension((max(1,2*npts-2))) :: twork

  grid = 0._dp
  do i=1,npts-1
    udiag(i) = gl_udiag(i-1)
  end do
  call dsteqr('I',npts,grid,udiag,eigvs,npts,twork,i)

  if (i /= 0) then
    print*,'WARNING, call to DSTEQR failed with INFO = ',i
  end if

  wg = 2._dp*eigvs(1,:)**2

  contains

    function gl_udiag(n)
      implicit none
      integer, parameter :: dp = selected_real_kind(15, 307)

      integer, intent(in) :: n
      real(dp) :: gl_udiag,an,anp1

      an = (2*n + 1._dp)/(n + 1._dp)
      anp1 = (2*n + 3._dp)/(n + 2._dp)
      gl_udiag = ( (n + 1._dp)/(n + 2._dp)/(an*anp1) )**(0.5_dp)

    end function gl_udiag


end subroutine gauss_legendre
