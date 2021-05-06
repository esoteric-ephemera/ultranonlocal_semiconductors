
program test_gl

  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)

  integer, parameter :: gsize = 200
  integer :: i
  real(dp), dimension(gsize) :: grid,wg

  call gauss_legendre(gsize,grid,wg)

  do i = 1,gsize
    print*,grid(i),wg(i)
  end do

  print*,'======================'
  print*,'Integral of x^2 from -1 < x < 1:'
  print*,dot_product(wg,grid**2)

end program test_gl
