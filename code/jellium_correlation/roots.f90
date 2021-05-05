

subroutine bracket(fun,bds,intervals,nivals)

  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)

  integer, parameter :: nstep = 500

  real(dp), external :: fun
  real(dp), dimension(2), intent(in) :: bds

  integer, intent(out) :: nivals
  real(dp), dimension(10,2),intent(out) :: intervals

  integer :: istep
  real(dp) :: tmp,step,cfun,ofun

  step = (bds(2)-bds(1))/(1._dp*nstep)
  tmp = bds(1)
  nivals = 0
  do istep = 1,nstep
    cfun = fun(tmp)
    if (istep == 1) then
      ofun = cfun
    end if
    if (ofun*cfun <= 0._dp) then
      nivals = nivals + 1
      intervals(nivals,1) = tmp-step
      intervals(nivals,2) = tmp
    end if
    ofun = cfun
    tmp = tmp + step
  end do

end subroutine bracket


subroutine bisect(fun,bds,mid,mtmp)

  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)

  integer,parameter :: maxstep=500
  real(dp),parameter :: tol=1.5d-7

  real(dp), external :: fun
  real(dp), dimension(2), intent(in) :: bds

  real(dp),intent(out) :: mid,mtmp

  integer :: istep
  real(dp) :: ltmp,utmp,lbd,ubd,spc

  ltmp = fun(bds(1))
  utmp = fun(bds(2))
  if (ltmp*utmp > 0._dp) then
    print*,"No root in bracket"
    stop
  end if

  if (ltmp < 0._dp) then
    lbd = bds(1)
    ubd = bds(2)
  else
    lbd = bds(2)
    ubd = bds(1)
  end if

  do istep =1,maxstep
    mid = (lbd + ubd)/2._dp
    mtmp = fun(mid)

    !if ((abs(mid - lbd)<tol*abs(mid)).or.(abs(mtmp)<tol)) then
    if (abs(mtmp)<tol) then
      exit
    end if

    if (mtmp < 0._dp) then
      lbd = mid
    else
      ubd = mid
    end if

  end do

end subroutine bisect
