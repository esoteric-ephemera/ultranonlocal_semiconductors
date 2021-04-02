       Program Ultranonlocality
       implicit double precision (a-h,o-z)
       complex(8):: fxcqu,fxcu,ffreq,alphag
       dimension rlvbas(3,3)
        parameter( pi=3.1415926535897932384626433832795d0)
c       m = 1
c       hbar = 1
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       rs = 3.93d0
       dens = 3.d0/(4.d0*pi*rs**3.d0)
       akf = ((9.d0*pi/4.d0)**(1.d0/3.d0))/rs
       aa = 4.91d0
       bb = 3.87d0

c Valence number of Na
         z = 1.d0
         Rp = 0.492d0

c Construct the bcc lattice


         al = 7.982d0
         con = 2.d0*pi/al
         rlvbas(1,1) = 0.d0
         rlvbas(1,2) = con
         rlvbas(1,3) = con
         rlvbas(2,1) = con
         rlvbas(2,2) = 0.d0
         rlvbas(2,3) = con
         rlvbas(3,1) = con
         rlvbas(3,2) = con
         rlvbas(3,3) = 0.d0

         gmin = 100.d0
         gmax = -100.d0

         dur = 0.1d0
         do jur = 1, 101
	       ur = (jur-1)*dur

         alphag = cmplx(0.d0, 0.d0)
         n = 10.d0
         m = 2.d0*n + 1

         do ik1 = 1, m
          ik1p = -n + (ik1 - 1)
          do  ik2 = 1, m
            ik2p = -n + (ik2 - 1)
            do ik3 = 1, m
              ik3p = -n + (ik3 - 1)

           Gvx=ik1p*rlvbas(1,1)+ik2p*rlvbas(2,1)+ik3p*rlvbas(3,1)
           Gvy=ik1p*rlvbas(1,2)+ik2p*rlvbas(2,2)+ik3p*rlvbas(3,2)
           Gvz=ik1p*rlvbas(1,3)+ik2p*rlvbas(2,3)+ik3p*rlvbas(3,3)
           Gv = dsqrt(Gvx**2 + Gvy**2 + Gvz**2)

c          fG = Fyf(akf,Gv,z1,z2)
c Here we evaluate the n(G) from perturbation theory using the dielectric function with an ALDA kernel
c Exclude G = 0
          y = Gv/(2.d0*akf)
          !if (y.lt.0.1d0) go to 783
          egrlv = 0.5d0*Gv**2
        !  if (egrlv.gt.20.d0) go to 783
        if ((Gv.gt.0.0d0).and.(egrlv.le.20.d0)) then
          gmin = min(gmin,gv)
          gmax = max(gmax,gv)
          wGv = Pseudo(Gv,z,Rp)
          chi0 = Dflind(akf,y)
          fxch = ALDAxc(akf,rs)
          vc = 4.d0*pi/(Gv**2)
          eps = 1.d0 -(vc + fxch)*chi0
          anG = (dens/z)*(chi0/eps)*wGv

          Call MCP07(Gv,akf,rs,fxcmcp07,f0,akn)
c          Call CP070(Gv,akf,rs,fxccp070)
c          Call CP07(Gv,ur,akf,rs,fxccp07)
c if CP07 then fxcqu = fxccp07
c           fxcqu = fxccp07
c           fxcmcp07 = fxccp070
          Call Frequency(rs,ur,fxcu)
c          Call Frequency_mod(rs,ur,fxcu)
c In the frequency-dependent versions we need fxcq = fxcmcp07/f0
          fxcq = fxcmcp07/f0
c          aknmod = akf*aa/(1.d0 + bb*sqrt(akf))
c           df = dexp(-akn*Gv**2)
c           df = 1.d0
          fxcqu = (f0 + (dexp(-akn*Gv**2))*(fxcu - f0))*fxcq
c           fxcqu = (f0 + df*(fxcu - f0))*fxcq
c          fxcqu = (f0 + (dexp(-(Gv/aknmod)**2))*(fxcu - f0))*fxcq
c for the dynamic LDA kernel we use fxcu and f0
c          ffreq = (fxcu - f0)*anG**2
          ffreq = (fxcqu - fxcmcp07)*anG**2
          alphag = alphag + (((1.d0/3.d0)*Gv**2)/(dens**2))*ffreq
        end if
783        enddo
         enddo
        enddo
        print*,gmin,gmax
        stop

100     write(*, 3) ur,alphag
3       format(3f20.5)

       enddo

       stop
       end
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        Double Precision Function Pseudo(Q,z,Rp)
         implicit double precision (a-h,o-z)
         parameter( pi=3.1415926535897932384626433832795d0)

! Evaluates the expression of 2.11 of Phys. Rev. B 51, 14001 (1995)
! w(Q) is the perturbation potential

         alpha = 3.517d0
         beta = (alpha**3 -2.d0*alpha)/(4.d0*(alpha**2 -1.d0))
         aa = 0.5d0*alpha**2 - alpha*beta

        sfac = 4.d0*pi*z*Rp**2
        denf = (Q*Rp)**2
        denss = (denf + alpha**2)
        den = (denf + 1.d0)**2
        wQ1 = -1.d0/denf +1.d0/denss + 2.d0*alpha*beta/(denss**2)
        wQ2 = 2.d0*aa/den
        wQ = sfac*(wQ1 + wQ2)
        Pseudo = wQ

        return
        end
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
         Double Precision Function Fyf(akf,Gv,z1,z2)
         implicit double precision (a-h,o-z)
         parameter( pi=3.1415926535897932384626433832795d0)

        yf = 2.d0*akf/Gv
        z1 = 1.d0 + yf
        z2 = 1.d0 - yf
        Fy1 = ((z1**2/2.d0)*dlog(z1) - z1*dlog(z1)-(z1**2/4.d0)+z1)
        Fy2 = ((z2**2/2.d0)*dlog(z2) - z2*dlog(z2)-(z2**2/4.d0)+z2)
        Fyf = Fy1 - Fy2
        return
        end
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        Double Precision Function Dflind(akf,y)
         implicit double precision (a-h,o-z)
         parameter( pi=3.1415926535897932384626433832795d0)

         Fy1 = (1.d0 - y**2)/(4.d0*y)
         xx = (1.d0 + y)/(1.d0 - y)
         zz = dabs(xx)
         Fy2 = dlog(zz)
         Fy = 0.5d0 + Fy1*Fy2
         chi0 = -(akf/pi**2)*Fy
         Dflind = chi0
         return
         end
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
         Subroutine MCP07(sq,akf,rs,fxcmcp07,f0,akn)
        implicit double precision (a-h,o-z)
         parameter( pi = 3.1415926535897932384626433832795d0)

        dens = 3.d0/(4.d0*pi*rs**3.d0)
        b = (3.d0/(4.d0*pi))**(1.d0/3.d0)
        cfac = 4.d0*pi/(akf**2)
        xx = dsqrt(rs)
c bn according to the parametrization of Eq. (7) of Phys. Rev. B 57, 14569 (1998)
         en =1.d0 + 2.15d0*xx + 0.435d0*xx**3.d0
         den = 3.d0 + 1.57d0*xx + 0.409d0*xx**3.d0
         bn = en/den
         f0 = ALDAxc(akf,rs)
         akn = -(f0/(4.d0*pi))/bn
c The uniform electron gas correlation energy according to Perdew and Zunger, Phys. Rev. B, 23, 5076 1981
c At first for rs>1
         gu =-0.1423d0
         b1u = 1.0529d0
         b2u = 0.3334d0
         gp = -0.0843d0
         b1p = 1.3981d0
         b2p = 0.2611d0
         ecu = gu/(1.d0 + b1u*xx + b2u*rs)
         ecp = gp/(1.d0 + b1p*xx + b2p*rs)
c The rs-dependent cn
         rsuff = ((1.d0/2.d0)*b1u*(rs**(-1.d0/2.d0))+b2u)
         rspff = ((1.d0/2.d0)*b1p*(rs**(-1.d0/2.d0))+b2p)
         rsfu = gu/(1.d0 + b1u*dsqrt(rs) + b2u*rs)
         gurs = gu*rs
         rssu = ((1.d0 + b1u*dsqrt(rs) + b2u*rs)**2.d0)
         rsecu = rsfu - gurs*rsuff/rssu
         rsfp = gp/(1.d0 + b1p*dsqrt(rs) + b2p*rs)
         gprs = gp*rs
         rssp = ((1.d0 + b1p*dsqrt(rs) + b2p*rs)**2.d0)
         rsecp = rsfp - gprs*rspff/rssp
c At second, for rs<1 (currently for unpolarized electron gas only)
         au = 0.0311d0
         bu = -0.048d0
         cu = 0.0020d0
         du = -0.0116d0
         rsecuh =au+bu+au*dlog(rs)+2.d0*cu*rs*dlog(rs)+cu*rs+2.d0*du*rs
         if (rs.ge.1.d0) cn = (-pi/(2.d0*akf))*rsecu
         if (rs.lt.1.d0) cn = (-pi/(2.d0*akf))*rsecuh
c The gradient term
         cxcn = 1.d0 + 3.138d0*rs + 0.3d0*rs**2.d0
         cxcd = 1.d0 + 3.d0*rs + 0.5334d0*rs**2.d0
         cxc = -0.00238d0 + 0.00423d0*(cxcn/cxcd)
         dd =(2.d0*cxc/(dens**(4.d0/3.d0)*(4.d0*pi*bn)))-0.5d0*akn**2.d0
c The MCP07 kernel
         vc = 4.d0*pi/sq**2
         cl = vc*bn
         zp = akn*(sq**2)
         grad = (1.d0 + dd*sq**4.d0)
         cutdown = 1.d0+1.d0/(akn*sq**2)**2
        fxcmcp07=cl*(dexp(-zp)*grad-1.d0)-cfac*cn/cutdown
c In this version we use the static-only component of MCP07, and evaluate the frequency-dependence in te main program
c         fxcq = fxcmcp07/f0

         return
         end
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        Subroutine CP070(sq,akf,rs,fxccp070)
        implicit double precision (a-h,o-z)
         parameter( pi = 3.1415926535897932384626433832795d0)

        dens = 3.d0/(4.d0*pi*rs**3.d0)
        cfac = 4.d0*pi/(akf**2)
        dpf = 3.d0/(4.d0*pi)
        dp = (1.d0/rs**3)
c The complex frequency in CP07
c         wp = dsqrt(3.d0/rs**3)
         del = 0.000001d0
         del = 0.d0
         u = 0.d0
         om = u
         omc = u + del
c The compressibility sum-rule
         alph = -0.025518491d0
         b = -0.691590707d0
         bet = -0.691590707d0
         cf = 4.d0*pi
         fxcl = cf*alph*b*(dens)**bet
c The high-frequency limit
         gam1 = -0.114548231d0
         gam2 = -0.614523371d0
         fxch = gam1*(dpf*dp)**gam2
c cnf and an
         cnf = fxch/fxcl
         an = 6.d0*dsqrt(cnf)
         xx = dsqrt(rs)
c bn according to the parametrization of Eq. (7) of Phys. Rev. B 57, 14569 (1998)
         en =1.d0 + 2.15d0*xx + 0.435d0*xx**3.d0
         den = 3.d0 + 1.57d0*xx + 0.409d0*xx**3.d0
         bn = en/den
         cc = -pi/2.d0*akf
c The uniform electron gas correlation energy according to Perdew and Zunger, Phys. Rev. B, 23, 5076 (1981)
         gu =-0.1423d0
         b1u = 1.0529d0
         b2u = 0.3334d0
         cu = 0.0020d0
         du = -0.0116d0
         gp = -0.0843d0
         b1p = 1.3981d0
         b2p = 0.2611d0
         cp =  0.0007d0
         dp = -0.0048d0
         ecu = gu/(1.d0 + b1u*dsqrt(rs) + b2u*rs)
         ecp = gp/(1.d0 + b1p*dsqrt(rs) + b2p*rs)
c         enfsig1 = (1.d0 + sig)**(4.d0/3.d0)
c         enfsig2 =(1.d0 - sig)**(4.d0/3.d0)-2.d0
c         fsig = (2.d0**(4.d0/3.d0))-2.d0
c         ec = ecu + fsig*(ecp - ecu)
c        rsec = rs*ec
c The rs-dependent cn
         rsuff = ((1.d0/2.d0)*b1u*(rs**(-1.d0/2.d0))+b2u)
         rspff = ((1.d0/2.d0)*b1p*(rs**(-1.d0/2.d0))+b2p)
         rsfu = gu/(1.d0 + b1u*dsqrt(rs) + b2u*rs)
         gurs = gu*rs
         rssu = ((1.d0 + b1u*dsqrt(rs) + b2u*rs)**2.d0)
         rsecu = rsfu - gurs*rsuff/rssu
         rsfp = gp/(1.d0 + b1p*dsqrt(rs) + b2p*rs)
         gprs = gp*rs
         rssp = ((1.d0 + b1p*dsqrt(rs) + b2p*rs)**2.d0)
         rsecp = rsfp - gprs*rspff/rssp
         cn = (-pi/(2.d0*akf))*rsecu
c The frequency-dependent kn wavevector
         aknn = (1.d0 + an*(-om) + cnf*(-(om**2)))
         aknd = (1.d0 - (omc**2))
         akn = -(fxcl/(cf*bn))*(aknn/aknd)
c The CP07 kernel
         vc = 4.d0*pi/sq**2
         cl = vc*bn
         zp = akn*(sq**2)
         fxccp070 = cl*(dexp(-zp)- 1.d0)-cfac*cn/(1.d0 + 1.d0/(sq**2))
         return
         end
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        Subroutine CP07(sq,u,akf,rs,fxccp07)
        implicit double precision (a-h,o-z)
         parameter( pi = 3.1415926535897932384626433832795d0)

        dens = 3.d0/(4.d0*pi*rs**3.d0)
        cfac = 4.d0*pi/(akf**2)
        dpf = 3.d0/(4.d0*pi)
        dp = (1.d0/rs**3)
c The complex frequency in CP07
c         wp = dsqrt(3.d0/rs**3)
         del = 0.000001d0
c         u = wp*u
         om = u
         omc = u + del
c The compressibility sum-rule
         alph = -0.025518491d0
         b = -0.691590707d0
         bet = -0.691590707d0
         cf = 4.d0*pi
         fxcl = cf*alph*b*(dens)**bet
c The high-frequency limit
         gam1 = -0.114548231d0
         gam2 = -0.614523371d0
         fxch = gam1*(dpf*dp)**gam2
c cnf and an
         cnf = fxch/fxcl
         an = 6.d0*dsqrt(cnf)
         xx = dsqrt(rs)
c bn according to the parametrization of Eq. (7) of Phys. Rev. B 57, 14569 (1998)
         en =1.d0 + 2.15d0*xx + 0.435d0*xx**3.d0
         den = 3.d0 + 1.57d0*xx + 0.409d0*xx**3.d0
         bn = en/den
         cc = -pi/2.d0*akf
c The uniform electron gas correlation energy according to Perdew and Zunger, Phys. Rev. B, 23, 5076 (1981)
         gu =-0.1423d0
         b1u = 1.0529d0
         b2u = 0.3334d0
         cu = 0.0020d0
         du = -0.0116d0
         gp = -0.0843d0
         b1p = 1.3981d0
         b2p = 0.2611d0
         cp =  0.0007d0
         dp = -0.0048d0
         ecu = gu/(1.d0 + b1u*dsqrt(rs) + b2u*rs)
         ecp = gp/(1.d0 + b1p*dsqrt(rs) + b2p*rs)
c         enfsig1 = (1.d0 + sig)**(4.d0/3.d0)
c         enfsig2 =(1.d0 - sig)**(4.d0/3.d0)-2.d0
c         fsig = (2.d0**(4.d0/3.d0))-2.d0
c         ec = ecu + fsig*(ecp - ecu)
c        rsec = rs*ec
c The rs-dependent cn
         rsuff = ((1.d0/2.d0)*b1u*(rs**(-1.d0/2.d0))+b2u)
         rspff = ((1.d0/2.d0)*b1p*(rs**(-1.d0/2.d0))+b2p)
         rsfu = gu/(1.d0 + b1u*dsqrt(rs) + b2u*rs)
         gurs = gu*rs
         rssu = ((1.d0 + b1u*dsqrt(rs) + b2u*rs)**2.d0)
         rsecu = rsfu - gurs*rsuff/rssu
         rsfp = gp/(1.d0 + b1p*dsqrt(rs) + b2p*rs)
         gprs = gp*rs
         rssp = ((1.d0 + b1p*dsqrt(rs) + b2p*rs)**2.d0)
         rsecp = rsfp - gprs*rspff/rssp
         cn = (-pi/(2.d0*akf))*rsecu
c The frequency-dependent kn wavevector
         aknn = (1.d0 + an*(-om) + cnf*(-(om**2)))
         aknd = (1.d0 - (omc**2))
         akn = -(fxcl/(cf*bn))*(aknn/aknd)
c The CP07 kernel
         vc = 4.d0*pi/sq**2
         cl = vc*bn
         zp = akn*(sq**2)
         fxccp07 = cl*(dexp(-zp)- 1.d0)-cfac*cn/(1.d0 + 1.d0/(sq**2))
         return
         end
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
         Subroutine Frequency(rs,u,fxcu)
         implicit double precision (a-h,o-z)
         complex(8):: fxcu, xk, hn, hd, hxf
         parameter( pi = 3.1415926535897932384626433832795d0)

c The exact constraints are the low and high-frequency limits
         Call Exact_Constraints(rs,cc,bn,f0,finf)

         sqrtbn = dsqrt(bn)

c The analytic hx frequency-dependent core kernel with the optimized parameter aj=0.63 determined in Analytic_Frequency.f

           aj = 0.63d0
              xk = sqrtbn*u
             h0 = 1.d0/1.311d0
             h0p = h0**(4.d0/7.d0)
             hn = (1.d0 - aj*xk**2.d0)
             hd =(1.d0+(aj**(4.d0/7.d0))*h0p*(xk**2.d0))**(7.d0/4.d0)
             hxf = hn/hd
             hx = h0*hxf
             refxc = finf - cc*(bn**(3.d0/4.d0))*hx
           aimfxc=-cc*bn**(3.d0/4.d0)*xk/((1.d0+xk**2.d0)**(5.d0/4.d0))
             fxcu = cmplx(refxc,aimfxc)

          return
          end
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
         Subroutine Frequency_mod(rs,u,fxcu)
         implicit double precision (a-h,o-z)
         complex(8):: fxcu, xk, hn, hd, hxf
         parameter( pi = 3.1415926535897932384626433832795d0)

c The exact constraints are the low and high-frequency limits
         Call Exact_Constraints(rs,cc,bn,f0,finf)

         sqrtbn = dsqrt(bn)

c The analytic hx frequency-dependent core kernel with the optimized parameter aj=0.63 determined in Analytic_Frequency.f

           aj = 0.1756d0
           bj = 1.0376d0
           c = 3.d0
           d = 7.d0/(2.d0*c)
           gam = 1.311d0
              xk = sqrtbn*u
             h0 = 1.d0/1.311d0
             h0p = h0**(4.d0/7.d0)
             hn = (1.d0 - aj*xk**2.d0)
             hd =(1.d0+ bj*xk**2+ ((aj/gam)**(1.d0/d))*(xk**c))**(d)
             hxf = (1.d0/gam)*hn/hd
             hx = h0*hxf
             refxc = finf - cc*(bn**(3.d0/4.d0))*hx
           aimfxc=-cc*bn**(3.d0/4.d0)*xk/((1.d0+xk**2.d0)**(5.d0/4.d0))
             fxcu = cmplx(refxc,aimfxc)

          return
          end
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       Subroutine Exact_Constraints(rs,cc,bn,f0,finf)
        implicit double precision (a-h,o-z)
        parameter( pi = 3.1415926535897932384626433832795d0)

         b = (3.d0/(4.d0*pi))**(1.d0/3.d0)
         akf = 1.91916d0/rs
         dens = 3.d0/(4.d0*pi*rs**3.d0)
c        sig = 0.d0
         ctil = -(3.d0/(4.d0*pi))*(3.d0*pi**2.d0)**(1.d0/3.d0)
c The compressibility sum rule
         f0 = ALDAxc(akf,rs)
c        f0 = (4.d0*ctil/9.d0)/(dens**(2.d0/3.d0))
c The high-frequency limit with the exact constraints (currently for spin-unpolarized version only)
c rs>1
         gu =-0.1423d0
         b1u = 1.0529d0
         b2u = 0.3334d0
         ffac1 = -(2.d0/3.d0)*dens**(-5.d0/3.d0)*gu
         df1= 1.d0 + (b1u*dsqrt(b))/(dens**(1.d0/6.d0))
         df2 = (b2u*(b/dens**(1.d0/3.d0)))
         d1 = (df1 + df2)**(-1.d0)
         dd11 =(-b1u*dsqrt(b)*(1.d0/6.d0)*(dens**(-7.d0/6.d0)))
         dd12 = (-b2u*b*(1.d0/3.d0)*(dens**(-4.d0/3.d0)))
         dd1 = dd11 + dd12
         ffac2 = gu*(dens**(-2.d0/3.d0))
         dd2 = (df1 + df2)**2.d0
         ft = ffac1*d1 - ffac2*dd1/dd2
         ffac3 = -(1.d0/3.d0)*dens**(-4.d0/3.d0)*gu
         ffac4 = gu*dens**(-1.d0/3.d0)
         st = ffac3*d1 - ffac4*dd1/dd2
         fxch1 = -(4.d0/5.d0)*(dens**(2.d0/3.d0))*ft
         fxch2 = 6.d0*(dens**(1.d0/3.d0))*st
         fxh = (4.d0*ctil/15.d0)/(dens**(2.d0/3.d0))
         finfg = fxh + fxch1 + fxch2
c         finf = fxh
c rs<1
          au = 0.0311d0
          bu =-0.048d0
          cu = 0.0020d0
          du = -0.0116d0
          threepi3 = (3.d0/pi)**(1.d0/3.d0)
          three23 = (3.d0)**(2.d0/3.d0)
          pi3 = pi**(1.d0/3.d0)
          six3 = (6.d0)**(1.d0/3.d0)
          two23 = (2.d0)**(2.d0/3.d0)
          dens3 = dens**(1.d0/3.d0)
          dens23 = dens**(2.d0/3.d0)
          dens43 = dens**(4.d0/3.d0)
          dens53 = dens**(5.d0/3.d0)
          tx1 = threepi3/(4.d0*dens43)
          tx2 = 0.d0
          ala = dlog(threepi3/(two23*dens3))
          alb = dlog(4.d0*dens*pi/3.d0)
          tc1=-six3*cu-3.d0*six3*du-2.d0*au*dens3*pi3-4.d0*bu*dens3*pi3
          tc1= tc1-4.d0*au*dens3*pi3*ala+six3*cu*alb
          tc1 = tc1/(6.d0*(dens**2)*pi3)
          tc2 = -six3*cu-2.d0*(six3*du+(au+bu)*dens3*pi3)
          tc2 = tc2 -2.d0*(six3*cu+au*dens3*pi3)*ala
          tc2 = tc2/(6.d0*dens53*pi3)
          finfh = -(4.d0/5.d0)*dens23*(tx1+tc1)+6.d0*dens3*(tx2+tc2)
          if (rs.ge.1.d0) finf = finfg
          if (rs.lt.1.d0) finf = finfh
c         write(*,2) rs, f0, finf
c 2        format(3f12.6, //)
c Gamma(1/4)
         gam = ((3.6256d0)**2.d0)/((32.d0*pi)**(1.d0/2.d0))
         cc = 23.d0*pi/15.d0
         bfac = (gam/cc)**(4.d0/3.d0)
         deltaf = finf - f0
         if (deltaf.lt.1.d-9) deltaf = 1.d-9
         bn = bfac*deltaf**(4.d0/3.d0)

         return
         end
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
         Double Precision Function ALDAxc(akf,rs)
         implicit double precision (a-h,o-z)
         parameter( pi = 3.1415926535897932384626433832795d0)

         dens = 3.d0/(4.d0*pi*rs**3.d0)
         aac = 1.d0/4.d0
         akc = akf**2/(4.d0*pi)
c The uniform electron gas correlation energy according to Perdew and Zunger, Phys. Rev. B, 23, 5076 (1981)
c rs>1
         gu =-0.1423d0
         b1u = 1.0529d0
         b2u = 0.3334d0
         gp = -0.0843d0
         b1p = 1.3981d0
         b2p = 0.2611d0
         b = (3.d0/(4.d0*pi))**(1.d0/3.d0)
         sqrs = dsqrt(rs)
         AA = b*b2u*(21.d0*b1u +16.d0*b2u*sqrs)
         BB = 5.d0*b1u+7.d0*(b1u**2)*sqrs+8.d0*b2u*sqrs
         decdrsn = gu*b**2*(AA + BB*b/rs)
         decdrsd = 36.d0*(b*b1u+(b*b2u+b/rs)*sqrs)**3*(b/rs)**3
         decdrs = decdrsn/decdrsd
c rs<1
         au = 0.0311d0
         bu = -0.048d0
         cu = 0.0020d0
         du = -0.0116d0
         two3 = (2.d0)**(1.d0/3.d0)
         three3 = (3.d0)**(1.d0/3.d0)
         pi3 = pi**(1.d0/3.d0)
         six3 = (6.d0)**(1.d0/3.d0)
         two23 = (2.d0)**(2.d0/3.d0)
         three23 = (3.d0)**(2.d0/3.d0)
         threepi3 = (3.d0/pi)**(1.d0/3.d0)
         dens3 = dens**(1.d0/3.d0)
         dens23 = dens**(2.d0/3.d0)
         dens43 =  dens**(4.d0/3.d0)
         ALDAx = -1.d0/(three23*pi3*dens3)
         ALDAc = -six3*(cu+du)-6.d0*pi3*au*dens3
         ALDAc = ALDAc - 2.d0*six3*cu*dlog(threepi3/(two23*dens))
         ALDAc = ALDAc/(18.d0*pi3*dens43)
         ALDAxch = ALDAx + ALDAc
         alrs = akc*decdrs
         aa = aac - alrs
         ALDAxc = -(1.d0/akc)*aa
         if(rs.lt.1.d0) ALDAxc = ALDAxch
         return
         end
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
