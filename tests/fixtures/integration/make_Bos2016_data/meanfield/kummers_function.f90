! compile with:
! f2py kummers_function.f90 -m kummers_function -h kummers_function1.pyf
! f2py -c kummers_function1.pyf kummers_function.f90
! instal slatec from source: https://launchpad.net/ubuntu/+source/slatec/4.1-4
! make soft link in /usr/bin/
! ln -sf gfortran g77

subroutine kummers_function(a, b, z, k)
 
  !implicit none
 
  complex , intent(in) :: a
  complex , intent(in) :: b
  complex , intent(in) :: z
  complex , intent(out) :: k

  complex :: cwhitm

  k = cwhitm(z,0.5*b-a,0.5*b-0.5)*exp(0.5*z)*z**(-0.5*b)

end subroutine kummers_function


