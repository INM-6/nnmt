
all:	kummers_function.so

kummers_function.so:	kummers_function.pyf kummers_function.f90
	f2py3 -c kummers_function.pyf -lmathlib kummers_function.f90
#	f2py -c kummers_function.pyf -L/home/helias/slatec-4.1 -lslatec kummers_function.f90



kummers_function.pyf: 	kummers_function.f90
	f2py3 kummers_function.f90 -m kummers_function -h kummers_function.pyf

.phony clean:
	rm kummers_function.pyf
	rm kummers_function.so
