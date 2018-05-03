        subroutine io_test
                real, dimension(:), allocatable :: x
                integer :: n

                open (unit=99, file='array.txt',&
                 status='old', action='read')
                read(99, *), n
                allocate(x(n))
                read(99,*) x

                write(*,*) x
        end subroutine io_test
