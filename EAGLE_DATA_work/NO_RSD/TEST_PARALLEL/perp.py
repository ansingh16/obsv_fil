from sympy import Point3D, Line3D
from sympy import *


p1, p2, p3 = Point3D(0.0, 0.0, 0.0), Point3D(1.0, 1.0, 1.0), Point3D(0.0, 2.0, 0.0)


l1 = Line3D(p1, p2)

s1 = l1.perpendicular_segment(p3)

P = l1.intersect(s1)

x,y,z = next(iter(P))

print N(x),N(y),N(z)