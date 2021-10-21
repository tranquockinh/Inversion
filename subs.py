from sympy.abc import x
from sympy import sin, pi
expr=sin(x)
expr1=expr.subs(x,pi)
print(expr1)
