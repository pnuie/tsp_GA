from test_set import *
from util import *
from GA import *
from LNS import *

my_tsp_solver = GA(10, 100, 500, 0.5)
my_tsp_solver.solve()

# my_LNS_solver = LNS(10, 5, 0.1)
# my_LNS_solver.solve()