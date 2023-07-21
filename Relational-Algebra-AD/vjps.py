'''
We register some derivative functions here. It can be substituted by a auto-diff kernel generation tool (e.g. JAX or PyTorch).
'''
from collections import defaultdict
from itertools import count
import operator
import numpy as np

coco = operator.attrgetter('co_code', 'co_consts')

multiply_lambda = lambda x,y:x*y
add_lambda = lambda x,y:x+y
identity_lambda = lambda x:x
square_lambda = lambda x:x*x
halfsquare_lambda = lambda x:1/2*x*x
ex_lambda = lambda x: np.log(x)
relu_lambda = lambda x: x * (x > 0)
minus_lambda = lambda x,y:x-y

primitive_vjps = defaultdict(dict)

def defvjp(fun, *vjps, **kwargs):
    """Registered functions for computing derivatives."""
    argnums = kwargs.get('argnums', count())
    for argnum, vjp in zip(argnums, vjps):
        primitive_vjps[coco(fun.__code__)][argnum] = vjp


defvjp(multiply_lambda,lambda x, y : y,
                    lambda x, y : x)
defvjp(add_lambda, lambda x, y : 1,
                lambda x, y : 1)
defvjp(minus_lambda,  lambda x, y : 1,
                      lambda x, y : -1)
defvjp(identity_lambda, lambda x : 1)

defvjp(square_lambda, lambda x: 2*x)

defvjp(halfsquare_lambda,lambda x: x)

defvjp(ex_lambda, lambda x: np.log(x))

defvjp(relu_lambda, lambda x: 1. * (x > 0))



