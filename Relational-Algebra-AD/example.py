import relationAD as ad
import numpy as np
import vjps
import unittest

# Here is a simple case for a two layer MLP implemented in a relational way to generate backprop:
if __name__ == '__main__':

    def relation_SGD(r1, r2, lr):
        for k1 in r1:
            r1[k1] = r1[k1] - lr*r2[k1]

    # define all the relations
    X = ad.Relation("X", [0])
    W1 = ad.Relation("W1", [0, 1])
    B1 = ad.Relation("B1",[0])
    W2 = ad.Relation("W2", [0, 1])
    B2 = ad.Relation("B2", [0])
    Truth = ad.Relation("T",[0])

    # define all the parameters
    Z1 = ad.join_op(W1,X,predicate=[[1],[0]],udf=lambda x, y: x * y)
    Z2 = ad.join_op(Z1,B1,predicate=[[0],[0]],udf=lambda x, y: x + y)
    Z3 = ad.select_op(Z2, udf=lambda x: x * (x > 0))
    Z4 = ad.join_op(W2,Z3,predicate=[[1],[0]],udf=lambda x, y: x * y)
    Z5 = ad.join_op(Z4,B2,predicate=[[0],[0]],udf=lambda x, y: x + y)
    Z6 = ad.join_op(Z5,Truth,predicate=[[0],[0]], udf=lambda x, y: x - y)
    Z7 = ad.select_op(Z6, udf=lambda x: 1 / 2 * x * x)
    L = ad.aggregation_op(Z7,groupby=[])

    grad_W1, grad_B1, grad_W2, grad_B2 = ad.gradients(L, [W1, B1, W2, B2])
    executor = ad.Executor([L, grad_W1, grad_B1, grad_W2, grad_B2],verbose=False)

    lr = 0.01

    feed = {W1: {(0, 0): 3, (0, 1): 4, (1, 0): 5, (1, 1): 6},
            B1: {(0,): 7, (1,): 8},
            W2: {(0, 0): 3, (0, 1): 4, (1, 0): 5, (1, 1): 6},
            B2: {(0,): 4, (1,): 9},
            X: {(0,): 1, (1,): 1},
            Truth: {(0,): 12.5, (1,): 13.5}}

    for i in range(100):
        Loss, grad_W1_val, grad_B1_val, grad_W2_val, grad_B2_val = executor.run(feed_dict=feed)
        print("iteration %d" % (i))
        relation_SGD(feed[W1], grad_W1_val, lr)
        relation_SGD(feed[B1], grad_B1_val, lr)
        relation_SGD(feed[W2], grad_W2_val, lr)
        relation_SGD(feed[B2], grad_B2_val, lr)


