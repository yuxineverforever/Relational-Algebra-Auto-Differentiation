import relationAD as ad
import numpy as np
import vjps
import unittest

class Test_forward_join_1(unittest.TestCase):
    def test_forward_join_1(self):
        leftInput = {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}
        rightInput = {(0, 0): 5, (0, 1): 6, (1, 0): 7, (1, 1): 8}
        lhsKey = [1]
        rhsKey = [0]
        f = lambda x, y: x * y
        assert ad.local_join(leftInput, rightInput, lhsKey, rhsKey, f) == {(0, 0, 0): 5, (0, 0, 1): 6, (0, 1, 0): 14,
                                                                           (0, 1, 1): 16, (1, 0, 0): 15, (1, 0, 1): 18,
                                                                           (1, 1, 0): 28, (1, 1, 1): 32}

class Test_forward_join_2(unittest.TestCase):
    def test_forward_join_2(self):
        leftInput = {(0, 0, 0): 5, (0, 0, 1): 6, (0, 1, 0): 14, (0, 1, 1): 16, (1, 0, 0): 15, (1, 0, 1): 18,
                     (1, 1, 0): 28, (1, 1, 1): 32}
        rightInput = {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}
        lhsKey = [1, 2]
        rhsKey = [0, 1]
        f = lambda x, y: y
        assert ad.local_join(leftInput, rightInput, lhsKey, rhsKey, f) == {(0, 0, 0): 1, (0, 0, 1): 2, (0, 1, 0): 3,
                                                                           (0, 1, 1): 4, (1, 0, 0): 1, (1, 0, 1): 2,
                                                                           (1, 1, 0): 3, (1, 1, 1): 4}

class Test_forward_aggregation_1(unittest.TestCase):
    def test_forward_aggregation_1(self):
        leftInput = {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}
        groupKey = [1]
        assert ad.local_aggregation(leftInput, groupKey) == {(0,): 4, (1,): 6}

class Test_forward_aggregation_2(unittest.TestCase):
    def test_forward_aggregation_2(self):
        leftInput = {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}
        groupKey = []
        assert ad.local_aggregation(leftInput, groupKey) == {(): 10}

class Test_forward_select(unittest.TestCase):
    def test_forward_select(self):
        leftInput = {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}
        f = lambda x: x * x
        assert ad.local_select(leftInput, f) == {(0, 0): 1, (0, 1): 4, (1, 0): 9, (1, 1): 16}

class Test_autodiff_join_1(unittest.TestCase):
    def test_autodiff_join_1(self):
        A = ad.Relation("A",[0,1])
        B = ad.Relation("B",[0,1])
        C = ad.Relation("C",[0,1])

        # loss = Agg((A join B) join C)
        D = ad.join_op(A, B, predicate=[[1], [0]], udf=lambda x, y: x * y)
        E = ad.aggregation_op(D, groupby=[0, 2])
        F = ad.join_op(E, C, predicate=[[1], [0]], udf=lambda x, y: x * y)
        G = ad.aggregation_op(F, groupby=[0, 2])
        H = ad.aggregation_op(G, groupby=[])

        grad_A, grad_B, grad_C = ad.gradients(H, [A, B, C])
        executor = ad.Executor([H, grad_A, grad_B, grad_C])

        h_val, grad_A_val, grad_B_val, grad_C_val = executor.run(feed_dict={
            A:{(0,0):3,(0,1):4,(1,0):5,(1,1):6},
            B:{(0,0):7,(0,1):8,(1,0):9,(1,1):10},
            C:{(0,0):1,(0,1):2,(1,0):3,(1,1):4}
        })

        assert h_val == {(): 1586}
        assert grad_A_val == {(0, 0): 77, (0, 1): 97, (1, 0): 77, (1, 1): 97}
        assert grad_B_val == {(0, 0): 24, (1, 0): 30, (0, 1): 56, (1, 1): 70}
        assert grad_C_val == {(0, 0): 146, (1, 0): 164, (0, 1): 146, (1, 1): 164}

class Test_autodiff_join_2(unittest.TestCase):
    def test_autodiff_join_2(self):
        A = ad.Relation("A",[0,1])
        B = ad.Relation("B",[0])

        # loss = Agg((A join B) join C)
        D = ad.join_op(A, B, predicate=[[1], [0]], udf=lambda x, y: x * y)
        E = ad.aggregation_op(D, groupby=[])

        grad_A, grad_B = ad.gradients(E, [A, B])
        executor = ad.Executor([E, grad_A, grad_B])

        e_val, grad_A_val, grad_B_val = executor.run(feed_dict={
            A:{(0,0):3,(0,1):4,(1,0):5,(1,1):6},
            B:{(0,):7,(1,):8}
        })
        assert e_val == {(): 136}
        assert grad_A_val == {(0, 0): 7, (0, 1): 8, (1, 0): 7, (1, 1): 8}
        assert grad_B_val == {(0,): 8, (1,): 10}

class Test_autodiff_join_3(unittest.TestCase):
    def test_autodiff_join_3(self):
        A = ad.Relation("A",[0,1])
        B = ad.Relation("B",[0,1])
        C = ad.Relation("C",[0,1])

        # loss = Agg((A join B) join C)
        D = ad.join_op(A, B, predicate=[[1], [0]], udf=lambda x, y: x * y)
        E = ad.aggregation_op(D, groupby=[0, 2])
        F = ad.join_op(E, C, predicate=[[1], [0]], udf=lambda x, y: x * y)
        G = ad.aggregation_op(F, groupby=[])

        grad_A, grad_B, grad_C = ad.gradients(G, [A, B, C])
        executor = ad.Executor([G, grad_A, grad_B, grad_C])

        g_val, grad_A_val, grad_B_val, grad_C_val = executor.run(feed_dict={
            A:{(0,0):3,(0,1):4,(1,0):5,(1,1):6},
            B:{(0,0):7,(0,1):8,(1,0):9,(1,1):10},
            C:{(0,0):1,(0,1):2,(1,0):3,(1,1):4}
        })

        assert g_val == {(): 1586}
        assert grad_A_val == {(0, 0): 77, (0, 1): 97, (1, 0): 77, (1, 1): 97}
        assert grad_B_val == {(0, 0): 24, (1, 0): 30, (0, 1): 56, (1, 1): 70}
        assert grad_C_val == {(0, 0): 146, (1, 0): 164, (0, 1): 146, (1, 1): 164}

class Test_autodiff_agg(unittest.TestCase):
    def test_autodiff_agg(self):
        A = ad.Relation("A",[0,1])
        B = ad.Relation("B",[0,1])
        C = ad.join_op(A, B, predicate=[[1], [0]], udf=lambda x, y: x * y)
        D = ad.aggregation_op(C, groupby=[0, 2])
        E = ad.aggregation_op(D, groupby=[])
        grad_A, grad_B = ad.gradients(E, [A, B])
        executor = ad.Executor([E, grad_A, grad_B])
        e_val, grad_A_val, grad_B_val = executor.run(feed_dict={
            A:{(0,0):3,(0,1):4,(1,0):5,(1,1):6},
            B:{(0,0):7,(0,1):8,(1,0):9,(1,1):10}
        })
        assert e_val == {(): 310}
        assert grad_A_val == {(0, 0): 15, (0, 1): 19, (1, 0): 15, (1, 1): 19}
        assert grad_B_val == {(0, 0): 8, (1, 0): 10, (0, 1): 8, (1, 1): 10}

class Test_vjps(unittest.TestCase):
    def test_vjps(self):
        mylambda1 = lambda x,y:x*y
        mylambda2 = lambda x,y:x*y
        assert vjps.primitive_vjps[vjps.coco(mylambda1.__code__)][0](3,2) == 2
        assert vjps.primitive_vjps[vjps.coco(mylambda1.__code__)][1](3,2) == 3









