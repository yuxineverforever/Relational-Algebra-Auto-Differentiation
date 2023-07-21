import numpy as np
import vjps
import copy

MaximumNameNumber = 1000
class Node(object):
    """Node in a computation graph."""
    def __init__(self):
        self.inputs = []
        self.op = None
        self.const_attr = None
        self.name = ""
        self.keys = []
        self.keysdim = 0
        self.mapped_keys = []

    def __add__(self, other):
        if isinstance(other, Node):
            new_node = add_op(self, other)
        else:
            new_node = add_byconst_op(self, other)
        return new_node

    def __mul__(self, other):
        if isinstance(other, Node):
            new_node = mul_op(self, other)
        else:
            new_node = mul_byconst_op(self, other)
        return new_node

    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        return self.name

    __repr__ = __str__

# Below is an example for a simple relation.
'''
my_dict = {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}
my_dict[(2, 1)] = 19
for key in my_dict:
    print(key[0], key[1], '->', my_dict[key])
    print(key, '->', my_dict[key])
'''
def Relation(name, keys):
    placeholder_node = placeholder_op ()
    placeholder_node.name = name
    placeholder_node.keys = keys
    placeholder_node.keysdim = len(keys)
    return placeholder_node

def getNameForRelation():
    name = 'T'
    for index in range(MaximumNameNumber):
        new_name = name+str(index)
        yield new_name

relationName = getNameForRelation()


class Op(object):
    def __call__(self):
        new_node = Node()
        new_node.op = self
        return new_node

    def compute(self, node, input_vals, verbose):
        raise NotImplementedError

    def gradient(self, node, output_grad):

        raise NotImplementedError


class AddOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s+%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, verbose):
        assert len(input_vals) == 2
        return input_vals[0] + input_vals[1]

    def gradient(self, node, output_grad):
        return [output_grad, output_grad]


class AddByConstOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s+%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals, verbose):
        assert len(input_vals) == 1
        return input_vals[0] + node.const_attr

    def gradient(self, node, output_grad):
        return [output_grad]


class MulOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s*%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, verbose):
        assert len(input_vals) == 2
        return input_vals[0] * input_vals[1]

    def gradient(self, node, output_grad):
        return [mul_op(node.inputs[1], output_grad), mul_op(node.inputs[0], output_grad)]


class MulByConstOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s*%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals, verbose):
        assert len(input_vals) == 1
        return input_vals[0] * node.const_attr

    def gradient(self, node, output_grad):
        return [mul_byconst_op(output_grad, node.const_attr)]


class MatMulOp(Op):
    def __call__(self, node_A, node_B, trans_A=False, trans_B=False):
        new_node = Op.__call__(self)
        new_node.matmul_attr_trans_A = trans_A
        new_node.matmul_attr_trans_B = trans_B
        new_node.inputs = [node_A, node_B]
        new_node.name = "MatMul(%s,%s,%s,%s)" % (node_A.name, node_B.name, str(trans_A), str(trans_B))
        return new_node

    def compute(self, node, input_vals, verbose):
        assert len(input_vals) == 2
        if node.matmul_attr_trans_A == True:
            input_vals[0] = np.transpose(input_vals[0])
        if node.matmul_attr_trans_B == True:
            input_vals[1] = np.transpose(input_vals[1])
        return np.dot(input_vals[0], input_vals[1])

    def gradient(self, node, output_grad):
        return [matmul_op(output_grad, node.inputs[1], trans_A=False, trans_B=True),
                matmul_op(node.inputs[0], output_grad, trans_A=True, trans_B=False)]


class PlaceholderOp(Op):
    def __call__(self):
        """Creates a variable node."""
        new_node = Op.__call__(self)
        return new_node

    def compute(self, node, input_vals, verbose):
        assert False, "placeholder values provided by feed_dict"

    def gradient(self, node, output_grad):
        return None


class ZerosLikeOp(Op):
    def __call__(self, node_A):
        """Creates a node that represents a np.zeros array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Zeroslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals, verbose):
        assert (isinstance(input_vals[0], np.ndarray))
        return np.zeros(input_vals[0].shape)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]


class OnesLikeOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Oneslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals, verbose):
        assert (isinstance(input_vals[0], np.ndarray))
        return np.ones(input_vals[0].shape)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]


class RelationOp(Op):
    def __call__(self):
        new_node = Op.__call__(self)
        return new_node

    def compute(self, node, input_vals, verbose):
        assert False, "relation values provided by feed_dict"

    def gradient(self, node, output_grad):
        return None

def local_join(lhsRel, rhsRel, lhsKey, rhsKey, lambdaFunction):
    result_rel = {}
    for lhs in lhsRel:
        for rhs in rhsRel:
            isMatched = True
            for lkey, rkey in zip(lhsKey, rhsKey):
                if lhs[lkey] != rhs[rkey]:
                    isMatched = False
            if isMatched == True:
                # create the new key for the join result
                res_key = list(lhs)
                # reverse() is necessary because when you delete some element in a list, the index for other elements will change.
                for lkey in reversed(lhsKey):
                    del res_key[lkey]
                res_key = tuple(res_key) + rhs
                # create the new value for the join result
                res_value = lambdaFunction(lhsRel[lhs], rhsRel[rhs])
                result_rel[res_key] = res_value
    return result_rel

# The default aggregation operator is (+).
def local_aggregation(inputRel, groupByKey):
    resultRel = {}
    for key in inputRel:
        groupKey=[]
        for i in groupByKey:
            groupKey.append(key[i])
        groupKey = tuple(groupKey)
        if groupKey in resultRel:
            resultRel[groupKey] += inputRel[key]
        else:
            resultRel[groupKey] = inputRel[key]
    return resultRel

def local_select(inputRel, lambdaFunction):
    for key in inputRel:
        inputRel[key] = lambdaFunction(inputRel[key])
    return inputRel

class JoinOp(Op):
    def __call__(self, node_A, node_B, predicate, udf):
        assert len(predicate[0]) == len(predicate[1])
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = next(relationName)
        new_node.predicate = predicate
        new_node.udf = udf
        new_node.keysdim = node_A.keysdim + node_B.keysdim - len(predicate[0])
        new_node.keys = list(range(0,new_node.keysdim))
        new_node.mapped_keys = []

        node_A_keys = copy.deepcopy(node_A.keys)
        node_B_keys = copy.deepcopy(node_B.keys)
        node_A_map = {}
        node_B_map = {}

        for i in reversed(predicate[0]):
            del node_A_keys[i]

        index = 0
        for idx, val in enumerate(node_A_keys):
            node_A_map[val] = index
            index = index + 1
        for idx, val in enumerate(node_B_keys):
            node_B_map[val] = index
            index = index + 1
        for lpred,rpred in zip(predicate[0],predicate[1]):
            node_A_map[lpred] = node_B_map[rpred]

        # mapped keys maintain the keys mapped from two input relation A and B
        new_node.mapped_keys.append(node_A_map)
        new_node.mapped_keys.append(node_B_map)
        return new_node

    def compute(self, node, input_vals, verbose):
        assert len(input_vals) == 2
        if verbose == True:
            print("(%s) = (%s) Join (%s)" % (node.name, node.inputs[0].name, node.inputs[1].name))
        return local_join(input_vals[0], input_vals[1], node.predicate[0], node.predicate[1], node.udf)
    # RJP for Join:
    def gradient(self, node, output_grad):
        multiply_lambda = lambda x,y:x*y
        derivative_function_0 = vjps.primitive_vjps[vjps.coco(node.udf.__code__)][0]
        derivative_function_1 = vjps.primitive_vjps[vjps.coco(node.udf.__code__)][1]
        gradients_result = []
        # This is a 1-1 join for input[0]
        if node.inputs[1].keysdim == len(node.predicate[1]):
            gradients_result.append(
                join_op(output_grad,
                        join_op(node.inputs[0],node.inputs[1],node.predicate,derivative_function_0),[output_grad.keys, output_grad.keys],multiply_lambda))
        else:
            # This is a 1-n join for input[0]
            # The rules are:
            # T = R. / (R.j=S.i, ∂F∂R) S;
            # R = Σ(R.i, R.j), (+) {Z. / (...,∗)(T)}
            # @final_groupby is to get the matched keys in (R.i, R.j) for final aggregation.
            final_groupby = []
            for k in node.mapped_keys[0]:
                final_groupby.append(node.mapped_keys[0][k])
            gradients_result.append(
                aggregation_op(
                    join_op(output_grad,
                            join_op(node.inputs[0],node.inputs[1],node.predicate,derivative_function_0),[output_grad.keys, output_grad.keys],multiply_lambda),final_groupby))

        # This is a 1-1 join for input[1]
        if node.inputs[0].keysdim == len(node.predicate[0]):
            gradients_result.append(
                join_op(output_grad,
                        join_op(node.inputs[0], node.inputs[1], node.predicate, derivative_function_1),[output_grad.keys, output_grad.keys], multiply_lambda))
        else:
            # This is a 1-n join for input[1]
            final_groupby = []
            for k in node.mapped_keys[1]:
                final_groupby.append(node.mapped_keys[1][k])
            gradients_result.append(
                aggregation_op(
                    join_op(output_grad,
                            join_op(node.inputs[0],node.inputs[1],node.predicate,derivative_function_1),[output_grad.keys, output_grad.keys],multiply_lambda),final_groupby))

        return gradients_result

class AggregationOp(Op):
    def __call__(self, node_A, groupby):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = next(relationName)
        new_node.groupby = groupby

        # dealing with all the key related issues.
        new_node.keysdim = len(groupby)
        new_node.keys = list(range(0,new_node.keysdim))
        new_node.mapped_keys = []
        node_A_map = {}

        for idx, val in enumerate(new_node.groupby):
            node_A_map[val] = idx

        new_node.mapped_keys.append(node_A_map)
        return new_node

    def compute(self, node, input_vals, verbose):
        assert len(input_vals) == 1
        if verbose == True:
            print("(%s) = Aggregate (%s)" % (node.name, node.inputs[0].name))
        return local_aggregation(input_vals[0], node.groupby)

    # RJP for Aggregate:
    def gradient(self, node, output_grad):
        # this rule is for n-1 aggregation
        if node.groupby == []:
            one_lambda = lambda x: 1
            return [select_op(node.inputs[0], one_lambda)]
        else:
            # this rule is for n-n aggregation
            lvalue_lambda = lambda x,y: x
            return [join_op(output_grad, node.inputs[0], [output_grad.keys, node.groupby], lvalue_lambda)]

class SelectOp(Op):
    def __call__(self, node_A, udf):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = next(relationName)
        new_node.udf = udf
        new_node.keysdim = node_A.keysdim
        new_node.keys = node_A.keys
        return new_node

    def compute(self, node, input_vals, verbose):
        assert len(input_vals) == 1
        if verbose == True:
            print("(%s) = Select (%s)" % (node.name,node.inputs[0].name))
        return local_select(input_vals[0], node.udf)

    # RJP for Select:
    def gradient(self, node, output_grad):
        multiply_lambda = lambda x,y:x*y
        derivative_function_0 = vjps.primitive_vjps[vjps.coco(node.udf.__code__)][0]
        return [join_op(select_op(node.inputs[0], derivative_function_0), output_grad, [node.keys, node.keys], multiply_lambda)]

# Create global singletons of operators.
add_op = AddOp()
mul_op = MulOp()
add_byconst_op = AddByConstOp()
mul_byconst_op = MulByConstOp()
matmul_op = MatMulOp()
placeholder_op = PlaceholderOp()
oneslike_op = OnesLikeOp()
zeroslike_op = ZerosLikeOp()

# relational operations
join_op = JoinOp()
aggregation_op = AggregationOp()
select_op = SelectOp()

class Executor:
    def __init__(self, eval_node_list, verbose=False):
        self.eval_node_list = eval_node_list
        self.verbose = verbose

    def run(self, feed_dict):
        node_to_val_map = dict(feed_dict)
        topo_order = find_topo_sort(self.eval_node_list)
        for node in topo_order:
            if node.op != placeholder_op:
                input_value = []
                for input_node in node.inputs:
                    input_value.append(node_to_val_map[input_node])
                node_to_val_map[node] = node.op.compute(node, input_value,self.verbose)
        node_val_results = [node_to_val_map[node] for node in self.eval_node_list]
        return node_val_results


def gradients(output_node, node_list):
    node_to_output_grads_list = {}
    node_to_output_grads_list[output_node] = [oneslike_op(output_node)]

    node_to_output_grad = {}
    reverse_topo_order = reversed(find_topo_sort([output_node]))

    for node in reverse_topo_order:
        grad = sum_node_list(node_to_output_grads_list[node])
        input_grad = node.op.gradient(node, grad)
        for idx, val in enumerate(node.inputs):
            if val in node_to_output_grads_list:
                node_to_output_grads_list[val].append(input_grad[idx])
            else:
                node_to_output_grads_list[val] = [input_grad[idx]]
        node_to_output_grad[node] = grad

    # Collect results for gradients requested.
    grad_node_list = [node_to_output_grad[node] for node in node_list]
    return grad_node_list


##############################
####### Helper Methods #######
##############################
def find_topo_sort(node_list):
    visited = set()
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order

def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)

def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce
    return reduce(add, node_list)
