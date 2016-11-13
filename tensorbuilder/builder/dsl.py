from utils import identity
import utils
import pprint
from abc import ABCMeta, abstractmethod


###############################
# Ref
###############################
class Ref(object):
    """docstring for Ref."""
    def __init__(self, value=None):
        super(Ref, self).__init__()
        self.value = value

    def __call__(self, *optional):
        return self.value

    def set(self, x):
        self.value = x
        return x

###############################
# DSL Elements
###############################

class Node(object):
    """docstring for Node."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def compile(self, refs):
        pass

class Function(Node):
    """docstring for Function."""
    def __init__(self, f):
        super(Function, self).__init__()
        self.f = f

    def __iter__(self):
        yield self

    def compile(self, refs):
        return self.f, refs

    def __str__(self):
        return "Fun({0})".format(self.f)


Identity = Function(identity)


class Tree(Node):
    """docstring for Tree."""

    def __init__(self, branches):
        self.branches = list(branches)

    def compile(self, refs):
        fs = []

        for node in self.branches:
            node_fs, refs = node.compile(refs)

            if type(node_fs) is list:
                fs += node_fs
            else:
                fs.append(node_fs)

        return fs, refs

    def __str__(self):
        return pprint.pformat(self.branches)


class Sequence(Node):
    """docstring for Sequence."""
    def __init__(self, left, right):
        super(Sequence, self).__init__()
        self.left = left
        self.right = right

    def compile(self, refs):
        f_left, refs = self.left.compile(refs)

        if type(f_left) is list:
            f_left = list_to_fn(f_left)

        f_right, refs = self.right.compile(refs)

        if type(f_right) is list:
            f_right = compose_list(f_right, f_left)
        else:
            f_right = utils.compose2(f_right, f_left)

        return f_right, refs

    def __str__(self):
        return "Seq({0}, {1})".format(self.left, self.right)


class Scope(Node):
    """docstring for Dict."""

    GLOBAL_SCOPE = None

    def __init__(self, scope_obj, body):
        super(Scope, self).__init__()
        self.scope_obj = scope_obj
        self.body = body

    def compile(self, refs):
        body_fs, refs = self.body.compile(refs)

        def scope_fun(x):
            with self.scope_obj as scope:
                with self.set_scope(scope):
                    return body_fs(x)

        left = Identity
        right = Function(scope_fun)

        return Sequence(left, right).compile(refs)

    def set_scope(self, new_scope):
        self.new_scope = new_scope
        self.old_scope = Scope.GLOBAL_SCOPE

        return self

    def __enter__(self):
        Scope.GLOBAL_SCOPE = self.new_scope

    def __exit__(self, *args):
        Scope.GLOBAL_SCOPE = self.old_scope

    def __str__(self):
        return "\{ {0}: {1}\}".format(pprint.pformat(self.scope_obj), pprint.pformat(self.body))

class Read(Node):
    """docstring for Read."""
    def __init__(self, name):
        super(Read, self).__init__()
        self.name = name

    def compile(self, refs):
        ref = refs[self.name]
        f = ref #ref is callable with an argument
        return f, refs


class Write(Node):
    """docstring for Read."""
    def __init__(self, name):
        super(Write, self).__init__()
        self.name = name

    def compile(self, refs):
        if self.name not in refs:
            refs[self.name] = ref = Ref()
        else:
            ref = refs[self.name]

        return ref.set, refs


class Input(Node):
    """docstring for Input."""
    def __init__(self, value):
        super(Input, self).__init__()
        self.value = value

    def compile(self, refs):
        f = lambda x: self.value
        return f, refs


class Apply(Node):
    """docstring for Read."""
    def __init__(self):
        super(Write, self).__init__()



#######################
### FUNCTIONS
#######################

def Compile(code, refs):
    ast = parse(code)
    fs, refs = ast.compile(refs)

    if type(fs) is list:
        fs = list_to_fn(fs)

    return fs, refs

def compose(f, g):
    return lambda x: f(g(x))

def compose_list(fs, g):
    return [ compose(f, g) for f in fs ]

def list_to_fn(fs):
    return lambda x: [ f(x) for f in fs ]

def parse(code):
    #if type(code) is tuple:
    if type(code) is str:
        return Read(code)
    elif type(code) is set:
        return parse_set(code)
    elif hasattr(code, '__call__'):
        return Function(code)
    elif type(code) is tuple:
        return parse_tuple(code)
    elif type(code) is dict:
        return parse_dictionary(code)
    else:
        return parse_iterable(code) #its iterable
        #raise Exception("Element has to be either a tuple for sequential operations, a list for branching, or a function from a builder to a builder, got %s, %s" % (type(code), type(code) is tuple))

def parse_set(code):
    if len(code) == 0:
        return Identity

    fst = iter(code).next()

    if len(code) == 1:
        if type(fst) is str:
            return Write(fst)
        elif type(fst) is tuple:
            value = fst[0]
            return Input(value)
        else:
            raise Exception("Not part of the language: {0}".format(code))
    else:
        raise Exception("Not part of the language: {0}".format(code))

def build_sequence(right, *prevs):
    left = prevs[0] if len(prevs) == 1 else build_sequence(*prevs)
    return Sequence(left, right)

def parse_tuple(tuple_code):
    nodes = [ parse(code) for code in tuple_code ]

    if len(nodes) == 1:
        return nodes[0]

    if len(nodes) == 0:
        return Identity

    nodes.reverse()

    return build_sequence(*nodes)


def parse_iterable(iterable_code):
    nodes = [ parse(code) for code in iterable_code ]

    if len(nodes) == 1:
        return nodes[0]

    if len(nodes) == 0:
        return Identity

    return Tree(nodes)

def parse_dictionary(dict_code):
    scope_obj, body_code = list(dict_code.items())[0]
    body = parse(body_code)
    return Scope(scope_obj, body)
