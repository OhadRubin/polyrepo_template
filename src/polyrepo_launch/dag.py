import networkx as nx
from contextvars import ContextVar
from dataclasses import dataclass
from itertools import product
import uuid
import json
import os
import tempfile

"""
with DAG() as dag:
    Branch(
        dataset("codeparrot") >> lt(2),
        dataset( "pg19") >> lt(4),
        dataset("arxiv") >> lt(8)
        ) >> Branch(
            ss_steps(3),
            ss_max(0.0),
            dng(True)
            ) >>epr(1)
"""


"""
from hpdag import DAG,Node,Branch
dataset = Node("dataset")
lr = Node("lr")



# Branch(
#   size("1b") >> fsdp(8),
#   size("250m") >>fsdp(1),
# ) >> sw(True,False)

with DAG() as dag:
    datasets = Branch(
        dataset("the_pile") >> lr(0.001), #for example, one dataset might require specific settings than the others
        dataset( "c4") >> lr(0.01),
        )
    ablations = Branch( #do a type of ablation on each dataset
            Node("use_glu")(True,False), #run the experiment with and without the glu
            Node("positional_enc")("alibi","rotary"), #run the experiment with two different positional encodings
            )
    sizes = Node("size")("7b","3b") #run the experiment with two different sizes
    datasets >> ablations >>sizes
print(dag)
print(dag)
Task(dataset=the_pile, lr=0.001, use_glu=True, size=7b)
Task(dataset=the_pile, lr=0.001, use_glu=True, size=3b)
Task(dataset=the_pile, lr=0.001, use_glu=False, size=7b)
Task(dataset=the_pile, lr=0.001, use_glu=False, size=3b)
Task(dataset=the_pile, lr=0.001, positional_enc=alibi, size=7b)
Task(dataset=the_pile, lr=0.001, positional_enc=alibi, size=3b)
Task(dataset=the_pile, lr=0.001, positional_enc=rotary, size=7b)
Task(dataset=the_pile, lr=0.001, positional_enc=rotary, size=3b)
Task(dataset=c4, lr=0.01, use_glu=True, size=7b)
Task(dataset=c4, lr=0.01, use_glu=True, size=3b)
Task(dataset=c4, lr=0.01, use_glu=False, size=7b)
Task(dataset=c4, lr=0.01, use_glu=False, size=3b)
Task(dataset=c4, lr=0.01, positional_enc=alibi, size=7b)
Task(dataset=c4, lr=0.01, positional_enc=alibi, size=3b)
Task(dataset=c4, lr=0.01, positional_enc=rotary, size=7b)
Task(dataset=c4, lr=0.01, positional_enc=rotary, size=3b)
"""

class TaskIterator:
    def __init__(self, tasks):
        self.tasks = tasks
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.tasks):
            raise StopIteration
        task = self.tasks[self.index]
        self.index += 1
        return task

_ACTIVE_DAG = ContextVar("_ACTIVE_DAG")


class _GraphExpr:
    def __init__(self, *, dag, entries, exits, nodes):
        self.dag = dag
        self.entries = tuple(dict.fromkeys(entries))
        self.exits = tuple(dict.fromkeys(exits))
        self.nodes = tuple(dict.fromkeys(nodes))

    def __rshift__(self, other):
        if not isinstance(other, _GraphExpr):
            return NotImplemented
        dag = _ACTIVE_DAG.get()
        assert self.dag is dag and other.dag is dag, "Expressions must belong to the active DAG"
        for from_node in self.exits:
            for to_node in other.entries:
                dag.link(from_node, to_node)
        return _GraphFragment(
            dag=dag,
            entries=self.entries,
            exits=other.exits,
            nodes=(*self.nodes, *other.nodes),
        )

    def __lshift__(self, other):
        if not isinstance(other, _GraphExpr):
            return NotImplemented
        return other >> self

    def __or__(self, other):
        assert isinstance(other, _GraphExpr), "Union operands must be graph expressions"
        return Branch(self, other)

    def __enter__(self):
        self.dag.enter_scope(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.dag.exit_scope(self)
        else:
            self.dag.abort_scope(self)


class _GraphFragment(_GraphExpr):
    pass


class _Node(_GraphExpr):
    def __init__(self,*, name, values):
        dag = _ACTIVE_DAG.get()
        if len(values)==0:
            values = [name]
        self.name = name
        self.uuid = uuid.uuid4()
        self.values = tuple(values)
        dag.add_param(self)
        super().__init__(
            dag=dag,
            entries=(self,),
            exits=(self,),
            nodes=(self,),
        )

    def __repr__(self):
        return f"Node({self.name}=[" + ", ".join(map(str, self.values)) + "])"

class Node:
    def __init__(self, name):
        self.name = name
    def __call__(self, *values,l=None):
        if l is None:
            return _Node(name=self.name, values=values)
        else:
            assert len(values)==0, "You can only specify a list of values for a node with no links"
            return _Node(name=self.name, values=list(l))



class Branch(_GraphExpr):
    def __init__(self, *arms):
        dag = _ACTIVE_DAG.get()
        assert len(arms) > 0, "Branch requires at least one arm"
        assert all(isinstance(arm, _GraphExpr) for arm in arms), "Every Branch arm must be a node, chain, or nested Branch"
        assert all(arm.dag is dag for arm in arms), "All Branch arms must belong to the active DAG"
        self.nodes = arms
        super().__init__(
            dag=dag,
            entries=(
                node
                for arm in arms
                for node in arm.entries
            ),
            exits=(
                node
                for arm in arms
                for node in arm.exits
            ),
            nodes=(
                node
                for arm in arms
                for node in arm.nodes
            ),
        )


@dataclass
class _NamedScope:
    expr: _GraphExpr
    label: str | None
    body_nodes: list[uuid.UUID]

class Task:
    def __init__(self, dag, params, named_fragments):
        self.params = {}
        for k, v in params.items():
            k = dag.node_dict[k].name
            self.params[k] = v
        self.named_fragments = named_fragments

    def __repr__(self):
        return "Task(" + ", ".join([f"{k}={v}" for k, v in self.params.items()]) + ")"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if not isinstance(other, Task):
            return False
        return self.params == other.params

    def __hash__(self):
        return hash(tuple(sorted(self.params.items())))

class GraphOperations(nx.DiGraph):
    def get_all_paths(self, start_node, end_node):
        return list(nx.all_simple_paths(self, start_node, end_node))



class DAG(GraphOperations):

    def __init__(self):
        super().__init__()
        self.params = {}
        self.node_dict = {}
        self.layers = []
        self.out = None
        self.scopes = []
        self.named_scopes = []

    def add_param(self,node):
        self.add_node(node.uuid)
        self.params[node.uuid] = node.values
        self.layers.append(node.uuid)
        self.node_dict[node.uuid] = node
        for scope in self.scopes:
            scope.body_nodes.append(node.uuid)



    def link(self, from_node, to_node):
        self.add_edge(from_node.uuid, to_node.uuid)

    def enter_scope(self, expr):
        self.scopes.append(_NamedScope(expr=expr, label=None, body_nodes=[]))

    def exit_scope(self, expr):
        scope = self.scopes.pop()
        assert scope.expr is expr, "Graph expression scopes must exit in stack order"
        assert scope.body_nodes, "Named graph expression scope must contain a body expression"
        body_node_set = set(scope.body_nodes)
        body_roots = [
            node_uuid
            for node_uuid in scope.body_nodes
            if not any(parent in body_node_set for parent in self.predecessors(node_uuid))
        ]
        for from_node in scope.expr.exits:
            for to_node_uuid in body_roots:
                self.link(from_node, self.node_dict[to_node_uuid])
        if scope.label is not None:
            self.named_scopes.append((set(node.uuid for node in scope.expr.nodes), scope.label))

    def name_active_scope(self, label):
        assert self.scopes, "named() must be used after a graph expression context manager"
        self.scopes[-1].label = label

    def abort_scope(self, expr):
        scope = self.scopes.pop()
        assert scope.expr is expr, "Graph expression scopes must exit in stack order"


    def cartesian_product(self, nodes):
        return list(product(*[self.params[node] for node in nodes]))

    def get_all_paths(self, start_nodes, end_nodes):
        all_paths = []
        for start_node in start_nodes:
            for end_node in end_nodes:
                all_paths.extend(super().get_all_paths(start_node, end_node))
        return all_paths

    def get_start_nodes(self):
        return [node_uuid for node_uuid, in_degree in self.in_degree if in_degree == 0]


    def get_end_nodes(self):
        return [node_uuid for node_uuid, out_degree in self.out_degree if out_degree == 0]

    def __str__(self):
        start_nodes = self.get_start_nodes()
        end_nodes = self.get_end_nodes()
        all_paths = self.get_all_paths(start_nodes, end_nodes)
        return '\n'.join(map(str, self.generate_tasks(all_paths)))

    def generate_tasks(self, all_paths):
        return [
            Task(
                self,
                {path[i]: combo[i] for i in range(len(combo))},
                self.named_fragments_for_path(path),
            )
            for path in all_paths
            for combo in self.cartesian_product(path)
        ]

    def named_fragments_for_path(self, path):
        path_node_set = set(path)
        return {
            self.node_dict[node_uuid].name: label
            for scope_node_uuids, label in self.named_scopes
            if scope_node_uuids <= path_node_set
            for node_uuid in scope_node_uuids
        }

    def task_iterator(self):
        all_paths = self.get_all_paths(self.get_start_nodes(), self.get_end_nodes())
        return TaskIterator(self.generate_tasks(all_paths))

    @property
    def tasks(self):
        if self.out is None:
            return list(self.task_iterator())
        else:
            return self.out

    def __enter__(self):
        self._active_dag_token = _ACTIVE_DAG.set(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type is None:
                assert not self.scopes, "All graph expression scopes must be closed before DAG exit"
                self.out = list(self.task_iterator())
        finally:
            _ACTIVE_DAG.reset(self._active_dag_token)


class _Named:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        dag = _ACTIVE_DAG.get()
        dag.name_active_scope(self.label)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def named(label):
    return _Named(label)


def parse_type(x):
    try:
      value = str(x).strip()
      parsed_value = eval(value)
    except:
      parsed_value = value
    return parsed_value

def parse_single(opt):
    opt = opt.strip()
    opt, *_ = opt.split("#",1)
    opt = opt.strip().split(",", 2)
    assert len(opt)==3, f"Invalid option: {opt}"
    opt = list(map(str.strip, list(opt)))
    return opt[0], opt[1], opt[2]

def handle_opt(options):
    options = [x.strip() for x in options.split("\n") if x]
    ARG_mapping = {}
    val_mapping = {}
    for opt in options:
        ARG, arg, default_val = parse_single(opt)
        # ARG, arg, default_val = map(str.strip, opt)
        ARG_mapping[arg] = ARG
        val_mapping[arg] = default_val
    return ARG_mapping,val_mapping


from copy import deepcopy


class Bunch(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def get_all_experiments(experiment, config, exp_count, stop_fn=lambda x: False):
    EXP_COUNT = f"v{exp_count}"
    var_dict = {}
    orig_task_dict = {}
    for task in experiment.tasks:
        if stop_fn(task):
            continue
        state = dict(task.params)
        state["WANDB_NAME"] = []
        ARG_VARS = Bunch()
        ARG_mapping, val_mapping, wandb_args = deepcopy(parse_config(config))
        explicit_wandb_args = list(wandb_args)
        wandb_args.extend(list(state.keys()))
        val_mapping_items = list(val_mapping.items())
        named_fragment_labels = set()
        for param_name, default_value in val_mapping_items[::-1]:
            if param_name in state:
                value = state[param_name]
            else:
                value = os.environ.get(f'_{param_name}', default_value)
            if param_name in wandb_args:
                if param_name in task.named_fragments and param_name not in explicit_wandb_args:
                    label = task.named_fragments[param_name]
                    if label not in named_fragment_labels:
                        state['WANDB_NAME'].append(label)
                        named_fragment_labels.add(label)
                else:
                    state['WANDB_NAME'].append(f"{param_name}={value}")
            state[param_name] = parse_type(value)
            ARG_VARS[ARG_mapping[param_name]] = parse_type(value)
        WANDB_NAME = ",".join(state['WANDB_NAME'])
        WANDB_NAME = f"{EXP_COUNT}_{WANDB_NAME}"
        ARG_VARS["WANDB_NAME"] = WANDB_NAME

        arg_list = []
        for var_name, var_value in ARG_VARS.items():
            arg_list.append(f'export {var_name}={var_value}')
        args = "\n".join(arg_list)
        assert WANDB_NAME not in var_dict
        var_dict[WANDB_NAME] = args
        orig_task_dict[WANDB_NAME] = deepcopy(ARG_VARS)

    return var_dict, orig_task_dict
def parse_config(config):
    v_opt,q_opt = config.strip().split("---")
    ARG_mapping,val_mapping = handle_opt(f"{v_opt}\n{q_opt}")
    wandb_args = list(handle_opt(v_opt)[0].keys())
    return ARG_mapping,val_mapping,wandb_args

import inspect
def load_config(config):
    ARG_mapping,val_mapping,*_ = parse_config(config)

    locals = inspect.getargvalues(inspect.getouterframes(inspect.currentframe())[1].frame).locals
    assert "named" not in locals, "Variable named already exists in the global namespace"
    locals["named"] = named
    for k,v in ARG_mapping.items():
        assert k not in locals, f"Variable {k} already exists in the global namespace"
        locals[f"_{k}"] = val_mapping[k]
        locals[k] = Node(k)
    return config

def add_one(l):
    return [(x*10) if x!=10 else x for x in l ]

# size("1b") >> fsdp(1)  >> aug_nei(True) >> aug_xnei(True) >> xnei_bias(True) >> cca_norm2(True)
# Branch(
#     aug_xnei(True) >> xnei_bias(True,False),
#     aug_xnei(False),
#   ) >> cca_norm2(True) >> size("250m") >> fsdp(1,8)
# aug_xnei(True) >> xnei_bias(True,False) >> dtype("bf16")
