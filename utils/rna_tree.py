import torch
from utils.rna_lib import Stack
import json
import os
from pyecharts import options as opts
from pyecharts.charts import Tree
from pyecharts.render import make_snapshot
import networkx
import numpy as np
# from snapshot_selenium import snapshot as driver


def tree2graph(rna_tree):
    node_cnt = 0
    loop_node = []
    final_node = -1
    graph = networkx.Graph()
    graph.add_nodes_from([(-1, {"type": "5", "pos": [], "color": "green"})])
    for i, node in zip(range(len(rna_tree.external_loop)), rna_tree.external_loop):
        if type(node) is int:
            loop_node.append(node)
            if i == len(rna_tree.external_loop) - 1:
                graph.add_nodes_from(
                    [(node_cnt, {
                        "type": "loop",
                        "pos": loop_node,
                        "color": "blue"
                    })]
                )
                loop_node = []
                graph.add_edge(final_node, node_cnt)
                final_node = node_cnt
                node_cnt += 1
        if type(node) is not int:
            if len(loop_node) > 0:
                graph.add_nodes_from(
                    [(node_cnt, {
                        "type": "loop",
                        "pos": loop_node,
                        "color": "blue"
                    })]
                )
                graph.add_edge(final_node, node_cnt)
                final_node = node_cnt
                node_cnt += 1
                loop_node = []
                
            last_node_cnt = node_cnt
            sub_graph, node_cnt = stemloop_graph(node_cnt, node)
            graph.add_nodes_from(sub_graph.nodes(data=True))
            graph.add_edges_from(sub_graph.edges)
            graph.add_edge(last_node_cnt-1, last_node_cnt)
            final_node = last_node_cnt
    graph.add_nodes_from([
        (-2, {"type": "3\'", "pos": [], "color" : "green"})
    ])
    graph.add_edge(-2, final_node)
    return graph

def stemloop_graph(node_cnt, org_node):
    graph = networkx.Graph()
    first_node = node_cnt
    graph.add_nodes_from(
        [(node_cnt,
        {
            "type": "stem",
            "pos": org_node.helice,
            "color": "red"
        })]
    )
    node_cnt += 1
    loop_node = []
    final_node = first_node
    for i, node in zip(range(len(org_node.loop)), org_node.loop):
        if type(node) is int:
            loop_node.append(node)
            if i == len(org_node.loop) - 1:
                graph.add_nodes_from(
                    [(node_cnt, {
                        "type": "loop",
                        "pos": loop_node,
                        "color": "blue"
                    })]
                )
                # if final_node > 0:
                graph.add_edge(final_node, node_cnt)
                final_node = node_cnt
                loop_node = []
                node_cnt += 1
        if type(node) is not int:
            if len(loop_node) > 0:
                graph.add_nodes_from(
                    [(node_cnt, {
                        "type": "loop",
                        "pos": loop_node,
                        "color": "blue"
                    })]
                )
                # if final_node > 0:
                graph.add_edge(final_node, node_cnt)
                final_node = node_cnt
                loop_node = []
                node_cnt += 1
            last_node_cnt = node_cnt
            sub_graph, node_cnt = stemloop_graph(node_cnt, node)
            graph.add_nodes_from(sub_graph.nodes(data=True))
            graph.add_edges_from(sub_graph.edges)
            graph.add_edge(last_node_cnt-1, last_node_cnt)
            final_node = last_node_cnt
    graph.add_edge(first_node, final_node)
    return graph, node_cnt


def get_relative_node(graph, pos):
    relative_pos = []
    relative_nodes = []
    pos_type = None
    center_node = None
    center_pos = []
    for node in graph.nodes():
        node_type = graph.nodes[node]["type"]
        if node_type == "loop":
            if pos in graph.nodes[node]["pos"]:
                relative_nodes = [node]
                pos_type = node_type
                center_node = node
                center_pos = graph.nodes[node]["pos"]
                break
        if node_type == "stem":
            near_pos = []
            for helix in graph.nodes[node]["pos"]:
                near_pos += helix 
            if pos in near_pos:
                relative_nodes = [node]
                pos_type = node_type
                center_node = node
                center_pos = near_pos
                break
    
    relative_nodes += graph.neighbors(relative_nodes[0])

    relative_nodes_out = []
    for i, re_node in zip(range(len(relative_nodes)), relative_nodes):
        if graph.nodes[re_node]["type"] == "loop":
            relative_pos += graph.nodes[re_node]["pos"]
            relative_nodes_out.append(re_node)
        # elif pos_type == "loop" or i == 0:
        elif i == 0:
            re_pos = []
            for helix in graph.nodes[re_node]["pos"]:
                re_pos += helix
            # re_pos = [min(helix) for helix in graph.nodes[re_node]["pos"]]
            relative_pos += re_pos
            relative_nodes_out.append(re_node)

    return relative_pos, relative_nodes_out, center_node, center_pos


class RNATree():
    def __init__(self, dotB):
        self.dotB = dotB
        self.length = len(dotB)
        self.branch_log = {}
        self.external_loop = []
        self.data = [{
            'name': str(0) + '-' + str(self.length-1),
            'children': [

            ]
        }]

        self.main_branch = self.Base_Branch(level=-1)
        self.main_branch.set_start(0)
        self.main_branch.set_end(self.length - 1)

        self.branch_log['-1'] = [self.main_branch]

    def branch_login(self, branch):
        level_str = str(branch.get_level())
        if level_str in self.branch_log.keys():
            self.branch_log[level_str] += [branch]
        else:
            self.branch_log[level_str] = [branch]

    class Base_Branch():
        def __init__(self, level=0):
            self.helice = []
            self.loop = []
            # self.bracket_stack = Stack()
            self.level = level
            self.start = None
            self.end = None
            self.n_sub = 0
            self.depth = 0
            self.settled = False
            self.settled_seq = None

        def __len__(self,):
            l = 0
            l += 2 * len(self.helice)
            for node in self.loop:
                if type(node) == int:
                    l += 1
                else:
                    l += len(node)
            return l

        def add_helice(self, pair):
            self.helice.insert(0, pair)

        def add_loop_node(self, node):
            self.loop.append(node)

        def set_helice(self, helice):
            self.helice = helice

        def set_loop(self, loop):
            self.loop = loop

        def get_level(self):
            return self.level

        def set_start(self, start):
            self.start = start

        def set_end(self, end):
            self.end = end

        def scope(self):
            return self.start, self.end

        def get_loop(self):
            return self.loop

        def get_helice(self):
            return self.helice

        def get_sub_branch(self):
            sub_branch = []
            for node in self.loop:
                if type(node) != int:
                    sub_branch.append(node)
            return sub_branch

        def add_sub_cnt(self):
            self.n_sub += 1

        def set_sub_cnt(self, n):
            self.n_sub = n

        def set_depth(self, depth):
            self.depth = depth

        def settle(self, seq_base):
            self.settled = True
            self.settled_seq = seq_base

        def reset(self):
            self.settled = False
            self.settled_seq = None

    def branch_serch(self, start, end, dotB, level=0):
        branch = self.Base_Branch(level)
        branch.set_start(start)
        # 一定是从(开始
        open_stack = Stack()
        open_stack.push(start)
        point = start + 1
        last_place = '('
        while point <= end and open_stack.size() > 0:
            if dotB[point] == '(':
                if last_place == '(':
                    open_stack.push(point)
                    last_place = '('
                elif last_place == '.':
                    sub, point = self.branch_serch(point, end, dotB, level=branch.get_level()+1)
                    branch.add_loop_node(sub)
                    branch.add_sub_cnt()
                    last_place = '.'
                elif last_place == ')':
                    sub, point = self.branch_serch(point, end, dotB, level=branch.get_level()+1)
                    branch.add_loop_node(sub)
                    branch.add_sub_cnt()
                    last_place = '.'
            elif dotB[point] == '.':
                if last_place == '(':
                    branch.add_loop_node(point)
                    last_place = '.'
                elif last_place == '.':
                    branch.add_loop_node(point)
                    last_place = '.'
                elif last_place == ')':
                    
                    if open_stack.size() > 0:
                        branch_tmp = self.Base_Branch(branch.get_level()+1)
                        branch_tmp.set_loop(branch.get_loop())
                        branch_tmp.set_helice(branch.get_helice())
                        s_tmp, e_tmp = tuple(branch.get_helice()[0])
                        branch_tmp.set_start(s_tmp)
                        branch_tmp.set_end(e_tmp)
                        branch_tmp.set_sub_cnt(branch.n_sub)
                        branch.set_loop([])
                        branch.set_helice([])
                        branch.add_loop_node(branch_tmp)
                        branch.set_sub_cnt(1)
                        last_place = '.'
                    else:
                        pass
                    branch.add_loop_node(point)
            elif dotB[point] == ')':
                if last_place == '(':
                    raise ValueError('() problem!')
                elif last_place == '.':
                    helice_open = open_stack.pop()
                    helice_close = point
                    branch.add_helice([helice_open, helice_close])
                    last_place = ')'
                elif last_place == ')':
                    helice_open = open_stack.pop()
                    helice_close = point
                    branch.add_helice([helice_open, helice_close])
                    last_place = ')'
            point += 1

        branch.set_end(point - 1)
        self.branch_login(branch)

        return branch, point - 1

    def get_scope(self):
        return 0, self.length - 1

    def external_loop_create(self):
        start = 0
        end = self.length - 1
        point = start
        while point <= end:
            if self.dotB[point] == '.':
                self.external_loop.append(point)
            else:
                branch, point = self.branch_serch(point, end, self.dotB, level=0)
                self.external_loop.append(branch)
            point += 1

    def check_depth(self, branch):
        sub_branchs = []
        for node in branch.get_loop():
            if type(node) != int:
                sub_branchs.append(node)
        if len(sub_branchs) == 0:
            branch.set_depth(0)
            return 0
        else:
            depths = list(map(self.check_depth, sub_branchs))
            depth = 1 + max(depths)
            branch.set_depth(depth)
            return depth

    def tree_depth(self):
        sub_branchs = []
        for node in self.external_loop:
            if type(node) != int:
                sub_branchs.append(node)
        for b in sub_branchs:
            self.check_depth(b)

    def traverse_branch(self, branch):
        scope = branch.scope()
        scope_str = str(scope[0]) + '-' + str(scope[1]) + ': ' + str(branch.depth)
        data = {
            'name': scope_str
        }
        base_loop = True
        for node in branch.get_loop():
            if type(node) != int:
                base_loop = False
                if 'children' not in data.keys():
                    data['children'] = [self.traverse_branch(node)]
                else:
                    data['children'] += [self.traverse_branch(node)]
        return data

    def show(self, root):
        tree_root = root + '/tree/'
        max_level = len(self.branch_log.items())
        data = [{
            'name': str(0) + '-' + str(self.length-1),
            'children': []
        }]
        for node in self.external_loop:
            if type(node) != int:
                data[0]['children'] += [self.traverse_branch(node)]

        c = (
            Tree().add(
                "",
                data,
                collapse_interval=100,
                layout='orthogonal'
            ).set_global_opts(title_opts=opts.TitleOpts(title="RNATree"))
        )
        c.render(tree_root + 'tree.html')
        # make_snapshot(driver, c.render(), tree_root + 'tree.png')

    def reset(self):
        for value in self.branch_log.values():
            for branch in value:
                branch.reset()


def get_variable_pos(aim_dotB, real_dotB, structure_graph):
    l = len(aim_dotB)
    diff_pos_list = [i for i in range(l) if aim_dotB[i] != real_dotB[i]]

    variable_pos_list = []
    center_pos_list = []
    relative_node_list = []
    for diff_pos in diff_pos_list:
        if diff_pos not in center_pos_list:
            relative_pos, relative_nodes, center_node, center_pos = get_relative_node(structure_graph, diff_pos)
            variable_pos_list += relative_pos
            center_pos_list += center_pos
            relative_node_list += relative_nodes
    variable_pos_list = list(
        set(variable_pos_list)
    )

    relative_node_list = list(
        set(relative_node_list)
    )

    return variable_pos_list, relative_node_list


def get_variable_pos_from_graph(org_graph, structure_graph):
    l = len(org_graph.y["dotB"])
    diff_adj = org_graph.y['aim_adj'] - org_graph.y['real_adj']
    
    diff_pos_list = [i for i in range(l) if not np.all(diff_adj[i] == 0)]

    variable_pos_list = []
    center_pos_list = []
    relative_node_list = []
    for diff_pos in diff_pos_list:
        if diff_pos not in center_pos_list:
            relative_pos, relative_nodes, center_node, center_pos = get_relative_node(structure_graph, diff_pos)
            variable_pos_list += relative_pos
            center_pos_list += center_pos
            relative_node_list += relative_nodes
    variable_pos_list = list(
        set(variable_pos_list)
    )

    relative_node_list = list(
        set(relative_node_list)
    )

    return variable_pos_list, relative_node_list


def get_freeze_pos_from_structure_graph(rna_graph, structure_graph):
    variable_poses, variable_nodes = get_variable_pos_from_graph(rna_graph, structure_graph)
    l = len(rna_graph.y['dotB'])

    freeze_loacations = [i for i in range(l) if i not in variable_poses]

    freeze_out_s = []
    freeze_out_p = []
    if len(rna_graph.y['pair_index']) == 0:
        pair_index_0 = []
    else:
        pair_index_0 = torch.tensor(rna_graph.y['pair_index'])[:, 0].view(-1,)
        max_loc = torch.max(pair_index_0)

    for freeze_loc in freeze_loacations:
        if rna_graph.y['dotB'][freeze_loc] == '.':
            loc = torch.nonzero(torch.tensor(rna_graph.y['single_index']) == freeze_loc).item()
            freeze_out_s.append(loc)
        else:
            if freeze_loc <= max_loc and freeze_loc in pair_index_0:
                loc = torch.nonzero(pair_index_0 == freeze_loc).item()
                freeze_out_p.append(loc)

    return freeze_out_s, freeze_out_p












