from torch_geometric.utils import from_networkx
from torch_geometric.data import Data

from common.utils import *
from common.inconsistents_parser import *
import networkx as nx



def serialize_w_terms(w: str):
    w_terms = w[1:-1].split(',')
    w_serial = []
    for term in w_terms:
        term_data = []
        ops = term.strip().split('*')
        for op in ops:
            op_exp = op.strip().split('^')
            index_start, index_len = find_number(op_exp[0])

            op_letter = op_exp[0][:index_start].strip()
            index = int(op_exp[0][index_start:index_start + index_len])
            exponent = 1
            if len(op_exp) == 2:
                exponent = int(op_exp[1].strip())

            term_data.append((op_letter, index, exponent))

        w_serial.append(term_data)

    return w_serial


def build_dynkin_diagram(graph: nx.MultiGraph, ade_class: int, rank: int) -> None:
    """
    Build a dynkin diagram.
    :param graph: Graph to build dynkin diagram. The graph is cleared before build.
    :param ade_class: ADE class of the algebra. A=1, B=2, C=3, D=4, E=5, F=6, G=7.
    :param rank: Rank of the algebra.
    :return:
    """
    # Dynkin diagram attributes
    # [simple_root, short, mark, comark]
    # simple_root: index of the simple root
    # short: 0 if long root, 1 if short root
    # mark, comark: mark and comark of the simple root

    # equivalent algebra check
    if (ade_class == 2 and rank == 1) or (ade_class == 3 and rank == 1):
        # A1=B1=C1
        ade_class = 1
        rank = 1
    elif ade_class == 3 and rank == 2:
        # B2=C2
        ade_class = 2
        rank = 2
    elif ade_class == 4 and rank == 3:
        # A3=D3
        ade_class = 1
        rank = 3

    if ade_class == 1:  # An
        graph.add_nodes_from(
            [(i, {"simple_root": i, "short": 0, "mark": 1, "comark": 1})
             for i in range(1, rank + 1)]
        )
        graph.add_edges_from(
            [(i, i + 1) for i in range(1, rank)]
        )
    elif ade_class == 2:  # Bn
        assert rank >= 2
        if rank == 2:
            graph.add_nodes_from(
                [
                    (1, {"simple_root": 1, "short": 0, "mark": 1, "comark": 1}),
                    (2, {"simple_root": 2, "short": 1, "mark": 2, "comark": 1})
                ]
            )
            graph.add_edge(1, 2)
            graph.add_edge(1, 2)
        else:
            graph.add_nodes_from(
                [(1, {"simple_root": 1, "short": 0, "mark": 1, "comark": 1})]
                + [
                    (i, {"simple_root": i, "short": 0, "mark": 2, "comark": 2})
                    for i in range(2, rank)
                ]
                + [(rank, {"simple_root": rank, "short": 1, "mark": 2, "comark": 1})]
            )
            graph.add_edges_from(
                [(i, i + 1) for i in range(1, rank)]
            )
            graph.add_edge(rank - 1, rank)
    elif ade_class == 3:  # Cn
        assert rank >= 3
        graph.add_nodes_from(
            [
                (i, {"simple_root": i, "short": 1, "mark": 2, "comark": 1})
                for i in range(1, rank)
            ]
            + [(rank, {"simple_root": rank, "short": 0, "mark": 1, "comark": 1})]
        )
        graph.add_edges_from(
            [(i, i + 1) for i in range(1, rank - 1)]
        )
        graph.add_edge(rank - 1, rank)
        graph.add_edge(rank - 1, rank)
    elif ade_class == 4:  # Dn
        assert rank == 2 or rank >= 4
        if rank == 2:  # D2=A1 x A1
            graph.add_nodes_from(
                [(i, {"simple_root": i, "short": 0, "mark": 1, "comark": 1})
                 for i in range(1, 3)]
            )
        else:
            graph.add_nodes_from(
                [(1, {"simple_root": 1, "short": 0, "mark": 1, "comark": 1})]
                + [
                    (i, {"simple_root": i, "short": 0, "mark": 2, "comark": 2})
                    for i in range(2, rank - 1)
                ]
                + [
                    (rank - 1, {"simple_root": rank - 1, "short": 0, "mark": 1, "comark": 1}),
                    (rank, {"simple_root": rank, "short": 0, "mark": 1, "comark": 1})
                ]
            )
            graph.add_edges_from(
                [(i, i + 1) for i in range(1, rank - 1)]
            )
            graph.add_edge(rank - 2, rank)
    elif ade_class == 5:  # En
        assert 6 <= rank <= 8
        if rank == 6:
            graph.add_nodes_from(
                [
                    (1, {"simple_root": 1, "short": 0, "mark": 1, "comark": 1}),
                    (2, {"simple_root": 2, "short": 0, "mark": 2, "comark": 2}),
                    (3, {"simple_root": 3, "short": 0, "mark": 3, "comark": 3}),
                    (4, {"simple_root": 4, "short": 0, "mark": 2, "comark": 2}),
                    (5, {"simple_root": 5, "short": 0, "mark": 1, "comark": 1}),
                    (6, {"simple_root": 6, "short": 0, "mark": 2, "comark": 2})
                ]
            )
            graph.add_edges_from(
                [(1, 2), (2, 3), (3, 4), (4, 5), (3, 6)]
            )
        elif rank == 7:
            graph.add_nodes_from(
                [
                    (1, {"simple_root": 1, "short": 0, "mark": 2, "comark": 2}),
                    (2, {"simple_root": 2, "short": 0, "mark": 3, "comark": 3}),
                    (3, {"simple_root": 3, "short": 0, "mark": 4, "comark": 4}),
                    (4, {"simple_root": 4, "short": 0, "mark": 3, "comark": 3}),
                    (5, {"simple_root": 5, "short": 0, "mark": 2, "comark": 2}),
                    (6, {"simple_root": 6, "short": 0, "mark": 1, "comark": 1}),
                    (7, {"simple_root": 7, "short": 0, "mark": 2, "comark": 2})
                ]
            )
            graph.add_edges_from(
                [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (3, 7)]
            )
        else:
            graph.add_nodes_from(
                [
                    (1, {"simple_root": 1, "short": 0, "mark": 2, "comark": 2}),
                    (2, {"simple_root": 2, "short": 0, "mark": 3, "comark": 3}),
                    (3, {"simple_root": 3, "short": 0, "mark": 4, "comark": 4}),
                    (4, {"simple_root": 4, "short": 0, "mark": 5, "comark": 5}),
                    (5, {"simple_root": 5, "short": 0, "mark": 6, "comark": 6}),
                    (6, {"simple_root": 6, "short": 0, "mark": 4, "comark": 4}),
                    (7, {"simple_root": 7, "short": 0, "mark": 2, "comark": 2}),
                    (8, {"simple_root": 8, "short": 0, "mark": 3, "comark": 3})
                ]
            )
            graph.add_edges_from(
                [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (5, 8)]
            )
    elif ade_class == 6:  # Fn
        assert rank == 4
        graph.add_nodes_from(
            [
                (1, {"simple_root": 1, "short": 0, "mark": 2, "comark": 2}),
                (2, {"simple_root": 2, "short": 0, "mark": 3, "comark": 3}),
                (3, {"simple_root": 3, "short": 1, "mark": 4, "comark": 2}),
                (4, {"simple_root": 4, "short": 1, "mark": 2, "comark": 1})
            ]
        )
        graph.add_edges_from(
            [(1, 2), (2, 3), (2, 3), (3, 4)]
        )
    elif ade_class == 7:  # Gn
        assert rank == 2
        graph.add_nodes_from(
            [
                (1, {"simple_root": 1, "short": 0, "mark": 2, "comark": 2}),
                (2, {"simple_root": 2, "short": 1, "mark": 3, "comark": 1})
            ]
        )
        graph.add_edges_from(
            [(1, 2), (1, 2), (1, 2)]
        )
    else:
        assert False


matter_fields = ['q', 'qb', 'phi', 'S', 'Sb', 'A', 'Ab']


class Superpotential:
    def __init__(self, theory: str, superpotential: str):
        """
        Create superpotential object.
        :param theory: Theory of the superpotential.
        :param superpotential: Raw superpotential string.
        """
        self.theory = serialize_theory_name(theory)
        self.superpotential = serialize_w_terms(superpotential)
        self.__build_graph()

    def __build_graph(self) -> None:
        self.dynkin_diagram = nx.MultiGraph()
        self.superpotential_graph = nx.MultiDiGraph()

        ade_class = self.theory[0]
        rank = self.theory[1]
        build_dynkin_diagram(self.dynkin_diagram, ade_class, rank)
        self.__build_superpotential_graph()

    def __build_superpotential_graph(self) -> None:
        # superpotential attributes
        # [node_type, matter, index]
        # node_type
        # 0: peripheral node connected to central nodes
        # 1: central node which represents matter fields
        # 2: central node which represents flipping fields
        # 3: central node which represents terms in superpotential
        # matter
        # for peripheral nodes it is always 0
        # for matter fields 1: fundamental, 2: antifundamental, 3: adjoint, 4: rank-2 symmetric tensor,
        # 5: conjugate of rank-2 symmetric tensor, 6: rank-2 antisymmetric tensor, 7: conjugate of rank-2 antisymmetric tensor
        # for flipping fields 1: M, 2: X
        # for terms it is always 0
        # index is the index of the fields of peripheral nodes connected to central field nodes starting from 1
        # for non-peripheral nodes and peripheral terms index is always 0

        M_num = 0
        X_num = 0
        for term in self.superpotential:
            for op, index, _ in term:
                if op == 'M' and index > M_num:
                    M_num = index
                elif op == 'X' and index > X_num:
                    X_num = index

        term_num = len(self.superpotential)

        for i in range(len(matter_fields)):
            field = matter_fields[i]
            field_num = self.theory[i + 2]
            if field_num > 0:
                self.superpotential_graph.add_node(field, node_type=1, matter=i + 1, index=0)
                self.superpotential_graph.add_nodes_from(
                    [
                        (f'{field}{j}', {'node_type': 0, 'matter': 0, 'index': j}) for j in range(1, field_num + 1)
                    ]
                )
                self.superpotential_graph.add_edges_from(
                    [
                        (field, f'{field}{j}') for j in range(1, field_num + 1)
                    ]
                )

        if M_num > 0:
            self.superpotential_graph.add_node('M', node_type=2, matter=1, index=0)
            self.superpotential_graph.add_nodes_from(
                [
                    (f'M{i}', {'node_type': 0, 'matter': 0, 'index': i}) for i in range(1, M_num + 1)
                ]
            )
            self.superpotential_graph.add_edges_from(
                [
                    ('M', f'M{i}') for i in range(1, M_num + 1)
                ]
            )

        if X_num > 0:
            self.superpotential_graph.add_node('X', node_type=2, matter=2, index=0)
            self.superpotential_graph.add_nodes_from(
                [
                    (f'X{i}', {'node_type': 0, 'matter': 0, 'index': i}) for i in range(1, X_num + 1)
                ]
            )
            self.superpotential_graph.add_edges_from(
                [
                    ('X', f'X{i}') for i in range(1, X_num + 1)
                ]
            )

        if term_num > 0:
            self.superpotential_graph.add_node('Term', node_type=3, matter=0, index=0)
            self.superpotential_graph.add_nodes_from(
                [
                    (f'Term{i}', {'node_type': 0, 'matter': 0, 'index': 0}) for i in range(1, term_num + 1)
                ]
            )
            self.superpotential_graph.add_edges_from(
                [
                    ('Term', f'Term{i}') for i in range(1, term_num + 1)
                ]
            )

        for i in range(len(self.superpotential)):
            term = self.superpotential[i]
            term_node = f'Term{i + 1}'

            for op, index, exponent in term:
                op_node = f'{op}{index}'
                self.superpotential_graph.add_edges_from(
                    [
                        (term_node, op_node) for _ in range(exponent)
                    ]
                )

    def set_theory(self, theory: str) -> None:
        """
        Sets the theory of the superpotential.
        :param theory: Theory of the superpotential.
        """
        self.theory = serialize_theory_name(theory)
        self.dynkin_diagram.clear()

        ade_class = self.theory[0]
        rank = self.theory[1]
        build_dynkin_diagram(self.dynkin_diagram, ade_class, rank)

    def set_superpotential(self, superpotential: str) -> None:
        """
        Sets the superpotential. Note that if the superpotential and theory does not match,
        it will cause error.
        :param superpotential: Raw superpotential string.
        """
        self.superpotential = serialize_w_terms(superpotential)
        self.superpotential_graph.clear()

        self.__build_superpotential_graph()

    def get_theory_data(self) -> Data:
        """
        Gets the data of the theory by dynkin diagram graph with node attribute
        [short, mark, comark].
        :return: PyG graph data of the dynkin diagram of the theory.
        """
        return from_networkx(self.dynkin_diagram, group_node_attrs=['short', 'mark', 'comark'])

    def get_superpotential_data(self) -> Data:
        """
        Gets the data of the superpotential by graph with node attribute
        [node_type, matter, index].
        :return: PyG graph data of the superpotential graph.
        """
        return from_networkx(self.superpotential_graph, group_node_attrs=['node_type', 'matter', 'index'])