from torch_geometric.utils import from_networkx

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


def build_dynkin_diagram(graph: nx.MultiDiGraph, ade_class: int, rank: int) -> None:
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

    # Arrow goes from smaller index to larger index, and from long root to short root
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
        graph.add_edge(rank, rank - 1)
        graph.add_edge(rank, rank - 1)
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
        self.dynkin_diagram = nx.MultiDiGraph()
        self.superpotential_graph = nx.MultiDiGraph()

        ade_class = self.theory[0]
        rank = self.theory[1]
        build_dynkin_diagram(self.dynkin_diagram, ade_class, rank)
