from common.utils import *

marginal = Monomial('t', 6)
spectral = Monomial('y', 0)


class SuperConformalIndex:
    """
    Class which contains the information of a superconformal index.
    """
    def __init__(self, index: str) -> None:
        """
        Converts the index string into a polynomial object and parses information.
        :param index: index string.
        """
        self.index = to_poly(index)
        # terms with exponent of t less than 6
        self.short_index = Polynomial(
            *filter(lambda term: term.exponent('t') < 6,
                    filter(lambda term2: term2.exponent('t') is not None, self.index.terms))
        )
        # # of marginal operators - rank of IR flavor symmetry
        self.num_dim3_minus_f = 0
        marginal_term = self.index.find_with(marginal)
        if len(marginal_term) > 0:
            self.num_dim3_minus_f = round(self.index.find_with(marginal)[0].coefficient)

        # dimensions of operators in the order of increasing
        self.dims: list[float] = []
        # dimensions of relevant operators in the order of increasing
        self.relevant_dims: list[float] = []
        # the number of operators with each dimensions
        self.relevant_spectrum: dict[float, int] = dict()
        # total number of relevant operators
        self.num_relevant_ops = 0

        terms_spectrum = self.index.find_with(spectral)
        tmp_dims = set()
        for term in terms_spectrum:
            dim = term.exponent('t')
            if dim is not None:
                dim = dim / 2.0 # t^3R, dim = 3R/2
                cnt = round(term.coefficient)

                if dim < 3.0: # relevant
                    if dim in self.relevant_spectrum:
                        self.relevant_spectrum[dim] += cnt
                    else:
                        self.relevant_dims.append(dim)
                        self.relevant_spectrum[dim] = cnt
                    self.num_relevant_ops += cnt

                tmp_dims.add(dim)

        self.dims = list(tmp_dims)
        self.dims.sort()
        self.relevant_dims.sort()
        # smallest dimension among all operators
        self.smallest_dim = self.relevant_dims[0]
