import math
from typing import Union
from torch.utils.data import Dataset
import torch


def prime_numbers(n: int) -> list[int]:
    """
    Gets a list of prime number.
    :param n: the number of prime numbers.
    :return: a list of prime numbers.
    """
    if n <= 0:
        return []
    primes = [2]
    for i in range(1, n):
        prime = primes[-1] + 1
        while True:
            is_prime = True
            for p in primes:
                if p > math.sqrt(prime):
                    break
                if prime % p == 0:
                    is_prime = False
                    break

            if is_prime:
                primes.append(prime)
                break

            prime += 1

    return primes


def median_sorted(data):
    """
    Gets median from a sorted list.
    :param data: sorted data
    :return: median of the data
    """
    n = len(data)
    if n % 2 == 0:
        return (data[n // 2 - 1] + data[n // 2]) / 2
    else:
        return data[(n - 1) // 2]


class Monomial:
    """
    A class of monomials.
    """
    def __init__(self, indet: str, expo: float):
        """
        Construct a monomial.
        :param indet: indeterminate letter.
        :param expo: exponent of the monomial.
        """
        self.indeterminate = indet
        self.exponential = expo

    def __str__(self):
        if self.exponential == 1:
            return self.indeterminate

        return f'{self.indeterminate}^{self.exponential}'

    def __eq__(self, other):
        return self.indeterminate == other.indeterminate and self.exponential == other.exponential


class Term:
    """
    A class of terms.
    """
    def __init__(self, coeff: float, *args: Monomial):
        """
        Construct a term.
        :param coeff: Coefficient of the term.
        :param args: Monomials of the term.
        """
        self.coefficient = coeff
        self.monomials: dict[str, Monomial] = dict()
        for arg in args:
            self.__add_monomial(arg)

    def __add_monomial(self, monomial: Monomial) -> None:
        if monomial.indeterminate in self.monomials:
            self.monomials[monomial.indeterminate].exponential += monomial.exponential
        else:
            self.monomials[monomial.indeterminate] = monomial

    def indeterminates(self) -> set[str]:
        """
        Returns a set of indeterminates.
        :return: a set of indeterminates.
        """
        return set(self.monomials.keys())

    def exponent(self, indeterminate: str) -> Union[float, None]:
        """
        Returns the exponent of the indeterminate.
        :param indeterminate: indeterminate.
        :return: exponent. If the indeterminate does not exist, return None.
        """
        if indeterminate in self.monomials:
            return self.monomials[indeterminate].exponential

        return None

    def __str__(self) -> str:
        indeterminates = list(self.monomials.keys())
        indeterminates.sort()
        term = ""
        if self.coefficient == -1:
            term += '-'
        elif self.coefficient != 1:
            term += f'{self.coefficient}*'
        else:
            if len(self.monomials) == 0:
                term += '1'
                return term

        for letter in indeterminates:
            mon = self.monomials[letter]
            term += f'{mon}*'

        return term[:-1]

    def __contains__(self, item: Monomial) -> bool:
        if item.indeterminate in self.monomials:
            return item == self.monomials[item.indeterminate]

        return False


class Polynomial:
    """
    A class of polynomials.
    """
    def __init__(self, *args: Term):
        """
        Construct a polynomial.
        :param args: list of terms.
        """
        self.terms = list(args)

    def add_term(self, term: Term) -> None:
        """
        Adds a term to the polynomial.
        :param term: term to add.
        """
        self.terms.append(term)

    def find_with(self, mono: Monomial) -> list[Term]:
        """
        Finds all terms with the given monomial.
        :param mono: monomial to find.
        :return: list of terms with the given monomial.
        """
        result = []
        for term in self.terms:
            if mono.exponential == 0:
                if not (mono.indeterminate in term.indeterminates()):
                    result.append(term)
            elif mono in term:
                result.append(term)

        return result

    def __str__(self) -> str:
        poly = ""
        for term in self.terms:
            if term.coefficient < 0 and len(poly) != 0:
                poly = poly[:-1]
            poly += f'{term}+'
        if len(self.terms) != 0:
            return poly[:-1]

        return poly


def is_number(n: str) -> bool:
    """
    Checks if a string is a number.
    :param n: string to check.
    :return: True if the string is a number. Else, false.
    """
    try:
        float(n)
        return True
    except ValueError:
        return False


def to_poly(expr: str) -> Polynomial:
    """
    Converts the expression string to polynomial.
    :param expr: string of the expression.
    :return: polynomial.
    """

    if expr[0] != '-':
        expr = f'+{expr}'

    poly = Polynomial()

    while len(expr) > 0:
        next_expr_plus = expr.find('+', 1)
        next_expr_minus = expr.find('-', 1)
        next_expr = -1
        term_str = ''

        if next_expr_plus != -1:
            next_expr = next_expr_plus
        if next_expr_minus != -1:
            if next_expr != -1:
                next_expr = min(next_expr, next_expr_minus)
            else:
                next_expr = next_expr_minus

        if next_expr == -1:
            term_str = expr[:]
            expr = ''
        else:
            term_str = expr[:next_expr]
            expr = expr[next_expr:]

        fractions = term_str.split('/')
        numerator = fractions[0].strip()
        numerator = numerator.replace('(', '')
        numerator = numerator.replace(')', '')

        denominator = None
        if len(fractions) > 1:
            denominator = fractions[1].strip()
            if denominator[0] == '(' and denominator[-1] == ')':
                denominator = denominator[1:-1]
            if denominator[0] != '-':
                denominator = f'+{denominator}'

        coeff, monomials = __make_monomials(numerator)
        if denominator is not None:
            _, denominators = __make_monomials(denominator)
            for denom in denominators:
                denom.exponential = -denom.exponential
            monomials += denominators

        poly.add_term(Term(coeff, *monomials))

    return poly


def __make_monomials(term: str) -> tuple[float, list[Monomial]]:
    """
    Returns the coefficient and list of monomials
    """
    multiples = [s.strip() for s in term.split('*')]
    coefficient = 1
    monomials = []

    if is_number(multiples[0]):
        coefficient = float(multiples[0])
        multiples = multiples[1:]
    else:
        if multiples[0][0] == '-':
            coefficient = -1
        multiples[0] = multiples[0][1:]

    for i in range(len(multiples)):
        multiple = multiples[i]
        indet_expo = [s.strip() for s in multiple.split('^')]
        if len(indet_expo) == 1:
            monomials.append(Monomial(indet_expo[0], 1))
        else:
            monomials.append(Monomial(indet_expo[0], float(indet_expo[1])))

    return coefficient, monomials


class GenericDataset(Dataset):
    """
    General dataset class.
    """
    def __init__(self, input_data, output_data):
        """
        Create a new dataset.
        :param input_data: input data
        :param output_data: output data
        """
        self.x_data = torch.tensor(input_data)
        self.y_data = torch.tensor(output_data)

    def __getitem__(self, index):
        return self.x_data[index].float(), self.y_data[index].float()

    def __len__(self):
        return self.x_data.shape[0]