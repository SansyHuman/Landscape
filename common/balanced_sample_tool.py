import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import math


class TheorySampler():
    """
    Class to sample theories evenly.
    """
    def __init__(self, filename: str):
        self.filename = filename
        self.df = pl.read_csv(self.filename)
        print(self.df)

    def a_range(self) -> float:
        """
        Gets the range of a central charge
        :return: min_a, max_a
        """
        min_a = self.df["CentralChargeA"].min()
        max_a = self.df["CentralChargeA"].max()
        return min_a, max_a

    def c_range(self) -> float:
        """
        Gets the range of a central charge
        :return: min_c, max_c
        """
        min_c = self.df["CentralChargeC"].min()
        max_c = self.df["CentralChargeC"].max()
        return min_c, max_c

    def draw_central_charge_histogram(self, charge_interval: float) -> None:
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (16, 12)
        plt.rcParams['font.size'] = 15

        min_a, max_a = self.a_range()
        min_c, max_c = self.c_range()
        min_a = math.floor(min_a / charge_interval) * charge_interval
        min_c = math.floor(min_c / charge_interval) * charge_interval

        a_bins = math.ceil((max_a - min_a) / charge_interval)
        c_bins = math.ceil((max_c - min_c) / charge_interval)

        fig, ax = plt.subplots(1, 2, squeeze=True)
        fig.suptitle("Central Charge Distribution")

        ax[0].set_title("A charge")
        ax[0].hist(x=self.df.select(pl.col("CentralChargeA")), bins=a_bins, range=(min_a, min_a + charge_interval * a_bins))
        ax[0].set_xlabel("a")
        ax[0].set_ylabel("count")

        ax[1].set_title("C charge")
        ax[1].hist(x=self.df.select(pl.col("CentralChargeC")), bins=c_bins, range=(min_c, min_c + charge_interval * c_bins))
        ax[1].set_xlabel("c")
        ax[1].set_ylabel("count")

        plt.show()
