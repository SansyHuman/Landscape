import csv
import numpy as np


def inconsistents_parser(filename: str, inconsistents_path: str) -> None:
    """
    Parse consistent theories and inconsistent theories
    :param filename: File of consistent theories
    :param inconsistents_path: Folder of inconsistent theories
    :return:
    """
    csv.field_size_limit(np.iinfo(np.int32).max)

    data = None
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)

    field_content_index, a_index, c_index = -1, -1, -1
    for i in range(len(data[0])):
        if data[0][i] == "Name":
            field_content_index = i
        elif data[0][i] == "CentralChargeA":
            a_index = i
        elif data[0][i] == "CentralChargeC":
            c_index = i

    field_contents_index = dict()
    field_contents = []
    a_charges = []
    c_charges = []

    for i in range(1, len(data)):
        field_content = data[i][field_content_index]
        a, c = float(data[i][a_index]), float(data[i][c_index])

        if field_content not in field_contents_index:
            field_contents_index[field_content] = len(field_contents_index)
        field_contents.append(field_contents_index[field_content])
        a_charges.append(a)
        c_charges.append(c)

    # [GaugeGroup, GroupSize,
    index_group_fields = dict()