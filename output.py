from typing import List

import numpy


class Output:
    def __init__(self, predict_list: List[List[float]]):
        self.predict_list = predict_list

    def result(self):
        return 0 if numpy.sum(self.predict_list[0]) > numpy.sum(self.predict_list[1]) else 1

