import numpy as np
from typing import Callable
from abc import abstractmethod
from ._util import _Array


class _Indicator(_Array):
    def decimals(self, round:int):
        self[:] = np.around(self, decimals=round)
        return self

    def shifting(self, num:int):
        self[:] = np.roll(self, num)
        self[..., :num] = np.nan
        return self

    def color(self, color):
        self._opts['color'] = color
        return self

    def name(self, name:str):
        self._opts['name'] = name
        return self

    def plot(self, enable:bool):
        self._opts['plot'] = enable

        if self._opts['plot'] and self._opts['overlay'] is None and self._opts['auto_overlay'] is not None:
            self._opts['overlay'] = self._opts['auto_overlay']

        return self

    def scatter(self, enable:bool):
        self._opts['scatter'] = enable
        return self

    def overlay(self, enable:bool):
        self._opts['overlay'] = enable
        self._opts['columnar'] = False
        if self._opts['plot'] and self._opts['overlay'] is None and self._opts['auto_overlay'] is not None:
            self._opts['overlay'] = self._opts['auto_overlay']

        return self

    def columnar(self, index:int):
        self._opts['overlay'] = False
        self._opts['columnar'] = index
        if self._opts['plot'] and self._opts['overlay'] is None and self._opts['auto_overlay'] is not None:
            self._opts['overlay'] = self._opts['auto_overlay']

        return self

class IndicatorData(_Indicator):
    pass