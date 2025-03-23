from pymonntorch import *
from typing import Sequence

class CustomPattern(Behavior):
    def __init__(
        self,
        iters: int,
        pattern: Sequence[int],
        amp: int,
        *args, 
        **kwargs
    ):
        super().__init__(iters=iters, pattern=pattern, amp=amp, *args, **kwargs)
    
    def initialize(self, neurons):
        self.iters = self.parameter("iters", None, required=True)
        self.pattern = self.parameter("pattern", None, required=True)
        self.amp = self.parameter("amp", None, required=True)
        neurons.I_pattern = neurons.vector()
        neurons.I = neurons.vector()
        return super().initialize(neurons)
    
    def forward(self, neurons):
        neurons.I = neurons.vector()
        neurons.I_pattern = neurons.vector()
        if self.iters:
            neurons.I_pattern[self.pattern] = self.amp
            neurons.I[self.pattern] = self.amp
        self.iters -= 1
        return super().forward(neurons)


