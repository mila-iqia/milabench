from collections import defaultdict
import math

from .validation import ValidationLayer, Summary


class _Layer(ValidationLayer):
    """Makes sures the loss we receive is not Nan.
    
    Notes
    -----
    Show a warning if the loss is not decreasing.

    """

    def __init__(self, gv) -> None:
        super().__init__(gv)
        self.nan_count = 0
        self.previous_loss = defaultdict(float)
        self.increasing_loss= 0
    
    def on_event(self, pack, run, tag, keys, data):
        loss = data.get('loss')
        if loss is not None:
            prev = self.previous_loss[tag]
            
            self.nan_count += int(math.isnan(loss))
            self.previous_loss[tag] = loss
            
            if loss > prev:
                self.increasing_loss += 1
        
    def report(self, **kwargs):
        summary = Summary()

        with summary.section('Nan Check'):
            if self.increasing_loss:
                summary.add(f'* Loss increased {self.increasing_loss} times')
                
            if self.nan_count > 0:
                summary.add(f'* Loss was NaN {self.nan_count} times')
                
        summary.show()
        return self.nan_count == 0
