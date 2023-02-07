from collections import defaultdict

from .validation import ValidationLayer, Summary


class _Layer(ValidationLayer):
    """Makes sure the events we are receiving are consistent with the planning method
    
    Notes
    -----
    Check that we are receiving loss from the right number of processes

    """

    def __init__(self, gv) -> None:
        super().__init__(gv)
        from ..gpu import get_gpu_info
        gpus = get_gpu_info().values()
        
        self.method = None
        self.gpus = len(gpus)
        self.njobs = 0
        self.configs = defaultdict(int)

    def on_event(self, pack, run, tag, keys, data):
        cfg = pack.config
        plan = cfg["plan"]
        method = plan["method"].replace('-', '_')
        
        self.method  = method
        self.njobs = plan.get('n', 0)
        
        assert method in ('per_gpu', 'njobs')
        
        loss = data.get('loss')
        if loss is not None:
            self.configs[f'{tag}-loss'] += 1
    
    def report(self, **kwargs):
        config_count = len(self.configs)
        
        summary = Summary()
        with summary.section('Planning Checks'):
            
            if self.method == 'njobs' and config_count != self.njobs:
                summary.add(f'* Wrong number of configs')
                
            if self.method == 'per_gpu' and config_count != self.gpus:
                summary.add(f'* Wrong number of configs')
        
            value = None
            for k, v in self.configs.items():
                if value is None:
                    value = v
                    
                elif value != v:
                    summary.add(f'* {k} sent {v} events, exepcted {value}')    
                
        summary.show()
        return summary.is_empty()
