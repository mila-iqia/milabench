from collections import defaultdict

from .validation import ValidationLayer, Summary


class _Layer(ValidationLayer):
    """Checks that GPU utilisation is > 0.01 and that memory used is above 50%.
    
    Notes
    -----
    This is a sanity check that only runs when enabled

    """
    
    def __init__(self, gv) -> None:
        super().__init__(gv)
        self.load_avg = defaultdict(float)
        self.mem_avg = defaultdict(float)

        self.count = 0
        self.mem_threshold = 0.50
        self.load_threshold = 0.01
    
    def on_event(self, pack, run, tags, keys, data):
        gpudata = data.get('gpudata')
        
        if gpudata is not None:

            for device, data in gpudata.items():
                self.load_avg[device] += data.get('load', 0)
                
                usage, total = data.get('memory', [0, 1])
                self.mem_avg[device] += usage / total
                
            self.count += 1
                
    
    def report(self, **kwargs):
        summary = Summary()
        with summary.section('Accelerator Utilization'):
 
            for device in self.load_avg.keys():
                load = self.load_avg.get(device, 0) / self.count
                mem = self.mem_avg.get(device, 0) / self.count
  
                with summary.section(f'Device {device}'):
                
                    if load < self.load_threshold:
                        summary.add(f'* load is below threshold {load} < {self.load_threshold}')
                    
                    if mem < self.mem_threshold:
                        summary.add(f'* used memory is below threshold {mem} < {self.mem_threshold}')
        
        summary.show()
        return summary.is_empty()