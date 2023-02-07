from collections import defaultdict

from .validation import ValidationLayer


class GPUUtilization(ValidationLayer):
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
                
    
    def report(self):
        indent = '    '
        report = [
            'Accelerator Utilization',
            '-----------------------'
        ]
        
        for device in self.load_avg.keys():
            load = self.load_avg.get(device, 0) / self.count
            mem = self.mem_avg.get(device, 0) / self.count
            errors = []
            
            if load < self.load_threshold:
                errors.append(f'{indent * 2}* load is below threshold {load} < {self.load_threshold}')
            
            if mem < self.mem_threshold:
                errors.append(f'{indent * 2}* used memory is below threshold {mem} < {self.mem_threshold}')
                
            if errors:
                report.append(f'{indent}Device {device}')
                report.extend(errors)
                
                
        if report:
            report = [
                'Accelerator Utilization',
                '-----------------------'
            ] + report
            print('\n'.join(report))
            
        return not report
