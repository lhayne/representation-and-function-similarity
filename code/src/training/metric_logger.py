import os

class MetricLogger:
    def __init__(self):
        self.log = {}

    def add(self, name: str, epoch: int, it: int, value: float):
        self.log[(name, epoch, it)] = value

    def __str__(self):
        return '\n'.join(['{},{},{},{}'.format(k[0], k[1], k[2], v) for k, v in self.log.items()])

    @staticmethod
    def create_from_string(as_str):
        logger = MetricLogger()
        if len(as_str.strip()) == 0:
            return logger

        rows = [row.split(',') for row in as_str.strip().split('\n')]
        logger.log = {(name, int(epoch), int(iteration)): float(value) for name,epoch, iteration, value in rows}
        return logger

    def save(self, location):
        if not os.path.exists(location):
            os.makedirs(location)
        with open(os.path.join(location,'logger'), 'w') as fp:
            fp.write(str(self))

    def get_data(self, desired_name):
        d = {k[1]: v for k, v in self.log.items() if k[0] == desired_name}
        return [(k, d[k]) for k in sorted(d.keys())]