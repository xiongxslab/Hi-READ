# Hi-C dataset wrapper used for comparison against experimental contact maps.

import numpy as np

class HiCDataset():
    def __init__(self, hic_file_name, res):
        import cooler

        self.hic_dataset = cooler.Cooler(f"{hic_file_name}::/resolutions/{res}")
        if("chr" in self.hic_dataset.chromnames[0]):
            self.remove_chr = False
        else:
            self.remove_chr = True

    def get(self, chromosome, start, window = 2097152, res = 10000):
        if(self.remove_chr):
            chromosome = chromosome.replace("chr", "")
        return np.log(self.hic_dataset.matrix(field="count", balance=None).fetch(f"{chromosome}:{start}-{start+window}")+1)
