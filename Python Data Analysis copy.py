import pandas as pd
import numpy as np

counts = pd.Series([632, 1638, 569, 115])
print (counts)

print (counts.values)

print (counts.index)

bacteria = pd.Series([632, 1638, 569, 115],
    index=['Firmicutes', 'Proteobacteria', 'Actinobacteria', 'Bacteroidetes'])

print (bacteria)

print (bacteria['Actinobacteria'])
###
print (bacteria[[name.endswith('bacteria') for name in bacteria.index]])
##
print([name.endswith('bacteria') for name in bacteria.index])

print (bacteria[0])

bacteria.name = 'counts'
bacteria.index.name = 'phylum'
print (bacteria)

print (np.log(bacteria))

print (bacteria[bacteria>1000])

bacteria_dict = {'Firmicutes': 632, 'Proteobacteria': 1638, 'Actinobacteria': 569, 'Bacteroidetes': 115}
print (pd.Series(bacteria_dict))

bacteria2 = pd.Series(bacteria_dict, index=['Cyanobacteria', 'Firmicutes', 'Proteobacteria', 'Actinobacteria'])
print (bacteria2)

print (bacteria2.isnull())

