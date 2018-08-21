import pandas as pd
import matplotlib.pyplot as plt
import json

with open('array_count_of_rides_between_225_cluster_sets_of_1m_data.txt') as f:
    data = json.load(f)

data = pd.DataFrame(data, columns=['pickup_cluster', 'dropoff_cluster', 'count'])

print('total combinations: ', len(data['count']))
data['count'].plot.hist(bins=100)
print(data['count'].value_counts())
print('# of rides per cluster combo on average: ', data['count'].value_counts().mean())

plt.show()


