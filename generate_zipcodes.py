import pandas as pd
from uszipcode import ZipcodeSearchEngine


# Test set
test_cols = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
             'passenger_count']
test_types = {'pickup_longitude': 'float32', 'pickup_latitude': 'float32', 'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32', 'passenger_count': 'uint8'}

test_raw_df = pd.read_csv('data/test.csv', dtype=test_types, usecols=test_cols, infer_datetime_format=True,
                           parse_dates=["pickup_datetime"])
print('raw test set shape: ', test_raw_df.shape)

# Train set
total_nrows = 55423855
train_cols = ['fare_amount', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
             'passenger_count']
train_types = {'fare_amount': 'float32', 'pickup_longitude': 'float32', 'pickup_latitude': 'float32', 'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32', 'passenger_count': 'uint8'}

zip_search = ZipcodeSearchEngine()


def extract_all_zipcodes(inp_df, lat_log_indices, url):
    zipcodes = []
    cities = []
    states = []
    passed_zips = 0
    for i, row in test_raw_df.iterrows():
        for j, row2 in enumerate(lat_log_indices):
            try:
                zipcode = zip_search.by_coordinate(float(inp_df.iloc[i, row2[0]]),
                                                   float(inp_df.iloc[i, row2[1]]), radius=5, returns=1)
                try:
                    zipcode = zipcode[0]
                    if zipcode['Zipcode'] not in zipcodes: zipcodes.append(zipcode['Zipcode'])
                    if zipcode['City'] not in cities: cities.append(zipcode['City'])
                    if zipcode['State'] not in states: states.append(zipcode['State'])
                except IndexError:
                    passed_zips += 1

            except IndexError:
                pass

    pd.DataFrame(zipcodes).to_csv(url + 'all_zipcodes_in_test_dataset.csv', header=None, index=None)
    pd.DataFrame(cities).to_csv(url + 'all_cities_in_test_dataset.csv', header=None, index=None)
    pd.DataFrame(states).to_csv(url + 'all_states_in_test_dataset.csv', header=None, index=None)
    print('passed zipcodes: ', passed_zips)

#extract_all_zipcodes(test_raw_df, [[2, 1], [4, 3]], 'inferred/zipcode_data/')  # Test set

num_of_batches = 100000
batch_size = int(total_nrows/num_of_batches)
print('batch size: ', batch_size)

for i in range(num_of_batches):
    print('batch ', i, ' (skipped ', i*batch_size, ' rows )')
    train_batch = pd.read_csv('data/train.csv', dtype=train_types, usecols=train_cols, infer_datetime_format=True,
                              parse_dates=["pickup_datetime"], nrows=batch_size, skiprows=range(1, i*batch_size))
    train_batch.dropna(inplace=True)
    # removing outliers
    train_batch.drop(train_batch.loc[(train_batch.fare_amount <= 0) | (train_batch.fare_amount > 150)].index, inplace=True)
    train_batch.drop(
        train_batch.loc[(train_batch.pickup_longitude < -77.03) | (train_batch.pickup_longitude > -70.75)].index,
        inplace=True)
    train_batch.drop(
        train_batch.loc[(train_batch.dropoff_longitude < -77.03) | (train_batch.dropoff_longitude > -70.75)].index,
        inplace=True)
    train_batch.drop(train_batch.loc[(train_batch.pickup_latitude < 38.63) | (train_batch.pickup_latitude > 42.85)].index,
                    inplace=True)
    train_batch.drop(train_batch.loc[(train_batch.dropoff_latitude < 38.63) | (train_batch.dropoff_latitude > 42.85)].index,
                    inplace=True)
    train_batch.drop(train_batch.loc[(train_batch.passenger_count > 7)].index, inplace=True)


    extract_all_zipcodes(train_batch, [[3, 2], [5, 4]], 'inferred/zipcode_data/train_set/batch' + str(i) + '_')


