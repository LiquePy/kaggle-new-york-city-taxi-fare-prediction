import pandas as pd
from uszipcode import ZipcodeSearchEngine
import numpy as np

zip_search = ZipcodeSearchEngine()

# Train set
total_nrows = 55423855
train_cols = ['fare_amount', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
             'passenger_count']
train_types = {'fare_amount': 'float32', 'pickup_longitude': 'float32', 'pickup_latitude': 'float32', 'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32', 'passenger_count': 'uint8'}

new_train_df = pd.read_csv('data/new_train_data.csv')


def extract_features(pickup_coordinates, dropoff_coordinates, trip_time, num_of_passengers, fare_amount=None):
    global new_train_df

    try:
        pickup_zipcode = zip_search.by_coordinate(float(pickup_coordinates[0]),
                                                  float(pickup_coordinates[1]), radius=5, returns=1)
        dropoff_zipcode = zip_search.by_coordinate(float(dropoff_coordinates[0]),
                                                   float(dropoff_coordinates[1]), radius=5, returns=1)
        try:
            condition = (((new_train_df['zip1'] == int(pickup_zipcode[0]['Zipcode'])) & (new_train_df['zip2'] == int(dropoff_zipcode[0]['Zipcode'])))  |  ((new_train_df['zip2'] == int(pickup_zipcode[0]['Zipcode'])) & (new_train_df['zip1'] == int(dropoff_zipcode[0]['Zipcode'])))  )
            df_index = new_train_df.index[condition][0]
            # Update index
            new_train_df.loc[df_index, 'fare_total'] += fare_amount
            new_train_df.loc[df_index, 'fare_min'] = min(fare_amount, new_train_df.loc[df_index, 'fare_min'])
            new_train_df.loc[df_index, 'fare_max'] = max(fare_amount, new_train_df.loc[df_index, 'fare_max'])
            new_train_df.loc[df_index, 'fare_count'] += 1
            new_train_df.loc[df_index, 'fare_h' + str(trip_time.hour) + '_total'] += fare_amount
            new_train_df.loc[df_index, 'fare_h' + str(trip_time.hour) + '_count'] += 1
            new_train_df.loc[df_index, 'fare_y' + str(trip_time.year) + '_total'] += fare_amount
            new_train_df.loc[df_index, 'fare_y' + str(trip_time.year) + '_count'] += 1
            new_train_df.loc[df_index, 'fare_p' + str(num_of_passengers) + '_total'] += fare_amount
            new_train_df.loc[df_index, 'fare_p' + str(num_of_passengers) + '_count'] += 1

        except IndexError:
            # Add new row
            new_df = pd.DataFrame(columns=new_train_df.columns)
            new_df.loc[0, :] = 0
            new_df.loc[0, 'zip1'] = int(pickup_zipcode[0]['Zipcode'])
            new_df.loc[0, 'zip2'] = int(dropoff_zipcode[0]['Zipcode'])
            new_df.loc[0, 'fare_total'] = fare_amount
            new_df.loc[0, 'fare_min'] = fare_amount
            new_df.loc[0, 'fare_max'] = fare_amount
            new_df.loc[0, 'fare_count'] = 1
            new_df.loc[0, 'fare_h' + str(trip_time.hour) + '_total'] = fare_amount
            new_df.loc[0, 'fare_h' + str(trip_time.hour) + '_count'] = 1
            new_df.loc[0, 'fare_y' + str(trip_time.year) + '_total'] = fare_amount
            new_df.loc[0, 'fare_y' + str(trip_time.year) + '_count'] = 1
            new_df.loc[0, 'fare_p' + str(num_of_passengers) + '_total'] = fare_amount
            new_df.loc[0, 'fare_p' + str(num_of_passengers) + '_count'] = 1
            new_train_df = pd.concat([new_train_df, new_df], ignore_index=True)
    except IndexError:
        print('passed 2')
        pass



num_of_batches = 100000
batch_size = int(total_nrows/num_of_batches)
print('batch size: ', batch_size)
run_from_batch = 1000
run_to_batch = 1200
for i in range(run_from_batch, run_to_batch):
    print('batch ', i, ' (skipped ', i*batch_size, ' rows )')
    train_batch = pd.read_csv('data/train.csv', dtype=train_types, usecols=train_cols, infer_datetime_format=True,
                              parse_dates=["pickup_datetime"], nrows=batch_size, skiprows=range(1, i*batch_size))
    train_batch.dropna(inplace=True)
    # removing outliers
    train_batch.drop(train_batch.loc[(train_batch.fare_amount <= 0) | (train_batch.fare_amount > 200)].index, inplace=True)
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

    for j, row in train_batch.iterrows():
        extract_features([row[3], row[2]], [row[5], row[4]], row[1], row[6], row[0])


new_train_df.to_csv('data/new_train_data.csv', index=None)

