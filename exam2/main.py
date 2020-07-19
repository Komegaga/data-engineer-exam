import os
import pandas as pd

if __name__ == '__main__':
    if not os.path.exists('result'):
        os.mkdir('result')
        
    for data in ['train', 'test']:
    
        data_pd = pd.read_csv(f"./data/{data}_need_aggregate.csv")
        data_pd['datetime'] = data_pd['datetime'].apply(lambda x : x[:-4])
        data_pd = data_pd.groupby('datetime')['EventId'].apply(list).reset_index()
        
        data_pd.to_csv(f'./result/{data}.csv')

    pass
