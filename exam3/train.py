from model import Model
from argparse import ArgumentParser
import torch
import torch.nn as nn
import pandas as pd
import numpy
from sklearn.preprocessing import MinMaxScaler
import os

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw * 2]
        inout_seq.append((train_seq ,train_label))
    return inout_seq


if __name__ == '__main__':
    if not os.path.exists('model'):
        os.mkdir('model')

    parser = ArgumentParser()
    parser.add_argument("-e", "--epochs", dest="epochs")
    args = parser.parse_args()
    epochs = int(args.epochs)

    model = Model()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    print(model)

    train_pd = pd.read_csv("../exam2/result/train.csv", index_col=0)
    train_pd['EventId'] = train_pd['EventId'].apply(lambda x : [int(i) for i in x[1:-1].split(",")] )
    train_pd['length'] = train_pd['EventId'].apply(len)
    maxLength = train_pd['length'].max()
    train_pd['EventId'] = train_pd['EventId'].apply(lambda x : x + [0] * (maxLength - len(x)))
    flat_list = [item for sublist in train_pd['EventId'].tolist() for item in sublist]
    train_pd['length'] = train_pd['EventId'].apply(len)
    train_data = numpy.array(flat_list).astype(float)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data_normalized = scaler.fit_transform(train_data .reshape(-1, 1))
    train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

    train_window = maxLength
    train_inout_seq = create_inout_sequences(train_data_normalized, train_window)


    for i in range(epochs):
        for seq, labels in train_inout_seq[:100]:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if i%2 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

    torch.save(model.state_dict(), "./model/model.ph")
    pass
