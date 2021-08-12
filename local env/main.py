import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#from keras.models import Sequential
#from keras.optimizers import Adam
#from keras.callbacks import ModelCheckpoint
#from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
import torch
import torch.nn
from torch.optim import Adam
from utilis import INPUT_SHAPE, batch_generator
import argparse
import os


np.random.seed(0)


def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    print(os.path.join(args.data_dir,'driving_log.csv'))
    data_df = pd.read_csv(os.path.join(args.data_dir, 'driving_log.csv'))

    X = data_df[['center','left','right']].values
    y = data_df['steering'].values
    # X = X[:2000]
    # y = y[:2000]
    print(X.shape)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)
    #X_train = X_train / 127.5 - 1.0
    #X_valid = X_valid / 127.5 - 1.0
    return X_train, X_valid, y_train, y_valid


def build_model(args):
    """
    Modified NVIDIA model
    """
    model = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels = 3, out_channels = 24,kernel_size = (5,5), stride = (2,2)),
        torch.nn.ELU(),
        torch.nn.Conv2d(in_channels=24, out_channels=36, kernel_size=(5, 5), stride=(2, 2)),
        torch.nn.ELU(),
        torch.nn.Conv2d(in_channels=36, out_channels=48, kernel_size=(5, 5), stride=(2, 2)),
        torch.nn.ELU(),
        torch.nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3,3), stride=(1, 1)),
        torch.nn.ELU(),
        torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1, 1)),
        torch.nn.ELU(),
        torch.nn.Dropout(p=args.keep_prob, inplace=True),
        torch.nn.Flatten(),
        torch.nn.Linear(in_features = 1152, out_features = 100,bias = True),
        torch.nn.ELU(),
        torch.nn.Linear(in_features = 100,out_features = 50,bias = True),
        torch.nn.ELU(),
        torch.nn.Linear(in_features = 50,out_features = 10,bias = True),
        torch.nn.ELU(),
        torch.nn.Linear(in_features = 10,out_features = 1,bias = True)
    )
    #model = Sequential()
    #model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))
    #model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    #model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    #model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    #model.add(Conv2D(64, 3, 3, activation='elu'))
    #model.add(Conv2D(64, 3, 3, activation='elu'))
    #model.add(Dropout(args.keep_prob))
    #model.add(Flatten())
    #model.add(Dense(100, activation='elu'))
    #model.add(Dense(50, activation='elu'))
    #model.add(Dense(10, activation='elu'))
    #model.add(Dense(1))
    #model.summary()

    return model


def train_model(model, args, X_train, X_valid, y_train, y_valid):
    """
    Train the model
    """
    criterion = torch.nn.MSELoss()

    optimizer = Adam(model.parameters(),lr = args.learning_rate)

    training_loss = []
    validation_loss = []
    for epoch in range(args.nb_epoch):
        running_loss = 0
        total_loss_valid = 0
        model.train()
        total_loss = 0
        i = 0
        print('Epoch : {}'.format(epoch))
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        for imgs,steers in batch_generator(args.data_dir, X_train, y_train, args.batch_size, True):

            shape_imgs = imgs.shape
            imgs = imgs.reshape(shape_imgs[0],shape_imgs[3],shape_imgs[1],shape_imgs[2])
            imgs = torch.FloatTensor(imgs)
            steers = steers.reshape(args.batch_size,1)
            steers = torch.FloatTensor(steers)
            optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, steers)


            loss.backward()
            optimizer.step()
            i = i + 1
            running_loss += loss.item()
            del loss

            # print statistics
            if i % 5 == 0:  # print every 5 mini-batches
                print('The training loss: %.6f after %d' % (running_loss / (i),i))

        running_loss = running_loss / (X_train.shape[0]  / 40)



        model.eval()


        j = 0
        for imgs, steers in batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False):

            shape_imgs = imgs.shape
            imgs = imgs.reshape(shape_imgs[0], shape_imgs[3], shape_imgs[1], shape_imgs[2])
            imgs  = torch.FloatTensor(imgs)
            steers = steers.reshape(args.batch_size, 1)
            steers = torch.FloatTensor(steers)
            optimizer.zero_grad()

            j = j + 1
            outputs = model(imgs)
            loss = criterion(outputs, steers)
            total_loss_valid = total_loss_valid + loss.item()
            if j % 5 == 0:  # print every 5 mini-batches
                print('The Validation loss: %.6f after %d' % (total_loss_valid / (j), j))

        total_loss_valid = total_loss_valid / (X_valid.shape[0] / 40)

        training_loss.append(running_loss)

        print('The training loss after epoch{} is {}'.format(epoch,(running_loss)))
        validation_loss.append(total_loss_valid)
        print('The validation loss after epoch{} is {}'.format(epoch, total_loss_valid))



def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default='data')
    parser.add_argument('-t', help='test size fraction', dest='test_size', type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability', dest='keep_prob', type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs', dest='nb_epoch', type=int, default=10)
    parser.add_argument('-s', help='samples per epoch', dest='samples_per_epoch', type=int, default=20000)
    parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=40)
    parser.add_argument('-o', help='save best models only', dest='save_best_only', type=s2b, default='true')
    parser.add_argument('-l', help='learning rate', dest='learning_rate', type=float, default=1.0e-4)
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    data = load_data(args)
    model = build_model(args)
    train_model(model, args, *data)



    torch.save(model, 'model.pth')
if __name__ == '__main__':
    main()