import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
import numpy as np

#creating the model
class Model(object):
    def __init__(self, ninput, layers, model = None):
        #ninput will tells us the size of the input
        #layers is same as hidden layer that will be needed for our model. If you say 4 that means 4 hidden layers
        self.keras_model = model or self.build_model(ninput, layers)

    def build_model(self, ninput, layers):
        #functional model-> Multiple inputs and Multiple outputs
        #Input layer
        input_layer = Input(shape = (ninput, )) #Input layer for our NN Model
        #Hidden Layer
        x = input_layer
        #This code is for hidden layer
        for n in layers:
            x = Dense(n, activation = "relu")(x)

        #output layer
        #softmax will display the result in probability
        output_layer = Dense(3, activation = "softmax")(x)

        #building our model
        model = tf.keras.Model(input_layer, output_layer)
        #compile our model
        model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics = ["accuracy"]) #you can use adam
        return model
    
    #data create from scratch -> data preprocessing
    def preprocess(self, plays):
        dataset = [] #final data to train the model

        for states, winner in plays:
            #Game is lost by AI
            if winner == 0:
                continue
            #this is the case of winning. If current player win the game
            rows = [(move.state.cells, winner) for move in states] #It will extract all the moves that player winner. #List comprehension

            if states[-1].state.winner != 0:
                rows.extend(self._preprocess_critical_action(states, winner))

            dataset.extend(rows)

        #let's divide our dataset featured and targets
        np.random.shuffle(dataset)
        #tuple comprehension -> Zip is used to iterate through each items and we simply unpack the dataset using *
        features, targets = tuple(np.array(e) for e in zip(*dataset))
        targets = tf.keras.utils.to_categorical(targets, num_classes = 3) #num_class = 3 indicates 3 classes-: win, loss or draw
        return features, targets
        
    def _preprocess_critical_action(self, states, winner):
        critical_state = states[-3].state
        critical_action = states[-2].action

        data = [] #empty list
        for action in critical_state.actions():
            state = critical_state.move(action)
            if action != critical_action:
                data.append((state.cells, critical_state.player()))

        return data
    

#Train the model using features and targets

    def train(self, plays, split_ratio = 0.2, epochs = 1, batch_size = 128):
        features, targets = self.preprocess(plays)

        #let's divide the data into training and testing
        idx = int(split_ratio*len(features))

        #build a model to make model learn from data
        trainX, trainY = features[idx:], targets[idx:]

        #to evaluate the model, we use testing data  
        testX, testY = features[:idx], targets[:idx]

        #Training steps
        #epochs how many data you can to fit the model
        history = self.keras_model.fit(trainX, trainY, validation_data = (testX, testY), epochs = epochs, batch_size = batch_size)

        print(history) #expect loss function(error value), accuracy of the model

    
    def predict(self, states):
        return self.keras_model.predict(states)