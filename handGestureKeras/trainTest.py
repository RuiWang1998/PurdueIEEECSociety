import keras
from keras.models import model_from_json
from ModelKeras import Conv2Dense2, handCNN
from dataloader import input_shape
from matplotlib import pyplot as plt

# Hyper parameters
learning_rate = 0.0001
NUM_CLASS = 5
def displayHistory(history, epochs):
    
    train_loss = history.history['loss']
    val_loss   = history.history['val_loss']
    train_acc  = history.history['acc']
    val_acc    = history.history['val_acc']
    xc         = range(epochs)

    plt.figure()
    plt.plot(xc, train_loss)
    plt.plot(xc, val_loss)

    return

def loadAndTest(data_test, weight_file, json_name = 'model.json'):
    json_file = open(json_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weight_file)
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adam(lr = 0.0001, beta_1=0.99, beta_2=0.999, epsilon=1e-8, amsgrad=True), metrics=['accuracy'])
    validation_step = len(data_test)

    score = loaded_model.evaluate_generator(data_test, steps = validation_step)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def firstTrain(input_shape, data_train, data_test, dir_name_weight, dir_json, epochs = 2, model = handCNN(input_shape(), NUM_CLASS)):
    # introducing the model
    # net1 = Conv2Dense2(input_shape, NUM_CLASS)

    validation_step = len(data_test)
    train_step = len(data_train)

    # training
    history = model.fit_generator(
            generator=data_train,
            steps_per_epoch = train_step,
            epochs=epochs,
            validation_data=data_test,
            validation_steps=validation_step,
            callbacks = [
        keras.callbacks.ModelCheckpoint(dir_name_weight, monitor='val_acc', verbose=0, save_best_only=True, mode='auto')])

    # visualizing losses and accuracy
    displayHistory(history, epochs)

    score = model.evaluate_generator(data_test, steps = validation_step)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    weights_file = dir_name_weight
    json_file_name = dir_json

    save_model_keras(model, dir_json = json_file_name, dir_name_weight = dir_name_weight)

    return model

def loadAndTrain(data_train, data_test, weight_file, json_name = 'model.json', epoch = 1, lossfunc = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adam(lr = 0.0001, beta_1=0.99, beta_2=0.999, epsilon=1e-8, amsgrad=True)):
    json_file = open(json_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weight_file)
    print("Loaded model from disk")
 
    # evaluate loaded model on test data
    loaded_model.compile(loss=lossfunc, optimizer=optimizer, metrics=['accuracy'])

    validation_step = len(data_test)
    train_step = len(data_train)

    # training5
    history = loaded_model.fit_generator(
            generator=data_train,
            steps_per_epoch = train_step,
            epochs=epoch,
            validation_data=data_test,
            validation_steps=validation_step,
            callbacks = [
        keras.callbacks.ModelCheckpoint(weight_file, monitor='val_acc', verbose=0, save_best_only=True, mode='auto')])

    # visualize training history
    displayHistory(history, epoch)

    score = loaded_model.evaluate_generator(data_test, steps = validation_step)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    weights_file = weight_file
    json_file_name = json_name

    save_model_keras(loaded_model, dir_json = json_file_name, dir_name_weight = weights_file)
    
    return loaded_model

def save_model_keras(model, dir_json = "model.json", dir_name_weight = './models/first_try.h5'):
    # serialize model to JSON
    model_json = model.to_json()
    with open(dir_json, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(dir_name_weight)

    print("Saved model to disk")