import keras
from keras.models import model_from_json
from ModelKeras import Conv2Dense2, handCNN

# Hyper parameters
EPOCHS = 2
learning_rate = 0.0001
NUM_CLASS = 5

def firstTrain(input_shape, data_train, data_test, epochs = EPOCHS, model = handCNN):
    # introducing the model
    # net1 = Conv2Dense2(input_shape, NUM_CLASS)
    net1 = model(input_shape, NUM_CLASS)

    validation_step = len(data_test)
    train_step = len(data_train)

    # training
    net1.fit_generator(
            generator=data_train,
            steps_per_epoch = train_step,
            epochs=EPOCHS,
            validation_data=data_test,
            validation_steps=train_step)

    score = net1.evaluate_generator(data_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    return net1

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
    loaded_model.fit_generator(
            generator=data_train,
            steps_per_epoch = train_step,
            epochs=epoch,
            validation_data=data_test,
            validation_steps=train_step,
            callbacks = [
        keras.callbacks.ModelCheckpoint(weight_file, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')])

    score = loaded_model.evaluate_generator(data_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    return loaded_model

def save_model_keras(model, dir_json = "model.json", dir_name_weight = './models/first_try.h5'):
    # serialize model to JSON
    model_json = model.to_json()
    with open(dir_json, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(dir_name_weight)

    print("Saved model to disk")