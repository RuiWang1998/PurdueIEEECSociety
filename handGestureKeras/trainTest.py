import keras
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

def save_model_keras(model, dir_json = "model.json", dir_name_weight = './models/first_try.h5'):
    # serialize model to JSON
    model_json = model.to_json()
    with open(dir_json, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(dir_name_weight)

    print("Saved model to disk")