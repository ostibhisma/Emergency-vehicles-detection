import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense,Conv2D,MaxPooling2D,Dropout,BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class Model:
    """
    This Class Train the model and save the best model according to validation
    aaccuracy in the form of .h5 file

    Written By : Bhisma
    
    """
    def __init__(self,X_train,X_test,y_train,y_test,logging_obj):
        self.logging_obj = logging_obj
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def cnn_model(self):
        self.logging_obj.info("Initially Started Creating Model")
        """
        Method Name: cnn_model
        Description : This method defines the Conventional Neural Network Model
                    [1 x CONV (3x3)] — MAXP (2x2) — DROPOUT (0.2)
                    [1 x CONV (3x3)] — MAXP (2x2) — DROPOUT (0.3)
                    [1 x CONV (3x3)] — MAXP (2x2) — DROPOUT (0.5)
                    Dense (128) — DROPOUT (0.4)
                    Dense (64) — DROPOUT (0.5)
                    Dense (2) 

        """
        try:
            num_features = 64
            num_labels = 2
            width, height, channels = 224, 224, 3
            self.train_aug = ImageDataGenerator(rescale = 1./255,rotation_range=20,horizontal_flip=True,zoom_range=0.2,shear_range=0.2)
            self.test_aug = ImageDataGenerator(rescale = 1./255)

            model = Sequential()

            model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, channels), data_format='channels_last'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.3))
            
            model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.3))



            model.add(Flatten())
            
            model.add(Dense(2*2*num_features, activation='relu'))
            model.add(Dropout(0.4))

            model.add(Dense(2*num_features, activation='relu'))
            model.add(Dropout(0.4))
        
            model.add(Dense(num_features, activation='relu'))
            model.add(Dropout(0.5))

            model.add(Dense(num_labels, activation='softmax'))

            return model

            self.logging_obj.info("Successfully Completed Creating Model")

        except Exception as e:
            self.logging_obj.error(f"Error Occurs as : {e}")

            
            
            
    def training(self,model):
        self.logging_obj.info("Started Training the model")
        """
        Method Name: training
        Description : Now the model that is pass from above method started to train
                      and the accuracy and loss of training process is saved in
                      "training_images/" folders.I have use "adam" as an optimizer
                      "categorical_crossentropy" as a loss .

        """
        try:
            BATCH_SIZE = 4
            EPOCHS = 20       
            
            model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

            lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, verbose=1)

            checkpoint = ModelCheckpoint('training/vehicles_model.h5',monitor='val_loss',verbose=0,
                                        save_best_only='True',mode='auto')

            history = model.fit(self.train_aug.flow(self.X_train,self.y_train,batch_size=BATCH_SIZE),
                               callbacks=[lr_reducer,checkpoint],epochs=EPOCHS,
                               validation_data=self.test_aug.flow(self.X_test,self.y_test))
            #plotting training and Validation (accuraccy and loss)
            plt.plot(history.history["accuracy"])
            plt.title("Model Accuracy")
            plt.ylabel("Accuracy")
            plt.xlabel("Epochs")
            plt.legend(["Train","Validation"],loc = 'upper_left')
            plt.savefig("training_images/accuracy.png")

            plt.plot(history.history["loss"])
            plt.title("Model Loss")
            plt.ylabel("Loss")
            plt.xlabel("Epochs")
            plt.legend(["Train","Validation"],loc = 'upper_left')
            plt.savefig("training_images/loss.png")

            self.logging_obj.info("Successfully Completed Training the Model")
            

        except Exception as e:
            self.logging_obj.error(f"Error Occurs as : {e}")
