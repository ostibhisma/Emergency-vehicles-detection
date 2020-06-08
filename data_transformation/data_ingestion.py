import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

class Dataloader:
    """
    Class : Dataloader
    
    input : path_to_data = (train.csv) and path_to_image = (images folder)
    
    Description : This class shall obtain the data as input and transform 
                these data in the training format.

    """
    def __init__(self,path_to_data,path_to_image,logging_obj):
        # reading input files
        self.obj = logging_obj
        self.obj.info("Data Loading and manipulations starts initially ")
        self.path_to_data = path_to_data
        self.path_to_image = path_to_image
        # pandas dataframe for train.csv files
        self.df = pd.read_csv(os.path.join(self.path_to_data,"train.csv"))
        # initializing additional attributes
        self.images = []
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []

    def imageData(self):
        """
        Method Name : imageData

        Description : This method reads all the 'image_names' column of 
                      dataframe(df) and convert each image to (224,224)
                      size and append it on the images list.
        """
        try:
            # reading all the image_name from dataframe 
            self.obj.info("Started Reading the 'image_names' columns of dataframe")
            for image_name in tqdm(self.df['image_names']):
                image = os.path.join(self.path_to_image,str(image_name))
                # reading images using opencv
                image = cv2.imread(image)
                # resizing the image as height 224 and width 224
                image_resized = cv2.resize(image,(224,224))
                # appending image to the images list
                self.images.append(np.array(image_resized,"float32"))
            return self.images
        
        except Exception as e:
            self.obj.error(f"Error occusrs {e}")
        self.obj.info("Successfully completed reading 'images_name' columns and al images are appended to list ")
    
    def getData(self,images):
        """
        Method Name : getData

        Input : Images list from the 'imageData' method

        Description : This method takes input as all images list from above
                       method and reads the target column of dataframe i.e
                       'emergency_or_not' column . It then splits the images and
                       target column using train_test_split from sklearn with 
                       the test_size of 0.2. It then convert all of the four 
                       data to numpy array and returns all of them.


        """
        try:
            self.images = images
            # target column
            labels = self.df['emergency_or_not'].values
            # splitting the data
            self.obj.info("Start Splitting train and test data")
            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(
                        self.images,labels,test_size=0.2,random_state=42
            )
            self.obj.info("Successfully Completed Splitting train and test data")
            # convert to numpy array
            self.X_train = np.array(self.X_train)/255.0
            self.X_test = np.array(self.X_test)/255.0
            self.y_train = np.array(self.y_train)
            self.y_test = np.array(self.y_test)
            # reshape the X_train and X_test data
            self.X_train = np.reshape(self.X_train,(self.X_train.shape[0],224,224,3))
            self.X_test = np.reshape(self.X_train,(self.X_train.shape[0],224,224,3))
            self.y_train = to_categorical(self.y_train)
            self.y_test = to_categorical(self.y_test)
            # returning the four values
            return self.X_train,self.X_test,self.y_train,self.y_test
        
        except Exception as e:
            self.obj.error(f"Error occusrs {e}")

        self.obj.info("Successfully Completed Loading and manipulating all data")













            
