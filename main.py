# importing Dataloader class from data_transformation
from data_transformation.data_ingestion import Dataloader
from training.model import Model
from application_logging.logger import Applog
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    def load_data(func):
        logg_data_transform = Applog("data_transformation/dataloading.log")
        logger = logg_data_transform.write(logg_data_transform)
        dataloader_obj = Dataloader("data","data/images",logger)
        images = dataloader_obj.imageData()
        X_train,X_test,y_train,y_test = dataloader_obj.getData(images)
        func(X_train,X_test,y_train,y_test)
        
    @load_data
    def training_model(X_train,X_test,y_train,y_test):
        logg_training = Applog("training/training_model.log")
        logger = logg_training.write(logg_training)
        model_object = Model(X_train,X_test,y_train,y_test,logger)
        model = model_object.cnn_model()
        model_object.training(model)

