# importing Dataloader class from data_transformation
from data_transformation.data_ingestion import Dataloader
from application_logging.logger import Applog

def load_data():
    log = Applog("data_transformation/dataloading.log")
    logger = log.write(log)
    dataloader_obj = Dataloader("data","data/images",logger)
    images = dataloader_obj.imageData()
    X_train,X_test,y_train,y_test = dataloader_obj.getData(images)

    




if __name__ == '__main__':
    load_data()