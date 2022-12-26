from config import Config
from train import train
from detect import detect

train(path_dir=Config.trainpath,label_dir=Config.csvpath,batch_size=Config.train_batch_size,workers=Config.workers,
            EPOCH=Config.EPOCH,tensorboard_path=Config.tensorboardpath,model_cp=Config.model_cp,transform=Config.transform)

result=detect(model_path=Config.model_cp,image_path=Config.testpath,batch_size=Config.train_batch_size,workers=Config.workers,data_transform=Config.transform)
print(result)