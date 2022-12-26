from torchvision import transforms
class Config(object):
    num_classes = 2
    loss = 'softmax'

    trainpath=r"./data/BitmojiDataset/trainimages"
    testpath=r"./data/BitmojiDataset/testimages"
    csvpath=r"./data/train.csv"
    tensorboardpath =r"./tensorboard"
    model_cp =r"./model.pth"
    train_batch_size = 16  # batch size
    workers=0
    EPOCH = 20

    transform = transforms.Compose([
        transforms.Resize(84),  # 图片尺寸重设置
        transforms.CenterCrop(84),  # 中心剪裁
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

