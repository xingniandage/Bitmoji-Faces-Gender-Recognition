from torch import nn
 
 
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()

        # 第一层 卷积网络 输入：3 输出：16 卷积核：5*5 步长：1 padding：2
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # => 16 * 42 * 42

        # 第二层 卷积网络 输入：16 输出：32 卷积核：5*5 padding：2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # => 32 * 21 * 21

        # 第三层 卷积网络 输入：32 输出：64 卷积核：5*5
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # => 64 * 8 * 8

        # 第四层 卷积网络 输入：32 输出：64 卷积核：5*5
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # => 128 * 2 * 2

        self.relu = nn.ReLU()

        # 全连接层 第一层的输入 必须为卷积层的输出
        self.out1 = nn.Linear(128 * 2 * 2, 84)
        self.out2 = nn.Linear(84, 10)
        self.out3 = nn.Linear(10, 2) # 最终为两个类别

    def forward(self, x):
        x = self.conv1(x)
        # print(f"输出=>{len(x[0])}   长/宽=>{len(x[0][0])}") 查看输出 卷积层的输出
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        output = self.out1(x)
        output = self.relu(output)
        output = self.out2(output)
        output = self.relu(output)
        output = self.out3(output)
        return output, x

print("Model complicated")