import torch
import torch.nn.functional as F
import torchvision


class MyFirstNetwork(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim):
    super(MyFirstNetwork, self).__init__()

    self.input_to_hidden = torch.nn.Linear(input_dim, hidden_dim)
    self.hidden_to_output = torch.nn.Linear(hidden_dim, output_dim)
    self.activation = torch.nn.Sigmoid()

    self.input_to_hidden.bias.data.fill_(0.)
    self.hidden_to_output.bias.data.fill_(0.)
  
  def forward(self, x):
    x = x.view(x.shape[0], -1)
    x = self.input_to_hidden(x)
    x = self.activation(x)
    x = self.hidden_to_output(x)
    return x


class LeNet(torch.nn.Module):
  def __init__(self):
    super(LeNet, self).__init__()
    
    # input channel = 1, output channels = 6, kernel size = 5
    # input image size = (28, 28), image output size = (24, 24)
    self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5))
    
    # input channel = 6, output channels = 16, kernel size = 5
    # input image size = (12, 12), output image size = (8, 8)
    self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5))
    
    # input dim = 4 * 4 * 16 ( H x W x C), output dim = 120
    self.fc3 = torch.nn.Linear(in_features=29 * 13 * 16, out_features=120)
    
    # input dim = 120, output dim = 84
    self.fc4 = torch.nn.Linear(in_features=120, out_features=84)
    
    # input dim = 84, output dim = 10
    self.fc5 = torch.nn.Linear(in_features=84, out_features=32)
    
  def forward(self, x):
    
    # print("input ", x.shape)
    x = self.conv1(x)
    x = F.relu(x)
    # print("conv1 ", x.shape)
    # Max Pooling with kernel size = 2
    # output size = (12, 12)
    x = F.max_pool2d(x, kernel_size=2)
    # print("max_pool1 ", x.shape)
    
    x = self.conv2(x)
    # print("conv2 ", x.shape)
    x = F.relu(x)
    # Max Pooling with kernel size = 2
    # output size = (4, 4)
    x = F.max_pool2d(x, kernel_size=2)
    # print("max_pool2 ", x.shape)
    
    # flatten the feature maps into a long vector
    # x = torch.nn.Flatten()(x)
    x = x.view(x.shape[0], -1)
    # print("flattern ", x.shape)
    
    x = self.fc3(x)
    x = F.relu(x)
    # print("fc1 ", x.shape)
    
    x = self.fc4(x)
    x = F.relu(x)
    # print("fc2 ", x.shape)
    
    x = self.fc5(x)
    # print("fc3 ", x.shape)

    return x


class VGG16(torch.nn.Module):
  def __init__(self):
    super(VGG16, self).__init__()

    #Block 1
    self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=(1,1))
    self.conv1_bn = torch.nn.BatchNorm2d(64)
    self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1,1))
    self.conv2_bn = torch.nn.BatchNorm2d(64)

    #Block 2
    self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1,1))
    self.conv3_bn = torch.nn.BatchNorm2d(128)
    self.conv4 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1,1))
    self.conv4_bn = torch.nn.BatchNorm2d(128)

    #Block 3
    self.conv5 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(1,1))
    self.conv5_bn = torch.nn.BatchNorm2d(256)
    self.conv6 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1,1))
    self.conv6_bn = torch.nn.BatchNorm2d(256)
    self.conv7 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1,1))
    self.conv7_bn = torch.nn.BatchNorm2d(256)

    #Block 4
    self.conv8 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=(1,1))
    self.conv8_bn = torch.nn.BatchNorm2d(512)
    self.conv9 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(1,1))
    self.conv9_bn = torch.nn.BatchNorm2d(512)
    self.conv10 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(1,1))
    self.conv10_bn = torch.nn.BatchNorm2d(512)

    #Block 5
    self.conv11 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(1,1))
    self.conv11_bn = torch.nn.BatchNorm2d(512)
    self.conv12 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(1,1))
    self.conv12_bn = torch.nn.BatchNorm2d(512)
    self.conv13 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(1,1))
    self.conv13_bn = torch.nn.BatchNorm2d(512)

    self.fc1 = torch.nn.Linear(in_features=512*4*2, out_features=4096)
    self.fc1_drop = torch.nn.Dropout(0.5)
    self.fc2 = torch.nn.Linear(in_features=4096, out_features=4096)
    self.fc2_drop = torch.nn.Dropout(0.5)

    self.fc3 = torch.nn.Linear(in_features=4096, out_features=32)

  def forward(self, x):
    # print("input: ", x.shape)
    x = self.conv1(x)
    x = self.conv1_bn(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = self.conv2_bn(x)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2)
    # print("Block 1: ", x.shape)

    x = self.conv3(x)
    x = self.conv3_bn(x)
    x = F.relu(x)
    x = self.conv4(x)
    x = self.conv4_bn(x)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2)
    # print("Block 2: ", x.shape)

    x = self.conv5(x)
    x = self.conv5_bn(x)
    x = F.relu(x)
    x = self.conv6(x)
    x = self.conv6_bn(x)
    x = F.relu(x)
    x = self.conv7(x)
    x = self.conv7_bn(x)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2)
    # print("Block 3: ", x.shape)

    x = self.conv8(x)
    x = self.conv8_bn(x)
    x = F.relu(x)
    x = self.conv9(x)
    x = self.conv9_bn(x)
    x = F.relu(x)
    x = self.conv10(x)
    x = self.conv10_bn(x)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2)
    # print("Block 4: ", x.shape)

    x = self.conv11(x)
    x = self.conv11_bn(x)
    x = F.relu(x)
    x = self.conv12(x)
    x = self.conv12_bn(x)
    x = F.relu(x)
    x = self.conv13(x)
    x = self.conv13_bn(x)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2)
    # print("Block 5: ", x.shape)

    x = x.view(x.shape[0], -1)

    # print("Flattern: ", x.shape)
    x = self.fc1(x)
    x = self.fc1_drop(x)
    x = F.relu(x)
    # print("fc1: ", x.shape)

    x = self.fc2(x)
    x = self.fc2_drop(x)
    x = F.relu(x)
    # print("fc2: ", x.shape)

    x = self.fc3(x)
    x = F.relu(x)
    # print("output: ", x.shape)

    return x


def initialize_vgg16(num_classes):
  vgg16 = torchvision.models.vgg16_bn(pretrained=True)
  in_features = vgg16.classifier[6].in_features
  vgg16.classifier[6] = torch.nn.Linear(in_features, out_features=num_classes)
  return vgg16


def initialize_alexnet(num_classes):
  # load the pre-trained Alexnet
  alexnet = torchvision.models.alexnet(pretrained=True)
  
  # get the number of neurons in the penultimate layer
  in_features = alexnet.classifier[6].in_features
  
  # re-initalize the output layer
  alexnet.classifier[6] = torch.nn.Linear(in_features=in_features, 
                                          out_features=num_classes)
  
  return alexnet
