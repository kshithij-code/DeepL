import torch
import torch.nn as nn
from torchvision import models,transforms
from PIL import Image
import torch.nn.functional as f
import gradio as gr
from torchvision.transforms import transforms


# model=models.resnet18(pretrained=True)
# model.fc=nn.Linear(model.fc.in_features,10)
t=transforms.Compose([  transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                        transforms.RandomHorizontalFlip(0.5),
                        transforms.RandomRotation(10),
                        ])
class_name=["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship","truck"]

class CIFAR_Module(nn.Module):
    def __init__(self,in_channel):
        self.in_channel=in_channel
        super(CIFAR_Module,self).__init__()
        self.con1=nn.Conv2d(in_channel,6*in_channel,5)
        self.pool1=nn.MaxPool2d(5,stride=2)
        self.con2=nn.Conv2d(6*in_channel,16*in_channel,5)
        self.pool2=nn.MaxPool2d(5,stride=2)
        self.flat=nn.Flatten()
        self.fc1=nn.Linear(192,100*in_channel)
        self.fc2=nn.Linear(100*in_channel,40*in_channel)
        self.fc3=nn.Linear(40*in_channel,10)
    def forward(self,x):
        x=self.con1(x)
        x=f.relu(x)
        x=self.pool1(x)
        x=f.relu(x)
        x=self.con2(x)
        x=f.relu(x)
        x=self.pool2(x)
        x=self.flat(x)
        x=self.fc1(x)
        x=f.relu(x)
        x=self.fc2(x)
        x=f.relu(x)
        x=self.fc3(x)
        return x


model=CIFAR_Module(3)
model.load_state_dict(torch.load("model.pth",weights_only=True))
model.eval()


print(model)

def predict(image):
    image=image.resize((32,32))
    image=t(image).unsqueeze(0)
    with torch.no_grad():
        output=model(image)
        _,predicted=torch.max(output,1)
        print(output)
        predicted_class=class_name[predicted.item()-1]
        return predicted_class

interface=gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="cifar dataset prediction",
    description="upload an image to get its class"
)

interface.launch(share=True)