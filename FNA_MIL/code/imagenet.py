from torchvision import models
from torchvision.models import DenseNet121_Weights
from torch import nn



def set_parameter_requires_grad(model, use_pretrained):
    if use_pretrained:
        for param in model.parameters():
            param.requires_grad = False


def get_model(model_name, num_outputs, use_pretrained=False):
    if model_name == 'mobilenetv2':
        """ MobileNet V2
        """
        model_ft = models.mobilenet_v2(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, use_pretrained)
        model_ft.classifier[1] = nn.Linear(1280, 10)
    
    elif model_name == "mobilenetv3":
        """ mobilenet V3
        """
        model_ft = models.mobilenet_v3_small() # weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1
        set_parameter_requires_grad(model_ft, use_pretrained)
        model_ft.classifier[3] = nn.Linear(1024, num_outputs)
    
    elif model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_outputs)
    elif model_name == 'resnet50':
        """Resnet50"""
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_outputs)
    elif model_name == 'wideresnet50':
        model_ft = models.wide_resnet50_2(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_outputs)
    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_outputs)
    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_outputs)

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, use_pretrained)
        model_ft.classifier[1] = nn.Conv2d(512, num_outputs, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_outputs

    elif model_name == "densenet":
        """ Densenet
        """
        if use_pretrained:
            model_ft = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        else:
            model_ft = models.densenet121()
        set_parameter_requires_grad(model_ft, use_pretrained)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_outputs)

    else:
        raise RuntimeError("Invalid model name")

    return model_ft
