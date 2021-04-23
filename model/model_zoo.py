from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet


def get_model(model, backbone, nclass):
    if model == 'deeplabv3plus':
        model = DeepLabV3Plus(backbone, nclass)
    elif model == 'pspnet':
        model = PSPNet(backbone, nclass)
    else:
        exit("\nError: MODEL \'%s\' is not implemented!\n" % model)

    return model
