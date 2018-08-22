from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torchvision.models as models
from torchvision import transforms
from torch.nn import functional as F
from scipy.misc import imresize
from telebot import TeleBot
from time import ctime, sleep
from random import randint
from urllib.request import urlretrieve
import warnings

session = {}

token = '540737355:AAFxnG-XsPTdK3CEwjvw1e4M1FiS7z8INbc'
bot = TeleBot(token)

def load_labels():
    # prepare all the labels
    # scene category relevant
    file_name_category = 'categories_places365.txt'

    classes = list()
    with open(file_name_category) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # indoor and outdoor relevant
    file_name_IO = 'IO_places365.txt'

    with open(file_name_IO) as f:
        lines = f.readlines()
        labels_IO = []
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) - 1) # 0 is indoor, 1 is outdoor
    labels_IO = np.array(labels_IO)

    return classes, labels_IO

def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data.numpy()))

def load_model():
    model_file = 'wideresnet18_places365.pth.tar'
    import wideresnet
    model = wideresnet.resnet18(num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)

    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
    for name in features_names:
        model._modules.get(name).register_forward_hook(hook_feature)
    return model

def image_transformer():
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    output_cam = []
    cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    output_cam.append(imresize(cam_img, size_upsample))

    return output_cam


@bot.message_handler(content_types=['photo'])
def receive_image(msg):
    print('Got image')
    # download image
    file_id = msg.photo[-1].file_id
    url = "https://api.telegram.org/file/bot" + token + "/" + bot.get_file(file_id).file_path
    urlretrieve(url, 'test_image.jpg')

    print(1)
    image_transforme = image_transformer() # image transformer
    test_filename = 'test_image.jpg'
    img = Image.open(test_filename)
    input_img = image_transforme(img).unsqueeze(0)
    logit = model.forward(input_img)
    output = F.softmax(logit, 1).data.squeeze()
    probs, idx = output.sort(dim=0, descending=True)
    probs = probs.numpy()
    idx = idx.numpy()
    io_image = np.mean(labels_IO[idx[:10]]) # vote for the indoor or outdoor
    print(2)
    if io_image > 0.5:
        bot.send_message(msg.chat.id, 'Outdoor')
    else:
        bot.send_message(msg.chat.id, 'Indoor')

    CAMs = returnCAM(features_blobs[0], weight_softmax, idx[0])
    img = plt.imread(test_filename)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    result = (heatmap * 0.3 + img * 0.7).astype(float) / 255
    plt.imsave('result.jpg', result)
    ans = '--SCENE CATEGORIES:\n'
    for i in range(0, 5):
        ans += '{:.3f} -> {}\n'.format(probs[i], classes[idx[i]])
    bot.send_message(msg.chat.id, ans)
    print(3)
    bot.send_photo(msg.chat.id, open('result.jpg', 'rb'))
    print(4)


@bot.message_handler(content_types=['text'])
def handle(msg):
    bot.send_message(msg.chat.id, 'Hello! Send me image')
    # image_transformer = image_transformer() # image transformer
    '''
    try:
        if msg.text[0] == '/':
            id[msg.chat.id] = msg.text[1:]
            bot.send_message(msg.chat.id, dialog(msg))
        elif msg.text.isdigit():
            id[msg.chat.id] = msg.text
            bot.send_message(msg.chat.id, dialog(msg))
        elif msg.chat.id not in id:
            bot.send_message(msg.chat.id, "/login first")
        elif id[msg.chat.id] != 0:
            vk_all[msg.chat.id].messages.send(user_id=id[msg.chat.id], random_id=randint(0, 100000), message=msg.text)
            bot.send_message(msg.chat.id, dialog(msg))
    except Exception as e:
        bot.send_message(msg.chat.id, str(e))
        '''


warnings.filterwarnings("ignore")
classes, labels_IO = load_labels()
model = load_model()
features_blobs = []
params = list(model.parameters())
weight_softmax = params[-2].data.numpy()
weight_softmax[weight_softmax<0] = 0
while True:
    try:
        bot.polling(none_stop=True)
    except Exception as e:
        print(e)
        sleep(1)
