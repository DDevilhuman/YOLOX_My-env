import os
import random
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("no directory specified, please input target directory")
    exit()

root_path = sys.argv[1]

xmlfilepath = os.path.join(root_path, 'VOC2007', 'Annotations')
imagefilepath = os.path.join(root_path, 'VOC2007', 'JPEGImages')

# 必要なディレクトリを作成
os.makedirs(xmlfilepath, exist_ok=True)
os.makedirs(imagefilepath, exist_ok=True)

# Move annotations to annotations folder
for filename in os.listdir(root_path):
    file_path = os.path.join(root_path, filename)
    if filename.endswith('.xml'):
        Path(file_path).rename(os.path.join(xmlfilepath, filename))
    elif filename.endswith('.jpg'):
        Path(file_path).rename(os.path.join(imagefilepath, filename))

txtsavepath = os.path.join(root_path, 'VOC2007', 'ImageSets', 'Main')
os.makedirs(txtsavepath, exist_ok=True)

trainval_percent = 0.9
train_percent = 0.8
total_xml = os.listdir(xmlfilepath)
num = len(total_xml)
indices = list(range(num))
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(indices, tv)
train = random.sample(trainval, tr)

print("train and val size:", tv)
print("train size:", tr)

with open(os.path.join(txtsavepath, 'trainval.txt'), 'w') as ftrainval, \
     open(os.path.join(txtsavepath, 'test.txt'), 'w') as ftest, \
     open(os.path.join(txtsavepath, 'train.txt'), 'w') as ftrain, \
     open(os.path.join(txtsavepath, 'val.txt'), 'w') as fval:

    for i in indices:
        name = total_xml[i][:-4] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)
