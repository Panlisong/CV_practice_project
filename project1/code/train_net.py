import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.utils.data as Data
import torchvision
import load_data
import net

EPOCH = 5
BATCH_SIZE = 8
LR = 0.01

train_image, train_label = load_data.load_train_data()
train_image, train_label = torch.Tensor(train_image), torch.Tensor(train_label)

train_data = Data.TensorDataset(train_image, train_label)
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

car_detection = net.car_detection_net()
optimizer = torch.optim.Adam(car_detection.parameters(), LR)
loss_function = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    LR /= 10
    optimizer = torch.optim.Adam(car_detection.parameters(), LR)
    whole_loss = 0
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x)
        b_y = Variable(y)

        output = car_detection(b_x)
        loss = loss_function(output, b_y.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        whole_loss += loss.data.numpy()

    whole_loss /= step+1
    print(whole_loss)

torch.save(car_detection.state_dict(), 'project1\\Parameters.pth')

test_image, test_label = load_data.load_test_data()
test_image, test_label = torch.Tensor(test_image), torch.Tensor(test_label)
pred_y = car_detection(test_image)
_, pred_y = torch.max(pred_y, dim=1)
print(torch.sum(pred_y == test_label).numpy()/pred_y.shape[0])
