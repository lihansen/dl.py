import torch
from utils.Animator import Animator

class Accumulator:
    def __init__(self, n_vars) -> None:
        self.data = [.0] * n_vars

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def cuda():
    return torch.device("cuda")


def evaluate_accuracy(net, data_iter):  # @save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(torch.device("cuda"))
            y = y.to(torch.device("cuda"))
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def accuracy(outputs, labels):  # @save
    """计算预测正确的数量"""
    if len(outputs.shape) > 1 and outputs.shape[1] > 1:
        outputs = outputs.argmax(axis=1)
    cmp = outputs.type(labels.dtype) == labels
    return float(cmp.type(labels.dtype).sum())



def cal_acc(net, test_iter):
    net.eval()
    
    total_acc = 0
    
    with torch.no_grad():
        for x, labels in test_iter:
            x = x.to(cuda())
            labels = labels.to(cuda())

            outputs = net(x)
            batch_acc = accuracy(outputs, labels)
            total_acc += batch_acc

    
    return total_acc / len(test_iter.dataset)




def train_epoch_ch3(net, train_iter, loss, updater):  # @save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)

    for X, y in train_iter:
        # 计算梯度并更新参数
        X = X.to(torch.device("cuda"))
        y = y.to(torch.device("cuda"))
        # print(X.device, y.device)
        y_hat = net(X)
        l = loss(y_hat, y)
        
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  # @save
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
        # print(train_metrics[0])
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

def cuda():
    return torch.device("cuda")

def train(net, train_iter, criterion, n_epochs, updater):
    animator = Animator(xlabel='epoch', 
                        xlim=[1, n_epochs], 
                        ylim=[0.1, 0.9],
                        legend=['avg loss', "running loss"])
    for epoch in range(n_epochs):
        total_loss = 0.
        total_sample = 0
        running_loss = 0.
        for i, data in enumerate(train_iter):
            x = data[0].to(device=cuda())
            labels = data[1].to(device=cuda())

            outputs = net(x)
            loss = criterion(outputs, labels)

            updater.zero_grad()
            loss.mean().backward()
            updater.step()
            
            total_loss += float(loss.sum())
            total_sample +=  labels.numel()
            avg_loss = total_loss / total_sample
            running_loss = float(loss.sum()) / labels.numel()


        animator.add(epoch + 1, [avg_loss, running_loss])
        # print(f'[{epoch + 1}] loss: {running_loss:.3f}', x.shape)


