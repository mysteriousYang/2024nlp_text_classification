import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from sklearn.metrics import classification_report, confusion_matrix
from dl_modules.LSTM.model import LSTMModel
from dl_modules.LSTM.data_loader import CommentDataset
from dl_modules.LSTM.utils import read_file, evaluate
from dl_modules.LSTM.pre_processing import DataPreprocess
from logger import Enable_Logger,timer_logger
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用 {device}")
Enable_Logger()


# 各数据集的路径
path_prefix = '.\\data'
train_with_label = os.path.join(path_prefix, 'training_balanced.txt')
test_file = os.path.join(path_prefix, "test_balanced.txt")

# word2vec模型文件路径
w2v_path = ".\\word2vec_balanced.model"

# 定义句子长度、是否固定 embedding、batch 大小、定义训练次数 epoch、learning rate 的值、model 的保存路径
requires_grad = True
sen_len = 10
# model_dir = os.path.join(path_prefix, 'model/')
model_dir = '.'
batch_size = 32
epochs = 20
lr = 1e-8


def LSTM_load_and_test(
        train_x, train_y,
        test_x,  test_y,
):
    # load data
    # train_x, train_y = read_file(train_with_label)
    # test_x,  test_y  = read_file(test_file)
    # print(data_x,data_y)
    # data pre_processing
    preprocess = DataPreprocess(train_x, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding()
    train_x = preprocess.sentence_word2idx()
    train_y = preprocess.labels2tensor(train_y)

    preprocess = DataPreprocess(test_x, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding()
    test_x = preprocess.sentence_word2idx()
    test_y = preprocess.labels2tensor(test_y)

    # 构造Dataset
    # 训练集
    train_dataset = CommentDataset(train_x, train_y)

    # 测试集
    val_dataset = CommentDataset(test_x, test_y)

    # preparing the training loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    print('Training loader prepared.')
    # preparing the validation loader
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    print('Validation loader prepared.')

    # load model
    model = LSTMModel(
        embedding,
        embedding_dim=300,
        hidden_dim=256,
        num_layers=4,
        dropout=0.5,
        requires_grad=requires_grad
    ).to(device)

    # 返回model中的参数的总数目
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))

    # loss function is binary cross entropy loss, 常见的二分类损失函数
    criterion = nn.BCELoss(reduction="sum")

    # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.

    # run epochs
    for epoch in range(epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        total_acc = validate(val_loader, model, criterion)

        if total_acc > best_acc:
            # 如果 validation 的结果好于之前所有的结果，就把当下的模型保存
            best_acc = total_acc
            # torch.save(model, "{}/val_acc_{:.3f}.model".format(model_dir,total_acc/v_batch*100))
            torch.save(model, "{}\\ckpt.model".format(model_dir))
            print('saving model with acc {:.3f}'.format(total_acc))

@timer_logger
def train(train_loader, model, criterion, optimizer, epoch):
    # 將 model 的模式设定为 train，这样 optimizer 就可以更新 model 的参数
    model.train()

    train_len = len(train_loader)
    total_loss, total_acc = 0, 0

    for i, (inputs, labels) in enumerate(train_loader):
        # print(inputs,labels)
        # 1. 放到GPU上
        inputs = inputs.to(device, dtype=torch.long)
        labels = labels.to(device, dtype=torch.float)  # 类型为float
        # 2. 清空梯度
        optimizer.zero_grad()
        # 3. 计算输出
        outputs = model(inputs)
        outputs = outputs.squeeze()  # 去掉最外面的 dimension
        # 4. 计算损失
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        # 5. 预测结果
        # outputs.clone().detach(): 非常重要, 因为evaluate对outputs进行了修改
        correct = evaluate(outputs.clone().detach(), labels)
        total_acc += (correct / batch_size)
        # 6. 反向传播
        loss.backward()
        # 7. 更新梯度
        optimizer.step()
        if i % 100 == 0:
            print('[ Epoch {}: {}/{} ] loss:{:.3f} acc:{:.3f} '.
                  format(epoch + 1, i + 1, train_len, loss.item(), correct * 100 / batch_size), end='\n')
    print('\n训练 | Loss:{:.5f} Acc: {:.3f}'.format(total_loss / train_len, total_acc / train_len * 100))


def validate(val_loader, model, criterion):
    model.eval()  # 將 model 的模式设定为 eval，固定model的参数

    val_len = len(val_loader)

    with torch.no_grad():
        total_loss, total_acc = 0, 0
        tp, tn, fp, fn = 0, 0, 0, 0
        for i, (inputs, labels) in enumerate(val_loader):
            # print(inputs, labels)
            # 1. 放到GPU上
            inputs = inputs.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float)
            # 2. 计算输出
            outputs = model(inputs)
            outputs = outputs.squeeze()
            # 3. 计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            # 4. 预测结果
            correct = evaluate(outputs, labels)
            total_acc += (correct / batch_size)
            # 5. 计算混淆矩阵
            # outputs[outputs >= 0.5] = 1
            # outputs[outputs < 0.5] = 0
            for j in range(len(outputs)):
                if  (outputs[j] > 0.5 and labels[j] == 1):
                    tp += 1
                elif(outputs[j] > 0.5 and labels[j] == 0):
                    fp += 1
                elif(outputs[j] < 0.5 and labels[j] == 0):
                    tn += 1
                elif(outputs[j] < 0.5 and labels[j] == 1):
                    fn += 1
        print(tp, tn, fp, fn," ")
        print("测试 | Loss:{:.5f} Acc: {:.3f} ".format(total_loss / val_len, total_acc / val_len * 100))
    prec = tp / (tp+fp)
    recl = tp / (tp+fn)
    print("准确率: {:.3f}, 查准率: {:.3f}, F1: {:.3f}".format(prec,recl,2*prec*recl / (prec+recl)))
    print('-----------------------------------------------\n')

    return total_acc / val_len * 100


if __name__ == '__main__':
    LSTM_load_and_test()
