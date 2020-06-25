# -*- coding: utf-8 -*-
import os
import torch
from config import parse_config
from data_loader import DataBatchIterator
from data_loader import PAD
from model import TextCNN
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import logging


def build_textcnn_model(vocab, config, train=True):
    model = TextCNN(vocab.vocab_size, config)
    if train:
        model.train()
    else:
        model.eval()

    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
    return model


def train_textcnn_model(model, train_data, valid_data, padding_idx, config):
    # Build optimizer.
    # params = [p for k, p in model.named_parameters(
    # ) if p.requires_grad and "embed" not in k]
    params = [p for k, p in model.named_parameters() if p.requires_grad]
    optimizer = Adam(params, lr=config.lr)
    criterion = CrossEntropyLoss(reduction="sum")
    model.train()

    for epoch in range(1, config.epochs + 1):
        train_data_iter = iter(train_data)
        for idx, batch in enumerate(train_data_iter):
            model.zero_grad()
            ground_truth = batch.label
            # batch_first = False
            outputs = model(batch.sent)
            loss = criterion(outputs, ground_truth)
            loss.backward()
            optimizer.step()

            if idx % 20 == 0:
                valid_loss = valid_textcnn_model(
                    model, valid_data, criterion, config)
                # 处理
                print("epoch {0:d} [{1:d}/{2:d}], valid loss: {3:.2f}".format(
                    epoch, idx, train_data.num_batches, valid_loss))
                model.train()


def valid_textcnn_model(model,  valid_data, criterion, config):
    # Build optimizer.
    # params = [p for k, p in model.named_parameters(
    # ) if p.requires_grad and "embed" not in k]
    model.eval()
    total_loss = 0
    valid_data_iter = iter(valid_data)
    for idx, batch in enumerate(valid_data_iter):
        model.zero_grad()
        ground_truth = batch.label
        # batch_first = False
        outputs = model(batch.sent)
        # probs = model.generator(decoder_outputs)
        loss = criterion(outputs, batch.label)
        # loss 打印
        # 处理
        total_loss += loss
        break
    return loss

def test_textcnn_model(model,  valid_data, criterion, config):
    # Build optimizer.
    # params = [p for k, p in model.named_parameters(
    # ) if p.requires_grad and "embed" not in k]
    model.eval()
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    valid_data_iter = iter(valid_data)
    for idx, batch in enumerate(valid_data_iter):
        model.zero_grad()
        ground_truth = batch.label
        # batch_first = False
        outputs = model(batch.sent)
        # probs = model.generator(decoder_outputs)
        print(accuracy_score(batch.label, outputs.max(1).indices, True))
        total_accuracy += accuracy_score(batch.label, outputs.max(1).indices, True)
        total_precision += precision_score(batch.label, outputs.max(1).indices, None, 1, 'macro')
        #total_precision += precision_score(batch.label, outputs.max(1).indices, None, 1, 'macro')
        #total_precision += precision_score(batch.label, outputs.max(1).indices, None, 1, 'micro')
        total_recall += recall_score(batch.label, outputs.max(1).indices, None, 1, 'macro')
        total_f1 += f1_score(batch.label, outputs.max(1).indices, None, 1, 'macro')
    print("accuracy  ,  precision  ,  recall  ,  f1  ")
    print(total_accuracy/157, total_precision/157, total_recall/157, total_f1/157)


def test_valid(model , config ,valid_data):

    criterion = CrossEntropyLoss(reduction="sum")
    test_textcnn_model(
        model, valid_data, criterion, config)
    # 处理

def main():
    # 读配置文件
    config = parse_config()
    # 载入训练集合
    train_data = DataBatchIterator(
        config=config,
        is_train=True,
        dataset="train",
        batch_size=config.batch_size,
        shuffle=True)
    train_data.load()

    vocab = train_data.vocab

    # 载入测试集合
    valid_data = DataBatchIterator(
        config=config,
        is_train=False,
        dataset="dev",
        batch_size=config.batch_size)
    valid_data.set_vocab(vocab)
    valid_data.load()

    # 构建textcnn模型
    model = build_textcnn_model(
        vocab, config, train=True)

    print(model)
    # Do training.
    padding_idx = vocab.stoi[PAD]
    #train_textcnn_model(model, train_data,
    #                    valid_data, padding_idx, config)
    #torch.save(model, '%s.pt' % (config.save_model))


    # 测试时
    #加载测试集
    test_data = DataBatchIterator(
        config=config,
        is_train=False,
        dataset="test",
        batch_size=config.batch_size)
    test_data.load()

    #读取训练好的模型
    checkpoint = torch.load(config.save_model+".pt",
                         map_location = config.device)
    #测试并打印评价
    test_valid(checkpoint , config , test_data)


if __name__ == "__main__":
    main()
