import os
import time
import shutil
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.utils.data.sampler import SequentialSampler
from torch.utils.tensorboard import SummaryWriter

from dataset import TSNDataSet
from models import TSN
from transforms import *
from opts import parser
import datasets_video
from main import AverageMeter, accuracy


class GestureSystem:
    """
    手势识别系统
    1. 可以加载jester数据集进行训练
    2. 可以加载jester数据集进行验证
    3. 可以处理某视频得到连续帧数据并作为一组数据进行识别
    4. 可以加载之前训练好的模型
    """

    def __init__(self):
        # 加载参数
        num_class, prefix = self._set_args()

        # 设置日志（包括输出日志和tensorboard）
        self._set_log()

        # 设置模型
        crop_size, input_mean, input_std, scale_size, train_augmentation = self._set_model(num_class)

        # 令cudnn根据卷积网络的实际结构选择最佳的实现算法
        cudnn.benchmark = True

        # 设置数据加载器
        self._set_data_loader(crop_size, input_mean, input_std, prefix, scale_size, train_augmentation)

        # 设置损失函数
        self._set_loss_function()

        # 验证 (若命令行未写该参数，默认为false)
        if self.args.evaluate:
            self.validate()
            return
        else:
            # 训练
            self._train_all()

    def __del__(self):
        self.board_writer.close()
        self.output_log.close()

    def _set_loss_function(self):
        # define loss function (criterion) and optimizer
        if self.args.loss_type == 'nll':
            self.criterion = torch.nn.CrossEntropyLoss().cuda()
        else:
            raise ValueError("Unknown loss type")

    def _set_data_loader(self, crop_size, input_mean, input_std, prefix, scale_size, train_augmentation):
        # Data loading code
        if (self.args.modality != 'RGBDiff') | (self.args.modality != 'RGBFlow'):
            normalize = GroupNormalize(input_mean, input_std)
        else:
            normalize = IdentityTransform()

        if self.args.modality == 'RGB':
            data_length = 1
        elif self.args.modality in ['Flow', 'RGBDiff']:
            data_length = 5
        elif self.args.modality == 'RGBFlow':
            data_length = self.args.num_motion
        else:
            raise Exception("ars.modality is not allowed.")

        # 训练数据加载器
        self.train_loader = torch.utils.data.DataLoader(
            TSNDataSet(self.args.root_path, self.args.train_list, num_segments=self.args.num_segments,
                       new_length=data_length,
                       modality=self.args.modality,
                       image_tmpl=prefix,
                       dataset=self.args.dataset,
                       transform=torchvision.transforms.Compose([
                           train_augmentation,
                           Stack(roll=(self.args.arch in ['BNInception', 'InceptionV3']),
                                 isRGBFlow=(self.args.modality == 'RGBFlow')),
                           ToTorchFormatTensor(div=(self.args.arch not in ['BNInception', 'InceptionV3'])),
                           normalize,
                       ])),
            batch_size=self.args.batch_size, shuffle=True,
            num_workers=self.args.workers, pin_memory=False)

        # 验证数据加载器
        self.val_loader = torch.utils.data.DataLoader(
            TSNDataSet(self.args.root_path, self.args.val_list, num_segments=self.args.num_segments,
                       new_length=data_length,
                       modality=self.args.modality,
                       image_tmpl=prefix,
                       dataset=self.args.dataset,
                       random_shift=False,
                       transform=torchvision.transforms.Compose([
                           GroupScale(int(scale_size)),
                           GroupCenterCrop(crop_size),
                           Stack(roll=(self.args.arch in ['BNInception', 'InceptionV3']),
                                 isRGBFlow=(self.args.modality == 'RGBFlow')),
                           ToTorchFormatTensor(div=(self.args.arch not in ['BNInception', 'InceptionV3'])),
                           normalize,
                       ])),
            batch_size=self.args.batch_size, shuffle=False,
            num_workers=self.args.workers, pin_memory=False)

    def _set_args(self):
        self.best_prec1 = 0
        self.args = parser.parse_args()  # 导入配置参数
        self._check_rootfolders()  # 创建日志和模型文件夹

        # 标签列表，训练集txt路径，验证集txt路径，数据根路径（datasets/jester），图片文件名（{:05d}.jpg）
        categories, self.args.train_list, self.args.val_list, self.args.root_path, prefix = \
            datasets_video.return_dataset(self.args.dataset, self.args.modality)
        num_class = len(categories)

        self.args.store_name = '_'.join(['MFF', self.args.dataset, self.args.modality, self.args.arch,
                                         'segment%d' % self.args.num_segments, '%df1c' % self.args.num_motion])
        print('storing name: ' + self.args.store_name)

        return num_class, prefix

    def _set_log(self):
        # tensorboard log
        self.board_writer = SummaryWriter("./log/tensorboard")
        # 输出日志
        self.output_log = open(os.path.join(self.args.root_log, '%s.csv' % self.args.store_name), 'w')

    def _set_model(self, num_class):
        # 数据模型
        self.model = TSN(num_class, self.args.num_segments, self.args.modality,
                         base_model=self.args.arch,
                         consensus_type=self.args.consensus_type,
                         dropout=self.args.dropout, num_motion=self.args.num_motion,
                         img_feature_dim=self.args.img_feature_dim,
                         partial_bn=not self.args.no_partialbn,
                         dataset=self.args.dataset)
        crop_size = self.model.crop_size
        scale_size = self.model.scale_size
        input_mean = self.model.input_mean
        input_std = self.model.input_std
        train_augmentation = self.model.get_augmentation()

        # 根据学习率定义优化器（SGD）
        self._set_optimizer()

        # 数据模型（转化为cuda）
        self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpus).cuda()

        # 加载训练好的模型
        if self.args.resume:
            self._load_model()

        print(self.model)

        return crop_size, input_mean, input_std, scale_size, train_augmentation

    def _set_optimizer(self):
        # 学习率调增策略
        policies = self.model.get_optim_policies()

        for group in policies:
            print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
                group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

        # 优化器
        self.optimizer = torch.optim.SGD(policies,
                                         self.args.lr,
                                         momentum=self.args.momentum,
                                         weight_decay=self.args.weight_decay)

    def _load_model(self):
        if os.path.isfile(self.args.resume):
            print(("=> loading checkpoint '{}'".format(self.args.resume)))
            checkpoint = torch.load(self.args.resume)
            self.args.start_epoch = checkpoint['epoch']
            self.best_prec1 = checkpoint['best_prec1']
            self.model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(self.args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(self.args.resume)))

    def _train_all(self):
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self._adjust_learning_rate(epoch, self.args.lr_steps)

            # train for one epoch
            self.train(epoch)

            # 在训练结束后评估模型，完成后退出
            # evaluate on validation set
            if (epoch + 1) % self.args.eval_freq == 0 or epoch == self.args.epochs - 1:
                prec1 = self.validate()

                # remember best prec@1 and save checkpoint
                is_best = prec1 > self.best_prec1
                self.best_prec1 = max(prec1, self.best_prec1)
                self._save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'best_prec1': self.best_prec1,
                }, is_best)
            else:
                # 每次存储检查点而不验证
                self._save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'best_prec1': self.best_prec1,
                }, False)

    def _check_rootfolders(self):
        """Create log and model folder"""
        folders_util = [self.args.root_log, self.args.root_model, self.args.root_output]
        for folder in folders_util:
            if not os.path.exists(folder):
                print('creating folder ' + folder)
                os.mkdir(folder)

    def _save_checkpoint(self, state, is_best):
        # 存在log/..._checkpoint.pth.tar中
        torch.save(state, '%s/%s_checkpoint.pth.tar' % (self.args.root_model, self.args.store_name))
        if is_best:
            shutil.copyfile('%s/%s_checkpoint.pth.tar' % (self.args.root_model, self.args.store_name),
                            '%s/%s_best.pth.tar' % (self.args.root_model, self.args.store_name))

    def _adjust_learning_rate(self, epoch, lr_steps):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        decay = 0.5 ** (sum(epoch >= np.array(lr_steps)))
        lr = self.args.lr * decay
        decay = self.args.weight_decay
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr * param_group['lr_mult']
            param_group['weight_decay'] = decay * param_group['decay_mult']

    def train(self, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        if self.args.no_partialbn:
            self.model.module.partialBN(False)
        else:
            self.model.module.partialBN(True)

        # switch to train mode
        self.model.train()

        end = time.time()
        for i, (input_data, target) in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            target = target.cuda()
            input_var = Variable(input_data)
            target_var = Variable(target)

            # compute output
            output = self.model(input_var)
            loss = self.criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), input_data.size(0))
            top1.update(prec1.item(), input_data.size(0))
            top5.update(prec5.item(), input_data.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()

            loss.backward()

            if self.args.clip_gradient is not None:
                total_norm = clip_grad_norm(self.model.parameters(), self.args.clip_gradient)

            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.args.print_freq == 0:
                output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(self.train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5, lr=self.optimizer.param_groups[-1]['lr']))
                print(output)

                self.output_log.write(output + '\n')
                self.output_log.flush()

                self.board_writer.add_scalar("loss", losses.val, global_step=epoch * len(self.train_loader) + i)
                self.board_writer.add_scalar("prec@1", top1.val, global_step=epoch * len(self.train_loader) + i)
                self.board_writer.add_scalar("prec@5", top5.val, global_step=epoch * len(self.train_loader) + i)
                self.board_writer.add_scalar("lr", self.optimizer.param_groups[-1]['lr'],
                                             global_step=epoch * len(self.train_loader) + i)
                self.board_writer.flush()

    def validate(self):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        for i, (input_data, target) in enumerate(self.val_loader):
            target = target.cuda()
            with torch.no_grad():
                input_var = Variable(input_data)
                target_var = Variable(target)

            # compute output
            output = self.model(input_var)
            loss = self.criterion(output, target_var)

            # measure accuracy and record loss
            # 测量精度并记录损失(最高值为正确的精确度，前5大值包含预测正确的精确度)
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.data.item(), input_data.size(0))
            top1.update(prec1.item(), input_data.size(0))
            top5.update(prec5.item(), input_data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                          .format(i, len(self.val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5))
                print(output)
                self.output_log.write(output + '\n')
                self.output_log.flush()

        output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                  .format(top1=top1, top5=top5, loss=losses))
        print(output)
        output_best = '\nBest Prec@1: %.3f' % self.best_prec1
        print(output_best)
        self.output_log.write(output + ' ' + output_best + '\n')
        self.output_log.flush()

        return top1.avg


def main():
    pass


if __name__ == '__main__':
    main()
