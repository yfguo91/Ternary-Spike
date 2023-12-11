import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from models.spike_block import is_normal_blk, is_spike_blk
from models.spike_layer import SpikeConv, SpikeModule


class LayerReconstructionController:

    def __init__(self, student: nn.Module, teacher: nn.Module,):

        self.student = student
        self.teacher = teacher

        self.break_list = ['layer1', 'layer2', 'layer3', 'layer4', 'fc']

    def recon_model(self, train_loader, val_loader, epoch=20, lr=0.1, wd=5e-5, batch_size=128):
        for idx in range(len(self.break_list)):
            print('Reconstruction Layer: {}/{}\n\n'.format(idx, len(self.break_list)))
            self.recon_layer(idx, train_loader, val_loader, epoch, lr, wd, batch_size)
        self.student.set_spike_state(True)

    def recon_layer(self, recon_idx, train_loader, val_loader, epoch, lr, wd, batch_size):
        self.student.set_spike_before(self.break_list[recon_idx])
        saver = DataSaverHook()
        handle_t = self.teacher.register_forward_hook(saver)
        handle_s = self.student.register_forward_hook(saver)

        # initialize training envrionments
        params = []
        for n, m in self.student.model.named_modules():
            if isinstance(m, SpikeModule):
                if m._spiking is True:
                    params += m.parameters()
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=epoch*len(train_loader))
        device = 'cuda'
        self.student.train()

        for e in range(epoch):
            running_loss = 0.
            start_time = time.time()
            for i, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                images = images.to(device)
                loss = self.compute_mse_loss_given_data(images, saver=saver)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
                if (i + 1) % 100 == 0:
                    print('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                          % (epoch + 1, e, i + 1, len(train_loader) // batch_size, running_loss))
                    running_loss = 0
                    print('Time elapsed:', time.time() - start_time)
                scheduler.step()

            self.validate(val_loader)

        handle_s.remove()
        handle_t.remove()
        del saver

    def compute_mse_loss_given_data(self, model_input, saver):
        saver.get_teacher = True
        with torch.no_grad():
            _ = self.teacher(model_input)
        saver.get_teacher = False
        with torch.enable_grad():
            _ = self.student(model_input)
        return saver.get_mse_loss()

    def validate(self, test_loader):
        correct = 0
        total = 0
        self.student.eval()
        device = next(self.student.parameters()).device
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = self.student(inputs)
                _, predicted = outputs.cpu().max(1)
                total += float(targets.size(0))
                correct += float(predicted.eq(targets.cpu()).sum().item())
                if batch_idx % 100 == 0:
                    acc = 100. * float(correct) / float(total)
                    print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)

        print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))
        self.student.train()


class DataSaverHook:
    """
    Forward hook that stores the input and output of a block
    """
    def __init__(self, stop_forward=True):
        self.get_teacher = True
        self.stop_forward = stop_forward

        self.output_student = None
        self.output_teacher = None

    def __call__(self, module, input_batch, output_batch):
        if self.get_teacher is True:
            self.output_teacher = output_batch
        else:
            self.output_student = output_batch

    def get_mse_loss(self):
        return F.kl_div(F.log_softmax(self.output_student, dim=1),
                        F.softmax( self.output_teacher, dim=1), reduction='batchmean')


# from models.resnet import resnet20_cifar_modified
# from models.spike_model import SpikeModel
#
# snn = SpikeModel(resnet20_cifar_modified(), step=2)
# teacher = resnet20_cifar_modified()
#
# snn.set_spike_before('fc')
# control = LayerReconstructionController(snn, teacher)
# print(control.teacher_module_list)
