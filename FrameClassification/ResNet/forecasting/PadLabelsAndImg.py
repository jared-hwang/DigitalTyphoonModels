import torch
import numpy as np
from torch import nn


class PadSequence(object):

    def __init__(self, max_length, pad_token, sos_token, eos_token, label_range=(860, 1020), img_range=(170, 300)):
        self.max_length = max_length
        self.PAD_token = pad_token
        self.SOS_token = sos_token
        self.EOS_token = eos_token

        self.label_range = label_range
        self.img_range = img_range

    def __call__(self, received_sample):
        sample, labels = received_sample

        # Prepare the images
        sample = torch.Tensor(sample).long()
        sample = self.map_img_to_rep(sample)
        sample = torch.flatten(sample, start_dim=1, end_dim=2)

        # pad_length = self.max_length - sample.size()[0]
        # pad = torch.full(torch.Size([self.max_length - sample.size()[0], sample.size()[1]]), self.PAD_token)
        # sample = torch.cat((sample, pad), dim=0)            

        # Transform labels and add the convolved images
        labels = np.clip(labels, self.label_range[0], self.label_range[1])
        labels = self.map_label_to_rep(labels)
        labels = torch.Tensor(labels)
        labels = torch.reshape(labels, [labels.size()[0], 1])

        # src: sos, 1, 2, 3, 4, 5
        # tgt: sos, 1, 2, 3, 4
        # tgt_expected: 1, 2, 3, 4, 5
        src = torch.cat((labels, sample), dim=1)
        tgt = src[:-1]
        tgt_expected = src[1:]

        src_pad = torch.full(torch.Size([self.max_length - src.size()[0], src.size()[1]]), self.PAD_token)
        src = torch.cat((src, src_pad), dim=0)

        tgt_pad = torch.full(torch.Size([self.max_length - 1 - tgt.size()[0], tgt.size()[1]]), self.PAD_token)
        tgt = torch.cat((tgt, tgt_pad), dim=0)

        tgt_expected_pad = torch.full(torch.Size([self.max_length - tgt_expected.size()[0], tgt_expected.size()[1]]), self.PAD_token)
        tgt_expected = torch.cat((tgt_expected, tgt_expected_pad), dim=0)


        sos_row = torch.full(torch.Size([1, src.size()[1]]), int(self.SOS_token))
        src = torch.cat((sos_row, src), dim=0)
        tgt = torch.cat((sos_row, tgt), dim=0)
        
        src, tgt, tgt_expected = src.long(), tgt.long(), tgt_expected.long()
        return src, tgt, tgt_expected

    def map_label_to_rep(self, labels):
        # -min_val to bring to 0, +5 to account
        # for eos, sos, and pad (2, 3, 4)
        # *10 to turn single dec float to int
        # + temp_range to include img in the mapping rep
        labels = labels * 10
        labels = labels - (10*self.label_range[0]) + 5 + (self.img_range[1] - self.img_range[0])
        return labels
    
    def map_label_rep_to_label(self, label_rep):
        labels = label_rep - 5 + (10*self.label_range[0]) - (self.img_range[1] - self.img_range[0])
        labels = labels / 10
        return labels
    
    def map_img_to_rep(self, img):
        img = img - self.img_range[0] + 5
        return img

    def map_img_rep_to_label(self, img_rep):
        img = img - 5 + self.img_range[0]
        img = img / 10
        return img


class ConvolveAndFlatten(object):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
    
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
    
    def __call__(self, received_sample):
        sample, labels = received_sample        
        print('1', sample)
        sample = torch.Tensor(sample)
        num_images = sample.size()[0]
        img_size = sample.size()[1]
        sample = torch.reshape(sample, (num_images, 1, img_size, img_size))
        sample = self.conv(sample)
        convolved_img_size = sample.size()[2]
        sample = torch.reshape(sample, (num_images, convolved_img_size, convolved_img_size))
        sample = torch.flatten(sample, start_dim=1, end_dim=2)
        return sample, labels