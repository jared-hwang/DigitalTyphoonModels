import torch
import numpy as np
class PadSequence(object):

    def __init__(self, max_length, pad_token, sos_token, eos_token, label_range=(860, 1020), img_range=(170, 300)):
        self.max_length = max_length
        self.PAD_token = pad_token
        self.sos_token = torch.Tensor(sos_token)
        self.eos_token = torch.Tensor(eos_token)

        self.label_range = label_range
        self.img_range = img_range

    def __call__(self, received_sample):
        sample, labels = received_sample

        # Transform image
        if self.img_range == (0, 0):
            sample = torch.Tensor([])
        else:
            sample = torch.from_numpy(sample)
            

        # sample = torch.from_numpy(sample)
        # pad_length = self.max_length - sample.size()[0]
        # pad = torch.zeros(pad_length, sample.size(1), sample.size(2))
        # sample = torch.cat((pad, sample), dim=0)
        # sample = torch.reshape(sample, [sample.size()[0], 1, sample.size()[1], sample.size()[2]])

        # Transform labels 
        labels = np.clip(labels, self.label_range[0], self.label_range[1]) 
        labels = self.map_label_to_rep(labels)
        labels = torch.Tensor(labels)
        
        src = torch.cat((self.sos_token, labels, self.eos_token), dim=0)
        tgt = torch.cat((self.sos_token, labels), dim=0)
        tgt_expected = torch.cat((labels, self.eos_token), dim=0)

        src_pad = torch.full(torch.Size([self.max_length - src.size()[0]]), self.PAD_token)
        src = torch.cat((src, src_pad), dim=0)

        tgt_pad = torch.full(torch.Size([self.max_length - tgt.size()[0]]), self.PAD_token)
        tgt = torch.cat((tgt, tgt_pad), dim=0)

        tgt_expected_pad = torch.full(torch.Size([self.max_length - tgt_expected.size()[0]]), self.PAD_token)
        tgt_expected = torch.cat((tgt_expected, tgt_expected_pad), dim=0)

        src, tgt, tgt_expected = src.long(), tgt.long(), tgt_expected.long()
        return sample, src, tgt, tgt_expected

    def map_label_to_rep(self, labels):
        # -min_val to bring to 0, +5 to account
        # for eos, sos, and pad (2, 3, 4)
        # *10 to turn single dec float to int
        # + temp_range to include img in the mapping rep
        labels = labels * 10
        labels = labels - (10*self.label_range[0]) + 5 + self.img_range[1]
        return labels
    
    def map_label_rep_to_label(self, label_rep):
        labels = label_rep - 5 + (10*self.label_range[0]) - self.img_range[1]
        labels = labels / 10
        return labels
    
    def map_img_to_rep(self, img):
        img = img - self.img_range[0] + 5
        return img

    def map_img_rep_to_label(self, img_rep):
        img = img - 5 + self.img_range[0]
        img = img / 10
        return img
