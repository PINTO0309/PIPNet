import torch
import torch.nn as nn
import torch.nn.functional as F


class PIPNet(nn.Module):
    def __init__(self, args, params, resnet, reverse_index1, reverse_index2, max_len):
        super(PIPNet, self).__init__()
        self.args = args
        self.params = params
        self.resnet = resnet
        self.max_len = max_len
        self.reverse_index1 = reverse_index1
        self.reverse_index2 = reverse_index2
        self.init_resnet(self.resnet)

        self.cls_layer = nn.Conv2d(512, self.params['num_lms'], kernel_size=1, stride=1, padding=0)
        self.x_layer = nn.Conv2d(512, self.params['num_lms'], kernel_size=1, stride=1, padding=0)
        self.y_layer = nn.Conv2d(512, self.params['num_lms'], kernel_size=1, stride=1, padding=0)
        self.nb_x_layer = nn.Conv2d(512, self.params['num_nb'] * self.params['num_lms'], kernel_size=1, stride=1,
                                    padding=0)
        self.nb_y_layer = nn.Conv2d(512, self.params['num_nb'] * self.params['num_lms'], kernel_size=1, stride=1,
                                    padding=0)

        nn.init.normal_(self.cls_layer.weight, std=0.001)
        if self.cls_layer.bias is not None:
            nn.init.constant_(self.cls_layer.bias, 0)

        nn.init.normal_(self.x_layer.weight, std=0.001)
        if self.x_layer.bias is not None:
            nn.init.constant_(self.x_layer.bias, 0)

        nn.init.normal_(self.y_layer.weight, std=0.001)
        if self.y_layer.bias is not None:
            nn.init.constant_(self.y_layer.bias, 0)

        nn.init.normal_(self.nb_x_layer.weight, std=0.001)
        if self.nb_x_layer.bias is not None:
            nn.init.constant_(self.nb_x_layer.bias, 0)

        nn.init.normal_(self.nb_y_layer.weight, std=0.001)
        if self.nb_y_layer.bias is not None:
            nn.init.constant_(self.nb_y_layer.bias, 0)

    def init_resnet(self, resnet):
        self.conv, self.bn, self.maxpool = resnet.conv1, resnet.bn1, resnet.maxpool
        self.block1, self.block2, self.block3, self.block4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        outputs_cls = self.cls_layer(x)
        outputs_x = self.x_layer(x)
        outputs_y = self.y_layer(x)
        outputs_nb_x = self.nb_x_layer(x)
        outputs_nb_y = self.nb_y_layer(x)
        if self.training:
            return outputs_cls, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y

        tmp_batch, tmp_channel, tmp_height, tmp_width = outputs_cls.size()
        assert tmp_batch == 1

        n,c,h,w = outputs_cls.shape
        outputs_cls = outputs_cls.view(n, c, h*w)
        max_ids = torch.argmax(outputs_cls, 2, keepdim=True)
        max_cls = torch.max(outputs_cls, 2, keepdim=True)

        max_ids = max_ids.view(n, c, 1)
        max_ids_nb = max_ids.repeat(1, 1, self.params['num_nb']).view(1, c*self.params['num_nb'], 1)

        outputs_x = outputs_x.view(tmp_batch, tmp_channel, tmp_height * tmp_width)
        outputs_x_select = torch.gather(outputs_x, 1, max_ids)

        # outputs_x_select = outputs_x_select.squeeze(1)
        outputs_y = outputs_y.view(tmp_batch, tmp_channel, tmp_height * tmp_width)
        outputs_y_select = torch.gather(outputs_y, 1, max_ids)

        # outputs_y_select = outputs_y_select.squeeze(1)

        # outputs_nb_x = outputs_nb_x.view(tmp_batch * self.params['num_nb'] * tmp_channel, -1)
        outputs_nb_x = outputs_nb_x.view(tmp_batch, self.params['num_nb'] * tmp_channel, tmp_height * tmp_width)
        outputs_nb_x_select = torch.gather(outputs_nb_x, 1, max_ids_nb)
        outputs_nb_x_select = outputs_nb_x_select.view(n, c, self.params['num_nb'])

        outputs_nb_y = outputs_nb_y.view(tmp_batch, self.params['num_nb'] * tmp_channel, tmp_height * tmp_width)
        outputs_nb_y_select = torch.gather(outputs_nb_y, 1, max_ids_nb)
        outputs_nb_y_select = outputs_nb_y_select.view(n, c, self.params['num_nb'])

        tmp_x = (max_ids % tmp_width).view(n, c, 1).float() + outputs_x_select.view(n, c, 1)
        tmp_y = (max_ids // tmp_width).view(n, c, 1).float() + outputs_y_select.view(n, c, 1)
        tmp_x /= 1.0 * self.args.input_size / self.params['stride']
        tmp_y /= 1.0 * self.args.input_size / self.params['stride']

        # tmp_x: [68,1]
        # tmp_y: [68,1]
        tmp_nb_x = (max_ids % tmp_width).view(n, c, 1).float() + outputs_nb_x_select
        tmp_nb_y = (max_ids // tmp_width).view(n, c, 1).float() + outputs_nb_y_select
        tmp_nb_x = tmp_nb_x.view(n, c, self.params['num_nb'])
        tmp_nb_y = tmp_nb_y.view(n, c, self.params['num_nb'])
        tmp_nb_x /= 1.0 * self.args.input_size / self.params['stride']
        tmp_nb_y /= 1.0 * self.args.input_size / self.params['stride']

        # tmp_x.shape
        #   torch.Size([1, 68, 22]) -> 1496
        # tmp_y.shape
        #   torch.Size([1, 68, 22]) -> 1496
        # self.reverse_index1
        #   [1496]
        # self.reverse_index2
        #   [1496]

        # tmp_nb_x = tmp_nb_x.reshape(n, self.params['num_lms'] * self.params['num_nb'])
        # tmp_nb_y = tmp_nb_y.reshape(n, self.params['num_lms'] * self.params['num_nb'])

        """
        self.reverse_index1
            0000:
            1
            0001:
            2
            0002:
            17
            0003:
            18
            0004:
            36
            0005:
            1
            0006:
            2
            0007:
            17
            0008:
            18
            0009:
            36
            0010:
            1
            0011:
            2
            0012:
            17

            self.reverse_index2
            0000:
            0
            0001:
            3
            0002:
            1
            0003:
            7
            0004:
            8
            0005:
            0
            0006:
            3
            0007:
            1
            0008:
            7
            0009:
            8
            0010:
            0
            0011:
            3
            0012:
            1
        """
        # tmp_nb_x = tmp_nb_x[:, self.reverse_index1, self.reverse_index2].view(n, self.params['num_lms'], self.max_len)
        # tmp_nb_y = tmp_nb_y[:, self.reverse_index1, self.reverse_index2].view(n, self.params['num_lms'], self.max_len)

        rverse_index1_tensor = torch.tensor(self.reverse_index1)
        rverse_index2_tensor = torch.tensor(self.reverse_index2)
        combined_indices = rverse_index1_tensor * 10 + rverse_index2_tensor
        flat_tmp_nb_x = tmp_nb_x.view(n, self.params['num_lms'] * self.params['num_nb'])
        flat_tmp_nb_y = tmp_nb_y.view(n, self.params['num_lms'] * self.params['num_nb'])
        tmp_nb_x = flat_tmp_nb_x[:, combined_indices].view(n, self.params['num_lms'], self.max_len)
        tmp_nb_y = flat_tmp_nb_y[:, combined_indices].view(n, self.params['num_lms'], self.max_len)

        tmp_x = torch.mean(torch.cat((tmp_x, tmp_nb_x), dim=2), dim=2, keepdim=True)
        tmp_y = torch.mean(torch.cat((tmp_y, tmp_nb_y), dim=2), dim=2, keepdim=True)
        lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=2)

        # return (torch.flatten(lms_pred_merge), max_cls)
        return torch.cat([lms_pred_merge, max_cls[0]], dim=2)



