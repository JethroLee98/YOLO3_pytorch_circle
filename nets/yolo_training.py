import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn

class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, cuda, anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]):
        super(YOLOLoss, self).__init__()
        self.anchors        = anchors
        self.num_classes    = num_classes
        self.bbox_attrs     = 5 + num_classes
        self.input_shape    = input_shape
        self.anchors_mask   = anchors_mask

        self.giou           = True
        self.balance        = [0.4, 1.0, 4]
        self.box_ratio      = 0.05
        self.obj_ratio      = 5 * (input_shape[0] * input_shape[1]) / (416 ** 2)
        self.cls_ratio      = 1 * (num_classes / 80)

        self.ignore_threshold = 0.5
        self.cuda           = cuda

    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    def MSELoss(self, pred, target):
        return torch.pow(pred - target, 2)

    def BCELoss(self, pred, target):
        epsilon = 1e-7
        pred    = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)
        output  = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output

    def box_giou(self, b1, b2):
        b1_xy       = b1[..., :2]
        b1_wh       = b1[..., 2:4]
        b1_wh_half  = b1_wh/2.
        b1_mins     = b1_xy - b1_wh_half
        b1_maxes    = b1_xy + b1_wh_half
        b2_xy       = b2[..., :2]
        b2_wh       = b2[..., 2:4]
        b2_wh_half  = b2_wh/2.
        b2_mins     = b2_xy - b2_wh_half
        b2_maxes    = b2_xy + b2_wh_half
        intersect_mins  = torch.max(b1_mins, b2_mins)
        intersect_maxes = torch.min(b1_maxes, b2_maxes)
        intersect_wh    = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
        intersect_area  = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area         = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area         = b2_wh[..., 0] * b2_wh[..., 1]
        union_area      = b1_area + b2_area - intersect_area
        iou             = intersect_area / union_area
        enclose_mins    = torch.min(b1_mins, b2_mins)
        enclose_maxes   = torch.max(b1_maxes, b2_maxes)
        enclose_wh      = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
        enclose_area    = enclose_wh[..., 0] * enclose_wh[..., 1]
        giou            = iou - (enclose_area - union_area) / enclose_area
        
        return giou
    
    def box_giou_cirlce(self, b1, b2):
        b1_xy       = b1[..., :2]
        b1_wh       = b1[..., 2:4]
        b1_wh_half  = b1_wh/2.
        b1_mins     = b1_xy - b1_wh_half
        b1_maxes    = b1_xy + b1_wh_half
        b2_xy       = b2[..., :2]
        b2_wh       = b2[..., 2:4]
        b2_wh_half  = b2_wh/2.
        b2_mins     = b2_xy - b2_wh_half
        b2_maxes    = b2_xy + b2_wh_half
        r1 = torch.clamp(b1_wh[..., 0]/2, min=b1_wh[..., 1]/2)
        r2 = torch.clamp(b2_wh[..., 0]/2, min=b2_wh[..., 1]/2)
        R = torch.clamp(r1, min=r2)
        r = torch.clamp(r1, max=r2)
        d = ((b1_xy[..., 0]-b2_xy[..., 0])**2 + (b1_xy[..., 1]-b2_xy[..., 1])**2)**0.5
        
        part1 = r*r*torch.acos((d*d + r*r - R*R)/(2*d*r))
        part2 = R*R*torch.acos((d*d + R*R - r*r)/(2*d*R))
        part3 = 0.5*torch.sqrt((-d+r+R)*(d+r-R)*(d-r+R)*(d+r+R))
        intersect_area = part1 + part2 - part3
        intersect_area[R>=d+r] = math.pi*r[R>=d+r]**2
        intersect_area[d>=R+r] = 0
        b1_area         = math.pi*r1*r1
        b2_area         = math.pi*r2*r2
        union_area      = b1_area + b2_area - intersect_area
        iou             = intersect_area / union_area

        r3 = R+r+d
        r3[R>=r+d] = R[R>=r+d]
        enclose_area    = math.pi*r3*r3
        giou            = iou - (enclose_area - union_area) / enclose_area
        
        return giou
        
    def forward(self, l, input, targets=None):
        bs      = input.size(0)
        in_h    = input.size(2)
        in_w    = input.size(3)
        stride_h = self.input_shape[0] / in_h
        stride_w = self.input_shape[1] / in_w
        scaled_anchors  = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]
        prediction = input.view(bs, len(self.anchors_mask[l]), self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()
        
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])
        y_true, noobj_mask, box_loss_scale = self.get_target(l, targets, scaled_anchors, in_h, in_w)

        noobj_mask, pred_boxes = self.get_ignore(l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask)

        if self.cuda:
            y_true          = y_true.type_as(x)
            noobj_mask      = noobj_mask.type_as(x)
            box_loss_scale  = box_loss_scale.type_as(x)
        box_loss_scale = 2 - box_loss_scale
            
        loss        = 0
        obj_mask    = y_true[..., 4] == 1
        n           = torch.sum(obj_mask)
        if n != 0:
            if self.giou:
                giou        = self.box_giou(pred_boxes, y_true[..., :4]).type_as(x)
                loss_loc    = torch.mean((1 - giou)[obj_mask])
            else:
                loss_x      = torch.mean(self.BCELoss(x[obj_mask], y_true[..., 0][obj_mask]) * box_loss_scale[obj_mask])
                loss_y      = torch.mean(self.BCELoss(y[obj_mask], y_true[..., 1][obj_mask]) * box_loss_scale[obj_mask])
                loss_w      = torch.mean(self.MSELoss(w[obj_mask], y_true[..., 2][obj_mask]) * box_loss_scale[obj_mask])
                loss_h      = torch.mean(self.MSELoss(h[obj_mask], y_true[..., 3][obj_mask]) * box_loss_scale[obj_mask])
                loss_loc    = (loss_x + loss_y + loss_h + loss_w) * 0.1

            loss_cls    = torch.mean(self.BCELoss(pred_cls[obj_mask], y_true[..., 5:][obj_mask]))
            loss        += loss_loc * self.box_ratio + loss_cls * self.cls_ratio

        loss_conf   = torch.mean(self.BCELoss(conf, obj_mask.type_as(conf))[noobj_mask.bool() | obj_mask])
        loss        += loss_conf * self.balance[l] * self.obj_ratio
        # if n != 0:
        #     print(loss_loc * self.box_ratio, loss_cls * self.cls_ratio, loss_conf * self.balance[l] * self.obj_ratio)
        return loss

    def calculate_iou(self, _box_a, _box_b):
        b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
        b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
        b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
        b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2
        box_a = torch.zeros_like(_box_a)
        box_b = torch.zeros_like(_box_b)
        box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
        box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2

        A = box_a.size(0)
        B = box_b.size(0)

        max_xy  = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy  = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter   = torch.clamp((max_xy - min_xy), min=0)
        inter   = inter[:, :, 0] * inter[:, :, 1]
        area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
        union = area_a + area_b - inter
        return inter / union  # [A,B]

    def calculate_iou_circle(box_a, box_b):

        A = box_a.size(0)
        B = box_b.size(0)
        box_a = box_a.unsqueeze(1).expand(A, B, 4)
        box_b = box_b.unsqueeze(0).expand(A, B, 4)
        
        r1 = torch.clamp((box_a[:, :, 2]-box_a[:, :, 0])/2, min=(box_a[:, :, 3]-box_a[:, :, 1])/2)
        r2 = torch.clamp((box_b[:, :, 2]-box_b[:, :, 0])/2, min=(box_b[:, :, 3]-box_b[:, :, 1])/2)
        R = torch.clamp(r1, min=r2)
        r = torch.clamp(r1, max=r2)
        cx1 = (box_a[:, :, 2]+box_a[:, :, 0])/2
        cy1 = (box_a[:, :, 3]+box_a[:, :, 1])/2
        cx2 = (box_b[:, :, 2]+box_b[:, :, 0])/2
        cy2 = (box_b[:, :, 3]+box_b[:, :, 1])/2
        
        d = ((cx2-cx1)**2 + (cy2-cy1)**2)**0.5
        
        part1 = r*r*torch.acos((d*d + r*r - R*R)/(2*d*r))
        part2 = R*R*torch.acos((d*d + R*R - r*r)/(2*d*R))
        part3 = 0.5*torch.sqrt((-d+r+R)*(d+r-R)*(d-r+R)*(d+r+R))
        intersectionArea = part1 + part2 - part3
        intersectionArea[R>=d+r] = math.pi*r[R>=d+r]**2
        intersectionArea[d>=R+r] = 0
        area_a = torch.tensor(np.ones([A, B]))*math.pi*r1*r1
        area_b = torch.tensor(np.ones([A, B]))*math.pi*r2*r2
        union = area_a + area_b - intersectionArea
        return intersectionArea / union  # [A,B]
    
    def get_target(self, l, targets, anchors, in_h, in_w):
        bs              = len(targets)
        noobj_mask      = torch.ones(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad = False)
        box_loss_scale  = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad = False)
        y_true          = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, self.bbox_attrs, requires_grad = False)
        for b in range(bs):            
            if len(targets[b])==0:
                continue
            batch_target = torch.zeros_like(targets[b])
            batch_target[:, [0,2]] = targets[b][:, [0,2]] * in_w
            batch_target[:, [1,3]] = targets[b][:, [1,3]] * in_h
            batch_target[:, 4] = targets[b][:, 4]
            batch_target = batch_target.cpu()
            gt_box          = torch.FloatTensor(torch.cat((torch.zeros((batch_target.size(0), 2)), batch_target[:, 2:4]), 1))
            anchor_shapes   = torch.FloatTensor(torch.cat((torch.zeros((len(anchors), 2)), torch.FloatTensor(anchors)), 1))
            best_ns = torch.argmax(self.calculate_iou_circle(gt_box, anchor_shapes), dim=-1)

            for t, best_n in enumerate(best_ns):
                if best_n not in self.anchors_mask[l]:
                    continue
                k = self.anchors_mask[l].index(best_n)
                i = torch.floor(batch_target[t, 0]).long()
                j = torch.floor(batch_target[t, 1]).long()
                c = batch_target[t, 4].long()
                noobj_mask[b, k, j, i] = 0
                if not self.giou:
                    y_true[b, k, j, i, 0] = batch_target[t, 0] - i.float()
                    y_true[b, k, j, i, 1] = batch_target[t, 1] - j.float()
                    y_true[b, k, j, i, 2] = math.log(batch_target[t, 2] / anchors[best_n][0])
                    y_true[b, k, j, i, 3] = math.log(batch_target[t, 3] / anchors[best_n][1])
                    y_true[b, k, j, i, 4] = 1
                    y_true[b, k, j, i, c + 5] = 1
                else:
                    y_true[b, k, j, i, 0] = batch_target[t, 0]
                    y_true[b, k, j, i, 1] = batch_target[t, 1]
                    y_true[b, k, j, i, 2] = batch_target[t, 2]
                    y_true[b, k, j, i, 3] = batch_target[t, 3]
                    y_true[b, k, j, i, 4] = 1
                    y_true[b, k, j, i, c + 5] = 1
                box_loss_scale[b, k, j, i] = batch_target[t, 2] * batch_target[t, 3] / in_w / in_h
        return y_true, noobj_mask, box_loss_scale

    def get_ignore(self, l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask):
        bs = len(targets)
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(x.shape).type_as(x)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(y.shape).type_as(x)

        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]
        anchor_w = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([0])).type_as(x)
        anchor_h = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([1])).type_as(x)
        
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        pred_boxes_x    = torch.unsqueeze(x + grid_x, -1)
        pred_boxes_y    = torch.unsqueeze(y + grid_y, -1)
        pred_boxes_w    = torch.unsqueeze(torch.exp(w) * anchor_w, -1)
        pred_boxes_h    = torch.unsqueeze(torch.exp(h) * anchor_h, -1)
        pred_boxes      = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h], dim = -1)
        
        for b in range(bs):           
            pred_boxes_for_ignore = pred_boxes[b].view(-1, 4)
            if len(targets[b]) > 0:
                batch_target = torch.zeros_like(targets[b])
                batch_target[:, [0,2]] = targets[b][:, [0,2]] * in_w
                batch_target[:, [1,3]] = targets[b][:, [1,3]] * in_h
                batch_target = batch_target[:, :4].type_as(x)
                anch_ious = self.calculate_iou(batch_target, pred_boxes_for_ignore)
                anch_ious_max, _    = torch.max(anch_ious, dim = 0)
                anch_ious_max       = anch_ious_max.view(pred_boxes[b].size()[:3])
                noobj_mask[b][anch_ious_max > self.ignore_threshold] = 0
        return noobj_mask, pred_boxes

def weights_init(net, init_type='normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
