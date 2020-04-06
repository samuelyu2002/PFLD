import torch
from torch import nn
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PFLDLoss(nn.Module):
    def __init__(self):
        super(PFLDLoss, self).__init__()

    def forward(self, attribute_gt, landmark_gt, euler_angle_gt, angle, landmarks, train_batchsize):
        weight_angle = torch.sum(1 - torch.cos(angle - euler_angle_gt), axis=1)
        attributes_w_n = attribute_gt[:, 1:6].float()
        # print(attributes_w_n)
        mat_ratio = torch.mean(attributes_w_n, axis=0)
        # print(mat_ratio)
        mat_ratio = torch.Tensor([
            1.0 / (x) if x > 0 else train_batchsize for x in mat_ratio
        ]).to(device)
        weight_attribute = torch.sum(attributes_w_n.mul(mat_ratio), axis=1)
        # print(weight_attribute)
        # assert(False)
        l2_distant = torch.sum((landmark_gt - landmarks) * (landmark_gt - landmarks), axis=1)
        return torch.mean(weight_angle * weight_attribute * l2_distant), torch.mean(l2_distant)

def smoothL1(y_true, y_pred, beta = 1):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    mae = torch.abs(y_true - y_pred)
    loss = torch.sum(torch.where(mae>beta, mae-0.5*beta , 0.5*mae**2/beta), axis=-1)
    return torch.mean(loss)

class WingLoss(nn.Module):
    def __init__(self, omega=10.0, epsilon=2.0):
        super(WingLoss, self).__init__()
        self.epsilon=epsilon
        self.omega = omega

    def forward(self,y_true, y_pred):
        x = y_true - y_pred
        c = self.omega * (1.0 - math.log(1.0 + self.omega / self.epsilon))
        absolute_x = torch.abs(x)
        losses = torch.where(self.omega > absolute_x, self.omega * torch.log(1.0 + absolute_x/self.epsilon), absolute_x - c)
        losses = torch.mean(losses, axis = 1)
        return torch.mean(losses)

class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, target, pred):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = A * delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))