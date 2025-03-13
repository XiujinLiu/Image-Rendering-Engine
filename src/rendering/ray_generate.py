import torch
import math


def gen_ray(R, depth, K, position, h, w, DEVICE):
    ray_param = {}
    camera_z = R[:, 2]
    camera_x = R[:, 0]
    camera_y = R[:, 1]
    fovy = 2 * torch.atan(h / 2 / K[1, 1])
    fovx = 2 * torch.atan(w / 2 / K[0, 0])
    col, row = torch.meshgrid(torch.arange(h, device=DEVICE), torch.arange(w, device=DEVICE))
    u0 = K[0, 2]
    v0 = K[1, 2]
    screenHeight = depth * math.tan(fovy / 2.0) * 2.0
    screenWidth = depth * math.tan(fovx / 2.0) * 2.0
    screenCenter = torch.zeros((h,w,3),device=DEVICE)
    screenCenter += depth * camera_z + position
    screenPoint = screenCenter
    screenPoint = screenPoint + ((row - u0)/w)[:,:,None] * screenWidth * camera_x
    screenPoint = screenPoint + ((col - v0)/h)[:,:,None] * screenHeight * camera_y
    direction_mat = screenPoint - position
    norm_mat = torch.norm(direction_mat,dim=2)
    ray_param['origin'] = position
    ray_param['direction'] = direction_mat / norm_mat[:,:,None]
    # print(row)
    # print(col)
    return ray_param