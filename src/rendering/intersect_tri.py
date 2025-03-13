import torch


def intersect_triangle_mt(ray_origin, ray_direction, A_position, B_position, C_position):
    out_param = {}
    E1 = A_position - C_position
    E2 = B_position - C_position
    # print(ray_direction.shape)
    T = ray_origin - C_position
    direction_mat = ray_direction[:,:,None,:]
    E2 = E2[None,None,:,:]
    P = torch.linalg.cross(direction_mat, E2)
    Q = torch.linalg.cross(T, E1)
    inv_PE1 = 1.0 / torch.sum(P * E1[None,None,:,:],dim=-1)
    distance = (torch.sum(Q * E2,dim=-1) * inv_PE1)
    u = torch.sum(P * T,dim=-1) * inv_PE1
    v = torch.sum(Q * direction_mat,dim=-1) * inv_PE1
    valid = (distance>0.001) & (u>=0) & (v>=0) & (u + v <=1)
    mask = torch.ones_like(valid)
    mask[valid] = 0
    distance[mask]=99999999
    pos_all = ray_origin[None,None,None,:] + distance[:,:,:,None] * direction_mat
    out_param['intersect_position'] = pos_all
    out_param['ray_vector'] = distance[:,:,:,None] * direction_mat
    out_param['distance'] = distance
    print(mask.shape)
    return out_param, mask


