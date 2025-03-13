import torch


def calc_geometry(ray_origin, A_position, B_position, C_position):
    out_normal = torch.linalg.cross(B_position - A_position, C_position - B_position)
    norm_mat = torch.norm(out_normal,dim=1)
    normal = out_normal / norm_mat[:,None]
    E1 = A_position - C_position
    E2 = B_position - C_position
    T = ray_origin - C_position
    
