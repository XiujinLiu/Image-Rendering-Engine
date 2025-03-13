import torch
import torch.nn.functional as F
import torch.cuda
import matplotlib.pyplot as plt
import math
import numpy as np
import cv2
from smoothengine import SmoothEngine
import random
import json
from utils.mesh_loader import MeshLoader
from rendering.ray_generate import gen_ray
from rendering.intersect_tri import intersect_triangle_mt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def draw_approx_hull_polygon(mask):
    mask = np.array(mask, np.uint8)
    cnts, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_side_len = mask.shape[0] / 32
    min_poly_len = mask.shape[0] / 16
    min_side_num = 3
    min_area = 16.0
    approxs = [cv2.approxPolyDP(cnt, min_side_len, True) for cnt in cnts]
    approxs = [approx for approx in approxs if cv2.arcLength(approx, True) > min_poly_len] 
    approxs = [approx for approx in approxs if len(approx) > min_side_num]
    approxs = [approx for approx in approxs if cv2.contourArea(approx) > min_area]
    hulls = [cv2.convexHull(cnt) for cnt in cnts]
    return hulls


class MeshRender:
    def __init__(self):
        # self.smooth_engine = SmoothEngine()
        self.mesh_loader = MeshLoader()

 

    def intersect(self, ray_origin, ray_direction):
        all_hits, mask = intersect_triangle_mt(ray_origin, ray_direction, self.mesh_loader.v0, self.mesh_loader.v1, self.mesh_loader.v2)
        min_index = torch.argmin(all_hits['distance'],axis=2)
        N,M = all_hits['distance'].shape[0:2]
        n_idx, m_idx = torch.meshgrid(torch.arange(N), torch.arange(M), indexing='ij')
        min_index = [n_idx,m_idx,min_index]
        print(mask.shape)
        return all_hits, min_index, mask


    def render_image(self, input_param):
        # load data
        image_height = input_param["image_height"]
        image_width = input_param["image_width"]
        image = torch.zeros((image_height, image_width), dtype=torch.uint8, device = DEVICE)
        R = torch.tensor(input_param["camera_params"]["R"], dtype = torch.float32, device = DEVICE)
        K = torch.tensor(input_param["camera_params"]["K"], dtype = torch.float32, device = DEVICE)
        position = torch.tensor(input_param["camera_params"]["position"], dtype = torch.float32, device = DEVICE)
        camera_z = R[:, 2].view(3, -1)
        batch_size = input_param["batch_size"]
        self.mesh_loader.parse(input_param["model_path"], DEVICE) 
        light_pos = torch.tensor(input_param["light_pos"], dtype = torch.float32, device = DEVICE)


        ray_param = gen_ray(R, input_param["camera_params"]["Z_near"], K, position, image_height, image_width, DEVICE)
        distance = torch.zeros((image_height, image_width), dtype=torch.float32, device = DEVICE)
        ray_vector = torch.zeros((image_height, image_width, 3), dtype=torch.float32, device = DEVICE)
        intersect_pt = torch.zeros((image_height, image_width, 3), dtype=torch.float32, device = DEVICE)
        mask_all = torch.zeros((image_height, image_width, ), dtype=torch.float32, device = DEVICE)
        min_idxx = torch.zeros((image_height, image_width), dtype=torch.int32, device = DEVICE)


        out_normal = torch.linalg.cross(self.mesh_loader.v1 - self.mesh_loader.v0, self.mesh_loader.v2- self.mesh_loader.v1)
        norm_mat = torch.norm(out_normal,dim=1)
        normal = out_normal / norm_mat[:,None]

        # because of memory limitation we setup batch_size
        for i in range(int(image_width/batch_size)):
            for j in range(int(image_height/batch_size)):
                valid_hits, min_idx, mask = self.intersect(ray_param["origin"], 
                                            ray_param["direction"][i*batch_size:(i+1)*batch_size,j*batch_size:(j+1)*batch_size])
                intersect_pt[i*batch_size:(i+1)*batch_size, j*batch_size:(j+1)*batch_size] = valid_hits['intersect_position'][min_idx]
                distance[i*batch_size:(i+1)*batch_size, j*batch_size:(j+1)*batch_size] = valid_hits['distance'][min_idx]
                ray_vector[i*batch_size:(i+1)*batch_size, j*batch_size:(j+1)*batch_size] = valid_hits['ray_vector'][min_idx]
                min_idxx[i*batch_size:(i+1)*batch_size, j*batch_size:(j+1)*batch_size] = min_idx[2]
                mask_all[i*batch_size:(i+1)*batch_size, j*batch_size:(j+1)*batch_size]  =  mask
        for j in range(int(image_height/batch_size)):
            valid_hits, min_idx, mask = self.intersect(ray_param["origin"], 
                                        ray_param["direction"][int(image_width/batch_size)*batch_size:image_width, j*batch_size:(j+1)*batch_size])
            intersect_pt[int(image_width/batch_size)*batch_size:image_width, j*batch_size:(j+1)*batch_size] = valid_hits['intersect_position'][min_idx]
            distance[int(image_width/batch_size)*batch_size:image_width, j*batch_size:(j+1)*batch_size] = valid_hits['distance'][min_idx]
            ray_vector[int(image_width/batch_size)*batch_size:image_width, j*batch_size:(j+1)*batch_size] = valid_hits['ray_vector'][min_idx]
            min_idxx[int(image_width/batch_size)*batch_size:image_width, j*batch_size:(j+1)*batch_size] = min_idx[2]
            mask_all[int(image_width/batch_size)*batch_size:image_width, j*batch_size:(j+1)*batch_size]  =  mask
        valid_hits, min_idx, mask = self.intersect(ray_param["origin"], 
                                    ray_param["direction"][int(image_width/batch_size)*batch_size:image_width, int(image_height/batch_size)*batch_size:image_height])
        intersect_pt[int(image_width/batch_size)*batch_size:image_width, int(image_height/batch_size)*batch_size:image_height] = valid_hits['intersect_position'][min_idx]
        distance[int(image_width/batch_size)*batch_size:image_width, int(image_height/batch_size)*batch_size:image_height] = valid_hits['distance'][min_idx]
        ray_vector[int(image_width/batch_size)*batch_size:image_width, int(image_height/batch_size)*batch_size:image_height] = valid_hits['ray_vector'][min_idx]
        min_idxx[int(image_width/batch_size)*batch_size:image_width, int(image_height/batch_size)*batch_size:image_height] = min_idx[2]
        mask_all[int(image_width/batch_size)*batch_size:image_width, int(image_height/batch_size)*batch_size:image_height]  =  mask
        
        
        
        
        depth_cam = torch.matmul(ray_vector, camera_z[None, None, :])
        depth_cam = depth_cam.view(ray_vector.shape[0], ray_vector.shape[1])
        tmp = light_pos[None,None,:] - intersect_pt
        tmp = tmp / torch.norm(tmp,dim=2)[:,:,None]
        diffuse = torch.sum(tmp * normal[min_idxx],dim=-1)
        diffuse[diffuse<0] = 0
        invalid_part = torch.sum(mask_all,dim=-1)
        diffuse[invalid_part==mask_all.shape[2]] = 0
        image = torch.zeros_like(diffuse)
        image = diffuse * 220

        image = image.cpu().numpy()
        cv2.imwrite('../test.png', mask * 255)
        # depth_cam = depth_cam.cpu().numpy()
        # distance = distance.cpu().numpy()
        # mask = np.zeros((image_height, image_width))
        # mask[distance < 99990] = 255
        # blur=((3,3),1)
        # erode_=(5,5)
        # dilate_=(3, 3)
        # mask_merge = cv2.dilate(cv2.erode(cv2.GaussianBlur(mask/255, blur[0], blur[1]), np.ones(erode_)), np.ones(dilate_))*255
        # if train:
        #     image = self.smooth_engine.smooth_img(image)
        #     mask[mask_merge > 200] = 255
        #     mask[mask_merge <= 200] = 0
        # print(intersect[273, 191])
        # camera_point = np.dot(np.linalg.inv(camera_params['K'].cpu().numpy()), np.array([191,273, 1]) * depth_cam[273, 191])
        # # print(depth_cam[275, 191])
        # print('camera:', camera_point)
        # camera_point2 = np.dot(np.linalg.inv(camera_params['Ext'].cpu().numpy()), np.append(intersect[273, 191].cpu().numpy(), np.array([1])))
        # print(camera_point2)
        # world_point = np.dot(camera_params['Ext'].cpu().numpy(), np.append(camera_point, np.array([1])))
        # # print('camera2', camera_point2)
        # print(world_point)
        # # # find contour
        # hulls = draw_approx_hull_polygon(mask)
        # image[mask == 0] = 255
        # depth_cam = np.transpose(depth_cam)
        # # cv2.imwrite('/home/lxj/Downloads/mask.png', mask * 255)
        # return image, depth_cam, hulls, mask_merge



def main():
    input_param_json = '../input_param.json'
    with open(input_param_json, 'r') as f:
        json_data = f.read()
        f.close()
        input_param = json.loads(json_data)
    mesh = MeshRender()
    mesh.render_image(input_param)
    # image, depth_cam, hulls, mask_merge = mesh.render_image(model_path_list[type], image_width = 512, image_height = 512, camera_params = camera_params, type = type)
    
    # plt.imshow(depth_cam, cmap='gray')
    # plt.show()
    # print(depth_cam[depth_cam < 500])


    # cv2.polylines(image, hulls, True, 255, 1)
    # print(depth_cam.shape)
    # cv2.imwrite('/home/lxj/Downloads/test4.png', image)
    # np.savetxt('/home/lxj/Downloads/real1.txt',depth_cam)
    
    # np.savetxt('/home/lxj/Downloads/inter.txt',intersect)
    

    
if __name__ == "__main__":
    main()