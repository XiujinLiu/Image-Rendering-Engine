# ref: https://github.com/ZFhuang/MLAA-python/blob/main/mlaa-cpu.py
import numpy as np
import cv2

class SmoothEngine():
    def __init__(self):
        pass

    def _find_edges(self, image, th=0.1):
        buffer = np.zeros((image.shape[0], image.shape[1], 3))
        for y in range(1, image.shape[0]):
            for x in range(0, image.shape[1]):
                if abs(image[y, x]-image[y-1, x]) > th:
                    buffer[y, x, 1] = 1
        for y in range(0, image.shape[0]):
            for x in range(1, image.shape[1]):
                if abs(image[y, x]-image[y, x-1]) > th:
                    buffer[y, x, 0] = 1
        return buffer


    def _cal_aliasing_info_x(self, img_edges, start_x, start_y, mask):
        dis = 1
        for x in range(start_x, img_edges.shape[1]):
            if img_edges[start_y, x, 0] == 1 and img_edges[start_y-1, x, 0] == 1:
                pattern = 'H'
                return dis, pattern, mask
            if img_edges[start_y, x, 0] == 1:
                pattern = 'T'
                return dis, pattern, mask
            if img_edges[start_y-1, x, 0] == 1:
                pattern = 'B'
                return dis, pattern, mask
            if img_edges[start_y, x, 1] == 0:
                break
            mask[start_y, x] = 1
            dis+=1
        pattern = 'L'
        return dis, pattern, mask


    def _cal_aliasing_info_y(self, img_edges, start_x, start_y, mask):
        dis = 1
        for y in range(start_y, img_edges.shape[0]):
            if img_edges[y, start_x, 1] == 1 and img_edges[y, start_x-1, 1] == 1:
                pattern = 'H'
                return dis, pattern, mask
            if img_edges[y, start_x, 1] == 1:
                pattern = 'T'
                return dis, pattern, mask
            if img_edges[y, start_x-1, 1] == 1:
                pattern = 'B'
                return dis, pattern, mask
            if img_edges[y, start_x, 0] == 0:
                break
            mask[y, start_x] = 1
            dis+=1
        pattern = 'L'
        return dis, pattern, mask


    def _find_aliasings_x(self, img_edges):
        list_aliasings = []
        mask = np.zeros((img_edges.shape[0], img_edges.shape[1], 1))
        for y in range(1, img_edges.shape[0]):
            for x in range(0, img_edges.shape[1]):
                if mask[y, x] == 0:
                    if img_edges[y, x, 1] == 1:
                        if img_edges[y, x, 0] == 1 and img_edges[y-1, x, 0] == 1:
                            start_pattern = 'H'
                        elif img_edges[y, x, 0] == 1:
                            start_pattern = 'T'
                        elif img_edges[y-1, x, 0] == 1:
                            start_pattern = 'B'
                        else:
                            start_pattern = 'L'
                        dis, end_pattern, mask = self._cal_aliasing_info_x(
                                img_edges, x+1, y, mask)
                        list_aliasings.append(
                            [y, x, dis, start_pattern+end_pattern])
        return list_aliasings


    def _find_aliasings_y(self, img_edges):
        list_aliasings = []
        mask = np.zeros((img_edges.shape[0], img_edges.shape[1], 1))
        for x in range(1, img_edges.shape[1]):
            for y in range(0, img_edges.shape[0]):
                if mask[y, x] == 0:
                    if img_edges[y, x, 0] == 1:
                        if img_edges[y, x, 1] == 1 and img_edges[y, x-1, 1] == 1:
                            start_pattern = 'H'
                        elif img_edges[y, x, 1] == 1:
                            start_pattern = 'T'
                        elif img_edges[y, x-1, 1] == 1:
                            start_pattern = 'B'
                        else:
                            start_pattern = 'L'
                        dis, end_pattern, mask = self._cal_aliasing_info_y(
                                img_edges, x, y+1, mask)
                        list_aliasings.append(
                            [y, x, dis, start_pattern+end_pattern])
        return list_aliasings


    def _analyse_pattern(self, pattern):
        if pattern[0] == 'H':
            if pattern[1] == 'H':
                start = 0
                end = 0
            elif pattern[1] == 'T':
                start = -0.5
                end = 0.5
            elif pattern[1] == 'B':
                start = 0.5
                end = -0.5
            elif pattern[1] == 'L':
                start = 0
                end = 0
        elif pattern[0] == 'T':
            if pattern[1] == 'H':
                start = 0.5
                end = -0.5
            elif pattern[1] == 'T':
                start = 0.5
                end = 0.5
            elif pattern[1] == 'B':
                start = 0.5
                end = -0.5
            elif pattern[1] == 'L':
                start = 0.5
                end = 0
        elif pattern[0] == 'B':
            if pattern[1] == 'H':
                start = -0.5
                end = 0.5
            elif pattern[1] == 'T':
                start = -0.5
                end = 0.5
            elif pattern[1] == 'B':
                start = -0.5
                end = -0.5
            elif pattern[1] == 'L':
                start = -0.5
                end = 0
        elif pattern[0] == 'L':
            if pattern[1] == 'H':
                start = 0
                end = 0
            elif pattern[1] == 'T':
                start = 0
                end = 0.5
            elif pattern[1] == 'B':
                start = 0
                end = -0.5
            elif pattern[1] == 'L':
                start = 0
                end = 0
        return start, end


    def _cal_area_list(self, dis, pattern):
        start, end = self._analyse_pattern(pattern)

        if start == 0 and end == 0:
            return None
        elif end == 0:
            h = start
            tri_len = dis
        elif start == 0:
            h = end
            tri_len = dis
        else:
            h = start
            tri_len = dis/2.0

        list_area = np.zeros((dis, 2))
        tri_area = abs(h)*tri_len/2

        if start==0:
            for i in range(0, dis):
                area = (end*2)*(tri_area*(((i+1)/tri_len)**2) -
                                tri_area*(((i)/tri_len)**2))
                if area > 0:
                    list_area[i, 0] = area
                else:
                    list_area[i, 1] = -area
        elif end==0:
            for i in range(0, dis):
                area = (start*2)*(tri_area*(((tri_len-i)/tri_len)
                                            ** 2)-tri_area*(((tri_len-i-1)/tri_len)**2))
                if area > 0:
                    list_area[i, 0] = area
                else:
                    list_area[i, 1] = -area
        elif tri_len % 2 == 0:
            for i in range(0, dis+1):
                if i == tri_len:
                    continue
                elif i < tri_len:
                    area = (start*2)*(tri_area*(((tri_len-i)/tri_len)
                                                ** 2)-tri_area*(((tri_len-i-1)/tri_len)**2))
                    if area > 0:
                        list_area[i, 0] = area
                    else:
                        list_area[i, 1] = -area
                elif i > tri_len:
                    area = (end*2)*(tri_area*(((i-tri_len)/tri_len)**2) -
                                    tri_area*(((i-tri_len-1)/tri_len)**2))
                    if area > 0:
                        list_area[i-1, 0] = area
                    else:
                        list_area[i-1, 1] = -area
        else:
            for i in range(0, dis+1):
                if abs(i-tri_len) <= 0.5:
                    if i < tri_len:
                        area = (start*2)*(tri_area*(((tri_len-i)/tri_len)**2))
                        if area > 0:
                            list_area[i, 0] += area
                        else:
                            list_area[i, 1] -= area
                    else:
                        area = (end*2)*(tri_area*(((i-tri_len)/tri_len)**2))
                        if area > 0:
                            list_area[i-1, 0] += area
                        else:
                            list_area[i-1, 1] -= area
                elif i < tri_len:
                    area = (start*2)*(tri_area*(((tri_len-i)/tri_len)
                                                ** 2)-tri_area*(((tri_len-i-1)/tri_len)**2))
                    if area > 0:
                        list_area[i, 0] = area
                    else:
                        list_area[i, 1] = -area
                elif i > tri_len:
                    area = (end*2)*(tri_area*(((i-tri_len)/tri_len)**2) -
                                    tri_area*(((i-tri_len-1)/tri_len)**2))
                    if area > 0:
                        list_area[i-1, 0] = area
                    else:
                        list_area[i-1, 1] = -area
        return list_area


    def _update_weights_x(self, weights, list_area, start_y, start_x):
        for x in range(start_x, start_x+len(list_area)):
            weights[start_y, x, 0] = list_area[x-start_x, 0]
            weights[start_y, x, 1] = list_area[x-start_x, 1]
        return weights


    def _update_weights_y(self, weights, list_area, start_y, start_x):
        for y in range(start_y, start_y+len(list_area)):
            weights[y, start_x, 2] = list_area[y-start_y, 0]
            weights[y, start_x, 3] = list_area[y-start_y, 1]
        return weights


    def _get_weights(self, img_shape, list_aliasing_x, list_aliasing_y):
        weights = np.zeros((img_shape[0], img_shape[1], 4))
        for [start_y, start_x, dis, pattern] in list_aliasing_x:
            list_area = self._cal_area_list(dis, pattern)
            if list_area is None:
                continue
            weights = self._update_weights_x(
                weights, list_area, start_y, start_x)
        for [start_y, start_x, dis, pattern] in list_aliasing_y:
            list_area = self._cal_area_list(dis, pattern)
            if list_area is None:
                continue
            weights = self._update_weights_y(
                weights, list_area, start_y, start_x)
        return weights


    def _blend_color(self,img_in, img_weight):
        img_blended= np.zeros((img_in.shape[0],img_in.shape[1]))
        for y in range(0, img_in.shape[0]):
            for x in range(0, img_in.shape[1]):
                img_blended[y, x]=(2-img_weight[y,x,0]-img_weight[y,x,2])*img_in[y,x]
                if y!=0:
                    img_blended[y, x]+=img_in[y-1,x]*img_weight[y,x,0]
                if y!=img_in.shape[0]-1:
                    img_blended[y, x]+=(img_in[y+1,x]-img_in[y,x])*img_weight[y+1,x,1]
                if x!=0:
                    img_blended[y, x]+=img_in[y,x-1]*img_weight[y,x,2]
                if x!=img_in.shape[1]-1:
                    img_blended[y, x]+=(img_in[y,x+1]-img_in[y,x])*img_weight[y,x+1,3]
                img_blended[y, x]/=2
        return img_blended


    def smooth_img(self, img, num_th = 0.1):
        img = img / 255 
        img_edge = self._find_edges(img, num_th)
        list_aliasing_x = self._find_aliasings_x(img_edge)
        list_aliasing_y = self._find_aliasings_y(img_edge)
        img_weight = self._get_weights(img.shape, list_aliasing_x, list_aliasing_y)
        img = self._blend_color(img, img_weight)
        img = img * 255
        return img
    

def main():
    img_path = '/home/lxj/Documents/rob590/test/testimg.png'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = img / 255   
    smoothimg = SmoothEngine()
    img = smoothimg.smooth_img(img, num_th = 0.1)
    print(img.shape)
    img = img * 255
    cv2.imwrite('/home/lxj/Documents/rob590/test/test2.png', img)

if __name__ == "__main__":
    main()