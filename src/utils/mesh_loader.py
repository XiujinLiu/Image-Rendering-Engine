import torch



class MeshLoader():
    def __init__(self):
        self.v0 = None
        self.v1 = None
        self.v2 = None
        self.t0 = None
        self.t1 = None
        self.t2 = None


        
    def parse(self, objfile: str, DEVICE):
        vertices = []
        faces_v = []
        faces_t = []
        textures = []
        with open(objfile, 'r') as file:
            for line in file:
                segs = line.strip().split()
                if segs:
                    if segs[0] == 'v':
                        vertices.append([float(segs[1]), float(segs[2]), float(segs[3])])

                    elif segs[0] == 'f':
                        face_v = [int(segs[i].split('/')[0]) for i in range(1, len(segs))]
                        face_t = [int(segs[i].split('/')[1]) for i in range(1, len(segs))]
                        faces_v.append(face_v)
                        faces_t.append(face_t)

                    elif segs[0] == 'vt':
                        textures.append([float(segs[1]), float(segs[2])])
        
        v0_indices = torch.tensor(faces_v,device=DEVICE)[:, 0].long() - 1
        v1_indices = torch.tensor(faces_v,device=DEVICE)[:, 1].long() - 1
        v2_indices = torch.tensor(faces_v,device=DEVICE)[:, 2].long() - 1
        t0_indices = torch.tensor(faces_t,device=DEVICE)[:, 0].long() - 1
        t1_indices = torch.tensor(faces_t,device=DEVICE)[:, 1].long() - 1
        t2_indices = torch.tensor(faces_t,device=DEVICE)[:, 2].long() - 1
        # convert unit from  m to mm
        self.v0 = torch.tensor(vertices,device=DEVICE)[v0_indices]*1000
        self.v1 = torch.tensor(vertices,device=DEVICE)[v1_indices]*1000
        self.v2 = torch.tensor(vertices,device=DEVICE)[v2_indices]*1000
        self.t0 = torch.tensor(textures,device=DEVICE)[t0_indices]
        self.t1 = torch.tensor(textures,device=DEVICE)[t1_indices]
        self.t2 = torch.tensor(textures,device=DEVICE)[t2_indices]



    def check_components(self, objfile: str):
        comp = ''
        with open(objfile, 'r') as file:
            for line in file:
                segs = line.strip().split()
                if segs:
                    if segs[0] != comp:
                        comp = segs[0]
                        print(segs)



    def check_numbers(self):
        print(self.t0.shape)
        print(self.t1.shape)
        print(self.t2.shape)
        print(self.v0.shape)
        print(self.v1.shape)
        print(self.v2.shape)




def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mesh_loader = MeshLoader()
    mesh_path = "../../models/002_master_chef_can/textured_simple.obj"
    mesh_loader.check_components(mesh_path)
    mesh_loader.parse(mesh_path, DEVICE)
    mesh_loader.check_numbers()







if __name__ == "__main__":
    main()