import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import sys
sys.path.append('./pytorch')
sys.path.append('./')
from star.star import STAR
star = STAR(gender='male')
import torch
import numpy as np 
import trimesh
from torch.autograd import Variable
import joblib
from tqdm import tqdm

smpl_para=joblib.load("20200901_zhh_y_t.pkl")
for i in tqdm(range(len(smpl_para[1]['frame_ids']))):
    pose=smpl_para[1]['pose'][i]
    pose=pose[np.newaxis, :]
    beta=smpl_para[1]['betas'][i]
    beta=beta[np.newaxis, :]

    poses = torch.cuda.FloatTensor(pose)
    poses = Variable(poses,requires_grad=True)
    betas = torch.cuda.FloatTensor(beta)
    betas = Variable(betas,requires_grad=True)
    trans = torch.cuda.FloatTensor(np.zeros((1,3)))
    trans = Variable(trans,requires_grad=True)
    d, faces = star(poses, betas, trans)

    f="%06d" % i

    vertices=d.detach().cpu().numpy().squeeze()
    # faces=d.f
    out_mesh = trimesh.Trimesh(vertices, faces)
    out_mesh.export("star_obj/"+f+".obj")

# batch_size=1
# poses = torch.cuda.FloatTensor(np.zeros((batch_size,72)))
# poses = Variable(poses,requires_grad=True)
# betas = torch.cuda.FloatTensor(np.zeros((batch_size,10)))
# betas = Variable(betas,requires_grad=True)

# trans = torch.cuda.FloatTensor(np.zeros((batch_size,3)))
# trans = Variable(trans,requires_grad=True)
# d = star(poses, betas,trans)
# vertices=d.v_posed.detach().cpu().numpy().squeeze()
# faces=d.f
# out_mesh = trimesh.Trimesh(vertices, faces)
# out_mesh.export("star_obj.obj")
# print(d)