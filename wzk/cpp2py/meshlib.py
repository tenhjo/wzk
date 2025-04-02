import numpy as np

# import meshlib

from meshlib import mrmeshpy as mlib
from meshlib import mrmeshnumpy as mlib_numpy

from rokin.robots import SmartHand_Thumb03

robot = SmartHand_Thumb03()

mesh_thumb = robot.meshes.files[-1]

mesh1 = mlib.loadMesh(mesh_thumb)
mesh2 = mlib.loadMesh(mesh_thumb)

mesh2.points.vec += 1.0

verts = mlib_numpy.getNumpyVerts(mesh1)
faces = mlib_numpy.getNumpyFaces(mesh1.topology)


mesh1b = mlib_numpy.meshFromFacesVerts(faces=faces, verts=verts+1)
z = mlib.findSignedDistance(mesh1b, mesh2)
print(z.signedDist)
