#### Python File Reader Name ####
# Description of your Python-based file reader for OVITO.

from ovito.io import FileReaderInterface
from ovito.data import DataCollection
from traits.api import Any
from .VmdParser import VmdParser
import numpy as np
from ovito.vis import TriangleMeshVis


class LammpsRegion2VmdReader(FileReaderInterface):
    @staticmethod
    def detect(filename: str):
        with open(filename, "r") as f:
            return (
                filename.endswith(".vmd")
                and f.readline().strip() == "# save old top molecule index"
                and f.readline().strip() == "set oldtop [molinfo top]"
            )

    def parse(self, data: DataCollection, filename: str, *args, **kwargs: Any):
        p = VmdParser()
        p.process_file(filename)

        for region in p.regions:
            mesh = data.triangle_meshes.create(
                identifier=region.name.replace(" ", "-"),
                vis_params={
                    "color": region.color,
                    "transparency": region.transparency,
                },
            )
            verts = []
            faces = []
            normals = []

            for prim in region.primitives:
                num_verts = sum(len(v) for v in verts)
                verts.append(prim.get_vertices())
                faces.append(prim.get_faces() + num_verts)
                normals.append(prim.get_vertex_normals())
                yield
            mesh.set_vertices(np.vstack(verts))
            mesh.set_faces(np.vstack(faces))
            mesh.set_normals(np.vstack(normals))
