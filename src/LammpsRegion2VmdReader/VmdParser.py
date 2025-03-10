from .Regions import Triangle, TriangleNormal, Cylinder, Ellipsoid, Sphere, Regions
import re
import numpy as np


class VmdParser:
    # from chatgpt
    colors = {
        "blue": np.array((0, 0, 255)) / 255,
        "red": np.array((255, 0, 0)) / 255,
        "gray": np.array((128, 128, 128)) / 255,
        "orange": np.array((255, 165, 0)) / 255,
        "yellow": np.array((255, 255, 0)) / 255,
        "tan": np.array((210, 180, 140)) / 255,
        "silver": np.array((192, 192, 192)) / 255,
        "green": np.array((0, 128, 0)) / 255,
        "white": np.array((255, 255, 255)) / 255,
        "pink": np.array((255, 192, 203)) / 255,
        "cyan": np.array((0, 255, 255)) / 255,
        "purple": np.array((128, 0, 128)) / 255,
        "lime": np.array((0, 255, 0)) / 255,
        "mauve": np.array((224, 176, 255)) / 255,
        "ochre": np.array((204, 119, 34)) / 255,
        "iceblue": np.array((200, 233, 233)) / 255,
        "black": np.array((0, 0, 0)) / 255,
    }

    _str_to_bool_map = {
        "y": True,
        "yes": True,
        "t": True,
        "true": True,
        "1": True,
        "n": False,
        "no": False,
        "f": False,
        "false": False,
        "0": False,
    }

    def __init__(self):
        self.vars = {}
        self.regions = []
        self.current_region = None

    def _str_to_bool(self, val: str | int):
        return self._str_to_bool_map[str(val).lower()]

    def process_file(self, fname: str):
        with open(fname) as f:
            for line in f:
                self.process_line(line)

    def process_line(self, line: str):
        # Create new region
        if line.startswith("draw"):
            self.process_draw(line[5:])
        elif match := re.match(r"set (\w+) \[mol new\]", line):
            var_name = match.group(1)
            self.regions.append(Regions())
            self.vars[var_name] = self.regions[-1]
            self.current_region = self.regions[-1]
        elif match := re.match(r"mol rename \$(\w+) \{(.+)\}", line):
            var_name = match.group(1)
            region_name = match.group(2)
            self.vars[var_name].name = region_name
        elif match := re.match(r"graphics \$(\w+) color (\w+)", line):
            var_name = match.group(1)
            rgb_tuple = self.colors[match.group(2)]
            self.vars[var_name].color = rgb_tuple
        elif match := re.match(r"graphics \$(\w+) material (\w+)", line):
            var_name = match.group(1)
            if match.group(2).lower() == "transparent":
                self.vars[var_name].transparency = 1.0 - 0.3
            else:
                self.vars[var_name].transparency = 1.0 - 1.0

    def process_draw(self, line: str):
        if line.startswith("triangle"):
            match = re.match(r"\{(.+)\} \{(.+)\} \{(.+)\}", line[9:])
            assert match
            tri = Triangle()
            for i in range(1, 4):
                match.group(i)
                tri.verts.append(np.array(tuple(map(float, match.group(i).split()))))
            self.current_region.primitives.append(tri)
        elif line.startswith("trinorm"):
            match = re.match(
                r"\{(.+)\} \{(.+)\} \{(.+)\} \{(.+)\} \{(.+)\} \{(.+)\}", line[8:]
            )
            assert match
            tri = TriangleNormal()
            for i in range(1, 4):
                tri.verts.append(np.array(tuple(map(float, match.group(i).split()))))
            for i in range(4, 7):
                tri.normals.append(np.array(tuple(map(float, match.group(i).split()))))
            self.current_region.primitives.append(tri)
        elif line.startswith("cylinder"):
            match = re.match(
                r"\{(.+)\} \{(.+)\} radius (.+) resolution (.+) filled (.+)", line[9:]
            )
            assert match
            cylinder = Cylinder()
            for i in range(1, 3):
                cylinder.verts.append(
                    np.array(tuple(map(float, match.group(i).split())))
                )
            cylinder.radius = float(match.group(3))
            cylinder.res = int(match.group(4))
            cylinder.filled = self._str_to_bool(match.group(5))
            self.current_region.primitives.append(cylinder)
        elif line.startswith("sphere"):
            match = re.match(r"\{(.+)\} radius (.+) resolution (.+)", line[7:])
            assert match
            sphere = Sphere()
            sphere.vert = np.array(tuple(map(float, match.group(1).split())))
            sphere.radius = float(match.group(2))
            sphere.res = int(match.group(3))
            self.current_region.primitives.append(sphere)
        elif line.startswith("ellipsoid"):
            match = re.match(r"(.+) \{(.+)\} \{(.+)\}", line[10:])
            assert match
            ellipsoid = Ellipsoid()
            ellipsoid.res = int(match.group(1))
            ellipsoid.vert = np.array(tuple(map(float, match.group(2).split())))
            ellipsoid.radius = np.array(tuple(map(float, match.group(3).split())))
            self.current_region.primitives.append(ellipsoid)
        else:
            raise NotImplementedError(f"Found new draw command:\n\t{line}")
