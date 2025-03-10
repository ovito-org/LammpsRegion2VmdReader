from dataclasses import dataclass, field
import numpy as np
from abc import abstractmethod


class Primitive:
    @abstractmethod
    def get_vertices(self): ...
    @abstractmethod
    def get_faces(self): ...
    @abstractmethod
    def get_vertex_normals(self): ...


@dataclass
class Regions:
    name: str = ""
    primitives: list[Primitive] = field(default_factory=list)
    color: tuple = ()
    transparency: float = 1.0


@dataclass
class Triangle(Primitive):
    verts: list[np.ndarray] = field(default_factory=list)

    def get_vertices(self):
        assert len(self.verts) == 3
        return self.verts

    def get_faces(self):
        return np.array((0, 1, 2))

    def get_vertex_normals(self):
        e1 = self.verts[1] - self.verts[0]
        e2 = self.verts[2] - self.verts[0]
        normal = np.cross(e1, e2)
        normal_norm = np.linalg.norm(normal)
        assert not np.isclose(normal_norm, 0)
        return np.array([normal / normal_norm] * 3)


@dataclass
class TriangleNormal(Primitive):
    verts: list = field(default_factory=list)
    normals: list = field(default_factory=list)

    def get_vertices(self):
        assert len(self.verts) == 3
        return self.verts

    def get_faces(self):
        return np.array((0, 1, 2))

    def get_vertex_normals(self):
        assert len(self.normals) == 3
        normals = []
        for f in self.get_faces():
            normals.append(self.normals[f])
        return np.vstack(normals)


@dataclass
class Cylinder(Primitive):
    verts: list = field(default_factory=list)
    radius: float = 1.0
    res: int = 6
    filled: bool = False
    _cache: dict = field(default_factory=dict, repr=False)

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        try:
            if key != "_cache":
                self._cache.clear()
        except AttributeError:
            pass

    def _generate_cylinder_mesh_with_flat_caps(self):
        """
        Generates a triangle mesh for a cylinder defined by two endpoints, a radius,
        and the number of edges used for the circular approximation. This version duplicates
        vertices so that the top and bottom caps have flat normals, separate from the sides.
        """
        assert len(self.verts) == 2
        p1 = self.verts[0]
        p2 = self.verts[1]

        # Compute cylinder axis and normalize
        axis = p2 - p1
        height = np.linalg.norm(axis)
        if height == 0:
            raise ValueError("p1 and p2 must be distinct points.")
        axis = axis / height

        # Compute an arbitrary perpendicular vector for the circle
        if np.allclose(axis, [0, 0, 1]) or np.allclose(axis, [0, 0, -1]):
            u = np.array([1, 0, 0])
        else:
            u = np.cross(axis, [0, 0, 1])
            u /= np.linalg.norm(u)
        # Second perpendicular vector
        v = np.cross(axis, u)
        v /= np.linalg.norm(v)

        # --- Create vertices for three parts ---
        # 1. Bottom cap: center + circle vertices
        bottom_center = p1
        bottom_circle = [
            p1 + self.radius * (np.cos(angle) * u + np.sin(angle) * v)
            for angle in np.linspace(0, 2 * np.pi, self.res, endpoint=False)
        ]

        # 2. Top cap: center + circle vertices
        top_center = p2
        top_circle = [
            p2 + self.radius * (np.cos(angle) * u + np.sin(angle) * v)
            for angle in np.linspace(0, 2 * np.pi, self.res, endpoint=False)
        ]

        # 3. Side: duplicate the circle vertices for the side surface
        side_bottom = bottom_circle[:]  # duplicate bottom circle vertices for the side
        side_top = top_circle[:]  # duplicate top circle vertices for the side

        # --- Combine vertices into one array, keeping track of indices ---
        vertices_bottom = []
        # Bottom cap
        idx_bottom_center = len(vertices_bottom)
        vertices_bottom.append(bottom_center)
        idx_bottom_circle = []
        for pt in bottom_circle:
            idx_bottom_circle.append(len(vertices_bottom))
            vertices_bottom.append(pt)

        # Top cap
        vertices_top = []
        idx_top_center = len(vertices_top)
        vertices_top.append(top_center)
        idx_top_circle = []
        for pt in top_circle:
            idx_top_circle.append(len(vertices_top))
            vertices_top.append(pt)

        # Side vertices (duplicate, so that they can have distinct normals)
        idx_side_bottom = []
        vertices_side = []
        for pt in side_bottom:
            idx_side_bottom.append(len(vertices_side))
            vertices_side.append(pt)

        idx_side_top = []
        for pt in side_top:
            idx_side_top.append(len(vertices_side))
            vertices_side.append(pt)

        vertices_bottom = np.array(vertices_bottom)
        vertices_top = np.array(vertices_top)
        vertices_side = np.array(vertices_side)

        # --- Create faces ---
        faces_bottom = []
        # Bottom cap (fan triangles)
        for i in range(self.res):
            next_i = (i + 1) % self.res
            faces_bottom.append(
                [idx_bottom_center, idx_bottom_circle[next_i], idx_bottom_circle[i]]
            )

        # Top cap (fan triangles, order reversed for outward normal)
        faces_top = []
        for i in range(self.res):
            next_i = (i + 1) % self.res
            faces_top.append(
                [idx_top_center, idx_top_circle[i], idx_top_circle[next_i]]
            )

        # Side faces: each side quadrilateral is split into two triangles
        faces_side = []
        for i in range(self.res):
            next_i = (i + 1) % self.res
            b0 = idx_side_bottom[i]
            b1 = idx_side_bottom[next_i]
            t0 = idx_side_top[i]
            t1 = idx_side_top[next_i]
            faces_side.append([b0, b1, t0])
            faces_side.append([t0, b1, t1])

        faces_bottom = np.array(faces_bottom, dtype=int)
        faces_top = np.array(faces_top, dtype=int)
        faces_side = np.array(faces_side, dtype=int)

        # --- Compute vertex normals ---
        vertex_normals_top = np.zeros_like(vertices_top)

        # For the bottom cap: assign flat normal (-axis)
        vertex_normals_top[idx_bottom_center] = -axis
        for idx in idx_bottom_circle:
            vertex_normals_top[idx] = -axis

        vertex_normals_bottom = np.zeros_like(vertices_bottom)
        # For the top cap: assign flat normal (axis)
        vertex_normals_bottom[idx_top_center] = axis
        for idx in idx_top_circle:
            vertex_normals_bottom[idx] = axis

        # For the side vertices: compute the radial normal.
        # The normal is the projection of (vertex - center_line) onto the plane perpendicular to the axis.
        vertex_normals_side = np.zeros_like(vertices_side)
        for idx in idx_side_bottom:
            radial = vertices_side[idx] - p1
            # Remove the component along the axis
            radial_component = np.dot(radial, axis) * axis
            normal_side = radial - radial_component
            norm_val = np.linalg.norm(normal_side)
            if norm_val != 0:
                normal_side /= norm_val
            vertex_normals_side[idx] = normal_side

        for idx in idx_side_top:
            radial = vertices_side[idx] - p2
            radial_component = np.dot(radial, axis) * axis
            normal_side = radial - radial_component
            norm_val = np.linalg.norm(normal_side)
            if norm_val != 0:
                normal_side /= norm_val
            vertex_normals_side[idx] = normal_side

        self._cache["vertices_top"] = vertices_top
        self._cache["vertices_bottom"] = vertices_bottom
        self._cache["vertices_side"] = vertices_side

        self._cache["faces_top"] = faces_top
        self._cache["faces_bottom"] = faces_bottom
        self._cache["faces_side"] = faces_side

        self._cache["vertex_normals_top"] = vertex_normals_top
        self._cache["vertex_normals_bottom"] = vertex_normals_bottom
        self._cache["vertex_normals_side"] = vertex_normals_side

    def get_vertices(self):
        if not self._cache:
            self._generate_cylinder_mesh_with_flat_caps()
        return np.vstack(
            (
                self._cache["vertices_top"],
                self._cache["vertices_bottom"],
                self._cache["vertices_side"],
            )
        )

    def get_faces(self):
        if not self._cache:
            self._generate_cylinder_mesh_with_flat_caps()
        return np.vstack(
            (
                self._cache["faces_top"],
                self._cache["faces_bottom"] + len(self._cache["vertices_top"]),
                self._cache["faces_side"]
                + len(self._cache["vertices_top"])
                + len(self._cache["vertices_bottom"]),
            )
        )

    def get_vertex_normals(self):
        if not self._cache:
            self._generate_cylinder_mesh_with_flat_caps()
        normals = []
        for f in self._cache["faces_top"]:
            normals.append(self._cache["vertex_normals_top"][f])
        for f in self._cache["faces_bottom"]:
            normals.append(self._cache["vertex_normals_bottom"][f])
        for f in self._cache["faces_side"]:
            normals.append(self._cache["vertex_normals_side"][f])
        print(np.vstack(normals).shape)
        return np.vstack(normals)


@dataclass
class Sphere(Primitive):
    vert: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radius: float = 1.0
    res: int = 6
    _cache: dict = field(default_factory=dict, repr=False)

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        try:
            if key != "_cache":
                self._cache.clear()
        except AttributeError:
            pass

    def _generate_sphere_mesh(self):
        """
        Generates a triangle mesh for a sphere.

        Parameters:
            center (array-like): The center of the sphere (x, y, z).
            radius (float): The radius of the sphere.
            stacks (int): Number of divisions along the polar (phi) direction.
                        (There will be one pole vertex at the top and bottom.)
            slices (int): Number of divisions along the azimuth (theta) direction.

        Returns:
            vertices (np.ndarray): Array of vertices (n, 3).
            faces (np.ndarray): Array of triangle faces (m, 3), where each face contains indices into vertices.
            vertex_normals (np.ndarray): Array of per-vertex normals (n, 3).
        """
        center = self.vert
        stacks = slices = self.res
        vertices = []

        # --- Create vertices ---
        # North pole.
        vertices.append(center + np.array([0, 0, self.radius]))

        # Rings of vertices between the poles.
        # i goes from 1 to stacks-1. (i=0 is north pole, i=stacks is south pole.)
        for i in range(1, stacks):
            phi = np.pi * i / stacks  # phi in (0, pi)
            for j in range(slices):
                theta = 2 * np.pi * j / slices
                x = self.radius * np.sin(phi) * np.cos(theta)
                y = self.radius * np.sin(phi) * np.sin(theta)
                z = self.radius * np.cos(phi)
                vertices.append(center + np.array([x, y, z]))

        # South pole.
        vertices.append(center + np.array([0, 0, -self.radius]))

        vertices = np.array(vertices)

        # --- Create faces ---
        faces = []
        # Top cap: north pole is vertex 0.
        # The first ring starts at index 1 and has 'slices' vertices.
        for j in range(slices):
            next_j = (j + 1) % slices
            faces.append([0, 1 + j, 1 + next_j])

        # Middle faces: connect adjacent rings.
        # There are (stacks - 2) rings between the poles.
        # For ring i (starting at 1) and ring i+1, vertices for ring i start at: 1 + (i-1)*slices.
        for i in range(1, stacks - 1):
            start_current = 1 + (i - 1) * slices
            start_next = 1 + i * slices
            for j in range(slices):
                next_j = (j + 1) % slices
                # Each quad on the sphere is split into two triangles.
                faces.append(
                    [start_current + j, start_next + j, start_current + next_j]
                )
                faces.append(
                    [start_current + next_j, start_next + j, start_next + next_j]
                )

        # Bottom cap: south pole is the last vertex.
        south_index = len(vertices) - 1
        # The last ring starts at index: 1 + (stacks - 2)*slices.
        start_last_ring = 1 + (stacks - 2) * slices
        for j in range(slices):
            next_j = (j + 1) % slices
            faces.append([south_index, start_last_ring + next_j, start_last_ring + j])

        faces = np.array(faces, dtype=int)

        # --- Compute vertex normals ---
        # For a sphere, the normal at a vertex is simply the (vertex - center) normalized.
        vertex_normals = np.zeros_like(vertices)
        for i, v in enumerate(vertices):
            n = v - center
            n_norm = np.linalg.norm(n)
            if n_norm != 0:
                vertex_normals[i] = n / n_norm
            else:
                vertex_normals[i] = n  # Fallback (should not happen for a sphere)

        self._cache["vertices"] = vertices
        self._cache["faces"] = faces
        self._cache["vertex_normals"] = vertex_normals

    def get_vertices(self):
        if not self._cache:
            self._generate_sphere_mesh()
        return self._cache["vertices"]

    def get_faces(self):
        if not self._cache:
            self._generate_sphere_mesh()
        return self._cache["faces"]

    def get_vertex_normals(self):
        if not self._cache:
            self._generate_sphere_mesh()
        normals = []
        for f in self.get_faces():
            normals.append(self._cache["vertex_normals"][f])
        return np.vstack(normals)


@dataclass
class Ellipsoid(Primitive):
    vert: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radius: np.ndarray = field(default_factory=lambda: np.zeros(3))
    res: int = 6
    _cache: dict = field(default_factory=dict, repr=False)

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        try:
            if key != "_cache":
                self._cache.clear()
        except AttributeError:
            pass

    def _generate_ellipsoid_mesh(self):
        """
        Generates a triangle mesh for an ellipsoid.

        Parameters:
            center (array-like): Center of the ellipsoid (x, y, z).
            radii (array-like): 3-component radii vector [r_x, r_y, r_z].
            stacks (int): Number of divisions along the polar direction (from north to south pole).
                        (A unique vertex is placed at the north and south poles.)
            slices (int): Number of divisions along the azimuth direction.

        Returns:
            vertices (np.ndarray): Array of vertices (n, 3).
            faces (np.ndarray): Array of triangle faces (m, 3) where each face contains indices into vertices.
            vertex_normals (np.ndarray): Array of per-vertex normals (n, 3).
        """
        center = self.vert
        rx, ry, rz = self.radius
        stacks = slices = 4 * self.res
        vertices = []

        # --- Vertex Generation ---
        # North pole (top). For phi=0, sin(0)=0 and cos(0)=1.
        vertices.append(center + np.array([0, 0, rz]))

        # Rings of vertices (phi from 0 to pi, excluding the poles)
        for i in range(1, stacks):
            phi = np.pi * i / stacks  # phi in (0, pi)
            for j in range(slices):
                theta = 2 * np.pi * j / slices
                # Use spherical parametrization and scale by radii
                x = rx * np.sin(phi) * np.cos(theta)
                y = ry * np.sin(phi) * np.sin(theta)
                z = rz * np.cos(phi)
                vertices.append(center + np.array([x, y, z]))

        # South pole (bottom). For phi=pi, sin(pi)=0 and cos(pi)=-1.
        vertices.append(center + np.array([0, 0, -rz]))

        vertices = np.array(vertices)

        # --- Face Construction ---
        faces = []
        # North cap: north pole is vertex 0.
        for j in range(slices):
            next_j = (j + 1) % slices
            faces.append([0, 1 + j, 1 + next_j])

        # Middle sections: connect vertices between rings.
        for i in range(1, stacks - 1):
            start_current = 1 + (i - 1) * slices
            start_next = 1 + i * slices
            for j in range(slices):
                next_j = (j + 1) % slices
                faces.append(
                    [start_current + j, start_next + j, start_current + next_j]
                )
                faces.append(
                    [start_current + next_j, start_next + j, start_next + next_j]
                )

        # South cap: south pole is the last vertex.
        south_index = len(vertices) - 1
        start_last_ring = 1 + (stacks - 2) * slices
        for j in range(slices):
            next_j = (j + 1) % slices
            faces.append([south_index, start_last_ring + next_j, start_last_ring + j])

        faces = np.array(faces, dtype=int)

        # --- Compute Vertex Normals ---
        # For each vertex, the unnormalized normal is given by:
        # n = [ (x-cx)/(rx^2), (y-cy)/(ry^2), (z-cz)/(rz^2) ]
        vertex_normals = np.zeros_like(vertices)
        for i, v in enumerate(vertices):
            rel = v - center  # relative coordinate
            n = np.array([rel[0] / (rx * rx), rel[1] / (ry * ry), rel[2] / (rz * rz)])
            norm_val = np.linalg.norm(n)
            if norm_val != 0:
                vertex_normals[i] = n / norm_val
            else:
                vertex_normals[i] = n  # This should not occur for a proper ellipsoid.

        self._cache["vertices"] = vertices
        self._cache["faces"] = faces
        self._cache["vertex_normals"] = vertex_normals

    def get_vertices(self):
        if not self._cache:
            self._generate_ellipsoid_mesh()
        return self._cache["vertices"]

    def get_faces(self):
        if not self._cache:
            self._generate_ellipsoid_mesh()
        return self._cache["faces"]

    def get_vertex_normals(self):
        if not self._cache:
            self._generate_ellipsoid_mesh()
        normals = []
        for f in self.get_faces():
            normals.append(self._cache["vertex_normals"][f])
        return np.vstack(normals)
