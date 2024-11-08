"""
The core functions used in DEISM extended to arbitrary room geometry
"""

import time
import os
import fnmatch
from copy import deepcopy
import numpy as np
from numpy.linalg import svd
from scipy import special as scy
from scipy.spatial import ConvexHull
import scipy.spatial as spatial
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sound_field_analysis.sph import sphankel2
import ray
from deism.utilities import (
    cart2sph,
    sph2cart,
)
from deism.core_deism import (
    rotation_matrix_ZXZ,
    SHCs_from_pressure_LS,
)
from deism import libroom_deism

libroom_eps = libroom_deism.get_eps()


# ----------- Geometries --------------
def clamp(value, min, max):
    if value < min:
        return min
    if value > max:
        return max
    return value


def ccw3p(p1, p2, p3):
    """
    Computes the orientation of three 2D points.

    p1: (array size 2) coordinates of a 2D point
    p2: (array size 2) coordinates of a 2D point
    p3: (array size 2) coordinates of a 2D point

    :returns: (int) orientation of the given triangle
        1 if triangle vertices are counter-clockwise
        -1 if triangle vertices are clockwise
        0 if vertices are collinear
    """
    d = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])

    if abs(d) < libroom_eps:
        return 0
    elif d > 0:
        return 1
    else:
        return -1


def intersection_2d_segments(p1, p2, p3, p4):
    """
    Computes the intersection between two line segments in 2D.
    This function computes the intersection between a line segment (defined
    by the coordinates of two points) and a surface (defined by an array of
    coordinates of corners of the polygon and a normal vector)
    If there is no intersection, None is returned.
    If the segment belongs to the surface, None is returned.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # lines are parallel

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    if t >= 0 and t <= 1 and u >= 0 and u <= 1:
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))  # intersection point
    else:
        return None


def intersection_3d_segment_plane(a1, a2, p, normal):
    """
    Computes the intersection between a line segment and a plane in 3D.

    a1: (array size 3) coordinates of the first endpoint of the segment
    a2: (array size 3) coordinates of the second endpoint of the segment
    p: (array size 3) coordinates of a point belonging to the plane
    normal: (array size 3) normal vector of the plane

    :returns:
    -1: no intersection
     0: intersection
     1: intersection and one of the end points of the segment is in the plane
     along with the intersection point
    """

    u = a2 - a1
    denom = normal @ u

    if abs(denom) > libroom_eps:
        w = a1 - p
        num = -normal @ w
        s = num / denom

        if -libroom_eps <= s <= 1 + libroom_eps:
            # compute intersection point
            intersection = s * u + a1

            # check limit case
            if abs(s) < libroom_eps or abs(s - 1) < libroom_eps:
                return 1, intersection  # a1 or a2 belongs to plane
            else:
                return 0, intersection  # plane is between a1 and a2

    return -1, None  # no intersection


def is_inside_2d_polygon(p, corners):
    """
    Checks if a given point is inside a given polygon in 2D.

    :param p: numpy.ndarray of size (2,) - coordinates of the point
    :param corners: numpy.ndarray of size (2, N) - coordinates of the corners of the polygon

    :return:
    -1 : if the point is outside
    0 : the point is inside
    1 : the point is on the boundary
    """
    is_inside = False  # initialize point not in the polygon
    n_corners = corners.shape[1]

    # find a point outside the polygon
    i_min = np.argmin(corners[0, :])
    p_out = np.array([corners[0, i_min] - 1, p[1]])

    # Now count intersections
    for i in range(n_corners):
        j = (i - 1) % n_corners  # ensures that j wraps around to 0 when i = n_corners

        # Check first if the point is on the segment
        # We count the border as inside the polygon
        c1c2p = ccw3p(corners[:, i], corners[:, j], p)
        if c1c2p == 0:
            # Here we know that p is co-linear with the two corners
            x_down = min(corners[0, i], corners[0, j])
            x_up = max(corners[0, i], corners[0, j])
            y_down = min(corners[1, i], corners[1, j])
            y_up = max(corners[1, i], corners[1, j])
            if x_down <= p[0] <= x_up and y_down <= p[1] <= y_up:
                return 1

        # Now check intersection with standard method
        c1c2p0 = ccw3p(corners[:, i], corners[:, j], p_out)
        if c1c2p == c1c2p0:  # no intersection
            continue

        pp0c1 = ccw3p(p, p_out, corners[:, i])
        pp0c2 = ccw3p(p, p_out, corners[:, j])
        if pp0c1 == pp0c2:  # no intersection
            continue

        # at this point we are sure there is an intersection
        c_max = max(corners[1, i], corners[1, j])
        if p[1] + libroom_eps < c_max:
            is_inside = not is_inside

    # for a odd number of intersections, the point is in the polygon
    if is_inside:
        return 0  # point strictly inside
    else:
        return -1  # point is outside


def area_2d_polygon(corners):
    """
    Computes the signed area of a 2D surface represented by its corners.

    :param corners: (numpy array of shape (2, N), N>2) list of coordinates of the corners forming the surface
    :return: (float) area of the surface
        positive area means anti-clockwise ordered corners.
        negative area means clockwise ordered corners.
    """
    area = 0
    for c1 in range(corners.shape[1]):
        c2 = 0 if c1 == corners.shape[1] - 1 else c1 + 1
        base = 0.5 * (corners[1, c2] + corners[1, c1])
        height = corners[0, c2] - corners[0, c1]
        area -= height * base
    return area


def Get_SPL(p):
    p0 = 20 * 10 ** (-6)
    p_rms = np.abs(np.sqrt(0.5 * p * p.conjugate()))
    return 20 * np.log10(p_rms / p0)


# -------------------------------------
# ----------- DEISM-ARG PYTHON --------
# -------------------------------------


# -------------------------------------
# ----------- Wall Class --------------
# -------------------------------------
class Wall_deism_python:
    def __init__(self, points, centroid, Z_S):
        self.points = points
        self.normal = np.cross(
            self.points[1, :] - self.points[0, :],
            self.points[2, :] - self.points[0, :],
        )
        self.normal = self.normal / np.linalg.norm(self.normal)
        self.origin = points[0, :]
        if np.dot(self.normal, centroid - self.points[0, :]) > 0:
            self.normal = -self.normal
        self.points = self.order_points(points)
        self.impedance = Z_S
        # The basis and normal are found by SVD
        U, s, Vh = svd(self.points.T - self.origin[np.newaxis, :].T)
        # U, s, Vh = svd(self.points - self.origin, full_matrices=True)
        if s[-1] > libroom_eps:
            raise RuntimeError("The corners of the wall do not lie in a plane")
        self.basis = U[:, :2]
        # self.normal = U[:,2]
        # Project the 3D vertices onto 2D plane defined by the basis.
        self.flat_corners = self.basis.T @ (
            self.points.T - self.origin[np.newaxis, :].T
        )
        # self.flat_corners = np.dot(
        #     self.basis.T, (self.points - self.origin).T
        # )
        a = area_2d_polygon(self.flat_corners)
        if a < 0:
            # exchange the other two basis vectors
            self.basis = self.basis[:, ::-1]
            self.flat_corners = self.flat_corners[::-1, :]
        # NTPRA !!!
        self.reflection_matrix = (
            np.identity(3) - 2 * self.normal[:, None] @ self.normal[:, None].T
        )

    def order_points(self, points):
        # Compute the centroid of the wall points
        wall_center = np.mean(points, axis=0)
        # Choose a reference vector
        ref_vector = points[0] - wall_center
        # Compute the dot product of the vectors from the centroid to each point with the reference vector
        dots = np.dot(points - wall_center, ref_vector)
        # Compute the cross product of the vectors from the centroid to each point with the reference vector
        cross = np.cross(points - wall_center, ref_vector)
        # Use the dot product and cross product to compute the angles and sort the points counter-clockwise
        angles = np.arctan2(np.linalg.norm(cross, axis=1), dots)
        # since the norm of the cross is positive, based on the definition of np.arctan2(), the
        angles = np.where(np.sign(cross @ self.normal) > 0, angles, 2 * np.pi - angles)
        return points[np.argsort(angles)]

    def reflect(self, point):
        # Calculate the reflection of a point over the wall
        """
        Reflects point across the wall.

        :param point: a point in space.
        :param self.origin: the origin of the wall.
        :param self.normal: the normal vector of the wall.
        :returns: the reflected point and a flag. The flag is 1 if reflection is in the same direction as the normal,
        0 if the point is within tolerance of the wall, -1 if the reflection is in the opposite direction of the normal.
        """
        # TODO: Implement the incidence angle calculation
        # Projection onto normal axis
        distance_wall2p = np.dot(self.normal, self.origin - point)

        # Compute reflected point
        reflected_point = point + 2 * distance_wall2p * self.normal

        # Check the direction of the reflection relative to the normal
        if distance_wall2p > libroom_eps:
            return reflected_point, 1
        elif distance_wall2p < -libroom_eps:
            return reflected_point, -1
        else:
            return reflected_point, 0

    # NTPRA !!! the whole function below
    def get_attenuation(self, theta):
        # Implement the wall attenuation calculation
        return (self.impedance * np.cos(theta) - 1) / (
            self.impedance * np.cos(theta) + 1
        )

    def intersection(self, p1, p2):
        """
        Computes the intersection between a line segment and a polygon surface in 3D.
        This function computes the intersection between a line segment (defined
        by the coordinates of two points) and a surface (defined by an array of
        coordinates of corners of the polygon and a normal vector)
        If there is no intersection, None is returned.
        If the segment belongs to the surface, None is returned.
        Two booleans are also returned to indicate if the intersection
        happened at extremities of the segment or at a border of the polygon,
        which can be useful for limit cases computations.

        :param p1: numpy.ndarray of size (3,) - coordinates of the first endpoint of the segment
        :param p2: numpy.ndarray of size (3,) - coordinates of the second endpoint of the segment

        :return:
               -1 if there is no intersection
                0 if the intersection striclty between the segment endpoints and in the polygon interior
                1 if the intersection is at endpoint of segment
                2 if the intersection is at boundary of polygon
                3 if both the above are true
        """
        ret1, ret2, ret = 0, 0, 0
        ret1, intersect_point = intersection_3d_segment_plane(
            p1, p2, self.origin, self.normal
        )
        if ret1 == -1:
            return -1, intersect_point  # there is no intersection
        if ret1 == 1:  # intersection at endpoint of segment
            ret = 1
        # project intersection into plane basis
        flat_intersection = self.basis.T @ (intersect_point - self.origin)
        # check in flatland if intersection is in the polygon
        ret2 = is_inside_2d_polygon(flat_intersection, self.flat_corners)  # !!!
        if ret2 < 0:  # intersection is outside of the wall
            return -1, intersect_point
        if ret2 == 1:  # intersection is on the boundary of the wall
            ret |= 2
        return ret, intersect_point  # no intersection


# -------------------------------------
# ----------- Room Class --------------
# -------------------------------------
class Room_deism_python:
    def __init__(
        self,
        params,
        *choose_wall_centers,
    ):
        self.points = params["vertices"]
        self.centroid = np.mean(self.points, axis=0)
        self.walls = []
        # self.obstructing_walls = obstructing_walls
        self.source = params["posSource"]
        # self.src_Psh_coords = src_Psh_coords
        # self.src_Psh_dirs = src_Psh_dirs
        self.microphones = [params["posReceiver"]]
        self.c = params["soundSpeed"]
        self.ism_order = params["maxReflOrder"]
        self.visible_sources = []
        # NTPRA !!!
        self.Z_S = params["acousImpend"]
        self.generate_walls(*choose_wall_centers)
        self.image_source_model()

    def generate_walls(self, *choose_wall_centers):
        # Find the unique normals
        hull = ConvexHull(self.points)
        normals = [tuple(face) for face in np.round(hull.equations[:, :3], decimals=5)]
        unique_normals = list(set(normals))

        # For each unique normal, find the points that belong to a face with that normal
        for normal in unique_normals:
            face_points = []
            for i, equation in enumerate(hull.equations):
                if tuple(np.round(equation[:3], decimals=5)) == normal:
                    face_points.extend(hull.points[hull.simplices[i]])
            face_points = np.unique(face_points, axis=0)
            new_wall = Wall_deism_python(face_points, self.centroid, self.Z_S)
            if not choose_wall_centers:
                self.walls.append(new_wall)
                print("The walls are generated")  # !!! remember to remove
            else:
                for wall_center in choose_wall_centers:
                    if (
                        np.linalg.norm(new_wall.points.mean(axis=0) - wall_center)
                        < 0.0001
                    ):
                        self.walls.append(new_wall)
                        print("Selected wall with center {}".format(wall_center))

    def image_source_model(self):
        self.image_sources_dfs(ImageSource(self.source), self.ism_order)
        self.fill_sources()

    def fill_sources(self):
        n_sources = len(self.visible_sources)
        if n_sources > 0:
            self.sources = np.zeros((len(self.visible_sources[0].loc), n_sources))
            self.gen_walls = np.zeros(n_sources, dtype=int)
            self.orders = np.zeros(n_sources, dtype=int)
            self.attenuations = np.zeros((n_sources), dtype=float)
            self.visible_mics = np.zeros((len(self.microphones), n_sources), dtype=bool)
            # NTPRA !!!
            self.reflection_matrix = np.zeros([3, 3, n_sources], dtype=float)
            for i in range(n_sources - 1, -1, -1):
                top = self.visible_sources.pop()
                self.sources[:, i] = top.loc
                self.gen_walls[i] = top.gen_wall
                self.orders[i] = top.order
                self.attenuations[i] = top.attenuation
                self.visible_mics[:, i] = top.visible_mics
                # NTPRA !!!
                self.reflection_matrix[:, :, i] = top.reflect_matrix
        return n_sources

    def image_sources_dfs(self, old_is, max_order):
        new_is = ImageSource()
        any_visible = False
        # TO DO: if later the microphone array is supported, changes the codes
        for m, mic in enumerate(self.microphones):
            is_visible, list_intercept_p_to_is = self.is_visible_dfs(mic, old_is)
            if is_visible and not any_visible:
                any_visible = is_visible
                old_is.visible_mics = np.zeros(len(self.microphones), dtype=bool)
            if any_visible:
                old_is.visible_mics[m] = is_visible
                # NTPRA !!!
                old_is.attenuation = self.get_image_attenuation(
                    old_is, list_intercept_p_to_is
                )
        if any_visible:
            self.visible_sources.append(deepcopy(old_is))  #!!!IMPORTANT
        if max_order == 0:
            return
        for wi, wall in enumerate(self.walls):
            reflected_point, dir_flag = wall.reflect(old_is.loc)
            if (
                dir_flag <= 0
            ):  # if reflected point is in the opposite direction of the normal
                continue
            new_is.loc = reflected_point
            new_is.order = old_is.order + 1
            new_is.gen_wall = wi
            new_is.parent = old_is
            # new_is.attenuation = self.get_image_attenuation(new_is)
            # NTPRA !!!
            new_is.reflect_matrix = wall.reflection_matrix @ old_is.reflect_matrix
            self.image_sources_dfs(new_is, max_order - 1)

    # NTPRA !!! the whole function below
    def get_image_attenuation(self, old_is, list_intercept_p_to_is):  # !!! Speed up?
        wall_id = old_is.gen_wall
        if wall_id >= 0:
            wall = self.walls[wall_id]
            intercept_p_to_is = list_intercept_p_to_is.pop(0)
            inc_angle = np.arccos(
                np.dot(intercept_p_to_is, wall.normal) / np.linalg.norm(intercept_p_to_is)
            )
            attenuation = wall.get_attenuation(inc_angle)
        else:
            return old_is.attenuation
        if old_is.parent is not None:
            return attenuation * self.get_image_attenuation(
                old_is.parent, list_intercept_p_to_is
            )

    def is_visible_dfs(self, p, old_is):
        # Most time consuming function !
        # if self.is_obstructed_dfs(p, old_is):
        #     return False
        # NTPRA !!!
        list_intercept_p_to_is = []
        if old_is.parent is not None:
            wall_id = old_is.gen_wall
            # Visibility check with the wall
            ret, intersect_p = self.walls[wall_id].intersection(p, old_is.loc)
            # vector from intersection point to IS
            # NTPRA !!!
            if ret >= 0:
                # NTPRA !!!
                list_intercept_p_to_is.append(old_is.loc - intersect_p)
                # NTPRA !!!
                ret_dfs, intercept_p_to_is_new = self.is_visible_dfs(
                    intersect_p, old_is.parent
                )
                # NTPRA !!!
                list_intercept_p_to_is = list_intercept_p_to_is + intercept_p_to_is_new
                # NTPRA !!!
                return ret_dfs, list_intercept_p_to_is
            else:
                # NTPRA !!!
                return False, list_intercept_p_to_is
        return True, list_intercept_p_to_is

    def plot_room(self):
        colors = ["r", "g", "b", "y", "c", "m", "orange", "purple"]
        print(
            "Number of images is {} for max. order {}".format(
                self.sources.shape[1], self.ism_order
            )
        )
        # for i in range(self.sources.shape[1]):
        #     print('Image at {}'.format(self.sources[:,i]))
        for i, view_angle in enumerate([(90, -90)]):  # (90, -90), (0, 0), (0, -90)
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(projection="3d")
            patches = []
            # Plot points
            ax.scatter(
                self.points[:, 0],
                self.points[:, 1],
                self.points[:, 2],
                c="k",
                marker="o",
            )
            ax.scatter(
                self.sources[0, 0],
                self.sources[1, 0],
                self.sources[2, 0],
                c="k",
                marker="<",
                label="source",
            )
            patches.append(mpatches.Patch(color="k", label=f"source"))
            ax.scatter(
                self.microphones[0][0],
                self.microphones[0][1],
                self.microphones[0][2],
                c="gray",
                marker=">",
                label="receiver",
            )
            patches.append(mpatches.Patch(color="w", label=f"receiver"))
            for i in range(1, self.sources.shape[1]):
                ax.scatter(
                    self.sources[0, i],
                    self.sources[1, i],
                    self.sources[2, i],
                    c=colors[self.gen_walls[i]],
                    marker="x",
                )

            # Plot walls and normals
            for i, wall in enumerate(self.walls):
                wall_center = np.mean(wall.points, axis=0)
                # Plot the normal vector as an arrow
                ax.quiver(
                    *wall_center,
                    *wall.normal,
                    length=0.5,
                    color=colors[i % len(colors)],
                )
                # Create a patch for legend
                patches.append(
                    mpatches.Patch(color=colors[i % len(colors)], label=f"Wall {i+1}")
                )
                # Plot the wall as a 3D polygon
                ax.add_collection3d(
                    Poly3DCollection(
                        [wall.points],
                        facecolors=colors[i % len(colors)],
                        linewidths=1,
                        edgecolors="r",
                        alpha=0.5,
                    )
                )
            ax.set_box_aspect([1, 1, 1])
            # Setting the legend
            plt.legend(handles=patches)

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            ax.set_xlim([-5, 10])
            ax.set_ylim([-5, 10])
            ax.set_zlim([-5, 10])

            ax.view_init(*view_angle)
        plt.show()


# -------------------------------------
# ----------- Image Class -------------
# -------------------------------------
class ImageSource:
    def __init__(self, loc=None):
        self.loc = loc if loc is not None else np.zeros(3)
        self.attenuation = 1
        self.order = 0
        self.gen_wall = -1
        self.parent = None
        self.visible_mics = None
        self.psh_dirs = None  # not useful
        self.v_intecp_p_to_is = []  # not useful
        self.inc_angle = None  # not useful
        # NTPRA !!!
        self.reflect_matrix = np.identity(3)
        # self.intersect_p = None
        # self.source_impact_dir = None
        # self.order_xyz = None


# -------------------------------------
# ----------- DEISM-ARG c++  ----------
# -------------------------------------

eps = libroom_deism.get_eps()


# Wall_deism class defined in python is only a wrapper of the C++ class Wall_deism
class Wall_deism_cpp:
    def __init__(self, points, centroid, Z_S):
        """
        The Wall_deism class calling functions from libroom
        parameters:
            points:     Nx2 or Nx3 NDArrays
            centroid:   algebraic center of the wall
            Z_S:        acoustic impedance,type:complex number, now only a integer number
        """

        self.energy_absorp_coef = np.array([0.15])  # -->new
        self.scatter_coef = np.array([0.1])  # -->new
        self.centroid = centroid  # -->new
        # the name of the wall
        self.name = ""  # -->new

        # initialize walls from corners
        self.libroom_walls = self._init_wall(
            points.T,
            self.centroid.T,
            Z_S,
            self.energy_absorp_coef,
            self.scatter_coef,
            self.name,
        )  # -->new

        # export properties from libroom_walls object
        self.normal = self.libroom_walls.normal
        self.origin = self.libroom_walls.origin
        # self.points=self.order_points(points)
        self.points = self.libroom_walls.corners.T
        self.impedance = Z_S
        self.basis = self.libroom_walls.basis
        self.flat_corners = self.libroom_walls.flat_corners
        self.reflection_matrix = self.libroom_walls.reflection_matrix
        self.dim = self.libroom_walls.dim

    def _init_wall(
        self, points, centroid, impedance, energy_absorp_coef, scatter_coef, name
    ):
        """
        to initialize a Wall_deism object from libroom.Wall_deism/Wall2D
        -----------
        parameters:
            points:     Nx2 or Nx3 NDArrays
            centoid:    Nx1 NDArrays
            energy_absorp_coef:   numpy arrays,length>=1
            scatter_coef:   numpy arrays,length>=1
        """
        walls = None
        if points.shape[0] == 2:
            walls = libroom_deism.Wall2D_deism(
                points, centroid, impedance, energy_absorp_coef, scatter_coef, name
            )
        elif points.shape[0] == 3:
            walls = libroom_deism.Wall_deism(
                points, centroid, impedance, energy_absorp_coef, scatter_coef, name
            )
        else:
            raise TypeError("The first dimension of points should be 2 or 3!")

        return walls

    def order_points(self, points):
        # order points such that they are arranged in counter-clockwise or clockwise
        # Compute the centroid of the wall points
        wall_center = np.mean(points, axis=0)
        # Choose a reference vector
        ref_vector = points[0] - wall_center
        # Compute the dot product of the vectors from the centroid to each point with the reference vector
        dots = np.dot(points - wall_center, ref_vector)
        # Compute the cross product of the vectors from the centroid to each point with the reference vector
        cross = np.cross(points - wall_center, ref_vector)
        # Use the dot product and cross product to compute the angles and sort the points counter-clockwise
        angles = np.arctan2(np.linalg.norm(cross, axis=1), dots)
        # since the norm of the cross is positive, based on the definition of np.arctan2(), the
        angles = np.where(np.sign(cross @ self.normal) > 0, angles, 2 * np.pi - angles)
        return points[np.argsort(angles)]

    def reflect(self, point):
        """
        -------
        return
        flags can be -1,0,1
        """
        reflected_point = np.zeros(self.dim, dtype=np.float32)
        flags = self.libroom_walls.reflect(point, reflected_point)

        return reflected_point, flags

    def get_attenuation(self, theta):
        return self.libroom_walls.get_attenuation(theta)

    def intersection(self, p1, p2):
        """
        -------
        parameters:
            -p1:    coordinate with shape (N,) NDArray
            -p2:    coordinate with shape (N,) NDArray
        """
        intersect_point = np.zeros(self.dim, dtype=np.float32)
        flag_int = self.libroom_walls.intersection(p1, p2, intersect_point)

        return flag_int, intersect_point


def find_non_convex_walls(walls):
    """
    Finds the walls that are not in the convex hull

    Parameters
    ----------
    walls: list of Wall_deism objects
        The walls that compose the room

    Returns
    -------
    list of int
        The indices of the walls no in the convex hull
    """

    all_corners = []
    for wall in walls[1:]:
        # -->new
        # dimensions of wall.corners(D,dynamic),which means each column is
        # a D-dimensional coordinate of one corner
        # all_corners.append(wall.corners.T)
        all_corners.append(wall.libroom_walls.corners.T)  # -->new
    X = np.concatenate(all_corners, axis=0)
    convex_hull = spatial.ConvexHull(X, incremental=True)

    # Now we need to check which walls are on the surface
    # of the hull
    in_convex_hull = [False] * len(walls)
    for i, wall in enumerate(walls):
        # We check if the center of the wall is co-linear or co-planar
        # with a face of the convex hull
        # point = np.mean(wall.corners, axis=1)
        point = np.mean(wall.libroom_walls.corners, axis=1)  # -->new

        for simplex in convex_hull.simplices:
            # -->new
            # if it's a 2D
            if point.shape[0] == 2:
                # check if co-linear
                p0 = convex_hull.points[simplex[0]]
                p1 = convex_hull.points[simplex[1]]
                if libroom_deism.ccw3p(p0, p1, point) == 0:
                    # co-linear point add to hull
                    in_convex_hull[i] = True

            elif point.shape[0] == 3:
                # Check if co-planar
                p0 = convex_hull.points[simplex[0]]
                p1 = convex_hull.points[simplex[1]]
                p2 = convex_hull.points[simplex[2]]

                normal = np.cross(p1 - p0, p2 - p0)
                if np.abs(np.inner(normal, point - p0)) < eps:
                    # co-planar point found!
                    in_convex_hull[i] = True
    # -->new return index of walls which are not in the convex hull
    return [i for i in range(len(walls)) if not in_convex_hull[i]]


class Room_deism_cpp:
    def __init__(self, params, *choose_wall_centers):
        """
        -----------
        parameters:
            points: Nx2 or Nx3 NDArrays
            params: dict containing multiple configuration parameters
            x_s: position of sound source
            x_r: position of receiver,should be a (3,) NDArray
            *choose_wall_centers: ?
        """
        # parameters initialization
        if not params["silentMode"]:
            print("[Calculating] DEISM-ARG image generation, ", end="")
        begin = time.time()
        self.points = params["vertices"]
        self.centroid = np.mean(self.points, axis=0)
        self.walls = []
        self.source = params["posSource"]
        self.microphones = [params["posReceiver"]]
        self.c = params["soundSpeed"]
        self.ism_order = params["maxReflOrder"]
        self.visible_sources = []
        self.impedance = params["acousImpend"]

        self.generate_walls(*choose_wall_centers)

        # -->new
        # something about ray tracing,which is essential for initialization
        # of Room_deism class in libroom
        self.rt_args = {}
        self.rt_args["energy_thres"] = 1e-7
        self.rt_args["time_thres"] = 10.0
        self.rt_args["receiver_radius"] = 0.5
        self.rt_args["hist_bin_size"] = 0.004
        self.simulationRequired = False
        # the above 6 parameters are default ones in pyroomacoustics, only used in
        # the initialization of Room_deism class, have nothing to do with our python class "Room_deism"

        self.dim = self.points.shape[1]
        # parameter to save a initialized "Room_deism" object
        self.room_engine = None
        # initialize the parameter self.room_engine
        self._init_room()
        self.room_engine.add_mic(self.microphones[0].T)
        self.room_engine.n_bands = 1
        self.room_engine.image_source_model(self.source.T)
        self.sources = self.room_engine.sources
        self.gen_walls = self.room_engine.gen_walls
        elapsed_pra_deism = time.time() - begin
        minutes, seconds = divmod(elapsed_pra_deism, 60)
        minutes = int(minutes)
        if not params["silentMode"]:
            print(f"Done [{minutes} minutes, {seconds:.3f} seconds]", end="\n\n")

    def _init_room(self, *args):
        args = list(args)
        if len(args) == 0:
            obstructing_walls = find_non_convex_walls(self.walls)
            # args+=[self.walls,obstructing_walls]
            # This step is crucial because the initialized Wall_deism must actually
            # be an object of the Wall_deism class from the libroom library,
            # rather than a wrapped Wall_deism object in python.
            args += [[f.libroom_walls for f in self.walls], obstructing_walls]

        args += [
            [],
            self.c,
            self.ism_order,
            self.rt_args["energy_thres"],
            self.rt_args["time_thres"],
            self.rt_args["receiver_radius"],
            self.rt_args["hist_bin_size"],
            self.simulationRequired,
            self.impedance,
        ]

        if self.dim == 2:
            self.room_engine = libroom_deism.Room2D_deism(*args)
        elif self.dim == 3:
            self.room_engine = libroom_deism.Room_deism(*args)
        else:
            raise TypeError("The room dimension should only be 2 or 3")

    def generate_walls(self, *choose_wall_centers):
        # Find the unique normals
        hull = ConvexHull(self.points)
        # find those unique normals
        normals = [tuple(face) for face in np.round(hull.equations[:, :3], decimals=5)]
        unique_normals = list(set(normals))

        # For each unique normal, find the points that belong to a face with that normal
        for normal in unique_normals:
            face_points = []
            for i, equation in enumerate(hull.equations):
                if tuple(np.round(equation[:3], decimals=5)) == normal:
                    # simplices are the indices of the faces of the convex hull, it is a 2D array, where each row is a face
                    # hull.points are the input points array
                    # hull.points[hull.simplices[i]]--> return an array containing all the vertices of the face
                    face_points.extend(hull.points[hull.simplices[i]])
            face_points = np.unique(face_points, axis=0)
            # The purpose should be to establish a unique surface based on the points passed in, by calculating the convex hull form,
            new_wall = Wall_deism_cpp(face_points, self.centroid, self.impedance)
            if not choose_wall_centers:
                self.walls.append(new_wall)
                # print("The walls are generated")  # !!! remember to remove
            else:
                for wall_center in choose_wall_centers:
                    # If the wall center attribute is given, the wall will only be added when the midpoint of the wall is the given center
                    if (
                        np.linalg.norm(new_wall.points.mean(axis=0) - wall_center)
                        < 0.0001
                    ):
                        self.walls.append(new_wall)
                        print("Selected wall with center {}".format(wall_center))

    def image_source_model(self):
        pass

    def image_source_dfs(self, old_is, max_order):
        pass
        # self.room_engine.image_source_dfs(ImageSource(self.source),self.ism_order)
        # self.fill_sources()

    def fill_sources(self):
        pass
        # return self.room_engine.fill_sources()

    def image_sources_dfs(self, old_is, max_order):
        pass
        # self.room_engine.image_sources_dfs(old_is,max_order)

    def get_image_attenuation(self, old_is, list_intercept_p_to_is):
        pass
        # atten=self.room_engine.get_image_attenuation(old_is,list_intercept_p_to_is)

        # return atten

    def is_visible_dfs(self, p, old_is):
        pass
        # list_intercept_p_to_is=[np.zeros(self.dim,dtype=np.float32)]
        # flag=self.room_engine.is_visible_dfs(p,old_is,list_intercept_p_to_is)

        # return flag,list_intercept_p_to_is

    def plot_room(self):
        colors = ["r", "g", "b", "y", "c", "m", "orange", "purple"]
        # print(
        #     "Number of images is {} for max. order {}".format(
        #         self.sources.shape[1], self.ism_order
        #     )
        # )
        # for i in range(self.sources.shape[1]):
        #     print('Image at {}'.format(self.sources[:,i]))
        for i, view_angle in enumerate([(90, -90)]):  # (90, -90), (0, 0), (0, -90)
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(projection="3d")
            patches = []
            # Plot points
            ax.scatter(
                self.points[:, 0],
                self.points[:, 1],
                self.points[:, 2],
                c="k",
                marker="o",
            )
            ax.scatter(
                self.sources[0, 0],
                self.sources[1, 0],
                self.sources[2, 0],
                c="k",
                marker="<",
                label="source",
            )
            patches.append(mpatches.Patch(color="k", label=f"source"))
            ax.scatter(
                self.microphones[0][0],
                self.microphones[0][1],
                self.microphones[0][2],
                c="gray",
                marker=">",
                label="receiver",
            )
            patches.append(mpatches.Patch(color="k", label=f"receiver"))
            for i in range(1, self.sources.shape[1]):
                ax.scatter(
                    self.sources[0, i],
                    self.sources[1, i],
                    self.sources[2, i],
                    # c=colors[self.gen_walls[i]],
                    c=colors[self.gen_walls[i] % len(colors)],
                    marker="x",
                )

            # Plot walls and normals
            for i, wall in enumerate(self.walls):
                wall_center = np.mean(wall.points, axis=0)
                # Plot the normal vector as an arrow
                ax.quiver(
                    *wall_center,
                    *wall.normal,
                    length=0.5,
                    color=colors[i % len(colors)],
                )
                # Create a patch for legend
                patches.append(
                    mpatches.Patch(color=colors[i % len(colors)], label=f"Wall {i+1}")
                )
                # Plot the wall as a 3D polygon
                ax.add_collection3d(
                    Poly3DCollection(
                        [wall.points],
                        facecolors=colors[i % len(colors)],
                        linewidths=1,
                        edgecolors="r",
                        alpha=0.5,
                    )
                )
            ax.set_box_aspect([1, 1, 1])
            # Setting the legend
            plt.legend(handles=patches)
            # add title
            ax.set_title("Room with image sources")

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            ax.set_xlim([-5, 10])
            ax.set_ylim([-5, 10])
            ax.set_zlim([-5, 10])

            ax.view_init(*view_angle)
        plt.show()


def find_wall_centers(vertices, *choose_wall_centers):
    """
    Find the centers of the walls of the room using the vertices
    This part is also similar to the one used in generate_walls in the class Room_deism_cpp
    """
    wall_centers = []
    centroid = np.mean(vertices, axis=0)
    # Find the unique normals
    hull = ConvexHull(vertices)
    # find those unique normals
    normals = [tuple(face) for face in np.round(hull.equations[:, :3], decimals=5)]
    unique_normals = list(set(normals))

    # For each unique normal, find the points that belong to a face with that normal
    for normal in unique_normals:
        face_points = []
        for i, equation in enumerate(hull.equations):
            if tuple(np.round(equation[:3], decimals=5)) == normal:
                # simplices are the indices of the faces of the convex hull, it is a 2D array, where each row is a face
                # hull.points are the input points array
                # hull.points[hull.simplices[i]]--> return an array containing all the vertices of the face
                face_points.extend(hull.points[hull.simplices[i]])
        face_points = np.unique(face_points, axis=0)
        if not choose_wall_centers:
            wall_centers.append(np.mean(face_points, axis=0))
            # print("The walls are generated")  # !!! remember to remove
        else:
            for wall_center in choose_wall_centers:
                # If the wall center attribute is given, the wall will only be added when the midpoint of the wall is the given center
                if np.linalg.norm(face_points.mean(axis=0) - wall_center) < 0.0001:
                    wall_centers.append(np.mean(face_points, axis=0))
                    print("Selected wall with center {}".format(wall_center))
    return wall_centers


def get_R_sI_to_r_from_room(receiver, sources):
    """
    Get the vectors from source images to receiver
    input:
    1. receiver: 1D numpy array, the receiver position
    2. sources: 2D numpy array, the image sources' positions
    """
    # calculate vectors from source images to receiver
    R_sI_to_r_all = receiver[:, None] - sources
    phi_x0, theta_x0, r_x0 = cart2sph(
        R_sI_to_r_all[0, :], R_sI_to_r_all[1, :], R_sI_to_r_all[2, :]
    )
    theta_x0 = np.pi / 2 - theta_x0
    return np.asarray([phi_x0, theta_x0, r_x0])


def get_ref_paths_ARG(params, room_pra_deism):
    """
    Get the reflection paths for DEISM-ARG
    """
    # get the reflection matrices to describe the reflected source directivity coefficients
    reflection_matrix = np.array(room_pra_deism.room_engine.reflection_matrix)
    reflection_matrix = np.moveaxis(reflection_matrix, 0, 2)
    # Get vectors from source images to receiver
    R_sI_r_all = get_R_sI_to_r_from_room(
        params["posReceiver"], room_pra_deism.room_engine.sources
    )
    # get attenuation values for each image source
    atten_all = room_pra_deism.room_engine.attenuations.flatten()
    # remove the direct path if params["ifRemoveDirectPath"] = 1
    if params["ifRemoveDirectPath"]:
        R_sI_r_all = R_sI_r_all[:, 1:]
        reflection_matrix = reflection_matrix[:, :, 1:]
        atten_all = atten_all[1:]
    # If using the MIX mode, we need to separate the early reflections and late reflections
    if params["DEISM_mode"] == "MIX":
        # Find the indices of the early reflections using the 1D numpy array room_pra_deism.room_engine.orders
        # This array contains the order of each image source
        # Find this indices of the early reflections whose order is less than or equal to params["maxEarlyOrder"]
        early_indices = np.where(
            room_pra_deism.room_engine.orders <= params["mixEarlyOrder"]
        )[0]
        # # Save the early reflections
        # R_sI_r_all_early = R_sI_r_all[:, early_indices]
        # reflection_matrix_early = reflection_matrix[:, :, early_indices]
        # atten_all_early = atten_all[early_indices]
        # Find the indices of the late reflections by excluding the indices of the early reflections
        late_indices = np.setdiff1d(np.arange(R_sI_r_all.shape[1]), early_indices)
        # # Save the late reflections
        # R_sI_r_all_late = R_sI_r_all[:, late_indices]
        # reflection_matrix_late = reflection_matrix[:, :, late_indices]
        # atten_all_late = atten_all[late_indices]
        # Save the early and late reflections in the params dictionary
        images = {
            # "R_sI_r_all_early": R_sI_r_all_early,
            # "atten_all_early": atten_all_early,
            "early_indices": early_indices,
            # "R_sI_r_all_late": R_sI_r_all_late,
            # "atten_all_late": atten_all_late,
            "late_indices": late_indices,
            "R_sI_r_all": R_sI_r_all,
            "atten_all": atten_all,
        }
    else:
        images = {
            "R_sI_r_all": R_sI_r_all,
            "atten_all": atten_all,
        }
    params["images"] = images
    params["reflection_matrix"] = reflection_matrix
    return params


def rotate_room_src_rec(params):
    """
    Rotate the room vertices and source/receiver positions
    """
    # Rotate the room vertices
    # Get the rotation matrix for the room
    room_rotation = params["room_rotation"] * np.pi / 180
    room_R = rotation_matrix_ZXZ(room_rotation[0], room_rotation[1], room_rotation[2])
    # Rotate the room vertices
    params["vertices"] = (room_R @ params["vertices"].T).T
    params["posSource"] = room_R @ params["posSource"]
    params["posReceiver"] = room_R @ params["posReceiver"]
    return params


# -------------------------------------
# Some functions checking the correctness of the implementation
# -------------------------------------
def check_distinct_floats(floats, tolerance=1e-5):
    # Check if all floats within the list are distinct
    for i in range(len(floats)):
        for j in range(i + 1, len(floats)):
            if abs(floats[i] - floats[j]) <= tolerance:
                print(
                    f"Values {floats[i]} at {i} and {floats[j]} at {j} are not distinct."
                )
                return False
    return True


def compare_float_lists(list1, list2, names, tolerance=1e-5):
    # Ensure list1 is always the shorter or equal in length
    if len(list1) > len(list2):
        list1, list2 = list2, list1
        names = names[::-1]

    # Step 1: Check if all floats within each list are distinct
    print("Checking distinctness in list1...")
    distinct_list1 = check_distinct_floats(list1, tolerance)

    print("Checking distinctness in list2...")
    distinct_list2 = check_distinct_floats(list2, tolerance)

    if not (distinct_list1 and distinct_list2):
        print(
            "Not all values are distinct within the lists. Comparison may yield unexpected results."
        )

    # Step 2: Proceed with comparing floats between the two lists
    matched_pairs = []
    unmatched_in_list1 = list1.copy()
    unmatched_in_list2 = list2.copy()

    # To track indices to remove after iteration
    indices_to_remove_list1 = []
    indices_to_remove_list2 = []

    # Iterate over each float in list1 and compare to all floats in list2
    for i, val1 in enumerate(list1):
        differences = np.array(
            [abs(val1 - val2) for val2 in list2]
        )  # Calculate the absolute difference
        min_difference = np.min(differences)
        min_index = np.argmin(differences)

        if min_difference <= tolerance:
            matched_pairs.append((val1, list2[min_index]))
            indices_to_remove_list1.append(i)
            indices_to_remove_list2.append(min_index)

    # Remove matched items by index, starting from the end to avoid reindexing issues
    for index in sorted(indices_to_remove_list1, reverse=True):
        if 0 <= index < len(unmatched_in_list1):
            unmatched_in_list1.pop(index)

    for index in sorted(indices_to_remove_list2, reverse=True):
        if 0 <= index < len(unmatched_in_list2):
            unmatched_in_list2.pop(index)

    # print the unmatched arrays using names
    print(f"Unmatched in {names[0]}:", unmatched_in_list1)
    print(f"Unmatched in {names[1]}:", unmatched_in_list2)
    return {
        "matches": matched_pairs,
        "unmatched_in_list1": unmatched_in_list1,
        "unmatched_in_list2": unmatched_in_list2,
    }


def check_distinct_arrays(arrays, tolerance=1e-5):
    # Check if all arrays within the list are distinct
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            if np.linalg.norm(arrays[i] - arrays[j]) <= tolerance:
                print(
                    f"Arrays {arrays[i]} at {i} and {arrays[j]} at {j} are not distinct."
                )
                return False
    return True


def compare_array_lists_by_distance(list1, list2, names, tolerance=1e-5):
    """
    Compare two lists of 1D numpy arrays by Euclidean distance
    Inputs:
    - list1: list of 1D numpy arrays
    - list2: list of 1D numpy arrays
    - names: list of strings, names of the two lists for printing
    """
    # Ensure list1 is always the shorter or equal in length
    if len(list1) > len(list2):
        list1, list2 = list2, list1
        names = names[::-1]
    # Step 1: Check if all arrays within each list are distinct
    # print(f"Checking distinctness in {names[0]}...")
    distinct_list1 = check_distinct_arrays(list1, tolerance)

    # print(f"Checking distinctness in {names[1]}...")
    distinct_list2 = check_distinct_arrays(list2, tolerance)

    # If both lists are distinct, i.e., not repetive arrays in each list
    if distinct_list1 and distinct_list2:
        print(f"All arrays in {names[0]} and {names[1]} are unique.")
    elif not (distinct_list1 and distinct_list2):
        print(
            "Not all arrays are unique within the lists. Comparison may yield unexpected results."
        )

    # Step 2: Proceed with comparing arrays between the two lists
    matched_pairs = []
    unmatched_in_list1 = list1.copy()
    unmatched_in_list2 = list2.copy()

    # To track indices to remove after iteration
    indices_to_remove_list1 = []
    indices_to_remove_list2 = []

    # Iterate over each array in list1 and compare to all arrays in list2
    for i, arr1 in enumerate(list1):
        distances = np.array(
            [np.linalg.norm(arr1 - arr2) for arr2 in list2]
        )  # Calculate the Euclidean distance
        min_distance = np.min(distances)
        min_index = np.argmin(distances)

        if min_distance <= tolerance:
            matched_pairs.append((arr1, list2[min_index]))
            indices_to_remove_list1.append(i)
            indices_to_remove_list2.append(min_index)

    # Remove matched items by index, starting from the end to avoid reindexing issues
    for index in sorted(indices_to_remove_list1, reverse=True):
        if 0 <= index < len(unmatched_in_list1):
            unmatched_in_list1.pop(index)

    # Sort indices for unmatched_in_list2 and remove them safely
    for index in sorted(indices_to_remove_list2, reverse=True):
        if 0 <= index < len(unmatched_in_list2):
            unmatched_in_list2.pop(index)

    # Print if unmatch arrays are both empty
    if not unmatched_in_list1 and not unmatched_in_list2:
        print(f"All arrays in {names[0]} and {names[1]} are matched.")
    else:
        # print the unmatched arrays using names
        print(f"Unmatched in {names[0]}:", unmatched_in_list1)
        print(f"Unmatched in {names[1]}:", unmatched_in_list2)

    return {
        "matches": matched_pairs,
        "unmatched_in_list1": unmatched_in_list1,
        "unmatched_in_list2": unmatched_in_list2,
    }
