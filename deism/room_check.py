"""
Helper functions for room geometry checking and updating.
Contributor:
Anjana
Zeyu Xu
"""

import gmsh
import numpy as np
from scipy.spatial import ConvexHull
import re
import os
import json

# -------------------------------
# Helper functions
# -------------------------------


def load_geometry(filename):
    gmsh.initialize()
    gmsh.open(filename)
    gmsh.model.mesh.generate(3)
    return gmsh


def get_points_and_tets(gmsh_obj):
    node_tags, coords, _ = gmsh_obj.model.mesh.getNodes()
    points = np.array(coords).reshape(-1, 3)

    elem_types, elem_tags, elem_node_tags = gmsh_obj.model.mesh.getElements(dim=3)
    if len(elem_node_tags) == 0:
        raise ValueError("No 3D elements (tetrahedra) found in mesh.")

    tets = np.array(elem_node_tags[0]).reshape(-1, 4) - 1  # zero-based
    return points, tets


def mesh_volume(points, tets):
    vol = 0.0
    for t in tets:
        p0, p1, p2, p3 = points[t]
        vol += abs(np.dot(np.cross(p1 - p0, p2 - p0), p3 - p0)) / 6.0
    return vol


def is_convex(points, tets, tol=1e-2):
    hull = ConvexHull(points)
    hull_vol = hull.volume
    mesh_vol = mesh_volume(points, tets)

    # If hull is much larger → non-convex (like L-shape)
    if (hull_vol - mesh_vol) / hull_vol > tol:
        return False
    return True


def get_corner_points(gmsh_obj):
    entities = gmsh_obj.model.getEntities(dim=0)
    points = []
    for dim, tag in entities:
        coords = gmsh_obj.model.getValue(dim, tag, [])
        points.append(coords)
    return np.array(points).reshape(-1, 3)


def is_shoebox_corners(corners, tol=1e-6):
    x_vals = np.unique(np.round(corners[:, 0] / tol) * tol)
    y_vals = np.unique(np.round(corners[:, 1] / tol) * tol)
    z_vals = np.unique(np.round(corners[:, 2] / tol) * tol)
    return len(x_vals) == 2 and len(y_vals) == 2 and len(z_vals) == 2


# -------------------------------
# Main
# -------------------------------
def get_room_geometry(geo_file):

    gmsh_obj = load_geometry(geo_file)
    points, tets = get_points_and_tets(gmsh_obj)
    volume = mesh_volume(points, tets)

    convex = is_convex(points, tets)
    corners = get_corner_points(gmsh_obj)
    shoebox = is_shoebox_corners(corners)

    if convex:
        room = "convex"
    else:
        room = "shoebox"
    # print("Mesh-based Volume:", volume)
    # print("Convex:", convex)
    # print("Shoebox:", shoebox)

    gmsh_obj.finalize()
    return volume, room


def update_surface_areas(json_file_path, geo_file_path):
    """
    Calculate surface areas from the Gmsh .geo file and write them
    into the JSON under "geometry[0]['room_areas']".
    """
    # Check that files exist
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")
    if not os.path.exists(geo_file_path):
        raise FileNotFoundError(f"GEO file not found: {geo_file_path}")

    # Initialize Gmsh
    gmsh.initialize()
    gmsh.open(geo_file_path)
    gmsh.model.mesh.generate(2)  # 2D surface mesh

    # Get all nodes
    node_tags_all, coords_all, _ = gmsh.model.mesh.getNodes()
    coords = coords_all.reshape((len(node_tags_all), 3))

    # Get all surface groups
    surface_group_tags = gmsh.model.getPhysicalGroups(dim=2)
    areas = {}

    for dim, tag in surface_group_tags:
        group_name = gmsh.model.getPhysicalName(dim, tag)
        if not group_name:
            group_name = f"Surface_{tag}"

        elem_types, elem_tags, node_tags = gmsh.model.mesh.getElements(dim, tag)
        if len(node_tags) == 0:
            areas[group_name] = 0.0
            continue

        faces = np.array(node_tags[0]).reshape(-1, 3) - 1  # 0-based
        v0 = coords[faces[:, 0], :]
        v1 = coords[faces[:, 1], :]
        v2 = coords[faces[:, 2], :]
        tri_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        areas[group_name] = float(np.sum(tri_areas))

    gmsh.finalize()

    # Update JSON
    with open(json_file_path, "r") as f:
        data = json.load(f)

    if "geometry" not in data:
        data["geometry"] = [{}]
    elif len(data["geometry"]) == 0:
        data["geometry"].append({})

    data["geometry"][0]["room_areas"] = areas

    with open(json_file_path, "w") as f:
        json.dump(data, f, indent=4)

    print("✅ Surface areas updated in JSON:")
    for name, area in areas.items():
        print(f"  {name}: {area:.2f} m²")

    # return areas


def update_wall_centers(json_path, geo_path):
    with open(geo_path, "r") as f:
        lines_geo = f.readlines()

    points = {}
    point_pattern = re.compile(
        r"Point\((\d+)\) = \{ ([\d\.\-e]+), ([\d\.\-e]+), ([\d\.\-e]+), [\d\.\-e]+ \};"
    )
    for line in lines_geo:
        match = point_pattern.match(line.strip())
        if match:
            pid = int(match.group(1))
            x, y, z = (
                float(match.group(2)),
                float(match.group(3)),
                float(match.group(4)),
            )
            points[pid] = [x, y, z]

    lines_dict = {}
    line_pattern = re.compile(r"Line\((\d+)\) = \{ (\d+), (\d+) \};")
    for line in lines_geo:
        match = line_pattern.match(line.strip())
        if match:
            lid = int(match.group(1))
            lines_dict[lid] = [int(match.group(2)), int(match.group(3))]

    line_loops = {}
    loop_pattern = re.compile(r"Line Loop\((\d+)\) = \{([^\}]+)\};")
    for line in lines_geo:
        match = loop_pattern.match(line.strip())
        if match:
            lid = int(match.group(1))
            line_ids = [int(x.strip()) for x in match.group(2).split(",")]
            line_loops[lid] = line_ids

    physical_surfaces = {}
    phys_pattern = re.compile(r'Physical Surface\("([^"]+)"\) = \{([^\}]+)\};')
    for line in lines_geo:
        match = phys_pattern.match(line.strip())
        if match:
            name = match.group(1)
            plane_ids = [int(x.strip()) for x in match.group(2).split(",")]
            if len(plane_ids) == 1:
                physical_surfaces[name] = plane_ids[0]

    def compute_center(line_ids):
        unique_points = set()
        for l in line_ids:
            l_id = abs(l)
            unique_points.update(lines_dict[l_id])
        x_sum = y_sum = z_sum = 0.0
        for p in unique_points:
            x, y, z = points[p]
            x_sum += x
            y_sum += y
            z_sum += z
        n = len(unique_points)
        return [x_sum / n, y_sum / n, z_sum / n]

    centers = {}
    for name, plane_id in physical_surfaces.items():
        line_loop = line_loops[plane_id]
        centers[name] = compute_center(line_loop)

    with open(json_path, "r") as f:
        data = json.load(f)

    if "geometry" not in data:
        data["geometry"] = [{}]
    # Add room vertices to the JSON
    data["geometry"][0]["vertices"] = [points[i + 1] for i in range(len(points))]
    # Add wall centers to the JSON
    data["geometry"][0]["wall_centers"] = {
        name: f"{c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}" for name, c in centers.items()
    }

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    print("✅ Wall centers updated successfully!")
