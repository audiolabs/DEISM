"""
Helper functions for room geometry checking and updating.
Contributor:
Anjana
Zeyu Xu
"""

import json
import os

import gmsh
import numpy as np
from scipy.spatial import ConvexHull

# -------------------------------
# Helper functions
# -------------------------------


def _open_gmsh_model(filename, mesh_dim=3):
    initialized_here = False
    if not gmsh.isInitialized():
        gmsh.initialize()
        initialized_here = True
    else:
        gmsh.clear()

    gmsh.open(filename)
    gmsh.model.mesh.generate(mesh_dim)
    return initialized_here


def _close_gmsh_model(initialized_here):
    gmsh.clear()
    if initialized_here:
        gmsh.finalize()


def get_points_and_tets():
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    points = np.array(coords).reshape(-1, 3)

    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=3)
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


def get_corner_points():
    entities = sorted(gmsh.model.getEntities(dim=0), key=lambda entity: entity[1])
    points = []
    for dim, tag in entities:
        coords = gmsh.model.getValue(dim, tag, [])
        points.append(coords)
    return np.array(points).reshape(-1, 3)


def is_shoebox_corners(corners, tol=1e-6):
    x_vals = np.unique(np.round(corners[:, 0] / tol) * tol)
    y_vals = np.unique(np.round(corners[:, 1] / tol) * tol)
    z_vals = np.unique(np.round(corners[:, 2] / tol) * tol)
    return len(x_vals) == 2 and len(y_vals) == 2 and len(z_vals) == 2


def _get_node_coordinates():
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = np.array(coords).reshape(-1, 3)
    return {int(tag): coords[index] for index, tag in enumerate(node_tags)}


def _get_triangles_for_entity(entity_tag, node_coordinates):
    elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(2, entity_tag)
    triangles = []

    for elem_type, flat_node_tags in zip(elem_types, elem_node_tags):
        (
            element_name,
            _,
            _,
            nodes_per_element,
            _,
            _,
        ) = gmsh.model.mesh.getElementProperties(elem_type)
        element_nodes = np.array(flat_node_tags, dtype=int).reshape(
            -1, nodes_per_element
        )
        lower_name = element_name.lower()

        if "triangle" in lower_name:
            corner_sets = element_nodes[:, :3]
            triangles.extend(
                [
                    [
                        node_coordinates[int(node_set[0])],
                        node_coordinates[int(node_set[1])],
                        node_coordinates[int(node_set[2])],
                    ]
                    for node_set in corner_sets
                ]
            )
        elif "quadrangle" in lower_name:
            corner_sets = element_nodes[:, :4]
            for node_set in corner_sets:
                triangles.append(
                    [
                        node_coordinates[int(node_set[0])],
                        node_coordinates[int(node_set[1])],
                        node_coordinates[int(node_set[2])],
                    ]
                )
                triangles.append(
                    [
                        node_coordinates[int(node_set[0])],
                        node_coordinates[int(node_set[2])],
                        node_coordinates[int(node_set[3])],
                    ]
                )
        else:
            raise ValueError(
                f"Unsupported surface element type '{element_name}' for entity {entity_tag}."
            )

    if not triangles:
        return np.empty((0, 3, 3), dtype=float)
    return np.array(triangles, dtype=float)


def _get_face_vertices_for_entity(entity_tag):
    boundary_entities = gmsh.model.getBoundary(
        [(2, entity_tag)], oriented=False, recursive=True
    )
    point_tags = sorted({tag for dim, tag in boundary_entities if dim == 0})
    face_points = [gmsh.model.getValue(0, tag, []) for tag in point_tags]
    if not face_points:
        return np.empty((0, 3), dtype=float)
    return np.unique(np.array(face_points, dtype=float), axis=0)


def _compute_surface_area(triangles):
    if len(triangles) == 0:
        return 0.0

    v0 = triangles[:, 0, :]
    v1 = triangles[:, 1, :]
    v2 = triangles[:, 2, :]
    tri_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    return float(np.sum(tri_areas))


def _collect_surface_metrics():
    node_coordinates = _get_node_coordinates()
    room_areas = {}
    wall_centers = {}

    for dim, physical_tag in sorted(
        gmsh.model.getPhysicalGroups(dim=2), key=lambda item: item[1]
    ):
        group_name = (
            gmsh.model.getPhysicalName(dim, physical_tag) or f"Surface_{physical_tag}"
        )
        entity_tags = gmsh.model.getEntitiesForPhysicalGroup(dim, physical_tag)

        triangles = []
        face_points = []
        for entity_tag in entity_tags:
            entity_triangles = _get_triangles_for_entity(entity_tag, node_coordinates)
            if len(entity_triangles) > 0:
                triangles.extend(entity_triangles.tolist())
            entity_face_points = _get_face_vertices_for_entity(entity_tag)
            if len(entity_face_points) > 0:
                face_points.extend(entity_face_points.tolist())

        area = _compute_surface_area(np.array(triangles, dtype=float))
        if face_points:
            face_center = np.mean(
                np.unique(np.array(face_points, dtype=float), axis=0), axis=0
            )
        else:
            face_center = np.zeros(3)
        room_areas[group_name] = area
        wall_centers[group_name] = np.round(face_center, 4).tolist()

    return room_areas, wall_centers


def _ensure_geometry_section(data):
    if "geometry" not in data:
        data["geometry"] = [{}]
    elif len(data["geometry"]) == 0:
        data["geometry"].append({})
    return data["geometry"][0]


def _write_geometry_fields(json_file_path, **fields):
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")

    with open(json_file_path, "r") as file_obj:
        data = json.load(file_obj)

    geometry = _ensure_geometry_section(data)
    geometry.update(fields)

    with open(json_file_path, "w") as file_obj:
        json.dump(data, file_obj, indent=4)


def collect_room_geometry_data(geo_file):
    if not os.path.exists(geo_file):
        raise FileNotFoundError(f"GEO file not found: {geo_file}")

    initialized_here = _open_gmsh_model(geo_file, mesh_dim=3)

    try:
        points, tets = get_points_and_tets()
        volume = mesh_volume(points, tets)

        convex = is_convex(points, tets)
        corners = get_corner_points()
        shoebox = is_shoebox_corners(corners)

        if convex:
            room = "convex"
        else:
            room = "shoebox"

        room_areas, wall_centers = _collect_surface_metrics()

        return {
            "vertices": corners.tolist(),
            "wall_centers": wall_centers,
            "room_areas": room_areas,
            "room_volumn": volume,
            "room": room,
            "shoebox": shoebox,
        }
    finally:
        _close_gmsh_model(initialized_here)


# -------------------------------
# Main
# -------------------------------
def get_room_geometry(geo_file):
    geometry_data = collect_room_geometry_data(geo_file)
    return geometry_data["room_volumn"], geometry_data["room"]


def sync_room_geometry(json_file_path, geo_file_path):
    geometry_data = collect_room_geometry_data(geo_file_path)
    _write_geometry_fields(
        json_file_path,
        vertices=geometry_data["vertices"],
        wall_centers=geometry_data["wall_centers"],
        room_areas=geometry_data["room_areas"],
        room_volumn=geometry_data["room_volumn"],
    )
    return geometry_data["room_volumn"], geometry_data["room"]


def update_surface_areas(json_file_path, geo_file_path):
    """
    Calculate surface areas from the Gmsh .geo file and write them
    into the JSON under "geometry[0]['room_areas']".
    """
    geometry_data = collect_room_geometry_data(geo_file_path)
    areas = geometry_data["room_areas"]
    _write_geometry_fields(json_file_path, room_areas=areas)

    print("✅ Surface areas updated in JSON:")
    for name, area in areas.items():
        print(f"  {name}: {area:.2f} m²")

    # return areas


def update_wall_centers(json_path, geo_path):
    geometry_data = collect_room_geometry_data(geo_path)
    _write_geometry_fields(
        json_path,
        vertices=geometry_data["vertices"],
        wall_centers=geometry_data["wall_centers"],
    )

    print("✅ Wall centers updated successfully!")
