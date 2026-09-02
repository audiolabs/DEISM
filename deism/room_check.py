"""
Helper functions for reading convex-room meshes and updating geometry data.

DEISM expects a pre-generated Gmsh ``.msh`` file. Meshio reads the mesh,
while the convex hull provides the room vertices and volume. Named physical
surface groups provide the wall areas and centers used by DEISM. The reader
rejects meshes whose physical surfaces do not tile the convex hull exactly,
because DEISM only supports convex rooms with one material per wall.

Contributor:
Anjana
Zeyu Xu
"""

import json
import os

import meshio
import numpy as np
from scipy.spatial import ConvexHull, QhullError

__all__ = [
    "collect_room_geometry_data",
    "get_corner_points",
    "get_room_geometry",
    "is_shoebox_corners",
    "sync_room_geometry",
    "update_surface_areas",
    "update_wall_centers",
]

# Corner-node counts of the meshio cell types accepted as wall surfaces and as
# volume cells. Gmsh lists corner nodes first, so slicing to this count drops
# the mid-edge and mid-face nodes of higher-order elements.
_SURFACE_CORNERS = {
    "triangle": 3,
    "triangle6": 3,
    "quad": 4,
    "quad8": 4,
    "quad9": 4,
}
_TETRA_CORNERS = {"tetra": 4, "tetra10": 4}

# Geometric tolerance relative to the room extent (for distances to wall
# planes) and to the hull surface (for wall areas).
_GEOMETRY_TOL = 1e-6
# Maximum relative gap between the tetrahedral volume and the hull volume.
# Matches the tolerance of the previous Gmsh-based convexity check.
_VOLUME_TOL = 1e-2


# -------------------------------
# Helper functions
# -------------------------------


def _load_mesh(filename):
    """Read a pre-generated Gmsh ``.msh`` file with meshio."""
    mesh_path = os.fspath(filename)
    if os.path.splitext(mesh_path)[1].lower() != ".msh":
        raise ValueError("DEISM geometry input must be a pre-generated .msh file.")
    # Call the Gmsh reader directly: meshio.read() answers an unparsable file
    # with sys.exit(1), which would take the host process down.
    try:
        return meshio.gmsh.read(mesh_path)
    except (meshio.ReadError, ValueError) as exc:
        raise ValueError(f"meshio could not read {mesh_path}: {exc}") from exc


def _mesh_points(mesh):
    points = np.asarray(mesh.points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError("The .msh file must contain at least four 3D points.")
    return points


def _convex_hull(points):
    try:
        return ConvexHull(points)
    except QhullError as exc:
        raise ValueError(
            "The .msh points do not define a three-dimensional convex room."
        ) from exc


def _sorted_hull_corners(points, hull):
    corners = points[hull.vertices]
    order = np.lexsort((corners[:, 2], corners[:, 1], corners[:, 0]))
    return corners[order]


def get_corner_points(mesh):
    """Return the extreme vertices of a convex-room mesh, sorted by x, y, z."""
    points = _mesh_points(mesh)
    return _sorted_hull_corners(points, _convex_hull(points))


def is_shoebox_corners(corners, tol=1e-6):
    x_vals = np.unique(np.round(corners[:, 0] / tol) * tol)
    y_vals = np.unique(np.round(corners[:, 1] / tol) * tol)
    z_vals = np.unique(np.round(corners[:, 2] / tol) * tol)
    return len(x_vals) == 2 and len(y_vals) == 2 and len(z_vals) == 2


def _surface_groups(mesh):
    """Collect named physical surface groups from a meshio mesh.

    Each group holds its name, its corner triangles (quads are split), and
    the ids of its corner nodes.
    """
    physical = mesh.cell_data.get("gmsh:physical")
    if physical is None:
        raise ValueError(
            "The .msh file contains no physical groups. Tag every wall with a "
            "named Physical Surface before meshing."
        )
    names = {
        int(tag_dim[0]): name
        for name, tag_dim in mesh.field_data.items()
        if len(tag_dim) > 1 and int(tag_dim[1]) == 2
    }

    groups = {}
    unnamed_tags = set()
    for block_index, block in enumerate(mesh.cells):
        corner_count = _SURFACE_CORNERS.get(block.type)
        if corner_count is None:
            continue

        tags = np.asarray(physical[block_index], dtype=int).reshape(-1)
        cells = np.asarray(block.data, dtype=int)[:, :corner_count]
        for tag in np.unique(tags):
            tag = int(tag)
            if tag not in names:
                # Tag 0 marks elements that belong to no physical group.
                if tag != 0:
                    unnamed_tags.add(tag)
                continue

            tagged_cells = cells[tags == tag]
            group = groups.setdefault(
                tag,
                {"name": names[tag], "triangles": [], "node_ids": set()},
            )
            group["node_ids"].update(map(int, tagged_cells.reshape(-1)))
            if corner_count == 3:
                group["triangles"].extend(map(tuple, tagged_cells))
            else:
                for a, b, c, d in tagged_cells:
                    group["triangles"].append((a, b, c))
                    group["triangles"].append((a, c, d))

    if not groups:
        if unnamed_tags:
            raise ValueError(
                "The physical surfaces in the .msh file have no names (missing "
                "$PhysicalNames section). DEISM matches walls to materials by "
                "name, so name every Physical Surface."
            )
        raise ValueError("The .msh file must contain named physical surface groups.")

    for group in groups.values():
        group["triangles"] = np.asarray(group["triangles"], dtype=int)
    return groups


def _triangle_areas(triangles):
    """Areas of an (N, 3, 3) array of triangle corner coordinates."""
    if len(triangles) == 0:
        return np.zeros(0)
    edges_a = triangles[:, 1, :] - triangles[:, 0, :]
    edges_b = triangles[:, 2, :] - triangles[:, 0, :]
    return 0.5 * np.linalg.norm(np.cross(edges_a, edges_b), axis=1)


def _hull_planes(points, hull):
    """Group the triangulated hull facets into the planar walls of the room.

    Returns a dict with the plane equations (P, 4), the wall area per plane
    (P,), the hull vertices on each plane (list of sets), and the distance
    and area tolerances used for the checks below.
    """
    distance_tol = _GEOMETRY_TOL * float(np.ptp(points, axis=0).max())
    rounded = np.round(hull.equations, 5)
    _, first, inverse = np.unique(
        rounded, axis=0, return_index=True, return_inverse=True
    )
    inverse = np.asarray(inverse).reshape(-1)
    equations = hull.equations[first]
    facet_areas = np.bincount(
        inverse, weights=_triangle_areas(points[hull.simplices]), minlength=len(first)
    )
    vertex_distances = np.abs(
        points[hull.vertices] @ equations[:, :3].T + equations[:, 3]
    )
    corner_ids = [
        set(map(int, hull.vertices[vertex_distances[:, index] <= distance_tol]))
        for index in range(len(first))
    ]
    return {
        "equations": equations,
        "areas": facet_areas,
        "corner_ids": corner_ids,
        "distance_tol": distance_tol,
        "area_tol": _GEOMETRY_TOL * float(hull.area),
    }


def _assign_wall_planes(points, planes, groups):
    """Map each surface group to the hull plane its triangles lie on.

    A triangle that lies on no hull plane sits inside the hull, which means a
    re-entrant wall. A group spread over several planes is not one wall.
    """
    equations = planes["equations"]
    inside = 0
    planes_per_group = {}
    for tag, group in groups.items():
        corners = points[group["triangles"]]
        distances = np.abs(corners @ equations[:, :3].T + equations[:, 3])
        on_plane = (distances <= planes["distance_tol"]).all(axis=1)
        inside += int((~on_plane.any(axis=1)).sum())
        planes_per_group[tag] = set(map(int, np.flatnonzero(on_plane.any(axis=0))))
    if inside:
        raise ValueError(
            f"Room geometry is not convex: {inside} wall triangles lie inside "
            "the convex hull of the mesh. Non-convex rooms are not supported."
        )

    for tag, group in groups.items():
        if len(planes_per_group[tag]) != 1:
            raise ValueError(
                f"Physical surface '{group['name']}' must span exactly one "
                "whole wall of the convex room."
            )
    return {tag: next(iter(indices)) for tag, indices in planes_per_group.items()}


def _tetra_volume(mesh, points):
    """Total volume of the tetrahedral cells, or ``None`` without any."""
    total = 0.0
    found = False
    for block in mesh.cells:
        corner_count = _TETRA_CORNERS.get(block.type)
        if corner_count is None:
            continue
        found = True
        tets = np.asarray(block.data, dtype=int)[:, :corner_count]
        p0, p1, p2, p3 = (points[tets[:, i]] for i in range(4))
        signed = np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), p3 - p0)
        total += float(np.abs(signed).sum()) / 6.0
    return total if found else None


def _require_convex_volume(mesh, points, hull):
    """Reject volume meshes that do not fill their convex hull.

    This is the check the previous Gmsh-based implementation performed; it
    only applies when the mesh carries tetrahedra.
    """
    tetra_volume = _tetra_volume(mesh, points)
    if tetra_volume is None:
        return
    if (hull.volume - tetra_volume) / hull.volume > _VOLUME_TOL:
        raise ValueError(
            "Room geometry is not convex: the tetrahedral mesh fills only "
            f"{tetra_volume / hull.volume:.1%} of its convex hull. Non-convex "
            "rooms are not supported."
        )


def _format_center(center):
    return "(" + ", ".join(f"{value:.3f}" for value in center) + ")"


def _collect_surface_metrics(points, hull, planes, groups, group_planes):
    """Wall areas and centers, checking that surfaces tile the hull exactly.

    DEISM identifies a wall by the mean of its corner vertices and needs one
    material per wall, so every hull plane must carry exactly one physical
    surface that reaches all corners of the wall and covers its whole area.
    """
    hull_vertex_ids = set(map(int, hull.vertices))
    surface_on_plane = {}
    room_areas = {}
    wall_centers = {}

    for tag in sorted(groups):
        group = groups[tag]
        name = group["name"]
        plane = group_planes[tag]
        area = float(_triangle_areas(points[group["triangles"]]).sum())
        corner_ids = group["node_ids"] & hull_vertex_ids
        wall_area = planes["areas"][plane]
        if corner_ids != planes["corner_ids"][plane] or (
            abs(area - wall_area) > planes["area_tol"]
        ):
            raise ValueError(
                f"Physical surface '{name}' must span exactly one whole wall "
                f"of the convex room (it covers {area:.4f} of {wall_area:.4f} m²)."
            )
        if plane in surface_on_plane:
            raise ValueError(
                f"Physical surfaces '{surface_on_plane[plane]}' and '{name}' "
                "lie on the same wall. Tag every wall with exactly one "
                "physical surface."
            )
        surface_on_plane[plane] = name

        room_areas[name] = area
        center = np.mean(points[sorted(corner_ids)], axis=0)
        wall_centers[name] = np.round(center, 4).tolist()

    missing = [
        _format_center(np.mean(points[sorted(planes["corner_ids"][plane])], axis=0))
        for plane in range(len(planes["equations"]))
        if plane not in surface_on_plane
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} wall(s) of the convex room have no physical surface "
            f"(wall centers {', '.join(missing)}). Tag every wall with exactly "
            "one named physical surface."
        )

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


def collect_room_geometry_data(mesh_file):
    """Extract the geometry DEISM needs from a convex-room ``.msh`` file."""
    mesh_path = os.fspath(mesh_file)
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"MSH file not found: {mesh_path}")

    mesh = _load_mesh(mesh_path)
    points = _mesh_points(mesh)
    hull = _convex_hull(points)
    groups = _surface_groups(mesh)
    planes = _hull_planes(points, hull)
    group_planes = _assign_wall_planes(points, planes, groups)
    _require_convex_volume(mesh, points, hull)
    corners = _sorted_hull_corners(points, hull)
    room_areas, wall_centers = _collect_surface_metrics(
        points, hull, planes, groups, group_planes
    )

    shoebox = is_shoebox_corners(corners)
    room = "shoebox" if shoebox else "convex"

    return {
        "vertices": corners.tolist(),
        "wall_centers": wall_centers,
        "room_areas": room_areas,
        "room_volume": float(hull.volume),
        "room": room,
        "shoebox": shoebox,
    }


# -------------------------------
# Public helpers
# -------------------------------
def get_room_geometry(mesh_file):
    geometry_data = collect_room_geometry_data(mesh_file)
    return geometry_data["room_volume"], geometry_data["room"]


def sync_room_geometry(json_file_path, mesh_file_path):
    geometry_data = collect_room_geometry_data(mesh_file_path)
    _write_geometry_fields(
        json_file_path,
        vertices=geometry_data["vertices"],
        wall_centers=geometry_data["wall_centers"],
        room_areas=geometry_data["room_areas"],
        room_volume=geometry_data["room_volume"],
    )
    return geometry_data["room_volume"], geometry_data["room"]


def update_surface_areas(json_file_path, mesh_file_path):
    """Calculate wall areas from a ``.msh`` file and write them to JSON."""
    geometry_data = collect_room_geometry_data(mesh_file_path)
    areas = geometry_data["room_areas"]
    _write_geometry_fields(json_file_path, room_areas=areas)

    print("✅ Surface areas updated in JSON:")
    for name, area in areas.items():
        print(f"  {name}: {area:.2f} m²")


def update_wall_centers(json_path, mesh_path):
    geometry_data = collect_room_geometry_data(mesh_path)
    _write_geometry_fields(
        json_path,
        vertices=geometry_data["vertices"],
        wall_centers=geometry_data["wall_centers"],
    )

    print("✅ Wall centers updated successfully!")
