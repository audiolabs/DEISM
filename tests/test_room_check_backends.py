"""Tests for the meshio-only convex-room geometry reader."""

import itertools
from pathlib import Path

import meshio
import numpy as np
import pytest

import deism
from deism.room_check import (
    _load_mesh,
    collect_room_geometry_data,
    get_corner_points,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEASUREMENT_ROOM_MSH = (
    PROJECT_ROOT / "examples" / "data" / "geometry" / "MeasurementRoom.msh"
)
# The same room meshed by Gmsh 4.13 as a refined MSH 4.1 volume mesh: one
# cell block per entity, tetrahedra, interior nodes, and a dim-1 physical
# group ("default") that the reader has to ignore.
MEASUREMENT_ROOM_GMSH41_MSH = (
    Path(__file__).resolve().parent / "data" / "geometry" / "MeasurementRoom_gmsh41.msh"
)

# Geometry of examples/data/geometry/MeasurementRoom.geo, as produced by the
# previous Gmsh-based reader and stored in examples/exampleInput_Deism.json.
REFERENCE_VOLUME = 88.68915
REFERENCE_AREAS = {
    "floor": 26.8755,
    "wall1": 20.81201453487865,
    "ceiling": 26.8755,
    "wall2": 18.216,
    "wall3": 16.83,
    "wall4": 13.394951623652844,
}
REFERENCE_CENTERS = {
    "floor": [2.9325, 2.275, 0.0],
    "wall1": [3.105, 4.55, 1.65],
    "ceiling": [2.9325, 2.275, 3.3],
    "wall2": [2.76, 0.0, 1.65],
    "wall3": [0.0, 2.55, 1.65],
    "wall4": [5.865, 2.0, 1.65],
}


# -------------------------------
# Mesh builders
# -------------------------------


def _unit_box():
    """Corner points and six named triangulated faces of the unit box."""
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )
    faces = {
        "floor": [[0, 1, 2], [0, 2, 3]],
        "ceiling": [[4, 5, 6], [4, 6, 7]],
        "front": [[0, 1, 5], [0, 5, 4]],
        "right": [[1, 2, 6], [1, 6, 5]],
        "back": [[2, 3, 7], [2, 7, 6]],
        "left": [[3, 0, 4], [3, 4, 7]],
    }
    return points, faces


def _mesh_from_faces(points, faces, extra_cells=(), named=True):
    """Build a meshio mesh with one physical surface group per named face.

    ``extra_cells`` adds cell blocks that belong to no physical group, the
    way Gmsh writes volume cells that were not tagged.
    """
    cells = []
    physical = []
    field_data = {}
    for tag, (name, triangles) in enumerate(faces.items(), start=1):
        cells.append(("triangle", np.asarray(triangles, dtype=int)))
        physical.append(np.full(len(triangles), tag))
        field_data[name] = np.array([tag, 2])
    for cell_type, data in extra_cells:
        cells.append((cell_type, np.asarray(data, dtype=int)))
        physical.append(np.zeros(len(data), dtype=int))

    return meshio.Mesh(
        np.asarray(points, dtype=float),
        cells,
        cell_data={"gmsh:physical": physical},
        field_data=field_data if named else {},
    )


def _unit_box_mesh():
    """A surface-only unit box with six named physical wall groups."""
    return _mesh_from_faces(*_unit_box())


def _refined_box_mesh(n=3):
    """Unit box whose faces are n x n grids, plus one unreferenced interior node.

    Edge and face nodes are not room corners, and the interior node mimics
    what a volume mesh carries; none of them may show up as vertices.
    """
    node_ids = {}
    points = []

    def node(x, y, z):
        key = (round(x, 9), round(y, 9), round(z, 9))
        if key not in node_ids:
            node_ids[key] = len(points)
            points.append([x, y, z])
        return node_ids[key]

    specs = {
        "left": (0, 0.0),
        "right": (0, 1.0),
        "front": (1, 0.0),
        "back": (1, 1.0),
        "floor": (2, 0.0),
        "ceiling": (2, 1.0),
    }
    faces = {}
    for name, (axis, value) in specs.items():
        u_axis, v_axis = [a for a in range(3) if a != axis]
        grid = np.empty((n + 1, n + 1), dtype=int)
        for i in range(n + 1):
            for j in range(n + 1):
                coord = [0.0, 0.0, 0.0]
                coord[axis] = value
                coord[u_axis] = i / n
                coord[v_axis] = j / n
                grid[i, j] = node(*coord)
        triangles = []
        for i in range(n):
            for j in range(n):
                a, b, c, d = (
                    grid[i, j],
                    grid[i + 1, j],
                    grid[i + 1, j + 1],
                    grid[i, j + 1],
                )
                triangles += [[a, b, c], [a, c, d]]
        faces[name] = triangles
    node(0.5, 0.5, 0.5)
    return _mesh_from_faces(np.array(points), faces)


def _l_room_mesh():
    """Surface mesh of a 6 x 6 x 3 box missing one 3 x 3 corner column."""
    footprint = [(0, 0), (6, 0), (6, 3), (3, 3), (3, 6), (0, 6)]
    points = np.array(
        [(x, y, z) for z in (0.0, 3.0) for x, y in footprint], dtype=float
    )
    faces = {
        "floor": [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5]],
        "ceiling": [[6, 7, 8], [6, 8, 9], [6, 9, 10], [6, 10, 11]],
    }
    for i in range(6):
        j = (i + 1) % 6
        faces[f"wall{i + 1}"] = [[i, j, j + 6], [i, j + 6, i + 6]]
    return _mesh_from_faces(points, faces)


def _kuhn_tetrahedra(points):
    """Split the unit box into the six Kuhn tetrahedra along its diagonal."""
    index = {tuple(p): i for i, p in enumerate(np.asarray(points).tolist())}
    tetrahedra = []
    for order in itertools.permutations(range(3)):
        corner = [0.0, 0.0, 0.0]
        ids = [index[tuple(corner)]]
        for axis in order:
            corner[axis] = 1.0
            ids.append(index[tuple(corner)])
        tetrahedra.append(ids)
    return np.asarray(tetrahedra, dtype=int)


def _write_msh(tmp_path, mesh, name="room.msh"):
    path = tmp_path / name
    meshio.write(path, mesh, file_format="gmsh22", binary=False)
    return path


# -------------------------------
# Accepted meshes
# -------------------------------


def test_msh_path_unit_box(tmp_path):
    """A pathlib .msh input reproduces exact convex-hull and wall metrics."""
    msh_file = tmp_path / "unit_box.msh"
    meshio.write(msh_file, _unit_box_mesh(), file_format="gmsh22")

    data = collect_room_geometry_data(msh_file)

    assert data["room"] == "shoebox"
    assert data["shoebox"] is True
    assert data["room_volume"] == pytest.approx(1.0)
    assert set(data["room_areas"]) == {
        "floor",
        "ceiling",
        "front",
        "right",
        "back",
        "left",
    }
    assert all(area == pytest.approx(1.0) for area in data["room_areas"].values())
    assert data["wall_centers"]["floor"] == pytest.approx([0.5, 0.5, 0.0])
    assert data["wall_centers"]["ceiling"] == pytest.approx([0.5, 0.5, 1.0])
    assert data["wall_centers"]["left"] == pytest.approx([0.0, 0.5, 0.5])
    assert set(map(tuple, data["vertices"])) == set(
        map(tuple, _unit_box_mesh().points)
    )


def test_refined_mesh_reports_only_room_corners(tmp_path):
    """Edge, face, and interior mesh nodes never become room vertices."""
    mesh = _refined_box_mesh(n=3)
    assert len(mesh.points) > 8

    data = collect_room_geometry_data(_write_msh(tmp_path, mesh))

    corners, _ = _unit_box()
    assert data["room"] == "shoebox"
    assert data["room_volume"] == pytest.approx(1.0)
    assert len(data["vertices"]) == 8
    assert set(map(tuple, data["vertices"])) == set(map(tuple, corners))
    assert all(area == pytest.approx(1.0) for area in data["room_areas"].values())
    assert data["wall_centers"]["ceiling"] == pytest.approx([0.5, 0.5, 1.0])
    assert data["wall_centers"]["right"] == pytest.approx([1.0, 0.5, 0.5])


def test_full_tetrahedral_mesh_is_accepted(tmp_path):
    points, faces = _unit_box()
    mesh = _mesh_from_faces(points, faces, [("tetra", _kuhn_tetrahedra(points))])

    data = collect_room_geometry_data(_write_msh(tmp_path, mesh))

    assert data["room_volume"] == pytest.approx(1.0)
    assert len(data["vertices"]) == 8


def test_measurement_room_msh_regression():
    """The checked-in room mesh preserves the geometry used by the example."""
    data = collect_room_geometry_data(MEASUREMENT_ROOM_MSH)

    assert data["room"] == "convex"
    assert data["shoebox"] is False
    assert data["room_volume"] == pytest.approx(REFERENCE_VOLUME)
    assert data["room_areas"] == pytest.approx(REFERENCE_AREAS)
    assert data["wall_centers"] == pytest.approx(REFERENCE_CENTERS)
    assert len(data["vertices"]) == 8


def test_gmsh41_volume_mesh_matches_reference():
    """A refined MSH 4.1 volume mesh written by Gmsh gives the same geometry."""
    mesh = meshio.read(MEASUREMENT_ROOM_GMSH41_MSH)
    assert len(mesh.points) > 8
    assert any(block.type == "tetra" for block in mesh.cells)
    assert sum(block.type == "triangle" for block in mesh.cells) > 1

    data = collect_room_geometry_data(MEASUREMENT_ROOM_GMSH41_MSH)

    assert data["room"] == "convex"
    assert data["room_volume"] == pytest.approx(REFERENCE_VOLUME)
    assert data["room_areas"] == pytest.approx(REFERENCE_AREAS)
    assert data["wall_centers"] == pytest.approx(REFERENCE_CENTERS)
    assert len(data["vertices"]) == 8


def test_get_corner_points_are_sorted_hull_vertices(tmp_path):
    mesh = meshio.read(_write_msh(tmp_path, _refined_box_mesh(n=2)))

    corners = get_corner_points(mesh)

    expected = np.array(sorted(map(tuple, _unit_box()[0])))
    np.testing.assert_allclose(corners, expected)


def test_public_api_is_exported_from_deism():
    for name in (
        "collect_room_geometry_data",
        "get_corner_points",
        "get_room_geometry",
        "is_shoebox_corners",
        "sync_room_geometry",
        "update_surface_areas",
        "update_wall_centers",
    ):
        assert hasattr(deism, name), name


# -------------------------------
# Rejected meshes
# -------------------------------


def test_non_msh_geometry_is_rejected(tmp_path):
    geometry_script = tmp_path / "room.geo"
    geometry_script.write_text("// mesh this file before using DEISM\n")

    with pytest.raises(ValueError, match="pre-generated .msh"):
        _load_mesh(geometry_script)


def test_unreadable_msh_reports_the_file(tmp_path):
    broken = tmp_path / "broken.msh"
    broken.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n8\n")

    with pytest.raises(ValueError, match="could not read .*broken.msh"):
        collect_room_geometry_data(broken)


def test_l_shaped_room_is_rejected(tmp_path):
    """Wall nodes inside the convex hull mean a re-entrant wall."""
    path = _write_msh(tmp_path, _l_room_mesh())

    with pytest.raises(ValueError, match="not convex"):
        collect_room_geometry_data(path)


def test_tetrahedra_must_fill_the_convex_hull(tmp_path):
    """A volume mesh smaller than its hull is a non-convex room."""
    points, faces = _unit_box()
    hollow = _mesh_from_faces(
        points, faces, [("tetra", _kuhn_tetrahedra(points)[:-1])]
    )

    with pytest.raises(ValueError, match="not convex.*convex hull"):
        collect_room_geometry_data(_write_msh(tmp_path, hollow))


def test_surface_patch_inside_a_wall_is_rejected(tmp_path):
    """A physical surface that touches no room corner cannot be a wall."""
    points, faces = _unit_box()
    patched_points = np.vstack(
        [points, [[0.4, 0.4, 1.0], [0.6, 0.4, 1.0], [0.5, 0.6, 1.0]]]
    )
    patched_faces = dict(faces, patch=[[8, 9, 10]])

    with pytest.raises(
        ValueError, match="'patch' must span exactly one whole wall"
    ):
        collect_room_geometry_data(
            _write_msh(tmp_path, _mesh_from_faces(patched_points, patched_faces))
        )


def test_surface_covering_two_walls_is_rejected(tmp_path):
    points, faces = _unit_box()
    merged = {name: tris for name, tris in faces.items() if name != "ceiling"}
    merged["floor"] = faces["floor"] + faces["ceiling"]

    with pytest.raises(
        ValueError, match="'floor' must span exactly one whole wall"
    ):
        collect_room_geometry_data(
            _write_msh(tmp_path, _mesh_from_faces(points, merged))
        )


def test_missing_wall_surface_is_rejected(tmp_path):
    """Every wall of the hull needs a physical surface, or DEISM lacks a material."""
    points, faces = _unit_box()
    without_ceiling = {
        name: tris for name, tris in faces.items() if name != "ceiling"
    }

    with pytest.raises(
        ValueError, match=r"no physical surface.*0\.500, 0\.500, 1\.000"
    ):
        collect_room_geometry_data(
            _write_msh(tmp_path, _mesh_from_faces(points, without_ceiling))
        )


def test_two_surfaces_on_one_wall_are_rejected(tmp_path):
    points, faces = _unit_box()
    doubled = dict(faces, floor_copy=faces["floor"])

    with pytest.raises(
        ValueError, match="'floor' and 'floor_copy' lie on the same wall"
    ):
        collect_room_geometry_data(
            _write_msh(tmp_path, _mesh_from_faces(points, doubled))
        )


def test_unnamed_physical_surfaces_are_rejected(tmp_path):
    mesh = _mesh_from_faces(*_unit_box(), named=False)

    with pytest.raises(ValueError, match="PhysicalNames"):
        collect_room_geometry_data(_write_msh(tmp_path, mesh))


def test_mesh_without_physical_groups_is_rejected(tmp_path):
    points, faces = _unit_box()
    cells = [("triangle", np.asarray(sum(faces.values(), []), dtype=int))]
    path = _write_msh(tmp_path, meshio.Mesh(points, cells))

    with pytest.raises(ValueError, match="physical"):
        collect_room_geometry_data(path)
