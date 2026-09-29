import numpy as np
import trimesh
from typing import Tuple


def _make_box(
    length: float, width: float, height: float,
    color: Tuple[int, int, int] = (180, 180, 180),
) -> trimesh.Trimesh:
    box = trimesh.creation.box(extents=[length, width, height])
    box.visual.vertex_colors = np.tile(
        np.array([*color, 255], dtype=np.uint8), (len(box.vertices), 1)
    )
    return box


def _colored_cylinder(radius: float, height: float, color: Tuple[int, int, int], sections: int = 16) -> trimesh.Trimesh:
    cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    cyl.visual.vertex_colors = np.tile(
        np.array([*color, 255], dtype=np.uint8), (len(cyl.vertices), 1)
    )
    return cyl


def _colored_sphere(radius: float, color: Tuple[int, int, int], subdivisions: int = 2) -> trimesh.Trimesh:
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    sphere.visual.vertex_colors = np.tile(
        np.array([*color, 255], dtype=np.uint8), (len(sphere.vertices), 1)
    )
    return sphere


def _apply_offset(mesh: trimesh.Trimesh, dx: float = 0, dy: float = 0, dz: float = 0) -> trimesh.Trimesh:
    T = np.eye(4)
    T[0, 3] = dx
    T[1, 3] = dy
    T[2, 3] = dz
    m = mesh.copy()
    m.apply_transform(T)
    return m


def _concat(parts):
    return trimesh.util.concatenate(parts)


def _ground_align(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    m = mesh.copy()
    m.apply_translation([0, 0, -m.bounds[0][2]])
    return m


def make_sedan(length=4.4, width=1.8, height=1.45):
    chassis_h = height * 0.42
    cabin_h = height * 0.58
    chassis = _make_box(length, width, chassis_h, (70, 70, 80))
    chassis = _apply_offset(chassis, dz=chassis_h / 2)

    cabin_l = length * 0.55
    cabin = _make_box(cabin_l, width * 0.92, cabin_h, (100, 100, 120))
    cabin = _apply_offset(cabin, dz=chassis_h + cabin_h / 2)

    wheel_r = chassis_h * 0.48
    wheel_w = width * 0.14
    wheel_color = (30, 30, 30)
    wheel_axle_inset = length * 0.28
    for lx in [wheel_axle_inset, -wheel_axle_inset]:
        for sy in [width / 2 + wheel_w / 2, -(width / 2 + wheel_w / 2)]:
            w = _colored_cylinder(wheel_r, wheel_w, wheel_color)
            w.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
            w = _apply_offset(w, dx=lx, dy=sy, dz=wheel_r)
            chassis = _concat([chassis, w])

    glass = _make_box(cabin_l * 0.7, width * 0.88, cabin_h * 0.5, (150, 200, 230))
    glass = _apply_offset(glass, dz=chassis_h + cabin_h * 0.6)

    mesh = _concat([chassis, cabin, glass])
    return _ground_align(mesh)


def make_suv(length=4.6, width=1.9, height=1.7):
    chassis_h = height * 0.40
    cabin_h = height * 0.60
    chassis = _make_box(length, width, chassis_h, (80, 60, 60))
    chassis = _apply_offset(chassis, dz=chassis_h / 2)

    cabin = _make_box(length * 0.72, width * 0.94, cabin_h, (110, 90, 90))
    cabin = _apply_offset(cabin, dz=chassis_h + cabin_h / 2)

    wheel_r = chassis_h * 0.52
    wheel_w = width * 0.15
    for lx in [length * 0.3, -length * 0.3]:
        for sy in [width / 2 + wheel_w / 2, -(width / 2 + wheel_w / 2)]:
            w = _colored_cylinder(wheel_r, wheel_w, (25, 25, 25))
            w.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
            w = _apply_offset(w, dx=lx, dy=sy, dz=wheel_r)
            chassis = _concat([chassis, w])

    mesh = _concat([chassis, cabin])
    return _ground_align(mesh)


def make_hatchback(length=4.0, width=1.75, height=1.5):
    chassis_h = height * 0.40
    cabin_h = height * 0.60
    chassis = _make_box(length, width, chassis_h, (60, 80, 60))
    chassis = _apply_offset(chassis, dz=chassis_h / 2)

    cabin = _make_box(length * 0.65, width * 0.93, cabin_h, (90, 110, 90))
    cabin = _apply_offset(cabin, dz=chassis_h + cabin_h / 2)

    wheel_r = chassis_h * 0.46
    wheel_w = width * 0.12
    for lx in [length * 0.28, -length * 0.28]:
        for sy in [width / 2 + wheel_w / 2, -(width / 2 + wheel_w / 2)]:
            w = _colored_cylinder(wheel_r, wheel_w, (30, 30, 30))
            w.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
            w = _apply_offset(w, dx=lx, dy=sy, dz=wheel_r)
            chassis = _concat([chassis, w])

    mesh = _concat([chassis, cabin])
    return _ground_align(mesh)


def make_coupe(length=4.5, width=1.85, height=1.3):
    chassis_h = height * 0.38
    cabin_h = height * 0.62
    chassis = _make_box(length, width, chassis_h, (100, 40, 40))
    chassis = _apply_offset(chassis, dz=chassis_h / 2)

    cabin = _make_box(length * 0.48, width * 0.90, cabin_h, (130, 60, 60))
    cabin = _apply_offset(cabin, dz=chassis_h + cabin_h / 2)

    wheel_r = chassis_h * 0.50
    wheel_w = width * 0.14
    for lx in [length * 0.3, -length * 0.3]:
        for sy in [width / 2 + wheel_w / 2, -(width / 2 + wheel_w / 2)]:
            w = _colored_cylinder(wheel_r, wheel_w, (25, 25, 25))
            w.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
            w = _apply_offset(w, dx=lx, dy=sy, dz=wheel_r)
            chassis = _concat([chassis, w])

    mesh = _concat([chassis, cabin])
    return _ground_align(mesh)


def make_sports_car(length=4.3, width=1.88, height=1.2):
    chassis_h = height * 0.35
    cabin_h = height * 0.65
    chassis = _make_box(length, width, chassis_h, (180, 20, 20))
    chassis = _apply_offset(chassis, dz=chassis_h / 2)

    cabin = _make_box(length * 0.42, width * 0.88, cabin_h, (200, 30, 30))
    cabin = _apply_offset(cabin, dz=chassis_h + cabin_h / 2)

    wheel_r = chassis_h * 0.55
    wheel_w = width * 0.16
    for lx in [length * 0.31, -length * 0.31]:
        for sy in [width / 2 + wheel_w / 2, -(width / 2 + wheel_w / 2)]:
            w = _colored_cylinder(wheel_r, wheel_w, (20, 20, 20))
            w.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
            w = _apply_offset(w, dx=lx, dy=sy, dz=wheel_r)
            chassis = _concat([chassis, w])

    mesh = _concat([chassis, cabin])
    return _ground_align(mesh)


def make_station_wagon(length=4.7, width=1.82, height=1.55):
    chassis_h = height * 0.40
    cabin_h = height * 0.60
    chassis = _make_box(length, width, chassis_h, (70, 70, 100))
    chassis = _apply_offset(chassis, dz=chassis_h / 2)

    cabin = _make_box(length * 0.78, width * 0.93, cabin_h, (100, 100, 130))
    cabin = _apply_offset(cabin, dz=chassis_h + cabin_h / 2)

    wheel_r = chassis_h * 0.47
    wheel_w = width * 0.13
    for lx in [length * 0.29, -length * 0.29]:
        for sy in [width / 2 + wheel_w / 2, -(width / 2 + wheel_w / 2)]:
            w = _colored_cylinder(wheel_r, wheel_w, (30, 30, 30))
            w.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
            w = _apply_offset(w, dx=lx, dy=sy, dz=wheel_r)
            chassis = _concat([chassis, w])

    mesh = _concat([chassis, cabin])
    return _ground_align(mesh)


def make_compact(length=3.8, width=1.68, height=1.48):
    chassis_h = height * 0.42
    cabin_h = height * 0.58
    chassis = _make_box(length, width, chassis_h, (80, 90, 70))
    chassis = _apply_offset(chassis, dz=chassis_h / 2)

    cabin = _make_box(length * 0.62, width * 0.91, cabin_h, (100, 115, 90))
    cabin = _apply_offset(cabin, dz=chassis_h + cabin_h / 2)

    wheel_r = chassis_h * 0.45
    wheel_w = width * 0.12
    for lx in [length * 0.27, -length * 0.27]:
        for sy in [width / 2 + wheel_w / 2, -(width / 2 + wheel_w / 2)]:
            w = _colored_cylinder(wheel_r, wheel_w, (30, 30, 30))
            w.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
            w = _apply_offset(w, dx=lx, dy=sy, dz=wheel_r)
            chassis = _concat([chassis, w])

    mesh = _concat([chassis, cabin])
    return _ground_align(mesh)


def make_box_truck(length=8.0, width=2.5, height=3.2):
    cab_l = length * 0.28
    cargo_l = length * 0.72
    chassis_h = height * 0.22
    cab_h = height * 0.55
    cargo_h = height * 0.78

    chassis = _make_box(length, width, chassis_h, (90, 90, 90))
    chassis = _apply_offset(chassis, dz=chassis_h / 2)

    cab = _make_box(cab_l, width, cab_h, (120, 100, 60))
    cab = _apply_offset(cab, dx=(length / 2 - cab_l / 2), dz=chassis_h + cab_h / 2)

    cargo = _make_box(cargo_l, width * 0.96, cargo_h, (200, 200, 200))
    cargo = _apply_offset(cargo, dx=-(length / 2 - cargo_l / 2), dz=chassis_h + cargo_h / 2)

    wheel_r = chassis_h * 0.85
    wheel_w = width * 0.12
    for lx in [length * 0.35, length * 0.2, -length * 0.35]:
        for sy in [width / 2 + wheel_w / 2, -(width / 2 + wheel_w / 2)]:
            w = _colored_cylinder(wheel_r, wheel_w, (25, 25, 25))
            w.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
            w = _apply_offset(w, dx=lx, dy=sy, dz=wheel_r)
            chassis = _concat([chassis, w])

    mesh = _concat([chassis, cab, cargo])
    return _ground_align(mesh)


def make_flatbed_truck(length=8.5, width=2.5, height=1.8):
    cab_l = length * 0.28
    bed_l = length * 0.72
    chassis_h = height * 0.38
    cab_h = height * 0.62
    bed_h = height * 0.08

    chassis = _make_box(length, width, chassis_h, (80, 80, 80))
    chassis = _apply_offset(chassis, dz=chassis_h / 2)

    cab = _make_box(cab_l, width, cab_h, (100, 80, 50))
    cab = _apply_offset(cab, dx=(length / 2 - cab_l / 2), dz=chassis_h + cab_h / 2)

    bed = _make_box(bed_l, width, bed_h, (160, 140, 100))
    bed = _apply_offset(bed, dx=-(length / 2 - bed_l / 2), dz=chassis_h + bed_h / 2)

    wheel_r = chassis_h * 0.7
    wheel_w = width * 0.12
    for lx in [length * 0.33, length * 0.18, -length * 0.33]:
        for sy in [width / 2 + wheel_w / 2, -(width / 2 + wheel_w / 2)]:
            w = _colored_cylinder(wheel_r, wheel_w, (25, 25, 25))
            w.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
            w = _apply_offset(w, dx=lx, dy=sy, dz=wheel_r)
            chassis = _concat([chassis, w])

    mesh = _concat([chassis, cab, bed])
    return _ground_align(mesh)


def make_pickup_truck(length=5.5, width=2.0, height=1.9):
    cab_l = length * 0.45
    bed_l = length * 0.50
    chassis_h = height * 0.40
    cab_h = height * 0.60
    bed_wall_h = height * 0.22

    chassis = _make_box(length, width, chassis_h, (85, 70, 55))
    chassis = _apply_offset(chassis, dz=chassis_h / 2)

    cab = _make_box(cab_l, width * 0.97, cab_h, (110, 90, 70))
    cab = _apply_offset(cab, dx=(length / 2 - cab_l / 2), dz=chassis_h + cab_h / 2)

    bed_floor = _make_box(bed_l, width * 0.95, chassis_h * 0.15, (140, 120, 100))
    bed_floor = _apply_offset(bed_floor, dx=-(length / 2 - bed_l / 2), dz=chassis_h + chassis_h * 0.075)

    for side_y in [width * 0.48, -width * 0.48]:
        wall = _make_box(bed_l, width * 0.05, bed_wall_h, (130, 110, 90))
        wall = _apply_offset(wall, dx=-(length / 2 - bed_l / 2), dy=side_y, dz=chassis_h + bed_wall_h / 2)
        chassis = _concat([chassis, wall])

    wheel_r = chassis_h * 0.58
    wheel_w = width * 0.13
    for lx in [length * 0.29, -length * 0.22]:
        for sy in [width / 2 + wheel_w / 2, -(width / 2 + wheel_w / 2)]:
            w = _colored_cylinder(wheel_r, wheel_w, (25, 25, 25))
            w.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
            w = _apply_offset(w, dx=lx, dy=sy, dz=wheel_r)
            chassis = _concat([chassis, w])

    mesh = _concat([chassis, cab, bed_floor])
    return _ground_align(mesh)


def make_cargo_van(length=5.2, width=2.0, height=2.2):
    chassis_h = height * 0.30
    body_h = height * 0.70

    chassis = _make_box(length, width, chassis_h, (80, 80, 80))
    chassis = _apply_offset(chassis, dz=chassis_h / 2)

    body = _make_box(length * 0.95, width * 0.98, body_h, (230, 230, 230))
    body = _apply_offset(body, dz=chassis_h + body_h / 2)

    wheel_r = chassis_h * 0.75
    wheel_w = width * 0.12
    for lx in [length * 0.30, -length * 0.30]:
        for sy in [width / 2 + wheel_w / 2, -(width / 2 + wheel_w / 2)]:
            w = _colored_cylinder(wheel_r, wheel_w, (25, 25, 25))
            w.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
            w = _apply_offset(w, dx=lx, dy=sy, dz=wheel_r)
            chassis = _concat([chassis, w])

    mesh = _concat([chassis, body])
    return _ground_align(mesh)


def make_city_bus(length=10.5, width=2.6, height=3.2):
    chassis_h = height * 0.18
    body_h = height * 0.82

    chassis = _make_box(length, width, chassis_h, (70, 70, 70))
    chassis = _apply_offset(chassis, dz=chassis_h / 2)

    body = _make_box(length * 0.99, width * 0.97, body_h, (200, 220, 200))
    body = _apply_offset(body, dz=chassis_h + body_h / 2)

    windows_per_side = 6
    win_h = body_h * 0.32
    win_w = length * 0.09
    win_d = 0.05
    for i in range(windows_per_side):
        wx = -length * 0.38 + i * (length * 0.13)
        for wy, wy_off in [(width / 2 + win_d / 2, 0), (-(width / 2 + win_d / 2), 0)]:
            win = _make_box(win_w, win_d, win_h, (140, 180, 230))
            win = _apply_offset(win, dx=wx, dy=wy, dz=chassis_h + body_h * 0.55)
            chassis = _concat([chassis, win])

    wheel_r = chassis_h * 1.2
    wheel_w = width * 0.1
    for lx in [length * 0.38, length * 0.22, -length * 0.38]:
        for sy in [width / 2 + wheel_w / 2, -(width / 2 + wheel_w / 2)]:
            w = _colored_cylinder(wheel_r, wheel_w, (25, 25, 25))
            w.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
            w = _apply_offset(w, dx=lx, dy=sy, dz=wheel_r)
            chassis = _concat([chassis, w])

    mesh = _concat([chassis, body])
    return _ground_align(mesh)


def make_coach_bus(length=12.0, width=2.6, height=3.5):
    chassis_h = height * 0.17
    body_h = height * 0.83

    chassis = _make_box(length, width, chassis_h, (60, 60, 60))
    chassis = _apply_offset(chassis, dz=chassis_h / 2)

    body = _make_box(length * 0.99, width * 0.97, body_h, (180, 180, 220))
    body = _apply_offset(body, dz=chassis_h + body_h / 2)

    wheel_r = chassis_h * 1.25
    wheel_w = width * 0.1
    for lx in [length * 0.40, length * 0.25, -length * 0.40]:
        for sy in [width / 2 + wheel_w / 2, -(width / 2 + wheel_w / 2)]:
            w = _colored_cylinder(wheel_r, wheel_w, (20, 20, 20))
            w.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
            w = _apply_offset(w, dx=lx, dy=sy, dz=wheel_r)
            chassis = _concat([chassis, w])

    mesh = _concat([chassis, body])
    return _ground_align(mesh)


def make_oak_tree(height=8.0, canopy_radius=3.5, trunk_radius=0.3):
    trunk_h = height * 0.45
    trunk = _colored_cylinder(trunk_radius, trunk_h, (100, 70, 40), sections=8)
    trunk = _apply_offset(trunk, dz=trunk_h / 2)

    canopy_h = height * 0.60
    canopy = _colored_sphere(canopy_radius, (60, 130, 60), subdivisions=2)
    canopy = _apply_offset(canopy, dz=trunk_h + canopy_h * 0.22)

    mesh = _concat([trunk, canopy])
    return _ground_align(mesh)


def make_pine_tree(height=12.0, base_radius=2.5, trunk_radius=0.25):
    trunk_h = height * 0.25
    trunk = _colored_cylinder(trunk_radius, trunk_h, (100, 70, 35), sections=6)
    trunk = _apply_offset(trunk, dz=trunk_h / 2)

    cone_h = height * 0.80
    cone = trimesh.creation.cone(radius=base_radius, height=cone_h, sections=12)
    cone.visual.vertex_colors = np.tile(
        np.array([40, 110, 50, 255], dtype=np.uint8), (len(cone.vertices), 1)
    )
    cone = _apply_offset(cone, dz=trunk_h)

    mesh = _concat([trunk, cone])
    return _ground_align(mesh)


def make_palm_tree(height=10.0, trunk_radius=0.2, frond_radius=3.0):
    trunk_h = height * 0.80
    trunk = _colored_cylinder(trunk_radius, trunk_h, (180, 150, 100), sections=8)
    trunk = _apply_offset(trunk, dz=trunk_h / 2)

    crown = _colored_sphere(frond_radius * 0.6, (50, 150, 50), subdivisions=2)
    crown = _apply_offset(crown, dz=trunk_h)

    for angle_deg in [0, 45, 90, 135, 180, 225, 270, 315]:
        a = np.radians(angle_deg)
        frond_len = frond_radius
        fx = np.cos(a) * frond_len
        fy = np.sin(a) * frond_len
        frond = _make_box(frond_len, 0.2, 0.05, (60, 160, 60))
        frond = _apply_offset(frond, dx=fx / 2, dy=fy / 2, dz=trunk_h)
        trunk = _concat([trunk, frond])

    mesh = _concat([trunk, crown])
    return _ground_align(mesh)


def make_round_deciduous(height=7.0, canopy_radius=3.0, trunk_radius=0.28):
    trunk_h = height * 0.38
    trunk = _colored_cylinder(trunk_radius, trunk_h, (110, 75, 42), sections=8)
    trunk = _apply_offset(trunk, dz=trunk_h / 2)

    for i, (dr, dh_frac, col) in enumerate([
        (canopy_radius, 0.15, (70, 140, 55)),
        (canopy_radius * 0.75, 0.35, (80, 150, 60)),
        (canopy_radius * 0.55, 0.52, (90, 160, 65)),
    ]):
        sphere = _colored_sphere(dr, col, subdivisions=2)
        sphere = _apply_offset(sphere, dz=trunk_h + height * dh_frac)
        trunk = _concat([trunk, sphere])

    return _ground_align(trunk)


def make_cypress_tree(height=9.0, max_radius=1.0, trunk_radius=0.15):
    trunk = _colored_cylinder(trunk_radius, height * 0.15, (100, 70, 40), sections=6)
    trunk = _apply_offset(trunk, dz=height * 0.075)

    cone_h = height * 0.90
    cone = trimesh.creation.cone(radius=max_radius, height=cone_h, sections=10)
    cone.visual.vertex_colors = np.tile(
        np.array([35, 90, 45, 255], dtype=np.uint8), (len(cone.vertices), 1)
    )
    cone = _apply_offset(cone, dz=height * 0.10)

    mesh = _concat([trunk, cone])
    return _ground_align(mesh)


def make_birch_tree(height=9.0, canopy_radius=2.5, trunk_radius=0.18):
    trunk_h = height * 0.50
    trunk = _colored_cylinder(trunk_radius, trunk_h, (230, 225, 215), sections=8)
    trunk = _apply_offset(trunk, dz=trunk_h / 2)

    canopy = _colored_sphere(canopy_radius, (130, 180, 80), subdivisions=2)
    canopy = _apply_offset(canopy, dz=trunk_h + canopy_radius * 0.40)

    mesh = _concat([trunk, canopy])
    return _ground_align(mesh)


CURATED_GENERATORS = {
    "sedan": (make_sedan, {"length": 4.4, "width": 1.8, "height": 1.45}),
    "suv": (make_suv, {"length": 4.6, "width": 1.9, "height": 1.7}),
    "hatchback": (make_hatchback, {"length": 4.0, "width": 1.75, "height": 1.5}),
    "coupe": (make_coupe, {"length": 4.5, "width": 1.85, "height": 1.3}),
    "sports_car": (make_sports_car, {"length": 4.3, "width": 1.88, "height": 1.2}),
    "station_wagon": (make_station_wagon, {"length": 4.7, "width": 1.82, "height": 1.55}),
    "compact": (make_compact, {"length": 3.8, "width": 1.68, "height": 1.48}),
    "box_truck": (make_box_truck, {"length": 8.0, "width": 2.5, "height": 3.2}),
    "flatbed_truck": (make_flatbed_truck, {"length": 8.5, "width": 2.5, "height": 1.8}),
    "pickup_truck": (make_pickup_truck, {"length": 5.5, "width": 2.0, "height": 1.9}),
    "cargo_van": (make_cargo_van, {"length": 5.2, "width": 2.0, "height": 2.2}),
    "city_bus": (make_city_bus, {"length": 10.5, "width": 2.6, "height": 3.2}),
    "coach_bus": (make_coach_bus, {"length": 12.0, "width": 2.6, "height": 3.5}),
    "oak_tree": (make_oak_tree, {"length": 7.0, "width": 7.0, "height": 8.0}),
    "pine_tree": (make_pine_tree, {"length": 5.0, "width": 5.0, "height": 12.0}),
    "palm_tree": (make_palm_tree, {"length": 6.0, "width": 6.0, "height": 10.0}),
    "round_deciduous": (make_round_deciduous, {"length": 6.0, "width": 6.0, "height": 7.0}),
    "cypress_tree": (make_cypress_tree, {"length": 2.0, "width": 2.0, "height": 9.0}),
    "birch_tree": (make_birch_tree, {"length": 5.0, "width": 5.0, "height": 9.0}),
}

SUBCATEGORY_TO_CLASS = {
    "sedan": "car", "suv": "car", "hatchback": "car", "coupe": "car",
    "sports_car": "car", "station_wagon": "car", "compact": "car",
    "box_truck": "truck", "flatbed_truck": "truck", "pickup_truck": "truck",
    "cargo_van": "van",
    "city_bus": "bus", "coach_bus": "bus",
    "oak_tree": "tree", "pine_tree": "tree", "palm_tree": "tree",
    "round_deciduous": "tree", "cypress_tree": "tree", "birch_tree": "tree",
}
