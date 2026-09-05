"""Held-out tasks: scientific computing, 3D geometry, and robotics.

This is the family the Multi-Scale Robot World capstone rests on, so the
validators here check the properties that make a design *fabricable* and its
physics *credible*, not that a formula was recalled. Two consequences shape
every task below.

First, the contracts are the ones a public standard or a theorem fixes --
binary STL's byte layout, the Euler characteristic, the parallel-axis
theorem, the Hamilton product, the standard Denavit-Hartenberg convention --
rather than contracts the prompt's own examples define. Reproducing an
example is not evidence of the capability; agreeing with a standard that the
prompt does not quote is.

Second, every validator drives the case where the memorised form of the
formula and the correct one diverge. A mesh whose volume is only right when
it happens to contain the origin, an inertia tensor that is only right on the
diagonal, a transform inverse that is only right without rotation, and a
quaternion product that is only right when the two rotations commute all
pass the textbook example and fail in an assembly. Those are the cases
asserted here, and they are what the paired mutations in
`tests/obstacle_references.py` reintroduce.

Float comparisons carry explicit tolerances. A geometry validator that
demands bit equality would fail a correct candidate that associates its
multiplications differently, which is the flaky case the contract refuses.
"""

from __future__ import annotations

from scripts.programming_obstacle_tasks import task
from scripts.programming_obstacle_tasks._support import LOAD_CANDIDATE, require

FAMILY = "scientific_3d_geometry_robotics"

TASKS = [
    task(
        f"{FAMILY}-0001", FAMILY,
        prompt=(
            "Implement three Python functions over 4x4 homogeneous rigid "
            "transforms, each represented as a tuple of four tuples of four "
            "floats in row-major order. compose(a, b) returns the matrix "
            "product a*b, so applying the result is equivalent to applying b "
            "first and then a. apply(t, point) returns the transformed 3D "
            "point as a tuple of three floats. invert(t) returns the inverse "
            "of a transform whose upper-left 3x3 block is a rotation. Return "
            "plain tuples rather than lists."
        ),
        validator=LOAD_CANDIDATE + require("compose") + require("apply")
        + require("invert") + """
import math

def close(a, b, tol=1e-9):
    return abs(a - b) <= tol

IDENTITY = tuple(
    tuple(1.0 if r == c else 0.0 for c in range(4)) for r in range(4)
)

def rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return ((c, -s, 0.0, 0.0), (s, c, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))

def with_translation(m, tx, ty, tz):
    rows = [list(row) for row in m]
    rows[0][3], rows[1][3], rows[2][3] = tx, ty, tz
    return tuple(tuple(row) for row in rows)

point = (1.0, 2.0, 3.0)

assert all(close(x, y) for x, y in zip(apply(IDENTITY, point), point)), \\
    'the identity transform moved a point'

translation = with_translation(IDENTITY, 5.0, -2.0, 0.5)
assert all(close(x, y) for x, y in
           zip(apply(translation, point), (6.0, 0.0, 3.5)))

# A quarter turn about +z carries +x to +y, fixing the handedness.
turn = rot_z(math.pi / 2)
assert all(close(a, b) for a, b in
           zip(apply(turn, (1.0, 0.0, 0.0)), (0.0, 1.0, 0.0)))

# compose must apply its RIGHT operand first.
shift = with_translation(IDENTITY, 1.0, 0.0, 0.0)
chained = apply(compose(turn, shift), point)
stepwise = apply(turn, apply(shift, point))
assert all(close(x, y) for x, y in zip(chained, stepwise)), \\
    'compose(a, b) is not the product a*b'
# ... so the reversed order must differ; equality here means the operands
# were commuted and the test above proved nothing.
assert not all(close(x, y) for x, y in
               zip(chained, apply(compose(shift, turn), point))), \\
    'compose is order-insensitive, so it cannot be a matrix product'

third = with_translation(rot_z(0.3), 2.0, 3.0, -1.0)
left = compose(compose(turn, shift), third)
right = compose(turn, compose(shift, third))
for row_l, row_r in zip(left, right):
    assert all(close(x, y) for x, y in zip(row_l, row_r)), \\
        'compose is not associative'

# The inverse must undo a transform that BOTH rotates and translates. A
# sign-flipped translation is correct without rotation and wrong with it,
# which is exactly the case an assembly hits and a bench example does not.
mixed = with_translation(rot_z(0.7), 4.0, -3.0, 2.0)
restored = compose(invert(mixed), mixed)
for r_index, row in enumerate(restored):
    for c_index, value in enumerate(row):
        assert close(value, 1.0 if r_index == c_index else 0.0), \\
            f'invert(t)*t is not the identity: row {r_index} = {row}'
assert all(close(x, y) for x, y in
           zip(apply(invert(mixed), apply(mixed, point)), point))

assert tuple(invert(mixed)[3]) == (0.0, 0.0, 0.0, 1.0), \\
    'the inverse lost its homogeneous bottom row'

result = compose(turn, shift)
assert isinstance(result, tuple) and isinstance(result[0], tuple), \\
    'compose did not return tuples'
assert isinstance(apply(turn, point), tuple), 'apply did not return a tuple'
""",
    ),
    task(
        f"{FAMILY}-0002", FAMILY,
        prompt=(
            "Implement Python functions for unit quaternions ordered "
            "(w, x, y, z). normalize(q) returns the unit quaternion and "
            "raises ValueError for the zero quaternion. multiply(q1, q2) "
            "returns the Hamilton product, so that the rotation it "
            "represents applies q2 first and then q1. to_matrix(q) returns "
            "the equivalent 3x3 rotation matrix as a tuple of three tuples, "
            "normalizing a non-unit input first. rotate(q, v) returns the "
            "rotated 3D vector as a tuple."
        ),
        validator=LOAD_CANDIDATE + require("normalize") + require("multiply")
        + require("to_matrix") + require("rotate") + """
import math

def close(a, b, tol=1e-9):
    return abs(a - b) <= tol

IDENTITY = (1.0, 0.0, 0.0, 0.0)
half = math.sqrt(0.5)
turn_z = (half, 0.0, 0.0, half)   # +90 degrees about z
turn_x = (half, half, 0.0, 0.0)   # +90 degrees about x

vector = (1.0, 2.0, 3.0)
assert all(close(a, b) for a, b in zip(rotate(IDENTITY, vector), vector)), \\
    'the identity quaternion rotated a vector'

assert all(close(a, b) for a, b in
           zip(rotate(turn_z, (1.0, 0.0, 0.0)), (0.0, 1.0, 0.0))), \\
    'a +90 degree turn about z did not carry +x to +y'
assert all(close(a, b) for a, b in
           zip(rotate(turn_x, (0.0, 1.0, 0.0)), (0.0, 0.0, 1.0))), \\
    'a +90 degree turn about x did not carry +y to +z'

# Composition order. These two rotations do NOT commute, so this separates
# the Hamilton product from its reverse -- the defect that silently mirrors
# an assembly built by chaining joint rotations.
composed = multiply(turn_z, turn_x)
assert all(close(a, b) for a, b in
           zip(rotate(composed, vector), rotate(turn_z, rotate(turn_x, vector)))), \\
    'multiply(q1, q2) does not apply q2 first'
assert not all(close(a, b) for a, b in
               zip(rotate(composed, vector),
                   rotate(turn_x, rotate(turn_z, vector)))), \\
    'the product is order-insensitive, so the test above proved nothing'

# A non-unit input must be normalized rather than scaling the vector.
assert all(close(a, b) for a, b in zip(rotate((2.0, 0.0, 0.0, 0.0), vector),
                                       vector)), \\
    'a non-unit quaternion scaled the rotated vector'

# q and -q are the same rotation: the double cover is not an error.
negated = tuple(-component for component in turn_z)
assert all(close(a, b) for a, b in
           zip(rotate(negated, vector), rotate(turn_z, vector))), \\
    'q and -q were treated as different rotations'

# The matrix must be a proper rotation: orthonormal with determinant +1.
matrix = to_matrix((0.3, -0.2, 0.5, 0.1))
for i in range(3):
    for j in range(3):
        product = sum(matrix[i][k] * matrix[j][k] for k in range(3))
        assert close(product, 1.0 if i == j else 0.0, 1e-9), \\
            'to_matrix did not return an orthonormal matrix'
determinant = (
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
    - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
    + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
)
assert close(determinant, 1.0, 1e-9), \\
    f'to_matrix returned a reflection, determinant {determinant}'

unit = normalize((0.0, 3.0, 4.0, 0.0))
assert close(sum(component * component for component in unit), 1.0)

try:
    normalize((0.0, 0.0, 0.0, 0.0))
except ValueError:
    pass
else:
    raise AssertionError('normalize accepted the zero quaternion')
""",
    ),
    task(
        f"{FAMILY}-0003", FAMILY,
        prompt=(
            "Implement a Python function mesh_report(vertices, triangles) "
            "for a triangle mesh, where triangles holds (i, j, k) index "
            "triples into vertices. Return a dict with keys: "
            "'boundary_edges', the number of undirected edges used by "
            "exactly one triangle; 'nonmanifold_edges', the number used by "
            "three or more; 'closed', true when both of those are zero; "
            "'consistent_winding', true when no directed edge is traversed "
            "by more than one triangle; and 'euler_characteristic', the "
            "quantity V - E + F counting undirected edges."
        ),
        validator=LOAD_CANDIDATE + require("mesh_report") + """
# A tetrahedron, wound outward. Every undirected edge is shared by exactly
# two faces and every directed edge is used once.
tetra_vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                  (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
tetra = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]

report = mesh_report(tetra_vertices, tetra)
assert report['closed'] is True, 'a closed tetrahedron was reported open'
assert report['consistent_winding'] is True
assert report['boundary_edges'] == 0
assert report['nonmanifold_edges'] == 0
assert report['euler_characteristic'] == 2, \\
    f"V - E + F is 2 for a tetrahedron, got {report['euler_characteristic']}"

# Removing a face opens exactly three edges and drops the characteristic.
open_report = mesh_report(tetra_vertices, tetra[:-1])
assert open_report['closed'] is False, 'an open mesh was reported watertight'
assert open_report['boundary_edges'] == 3
assert open_report['euler_characteristic'] == 1

# Reversing one face keeps the mesh closed but breaks its orientation, which
# is the defect that makes an exported normal point into the solid.
flipped = list(tetra)
flipped[3] = (1, 3, 2)
flipped_report = mesh_report(tetra_vertices, flipped)
assert flipped_report['closed'] is True
assert flipped_report['consistent_winding'] is False, \\
    'a reversed face was accepted as consistently wound'

# A cube: a different V, E and F reaching the same characteristic, so the
# count cannot have been special-cased to the tetrahedron.
cube_vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0),
                 (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 1.0),
                 (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)]
cube = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6),
        (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5)]
cube_report = mesh_report(cube_vertices, cube)
assert cube_report['closed'] is True, 'a closed cube was reported open'
assert cube_report['consistent_winding'] is True
assert cube_report['euler_characteristic'] == 2, \\
    f"V - E + F is 2 for a cube, got {cube_report['euler_characteristic']}"

# A third triangle on one edge is non-manifold: no printer can resolve it.
spur = list(tetra) + [(0, 1, 2)]
spur_report = mesh_report(tetra_vertices, spur)
assert spur_report['nonmanifold_edges'] > 0, \\
    'an edge shared by three faces was accepted as manifold'
assert spur_report['closed'] is False

empty = mesh_report([], [])
assert empty['euler_characteristic'] == 0 and empty['boundary_edges'] == 0
""",
    ),
    task(
        f"{FAMILY}-0004", FAMILY,
        prompt=(
            "Implement Python functions signed_volume(vertices, triangles) "
            "and centroid(vertices, triangles) for a closed triangle mesh "
            "whose faces are wound counter-clockwise seen from outside. "
            "signed_volume returns the enclosed volume, positive for that "
            "winding and negative when the mesh is inverted. centroid "
            "returns the centre of volume as a tuple of three floats, and "
            "raises ValueError when the enclosed volume is zero. Neither may "
            "assume the mesh contains or is near the coordinate origin."
        ),
        validator=LOAD_CANDIDATE + require("signed_volume") + require("centroid")
        + """
def close(a, b, tol=1e-9):
    return abs(a - b) <= tol

def box(ox, oy, oz, sx, sy, sz):
    vertices = [
        (ox, oy, oz), (ox + sx, oy, oz), (ox + sx, oy + sy, oz),
        (ox, oy + sy, oz), (ox, oy, oz + sz), (ox + sx, oy, oz + sz),
        (ox + sx, oy + sy, oz + sz), (ox, oy + sy, oz + sz),
    ]
    triangles = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
                 (0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6),
                 (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5)]
    return vertices, triangles

vertices, triangles = box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
assert close(signed_volume(vertices, triangles), 1.0), \\
    f'unit cube volume, got {signed_volume(vertices, triangles)}'
assert all(close(a, b) for a, b in
           zip(centroid(vertices, triangles), (0.5, 0.5, 0.5))), \\
    f'unit cube centroid, got {centroid(vertices, triangles)}'

# Translated far from the origin. A divergence-theorem sum is invariant here
# and a formula that quietly assumes the origin is inside the solid is not;
# a robot part is almost never modelled at the world origin.
vertices, triangles = box(10.0, -5.0, 3.0, 1.0, 1.0, 1.0)
assert close(signed_volume(vertices, triangles), 1.0), \\
    'volume changed when the mesh was translated'
assert all(close(a, b) for a, b in
           zip(centroid(vertices, triangles), (10.5, -4.5, 3.5))), \\
    f'translated centroid, got {centroid(vertices, triangles)}'

# Volume is cubic in scale and the centroid tracks the box.
vertices, triangles = box(0.0, 0.0, 0.0, 2.0, 3.0, 4.0)
assert close(signed_volume(vertices, triangles), 24.0)
assert all(close(a, b) for a, b in
           zip(centroid(vertices, triangles), (1.0, 1.5, 2.0)))

# A tetrahedron: one sixth, and a shape whose centroid is not its bounding
# box centre.
tetra_vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                  (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
tetra = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
assert close(signed_volume(tetra_vertices, tetra), 1.0 / 6.0), \\
    f'tetrahedron volume, got {signed_volume(tetra_vertices, tetra)}'
assert all(close(a, b) for a, b in
           zip(centroid(tetra_vertices, tetra), (0.25, 0.25, 0.25))), \\
    f'tetrahedron centroid, got {centroid(tetra_vertices, tetra)}'

# Inverted winding reports a negative volume rather than an absolute one.
inverted = [(a, c, b) for a, b, c in tetra]
assert close(signed_volume(tetra_vertices, inverted), -1.0 / 6.0), \\
    'an inverted mesh did not report a negative volume'

flat = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
try:
    centroid(flat, [(0, 1, 2), (0, 2, 1)])
except ValueError:
    pass
else:
    raise AssertionError('centroid accepted a zero-volume mesh')
""",
    ),
    task(
        f"{FAMILY}-0005", FAMILY,
        prompt=(
            "Implement box_inertia(mass, sx, sy, sz), returning the inertia "
            "tensor of a uniform rectangular solid about its own centre of "
            "mass as a 3x3 tuple of tuples with the axes aligned to its "
            "extents, and translate_inertia(inertia, mass, offset), applying "
            "the parallel-axis theorem to express that tensor about a point "
            "displaced from the centre of mass by offset. box_inertia raises "
            "ValueError unless the mass and all three extents are positive."
        ),
        validator=LOAD_CANDIDATE + require("box_inertia")
        + require("translate_inertia") + """
def close(a, b, tol=1e-9):
    return abs(a - b) <= tol

# Extents chosen distinct so a transposed or mismatched axis shows up.
tensor = box_inertia(12.0, 1.0, 2.0, 3.0)
assert close(tensor[0][0], 13.0), f'Ixx, got {tensor[0][0]}'
assert close(tensor[1][1], 10.0), f'Iyy, got {tensor[1][1]}'
assert close(tensor[2][2], 5.0), f'Izz, got {tensor[2][2]}'
for i in range(3):
    for j in range(3):
        if i != j:
            assert close(tensor[i][j], 0.0), \\
                'a principal-axis box has no products of inertia'

# Mass scales the tensor linearly.
doubled = box_inertia(24.0, 1.0, 2.0, 3.0)
assert close(doubled[0][0], 26.0)

# The parallel-axis theorem. Displacing along x leaves Ixx alone and adds
# m*d^2 to the two perpendicular moments -- an implementation that adds to
# all three is right on a sphere and wrong on every real part.
moved = translate_inertia(tensor, 12.0, (2.0, 0.0, 0.0))
assert close(moved[0][0], 13.0), \\
    f'Ixx must not change for a displacement along x, got {moved[0][0]}'
assert close(moved[1][1], 10.0 + 12.0 * 4.0), f'Iyy, got {moved[1][1]}'
assert close(moved[2][2], 5.0 + 12.0 * 4.0), f'Izz, got {moved[2][2]}'

# A zero displacement is the identity.
same = translate_inertia(tensor, 12.0, (0.0, 0.0, 0.0))
for i in range(3):
    for j in range(3):
        assert close(same[i][j], tensor[i][j]), \\
            'a zero offset changed the tensor'

# An off-axis displacement creates products of inertia with a NEGATIVE
# sign: the theorem adds m*(|d|^2 * delta_ij - d_i * d_j).
skew = translate_inertia(tensor, 12.0, (2.0, 3.0, 0.0))
assert close(skew[0][0], 13.0 + 12.0 * 9.0), f'Ixx, got {skew[0][0]}'
assert close(skew[1][1], 10.0 + 12.0 * 4.0), f'Iyy, got {skew[1][1]}'
assert close(skew[2][2], 5.0 + 12.0 * 13.0), f'Izz, got {skew[2][2]}'
assert close(skew[0][1], -12.0 * 6.0), \\
    f'Ixy must be -m*dx*dy, got {skew[0][1]}'
assert close(skew[0][2], 0.0) and close(skew[1][2], 0.0)

# The result stays symmetric, as any inertia tensor must.
for i in range(3):
    for j in range(3):
        assert close(skew[i][j], skew[j][i]), 'the tensor is not symmetric'

for bad in ((0.0, 1.0, 1.0, 1.0), (1.0, 0.0, 1.0, 1.0), (1.0, 1.0, 1.0, -2.0)):
    try:
        box_inertia(*bad)
    except ValueError:
        continue
    raise AssertionError(f'box_inertia accepted {bad}')
""",
    ),
    task(
        f"{FAMILY}-0006", FAMILY,
        prompt=(
            "Implement a Python function intersect(origin, direction, "
            "triangle) where triangle is a tuple of three 3D points. Return "
            "the parameter t such that origin + t*direction is the point "
            "where the ray meets the triangle, or None when it does not. "
            "direction need not be a unit vector, so t is a parameter rather "
            "than a distance. Only forward hits count: a triangle behind the "
            "origin is a miss. Points on an edge or vertex are hits, and a "
            "ray parallel to the triangle's plane or a zero-area triangle is "
            "always a miss."
        ),
        validator=LOAD_CANDIDATE + require("intersect") + """
def close(a, b, tol=1e-9):
    return a is not None and abs(a - b) <= tol

triangle = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))

hit = intersect((0.25, 0.25, 1.0), (0.0, 0.0, -1.0), triangle)
assert close(hit, 1.0), f'straight-down hit, got {hit}'

# t is a ray PARAMETER: doubling the direction halves it. A candidate that
# normalizes internally returns 1.0 here and is wrong.
hit = intersect((0.25, 0.25, 1.0), (0.0, 0.0, -2.0), triangle)
assert close(hit, 0.5), f't must scale with |direction|, got {hit}'

# Behind the origin is a miss, not a negative t.
assert intersect((0.25, 0.25, -1.0), (0.0, 0.0, -1.0), triangle) is None, \\
    'a triangle behind the ray origin was reported as a hit'

# Outside the triangle but inside its plane.
assert intersect((2.0, 2.0, 1.0), (0.0, 0.0, -1.0), triangle) is None
assert intersect((0.6, 0.6, 1.0), (0.0, 0.0, -1.0), triangle) is None, \\
    'a point beyond the hypotenuse was reported inside'

# Parallel to the plane.
assert intersect((0.25, 0.25, 1.0), (1.0, 0.0, 0.0), triangle) is None

# Boundary hits count: a vertex and an edge midpoint.
assert close(intersect((0.0, 0.0, 1.0), (0.0, 0.0, -1.0), triangle), 1.0), \\
    'a vertex hit was rejected'
assert close(intersect((0.5, 0.5, 1.0), (0.0, 0.0, -1.0), triangle), 1.0), \\
    'a hit on the hypotenuse was rejected'

# A degenerate triangle has no interior to hit.
collinear = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0))
assert intersect((0.5, 0.0, 1.0), (0.0, 0.0, -1.0), collinear) is None, \\
    'a zero-area triangle was reported as hit'

# An oblique ray against a triangle that is not axis-aligned.
slanted = ((0.0, 0.0, 1.0), (2.0, 0.0, 1.0), (0.0, 2.0, 3.0))
found = intersect((0.5, 0.5, -5.0), (0.0, 0.0, 1.0), slanted)
assert found is not None, 'an oblique triangle was missed'
z_hit = -5.0 + found * 1.0
assert abs(z_hit - 1.5) <= 1e-9, f'hit at z={z_hit}, expected the plane z=1.5'
""",
    ),
    task(
        f"{FAMILY}-0007", FAMILY,
        prompt=(
            "Implement write_binary_stl(triangles, header=b'') returning the "
            "bytes of a binary STL file, and read_binary_stl(data) returning "
            "the list of triangles it holds, each a tuple of three 3D vertex "
            "tuples. The binary STL layout is an 80-byte header, a "
            "little-endian uint32 triangle count, then per triangle twelve "
            "little-endian 32-bit floats -- the facet normal followed by the "
            "three vertices -- and a little-endian uint16 attribute byte "
            "count of zero. Write the unit-length normal implied by the "
            "vertex winding under the right-hand rule, or three zeros for a "
            "degenerate facet. Pad or truncate the supplied header to exactly "
            "80 bytes. read_binary_stl raises ValueError when the buffer is "
            "too short or its length disagrees with its own count."
        ),
        validator=LOAD_CANDIDATE + require("write_binary_stl")
        + require("read_binary_stl") + """
import struct

facet = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))

empty = write_binary_stl([])
assert len(empty) == 84, f'an empty STL is 84 bytes, got {len(empty)}'
assert struct.unpack_from('<I', empty, 80)[0] == 0
assert list(read_binary_stl(empty)) == []

one = write_binary_stl([facet])
assert len(one) == 134, f'84 + 50 bytes per facet, got {len(one)}'
assert struct.unpack_from('<I', one, 80)[0] == 1, \\
    'the triangle count is a little-endian uint32 at offset 80'

# The normal follows the right-hand rule and is unit length.
normal = struct.unpack_from('<3f', one, 84)
assert all(abs(a - b) <= 1e-6 for a, b in zip(normal, (0.0, 0.0, 1.0))), \\
    f'counter-clockwise winding implies +z, got {normal}'
reversed_normal = struct.unpack_from(
    '<3f', write_binary_stl([(facet[0], facet[2], facet[1])]), 84)
assert all(abs(a - b) <= 1e-6 for a, b in zip(reversed_normal, (0.0, 0.0, -1.0))), \\
    f'reversed winding implies -z, got {reversed_normal}'

# The attribute byte count occupies the final two bytes of the facet record.
assert struct.unpack_from('<H', one, 84 + 48)[0] == 0

# A degenerate facet gets a zero normal rather than a division by zero.
degenerate = write_binary_stl([((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                                (2.0, 0.0, 0.0))])
assert struct.unpack_from('<3f', degenerate, 84) == (0.0, 0.0, 0.0)

# Round trip, at float32 precision.
mesh = [facet,
        ((1.0, 2.0, 3.0), (4.5, -2.25, 0.5), (-1.0, 0.25, 7.5)),
        ((0.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0))]
encoded = write_binary_stl(mesh, b'wizard')
assert len(encoded) == 84 + 50 * 3
recovered = list(read_binary_stl(encoded))
assert len(recovered) == 3, f'round trip lost facets: {len(recovered)}'
for original, parsed in zip(mesh, recovered):
    for want, got in zip(original, parsed):
        assert all(abs(a - b) <= 1e-6 for a, b in zip(want, got)), \\
            f'round trip changed {want} into {got}'

# The header is exactly 80 bytes either way.
assert len(write_binary_stl([], b'x' * 200)) == 84
assert write_binary_stl([], b'wizard')[:6] == b'wizard'

try:
    read_binary_stl(b'\\x00' * 40)
except ValueError:
    pass
else:
    raise AssertionError('a truncated buffer was accepted')

try:
    read_binary_stl(one[:-10])
except ValueError:
    pass
else:
    raise AssertionError('a length disagreeing with the count was accepted')
""",
    ),
    task(
        f"{FAMILY}-0008", FAMILY,
        prompt=(
            "Implement forward_kinematics(links, joint_values) for a serial "
            "revolute manipulator. links is a sequence of (a, alpha, d, "
            "theta_offset) standard Denavit-Hartenberg parameters and "
            "joint_values holds one angle per link, added to that link's "
            "theta_offset. Compose the links under the standard convention, "
            "in which a link contributes Rot_z(theta) * Trans_z(d) * "
            "Trans_x(a) * Rot_x(alpha), and return the end-effector pose as a "
            "4x4 row-major tuple of tuples. An empty chain returns the "
            "identity, and a length mismatch raises ValueError."
        ),
        validator=LOAD_CANDIDATE + require("forward_kinematics") + """
import math

def close(a, b, tol=1e-9):
    return abs(a - b) <= tol

def position(pose):
    return (pose[0][3], pose[1][3], pose[2][3])

identity = forward_kinematics([], [])
for row in range(4):
    for column in range(4):
        assert close(identity[row][column], 1.0 if row == column else 0.0), \\
            'an empty chain is not the identity'

# A planar two-link arm, alpha = d = 0, unit links. The closed form is
# x = a1*cos(t1) + a2*cos(t1+t2), y = a1*sin(t1) + a2*sin(t1+t2).
planar = [(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)]

got = position(forward_kinematics(planar, [0.0, 0.0]))
assert all(close(a, b) for a, b in zip(got, (2.0, 0.0, 0.0))), \\
    f'straight-out arm, got {got}'

got = position(forward_kinematics(planar, [math.pi / 2, 0.0]))
assert all(close(a, b) for a, b in zip(got, (0.0, 2.0, 0.0))), \\
    f'arm turned a quarter turn at the shoulder, got {got}'

got = position(forward_kinematics(planar, [0.0, math.pi / 2]))
assert all(close(a, b) for a, b in zip(got, (1.0, 1.0, 0.0))), \\
    f'elbow bent a quarter turn, got {got}'

# Both joints bent. Neither link lies on an axis here, so an implementation
# that drops or transposes a rotation term in the link matrix diverges,
# where the three poses above would still agree.
got = position(forward_kinematics(planar, [math.pi / 2, math.pi / 2]))
assert all(close(a, b) for a, b in zip(got, (-1.0, 1.0, 0.0))), \\
    f'shoulder and elbow both turned, got {got}'

# theta_offset is added to the joint value, not ignored or replacing it.
offset_arm = [(1.0, 0.0, 0.0, math.pi / 2), (1.0, 0.0, 0.0, 0.0)]
assert all(close(a, b) for a, b in
           zip(position(forward_kinematics(offset_arm, [0.0, 0.0])),
               (0.0, 2.0, 0.0))), 'theta_offset was ignored'
assert all(close(a, b) for a, b in
           zip(position(forward_kinematics(offset_arm, [math.pi / 2, 0.0])),
               (-2.0, 0.0, 0.0))), 'theta_offset did not add to the joint value'

# A twist out of the plane: alpha carries the next link's d off the z axis.
twisted = [(0.0, math.pi / 2, 0.0, 0.0), (0.0, 0.0, 2.0, 0.0)]
got = position(forward_kinematics(twisted, [0.0, 0.0]))
assert all(close(a, b) for a, b in zip(got, (0.0, -2.0, 0.0))), \\
    f'a quarter-turn twist did not carry +z to -y, got {got}'

# The pose stays a valid homogeneous transform.
pose = forward_kinematics(planar, [0.4, -1.1])
assert tuple(pose[3]) == (0.0, 0.0, 0.0, 1.0), 'the bottom row is not [0 0 0 1]'
for i in range(3):
    for j in range(3):
        product = sum(pose[i][k] * pose[j][k] for k in range(3))
        assert close(product, 1.0 if i == j else 0.0, 1e-9), \\
            'the rotation block is not orthonormal'

try:
    forward_kinematics(planar, [0.0])
except ValueError:
    pass
else:
    raise AssertionError('a joint-value count mismatch was accepted')
""",
    ),
    task(
        f"{FAMILY}-0009", FAMILY,
        prompt=(
            "Implement slice_mesh(vertices, triangles, z) returning the "
            "cross-section of a closed triangle mesh cut by the horizontal "
            "plane at height z. triangles holds (i, j, k) index triples into "
            "vertices. Return a list of closed contours; each contour is a "
            "list of (x, y) tuples in order around the loop, and the first "
            "point is not repeated at the end. Contours may be returned in "
            "any order, may start at any point on their loop, and may run in "
            "either direction. A plane that misses the mesh yields an empty "
            "list. No vertex of the mesh lies exactly on the plane."
        ),
        validator=LOAD_CANDIDATE + require("slice_mesh") + """
def close(a, b, tol=1e-7):
    return abs(a - b) <= tol

def area(loop):
    total = 0.0
    for index in range(len(loop)):
        x0, y0 = loop[index]
        x1, y1 = loop[(index + 1) % len(loop)]
        total += x0 * y1 - x1 * y0
    return abs(total) / 2.0

def perimeter(loop):
    total = 0.0
    for index in range(len(loop)):
        x0, y0 = loop[index]
        x1, y1 = loop[(index + 1) % len(loop)]
        total += ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    return total

def box(x0, y0, z0, x1, y1, z1):
    corners = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
               (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    faces = [(0, 3, 2), (0, 2, 1), (4, 5, 6), (4, 6, 7),
             (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
             (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    return corners, faces

# A 2x2x2 cube. Every side face is split into two triangles, so a correct
# implementation has to chain eight segments into one loop rather than
# report eight fragments.
cube, cube_faces = box(0.0, 0.0, 0.0, 2.0, 2.0, 2.0)
loops = slice_mesh(cube, cube_faces, 0.5)
assert isinstance(loops, list), 'slice_mesh did not return a list'
assert len(loops) == 1, f'a cube has one cross-section, got {len(loops)}'
square = loops[0]
assert len(square) >= 3, 'the segments were not chained into a closed loop'
assert tuple(square[0]) != tuple(square[-1]), 'the first point was repeated'
assert close(area(square), 4.0), f'cross-section area {area(square)} is not 4'
assert close(perimeter(square), 8.0), 'the contour is not the 2x2 square'

# A prism's cross-section is height-independent, so the cube cannot detect a
# reversed interpolation parameter. A tetrahedron sliced off-centre can.
tetra = [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0)]
tetra_faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
loops = slice_mesh(tetra, tetra_faces, 1.0)
assert len(loops) == 1, f'one contour expected at z=1, got {len(loops)}'
triangle = loops[0]
assert len(triangle) == 3, f'the section is a triangle, got {len(triangle)}'
assert close(area(triangle), 4.5), (
    f'the section at z=1 has area 4.5, got {area(triangle)}')

# Two disjoint solids yield two independent contours.
first, first_faces = box(0.0, 0.0, 0.0, 2.0, 2.0, 2.0)
second, second_faces = box(5.0, 5.0, 0.0, 6.0, 6.0, 2.0)
both = list(first) + list(second)
both_faces = list(first_faces) + [
    tuple(index + len(first) for index in face) for face in second_faces]
loops = slice_mesh(both, both_faces, 0.5)
assert len(loops) == 2, f'two solids give two contours, got {len(loops)}'
areas = sorted(round(area(loop), 6) for loop in loops)
assert areas == [1.0, 4.0], f'contour areas {areas} are not [1.0, 4.0]'

for miss in (-1.0, 2.5, 7.0):
    assert slice_mesh(cube, cube_faces, miss) == [], (
        f'a plane at z={miss} misses the cube but returned a contour')
""",
    ),
    task(
        f"{FAMILY}-0010", FAMILY,
        prompt=(
            "Implement overhang_faces(vertices, triangles, "
            "max_overhang_degrees) for a mesh printed along +z. triangles "
            "holds (i, j, k) index triples wound counter-clockwise seen from "
            "outside, so the right-hand rule gives each face its outward "
            "normal. A face is downward-facing when the z component of that "
            "unit normal is negative; its overhang angle is the angle in "
            "degrees between the normal and straight down, (0, 0, -1), so a "
            "flat ceiling measures 0 and a vertical wall approaches 90. "
            "Return the sorted indices of the downward-facing faces whose "
            "overhang angle is strictly less than max_overhang_degrees. "
            "Raise ValueError for a degenerate triangle of zero area, and "
            "unless max_overhang_degrees lies in (0, 90]."
        ),
        validator=LOAD_CANDIDATE + require("overhang_faces") + """
import math

# Normals are fixed by 3-4-5 triples so the two sloped faces sit 8.13 degrees
# either side of the 45-degree threshold. Nothing here is decided at a
# boundary; the boundary arithmetic itself is asserted separately below.
vertices = [
    (0.0, 0.0, 0.0),   # 0
    (1.0, 0.0, 0.0),   # 1
    (0.0, 4.0, 3.0),   # 2
    (0.0, 3.0, 4.0),   # 3
    (0.0, 1.0, 0.0),   # 4
    (0.0, 0.0, 1.0),   # 5
    (0.0, 0.0, 5.0),   # 6
    (0.0, 1.0, 5.0),   # 7
    (1.0, 0.0, 5.0),   # 8
]
triangles = [
    (0, 2, 1),   # 0: normal (0, 3, -4)/5  -> 36.87 degrees
    (0, 3, 1),   # 1: normal (0, 4, -3)/5  -> 53.13 degrees
    (6, 7, 8),   # 2: normal (0, 0, -1)    -> 0 degrees, a flat ceiling
    (0, 1, 4),   # 3: normal (0, 0, 1)     -> faces up, never supported
    (0, 4, 5),   # 4: normal (1, 0, 0)     -> vertical wall, nz is 0
]

assert abs(math.degrees(math.acos(4.0 / 5.0)) - 36.8698976) < 1e-5
assert abs(math.degrees(math.acos(3.0 / 5.0)) - 53.1301024) < 1e-5

got = overhang_faces(vertices, triangles, 45.0)
assert list(got) == [0, 2], f'at 45 degrees the supported set is [0, 2], got {got}'

got = overhang_faces(vertices, triangles, 60.0)
assert list(got) == [0, 1, 2], f'at 60 degrees expected [0, 1, 2], got {got}'

got = overhang_faces(vertices, triangles, 30.0)
assert list(got) == [2], f'at 30 degrees only the ceiling qualifies, got {got}'

# An upward-facing facet is never an overhang, however shallow it is.
got = overhang_faces(vertices, triangles, 90.0)
assert 3 not in got, 'an upward-facing facet was reported as an overhang'
assert 4 not in got, 'a vertical wall was reported as an overhang'

try:
    overhang_faces(vertices, [(0, 1, 1)], 45.0)
except ValueError:
    pass
else:
    raise AssertionError('a degenerate zero-area triangle was accepted')

for bad in (0.0, -5.0, 120.0):
    try:
        overhang_faces(vertices, triangles, bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'max_overhang_degrees={bad} was accepted')
""",
    ),
    task(
        f"{FAMILY}-0011", FAMILY,
        prompt=(
            "Implement inverse_kinematics(l1, l2, x, y, elbow='up') for a "
            "planar two-link revolute arm. The first link leaves the origin "
            "at angle theta1 from +x; the second is attached at its tip and "
            "runs at theta1 + theta2, so the end effector sits at "
            "(l1*cos(theta1) + l2*cos(theta1+theta2), l1*sin(theta1) + "
            "l2*sin(theta1+theta2)). Return (theta1, theta2) in radians with "
            "theta1 in (-pi, pi]. elbow is 'up' for the solution with theta2 "
            ">= 0 and 'down' for theta2 <= 0. Raise ValueError when the "
            "target lies outside the reachable annulus, when either link "
            "length is not positive, and for any other elbow value."
        ),
        validator=LOAD_CANDIDATE + require("inverse_kinematics") + """
import math

def forward(l1, l2, theta1, theta2):
    return (l1 * math.cos(theta1) + l2 * math.cos(theta1 + theta2),
            l1 * math.sin(theta1) + l2 * math.sin(theta1 + theta2))

L1, L2 = 2.0, 1.0   # reachable annulus is 1 <= radius <= 3

# The expectation is recomputed from the returned angles rather than written
# down, so the fixture and the answer cannot drift apart.
targets = [(2.5, 0.0), (0.0, 1.5), (-1.2, 1.4), (1.0, -1.0), (-2.0, -1.5)]
for target in targets:
    for elbow in ('up', 'down'):
        theta1, theta2 = inverse_kinematics(L1, L2, target[0], target[1], elbow)
        assert -math.pi < theta1 <= math.pi + 1e-12, (
            f'theta1={theta1} is outside (-pi, pi]')
        if elbow == 'up':
            assert theta2 >= -1e-12, f'elbow up gave theta2={theta2}'
        else:
            assert theta2 <= 1e-12, f'elbow down gave theta2={theta2}'
        reached = forward(L1, L2, theta1, theta2)
        assert abs(reached[0] - target[0]) < 1e-9, (
            f'{elbow} at {target} reached x={reached[0]}')
        assert abs(reached[1] - target[1]) < 1e-9, (
            f'{elbow} at {target} reached y={reached[1]}')

# Away from the singular fully-extended pose the two branches are distinct.
up = inverse_kinematics(L1, L2, 2.5, 0.0, 'up')
down = inverse_kinematics(L1, L2, 2.5, 0.0, 'down')
assert abs(up[1] - down[1]) > 1e-6, 'both elbow branches gave the same pose'

for unreachable in ((0.5, 0.0), (0.0, 0.4), (4.0, 0.0), (2.5, 2.5)):
    try:
        inverse_kinematics(L1, L2, unreachable[0], unreachable[1])
    except ValueError:
        pass
    else:
        raise AssertionError(f'unreachable target {unreachable} was accepted')

for bad_links in ((0.0, 1.0), (1.0, -2.0)):
    try:
        inverse_kinematics(bad_links[0], bad_links[1], 0.5, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError(f'link lengths {bad_links} were accepted')

try:
    inverse_kinematics(L1, L2, 2.5, 0.0, 'sideways')
except ValueError:
    pass
else:
    raise AssertionError('an unknown elbow value was accepted')
""",
    ),
    task(
        f"{FAMILY}-0012", FAMILY,
        prompt=(
            "Implement fit_plane(points) over a sequence of 3D points. "
            "Return (centroid, normal): centroid is the arithmetic mean as a "
            "tuple of three floats, and normal is the unit vector that "
            "minimises the sum of squared perpendicular distances from the "
            "points to the plane through the centroid. Orient normal so that "
            "its component of largest absolute value is positive, breaking a "
            "tie in favour of the earlier index. Raise ValueError for fewer "
            "than three points and when the points are collinear, because no "
            "single best-fit plane exists then."
        ),
        validator=LOAD_CANDIDATE + require("fit_plane") + """
import math

def residual(points, centroid, normal):
    total = 0.0
    for p in points:
        total += sum((p[i] - centroid[i]) * normal[i] for i in range(3)) ** 2
    return total

def unit(v):
    length = math.sqrt(sum(c * c for c in v))
    return tuple(c / length for c in v)

# A plane parallel to the z axis. Fitting z = a*x + b*y + c cannot represent
# it at all, which is the near-miss this case exists to reject.
vertical = [(2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (2.0, 0.0, 1.0),
            (2.0, 3.0, -2.0), (2.0, -1.0, 4.0)]
centroid, normal = fit_plane(vertical)
assert abs(centroid[0] - 2.0) < 1e-9, f'centroid x {centroid[0]} is not 2'
for i in range(3):
    expected = sum(p[i] for p in vertical) / len(vertical)
    assert abs(centroid[i] - expected) < 1e-9, 'centroid is not the mean'
assert abs(normal[0] - 1.0) < 1e-7, f'normal {normal} is not +x'
assert abs(normal[1]) < 1e-7 and abs(normal[2]) < 1e-7, f'normal {normal}'

horizontal = [(0.0, 0.0, 5.0), (1.0, 0.0, 5.0), (0.0, 2.0, 5.0),
              (3.0, -1.0, 5.0)]
_, normal = fit_plane(horizontal)
assert abs(normal[2] - 1.0) < 1e-7, f'normal {normal} is not +z'

# Noisy samples about a tilted plane. The returned normal must actually be
# the minimiser, checked against a deterministic sweep of the sphere.
seed = 12345
def nextf():
    global seed
    seed = (1103515245 * seed + 12345) % 2147483648
    return seed / 2147483648.0 - 0.5

base = unit((1.0, 2.0, 2.0))
noisy = []
for _ in range(60):
    u, v = nextf() * 6.0, nextf() * 6.0
    point = (u, v, -(base[0] * u + base[1] * v) / base[2])
    offset = nextf() * 0.05
    noisy.append(tuple(point[i] + offset * base[i] for i in range(3)))

centroid, normal = fit_plane(noisy)
assert abs(math.sqrt(sum(c * c for c in normal)) - 1.0) < 1e-9, 'not a unit vector'
best = residual(noisy, centroid, normal)
for a in range(18):
    for b in range(36):
        theta = math.pi * (a + 0.5) / 18.0
        phi = 2.0 * math.pi * b / 36.0
        candidate_normal = (math.sin(theta) * math.cos(phi),
                            math.sin(theta) * math.sin(phi),
                            math.cos(theta))
        assert best <= residual(noisy, centroid, candidate_normal) + 1e-9, (
            'a swept direction fits the points better than the returned normal')

dominant = max(range(3), key=lambda i: (abs(normal[i]), -i))
assert normal[dominant] > 0.0, f'normal {normal} is not oriented as specified'

for degenerate in (
        [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (-3.0, -3.0, -3.0)],
        [(1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (5.0, 0.0, 0.0)],
        [(4.0, 4.0, 4.0)] * 5):
    try:
        fit_plane(degenerate)
    except ValueError:
        pass
    else:
        raise AssertionError('collinear points were given a unique plane')

for short in ([], [(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]):
    try:
        fit_plane(short)
    except ValueError:
        pass
    else:
        raise AssertionError('fewer than three points were accepted')
""",
    ),
    task(
        f"{FAMILY}-0013", FAMILY,
        prompt=(
            "Implement integrate(derivative, state, t0, t1, steps) advancing "
            "an ordinary differential equation with the classical "
            "fourth-order Runge-Kutta method at a fixed step size. state is "
            "a sequence of floats and derivative(t, state) returns the time "
            "derivative as a sequence of the same length. Take exactly steps "
            "equal steps from t0 to t1 and return the final state as a tuple "
            "of floats. Raise ValueError unless steps is a positive integer."
        ),
        validator=LOAD_CANDIDATE + require("integrate") + """
import math

result = integrate(lambda t, s: (s[0],), (1.0,), 0.0, 1.0, 16)
assert isinstance(result, tuple), 'integrate did not return a tuple'
assert len(result) == 1, f'the state changed length: {result}'
error_fine = abs(result[0] - math.e)
assert error_fine < 1e-4, (
    f'exp(1) came out as {result[0]}, off by {error_fine}')

# Halving the step must cut the error by about 2**4. Euler would give 2 and
# any second-order scheme 4, so the admitted band cannot be reached by
# accident yet is wide enough that rounding cannot push a correct answer out.
error_coarse = abs(integrate(lambda t, s: (s[0],), (1.0,), 0.0, 1.0, 8)[0]
                   - math.e)
ratio = error_coarse / error_fine
assert 10.0 <= ratio <= 22.0, (
    f'halving the step changed the error by {ratio}x, not about 16x')

# A vector field, integrated over a full period of a unit harmonic
# oscillator. Energy is the invariant a wrong stage weighting destroys.
state = integrate(lambda t, s: (s[1], -s[0]), (1.0, 0.0),
                  0.0, 2.0 * math.pi, 400)
assert len(state) == 2, f'the vector state changed length: {state}'
energy = state[0] ** 2 + state[1] ** 2
assert abs(energy - 1.0) < 1e-6, f'energy drifted to {energy}'
assert abs(state[0] - 1.0) < 1e-6, f'the oscillator returned to {state[0]}'
assert abs(state[1]) < 1e-6, f'the velocity returned to {state[1]}'

# The derivative may depend on t explicitly: y' = 2t integrates to t**2.
result = integrate(lambda t, s: (2.0 * t,), (0.0,), 0.0, 3.0, 12)
assert abs(result[0] - 9.0) < 1e-9, f'y(3) came out as {result[0]}'

for bad in (0, -3):
    try:
        integrate(lambda t, s: (s[0],), (1.0,), 0.0, 1.0, bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'steps={bad} was accepted')
""",
    ),
    task(
        f"{FAMILY}-0014", FAMILY,
        prompt=(
            "Implement boxes_overlap(a, b) for two oriented bounding boxes "
            "in 3D. Each box is a tuple (center, axes, half_extents): center "
            "is a 3D point, axes is a tuple of three orthonormal row vectors "
            "giving the box's local frame, and half_extents holds the three "
            "non-negative half-widths along those axes. Return True when the "
            "boxes share at least one point and False when a plane separates "
            "them. Raise ValueError for a negative half extent or for axes "
            "that are not orthonormal to within 1e-9."
        ),
        timeout_seconds=60.0,
        validator=LOAD_CANDIDATE + require("boxes_overlap") + """
import math
import itertools

# The expected verdict is computed by an algorithm unrelated to projection
# tests: a bounded intersection of half-spaces is non-empty exactly when some
# triple of its bounding planes meets at a feasible point. That keeps the
# oracle independent of whatever the candidate does.
def half_spaces(box):
    center, axes, half = box
    out = []
    for i in range(3):
        normal = axes[i]
        offset = sum(center[k] * normal[k] for k in range(3))
        out.append((normal, offset + half[i]))
        out.append((tuple(-c for c in normal), half[i] - offset))
    return out

def solve3(rows, rhs):
    def det(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    base = det(rows)
    if abs(base) < 1e-9:
        return None
    out = []
    for column in range(3):
        swapped = [list(row) for row in rows]
        for r in range(3):
            swapped[r][column] = rhs[r]
        out.append(det(swapped) / base)
    return tuple(out)

def oracle(a, b):
    planes = half_spaces(a) + half_spaces(b)
    for triple in itertools.combinations(range(len(planes)), 3):
        rows = [planes[i][0] for i in triple]
        rhs = [planes[i][1] for i in triple]
        point = solve3(rows, rhs)
        if point is None:
            continue
        if all(sum(point[k] * normal[k] for k in range(3)) <= bound + 1e-9
               for normal, bound in planes):
            return True
    return False

def rotation(axis, angle):
    length = math.sqrt(sum(c * c for c in axis))
    x, y, z = (c / length for c in axis)
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    return ((t * x * x + c, t * x * y - s * z, t * x * z + s * y),
            (t * x * y + s * z, t * y * y + c, t * y * z - s * x),
            (t * x * z - s * y, t * y * z + s * x, t * z * z + c))

IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
base = ((0.0, 0.0, 0.0), IDENTITY, (1.0, 1.0, 1.0))

cases = []
for angle in (0.3, 0.7854, 1.1, 2.0):
    for axis in ((0.0, 0.0, 1.0), (1.0, 1.0, 0.0), (1.0, 1.0, 1.0)):
        for shift in (0.6, 1.7, 2.05, 2.4, 3.1):
            other = ((shift, shift * 0.5, shift * 0.25),
                     rotation(axis, angle), (1.0, 1.0, 1.0))
            cases.append((base, other))
cases.append((base, ((0.5, 0.5, 0.5), IDENTITY, (1.0, 1.0, 1.0))))
cases.append((base, ((0.0, 0.0, 0.0), IDENTITY, (0.2, 0.2, 0.2))))
cases.append((base, ((3.5, 0.0, 0.0), IDENTITY, (1.0, 1.0, 1.0))))

agreed = {True: 0, False: 0}
for index, (first, second) in enumerate(cases):
    expected = oracle(first, second)
    got = boxes_overlap(first, second)
    assert isinstance(got, bool), f'case {index} returned {got!r}, not a bool'
    assert got == expected, (
        f'case {index} answered {got} where the half-space oracle says '
        f'{expected}')
    assert boxes_overlap(second, first) == expected, (
        f'case {index} is not symmetric in its arguments')
    agreed[expected] += 1

# A suite that never exercises one verdict proves nothing about it.
assert agreed[True] >= 5 and agreed[False] >= 5, (
    f'the fixtures are one-sided: {agreed}')

try:
    boxes_overlap(base, ((0.0, 0.0, 0.0), IDENTITY, (1.0, -1.0, 1.0)))
except ValueError:
    pass
else:
    raise AssertionError('a negative half extent was accepted')

skewed = ((1.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.0, 0.0, 1.0))
try:
    boxes_overlap(base, ((0.0, 0.0, 0.0), skewed, (1.0, 1.0, 1.0)))
except ValueError:
    pass
else:
    raise AssertionError('a non-orthonormal frame was accepted')
""",
    ),
    task(
        f"{FAMILY}-0015", FAMILY,
        prompt=(
            "Implement solve(matrix, rhs) returning the solution of a dense "
            "square linear system as a tuple of floats, using Gaussian "
            "elimination with partial pivoting. matrix is a sequence of rows "
            "and rhs a sequence of the same length. Raise ValueError when "
            "the matrix is empty or not square, when rhs does not match its "
            "size, and when the matrix is singular to working precision."
        ),
        validator=LOAD_CANDIDATE + require("solve") + """
def multiply(matrix, vector):
    return [sum(row[i] * vector[i] for i in range(len(vector)))
            for row in matrix]

def check(matrix, expected, tol=1e-9):
    rhs = multiply(matrix, expected)
    got = solve(matrix, rhs)
    assert isinstance(got, tuple), 'solve did not return a tuple'
    assert len(got) == len(expected), f'expected {len(expected)} unknowns'
    for i in range(len(expected)):
        assert abs(got[i] - expected[i]) < tol, (
            f'x[{i}] came out as {got[i]}, expected {expected[i]}')

# A zero in the leading position. Correct, but fatal without pivoting.
check([[0.0, 1.0], [1.0, 0.0]], [2.0, 1.0])

# Non-singular, yet the second pivot vanishes during elimination.
check([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [1.0, 3.0, 4.0]], [1.0, 2.0, -1.0])

# Pivoting on magnitude, not on a zero test. The leading entry here is
# 1e-5 -- far too large for any "treat this as zero and swap" rule to fire --
# but it is thirteen orders below the rest of its column, so eliminating with
# it loses four significant digits of the first unknown. Swapping to the
# largest available pivot returns both unknowns exactly. Measured: a solver
# that swaps only on a near-zero pivot is off by 1.3e-4 here for every zero
# threshold between 1e-14 and 1e-6.
check([[1e-5, 1e8], [1e8, 1e8]], [1.0, 1.0], tol=1e-6)

check([[4.0, -2.0, 1.0, 0.0, 3.0],
       [-2.0, 5.0, 0.0, 1.0, -1.0],
       [1.0, 0.0, 6.0, -3.0, 2.0],
       [0.0, 1.0, -3.0, 7.0, 1.0],
       [3.0, -1.0, 2.0, 1.0, 8.0]],
      [1.5, -2.25, 0.75, 3.0, -1.125])

for singular in ([[1.0, 2.0], [2.0, 4.0]],
                 [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [0.0, 1.0, 3.0]],
                 [[0.0, 0.0], [0.0, 0.0]]):
    try:
        solve(singular, [1.0] * len(singular))
    except ValueError:
        pass
    else:
        raise AssertionError(f'the singular matrix {singular} was solved')

try:
    solve([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [1.0, 2.0])
except ValueError:
    pass
else:
    raise AssertionError('a non-square matrix was accepted')

try:
    solve([[1.0, 0.0], [0.0, 1.0]], [1.0, 2.0, 3.0])
except ValueError:
    pass
else:
    raise AssertionError('a mismatched right-hand side was accepted')

try:
    solve([], [])
except ValueError:
    pass
else:
    raise AssertionError('an empty system was accepted')
""",
    ),
    task(
        f"{FAMILY}-0016", FAMILY,
        prompt=(
            "Implement natural_cubic_spline(xs, ys) returning a callable f "
            "that evaluates the natural cubic spline through the knots. The "
            "spline is a cubic on each interval, passes through every knot, "
            "is twice continuously differentiable across the interior knots, "
            "and has zero second derivative at both ends. Spacing between "
            "knots is not uniform. Raise ValueError when the inputs differ "
            "in length, when there are fewer than three knots, and when xs "
            "is not strictly increasing; f raises ValueError outside "
            "[xs[0], xs[-1]]."
        ),
        validator=LOAD_CANDIDATE + require("natural_cubic_spline") + """
# f is a cubic on each interval, so a central difference recovers its second
# derivative exactly and the five-point stencil its first. Both are then
# extrapolated to the knot along the interval that owns them -- a polynomial
# extrapolation, so it is exact too, and the one-sided limits it compares are
# the actual definition of C1 and C2 continuity.
def deriv1(f, x, h):
    return (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / (12 * h)

def deriv2(f, x, h):
    return (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)

def extrapolate(samples, target):
    total = 0.0
    for i, (xi, yi) in enumerate(samples):
        term = yi
        for j, (xj, _) in enumerate(samples):
            if i != j:
                term *= (target - xj) / (xi - xj)
        total += term
    return total

def limit(f, lo, hi, target, order):
    span = hi - lo
    h = 0.05 * span
    fractions = (0.2, 0.5, 0.8) if order == 1 else (0.25, 0.75)
    samples = []
    for fraction in fractions:
        x = lo + fraction * span
        value = deriv1(f, x, h) if order == 1 else deriv2(f, x, h)
        samples.append((x, value))
    return extrapolate(samples, target)

xs = [0.0, 1.0, 2.5, 3.0, 5.0]
ys = [1.0, 3.0, -2.0, 0.5, 4.0]
f = natural_cubic_spline(xs, ys)
assert callable(f), 'natural_cubic_spline did not return a callable'

for x, y in zip(xs, ys):
    assert abs(f(x) - y) < 1e-9, f'f({x}) is {f(x)}, not the knot value {y}'

# Natural end conditions.
start = limit(f, xs[0], xs[1], xs[0], 2)
end = limit(f, xs[-2], xs[-1], xs[-1], 2)
assert abs(start) < 1e-6, f'the second derivative at the left end is {start}'
assert abs(end) < 1e-6, f'the second derivative at the right end is {end}'

for k in range(1, len(xs) - 1):
    left1 = limit(f, xs[k - 1], xs[k], xs[k], 1)
    right1 = limit(f, xs[k], xs[k + 1], xs[k], 1)
    assert abs(left1 - right1) < 1e-6, (
        f'the slope jumps at x={xs[k]}: {left1} then {right1}')
    left2 = limit(f, xs[k - 1], xs[k], xs[k], 2)
    right2 = limit(f, xs[k], xs[k + 1], xs[k], 2)
    assert abs(left2 - right2) < 1e-6, (
        f'the curvature jumps at x={xs[k]}: {left2} then {right2}')

# Collinear knots have zero second derivative everywhere, so the natural
# spline is exactly the straight line through them.
line = natural_cubic_spline([0.0, 1.0, 3.0, 4.0], [1.0, 3.0, 7.0, 9.0])
for x in (0.25, 1.5, 2.0, 3.75):
    assert abs(line(x) - (1.0 + 2.0 * x)) < 1e-9, (
        f'the spline through collinear knots bends at {x}')

for outside in (-0.001, 5.001, -10.0, 12.0):
    try:
        f(outside)
    except ValueError:
        pass
    else:
        raise AssertionError(f'f evaluated outside its range at {outside}')

for bad_xs, bad_ys in (([0.0, 1.0], [1.0, 2.0]),
                       ([0.0, 1.0, 1.0, 2.0], [1.0, 2.0, 3.0, 4.0]),
                       ([0.0, 2.0, 1.0, 3.0], [1.0, 2.0, 3.0, 4.0]),
                       ([0.0, 1.0, 2.0], [1.0, 2.0])):
    try:
        natural_cubic_spline(bad_xs, bad_ys)
    except ValueError:
        pass
    else:
        raise AssertionError(f'xs={bad_xs} ys={bad_ys} was accepted')
""",
    ),
]
