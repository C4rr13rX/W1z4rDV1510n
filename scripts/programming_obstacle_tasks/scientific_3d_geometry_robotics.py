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
]
