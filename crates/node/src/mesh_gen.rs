//! crates/node/src/mesh_gen.rs — Neural activation → 3D geometry translation.
//!
//! Pure geometry module — zero imports from the neural layer. Takes a
//! `Vec<MeshPoint>` (labelled xyz + activation + colour) and produces
//! triangulated, UV-mapped, OBJ-serialisable meshes.
//!
//! Pipeline:
//!   NeuroSnapshot centroids + propagation activations
//!       → (caller filters / projects to Vec<MeshPoint>)
//!       → 3-D convex hull  (incremental O(n²) — fine for sparse centroid clouds)
//!       → per-vertex spherical UV + averaged colour
//!       → OBJ + MTL text output
//!
//! Developer starting point: scripts/mesh_from_neuro.py

use std::fmt::Write as FmtWrite;

// ── Public types ──────────────────────────────────────────────────────────────

/// A labelled 3-D point produced from the neural activation state.
#[derive(Debug, Clone)]
pub struct MeshPoint {
    pub label:      String,
    pub x:          f32,
    pub y:          f32,
    pub z:          f32,
    /// Relative activation strength 0..1.
    pub activation: f32,
    /// Linear RGB 0..1, derived from Hebbian colour links.
    pub color:      [f32; 3],
}

/// Triangulated mesh — vertices, normals, UVs, faces, per-vertex colours.
#[derive(Debug, Default)]
pub struct TriMesh {
    pub verts:       Vec<[f32; 3]>,
    pub normals:     Vec<[f32; 3]>,
    pub uvs:         Vec<[f32; 2]>,
    /// Triangle faces — 0-based indices into `verts`.
    pub faces:       Vec<[u32; 3]>,
    /// Per-vertex colour (matches `verts` length).
    pub vert_colors: Vec<[f32; 3]>,
    pub name:        String,
}

impl TriMesh {
    /// Serialise to Wavefront OBJ text.  `mtl_name` adds `mtllib` + `usemtl`
    /// references — call `to_mtl()` separately and save as `{mtl_name}.mtl`.
    pub fn to_obj(&self, mtl_name: Option<&str>) -> String {
        let mut s = String::with_capacity(self.verts.len() * 80 + self.faces.len() * 32);
        let _ = writeln!(s, "# W1z4rD neural mesh — {}", self.name);
        if let Some(m) = mtl_name {
            let _ = writeln!(s, "mtllib {m}.mtl");
        }
        let _ = writeln!(s, "o {}", self.name);
        for &[x, y, z] in &self.verts {
            let _ = writeln!(s, "v {x:.6} {y:.6} {z:.6}");
        }
        for &[u, v] in &self.uvs {
            let _ = writeln!(s, "vt {u:.6} {v:.6}");
        }
        for &[nx, ny, nz] in &self.normals {
            let _ = writeln!(s, "vn {nx:.6} {ny:.6} {nz:.6}");
        }
        if mtl_name.is_some() {
            let _ = writeln!(s, "usemtl {}", self.name);
        }
        for &[a, b, c] in &self.faces {
            // OBJ indices are 1-based; v/vt/vn share the same index
            let (ai, bi, ci) = (a + 1, b + 1, c + 1);
            let _ = writeln!(s, "f {ai}/{ai}/{ai} {bi}/{bi}/{bi} {ci}/{ci}/{ci}");
        }
        s
    }

    /// Companion MTL file (averaged vertex colour as diffuse Kd).
    pub fn to_mtl(&self, name: &str) -> String {
        let mut s = String::new();
        let n = self.vert_colors.len().max(1) as f32;
        let (r, g, b) = self.vert_colors.iter()
            .fold((0f32, 0f32, 0f32), |(ar, ag, ab), c| (ar + c[0], ag + c[1], ab + c[2]));
        let _ = writeln!(s, "newmtl {name}");
        let _ = writeln!(s, "Ka 0.15 0.15 0.15");
        let _ = writeln!(s, "Kd {:.4} {:.4} {:.4}", r / n, g / n, b / n);
        let _ = writeln!(s, "Ks 0.05 0.05 0.05");
        let _ = writeln!(s, "Ns 12.0");
        let _ = writeln!(s, "d 1.0");
        s
    }

    /// Vertex count.
    pub fn vertex_count(&self) -> usize { self.verts.len() }

    /// Face (triangle) count.
    pub fn face_count(&self) -> usize { self.faces.len() }
}

// ── Synthesizer ───────────────────────────────────────────────────────────────

/// Converts neural activation point clouds into triangle meshes.
///
/// Instantiate once, call `synthesize()` per mesh needed.
/// Fields are public so callers can tune thresholds per-request.
#[derive(Debug, Clone)]
pub struct MeshSynthesizer {
    /// Points below this activation are excluded (0..1).
    pub min_activation: f32,
    /// Minimum point count after filtering before synthesis proceeds.
    pub min_points: usize,
    /// Uniform scale applied to centroid positions.
    pub scale: f32,
}

impl Default for MeshSynthesizer {
    fn default() -> Self {
        Self { min_activation: 0.05, min_points: 4, scale: 1.0 }
    }
}

impl MeshSynthesizer {
    pub fn new(min_activation: f32) -> Self {
        Self { min_activation, ..Default::default() }
    }

    /// Synthesize a `TriMesh` from the activation cloud.
    /// Returns `None` when the cloud is too sparse to mesh.
    pub fn synthesize(&self, name: &str, points: &[MeshPoint]) -> Option<TriMesh> {
        let filtered: Vec<&MeshPoint> = points.iter()
            .filter(|p| p.activation >= self.min_activation)
            .collect();

        if filtered.len() < self.min_points {
            return None;
        }

        let verts: Vec<[f32; 3]> = filtered.iter()
            .map(|p| [p.x * self.scale, p.y * self.scale, p.z * self.scale])
            .collect();
        let vert_colors: Vec<[f32; 3]> = filtered.iter()
            .map(|p| p.color)
            .collect();

        let center = centroid(&verts);

        let hull_faces = incremental_hull_3d(&verts);
        if hull_faces.is_empty() {
            return None;
        }

        let u32_faces: Vec<[u32; 3]> = hull_faces.iter()
            .map(|&[a, b, c]| [a as u32, b as u32, c as u32])
            .collect();

        let uvs: Vec<[f32; 2]> = verts.iter()
            .map(|&v| spherical_uv(v, center))
            .collect();

        let normals = vertex_normals(&verts, &u32_faces);

        Some(TriMesh { verts, normals, uvs, faces: u32_faces, vert_colors, name: name.to_string() })
    }

    /// Synthesize one mesh per category label found in `points`.
    /// Useful for building multi-object scenes (e.g. cow_body + env separately).
    pub fn synthesize_by_category(
        &self,
        points: &[MeshPoint],
        category_fn: impl Fn(&str) -> String,
    ) -> Vec<TriMesh> {
        use std::collections::HashMap;
        let mut by_cat: HashMap<String, Vec<&MeshPoint>> = HashMap::new();
        for p in points {
            by_cat.entry(category_fn(&p.label)).or_default().push(p);
        }
        let mut meshes = Vec::new();
        for (cat, pts) in by_cat {
            let owned: Vec<MeshPoint> = pts.iter().map(|&&ref p| p.clone()).collect();
            if let Some(m) = self.synthesize(&cat, &owned) {
                meshes.push(m);
            }
        }
        meshes
    }
}

// ── Vec3 helpers ──────────────────────────────────────────────────────────────

#[inline] fn v3_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0]-b[0], a[1]-b[1], a[2]-b[2]]
}
#[inline] fn v3_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0]+b[0], a[1]+b[1], a[2]+b[2]]
}
#[inline] fn v3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}
#[inline] fn v3_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}
#[inline] fn v3_len(a: [f32; 3]) -> f32 {
    (a[0]*a[0] + a[1]*a[1] + a[2]*a[2]).sqrt()
}
#[inline] fn v3_norm(a: [f32; 3]) -> [f32; 3] {
    let l = v3_len(a) + 1e-12; [a[0]/l, a[1]/l, a[2]/l]
}
#[inline] fn v3_neg(a: [f32; 3]) -> [f32; 3] { [-a[0], -a[1], -a[2]] }

fn centroid(pts: &[[f32; 3]]) -> [f32; 3] {
    let n = pts.len() as f32;
    let s = pts.iter().fold([0f32; 3], |acc, p| v3_add(acc, *p));
    [s[0]/n, s[1]/n, s[2]/n]
}

fn dist_to_line(p: [f32; 3], a: [f32; 3], b: [f32; 3]) -> f32 {
    let ab = v3_sub(b, a);
    let ap = v3_sub(p, a);
    v3_len(v3_cross(ap, ab)) / (v3_len(ab) + 1e-12)
}

// ── Spherical UV mapping ──────────────────────────────────────────────────────

/// Project a 3-D point onto a sphere surface UV in [0,1]×[0,1].
/// Uses standard longitude/latitude parameterisation.
fn spherical_uv(pos: [f32; 3], center: [f32; 3]) -> [f32; 2] {
    let d = v3_norm(v3_sub(pos, center));
    let u = 0.5 + d[0].atan2(d[2]) / (2.0 * std::f32::consts::PI);
    let v = 0.5 - d[1].clamp(-1.0, 1.0).asin() / std::f32::consts::PI;
    [u.clamp(0.0, 1.0), v.clamp(0.0, 1.0)]
}

// ── Per-vertex normals ────────────────────────────────────────────────────────

fn vertex_normals(verts: &[[f32; 3]], faces: &[[u32; 3]]) -> Vec<[f32; 3]> {
    let mut acc = vec![[0.0f32; 3]; verts.len()];
    for &[a, b, c] in faces {
        let (ai, bi, ci) = (a as usize, b as usize, c as usize);
        let n = v3_cross(v3_sub(verts[bi], verts[ai]), v3_sub(verts[ci], verts[ai]));
        acc[ai] = v3_add(acc[ai], n);
        acc[bi] = v3_add(acc[bi], n);
        acc[ci] = v3_add(acc[ci], n);
    }
    acc.iter().map(|&n| v3_norm(n)).collect()
}

// ── 3-D Convex Hull (incremental O(n²)) ──────────────────────────────────────
// Reference: "The Quickhull Algorithm for Convex Hulls", Barber et al. 1996
// Implementation: incremental "beneath-beyond" variant, O(n) faces × O(n) points.
//
// Invariant: all faces in `hull` have outward-pointing normals (away from the
// bounding centroid of the initial tetrahedron, which is always interior).

#[derive(Clone)]
struct HullFace {
    verts:  [usize; 3],
    normal: [f32; 3],   // outward unit normal
    d:      f32,        // plane offset: dot(normal, verts[0]) — for signed distance
}

impl HullFace {
    /// Build a face, flipping winding so the normal points away from `interior`.
    fn new(pts: &[[f32; 3]], a: usize, b: usize, c: usize, interior: [f32; 3]) -> Self {
        let ab  = v3_sub(pts[b], pts[a]);
        let ac  = v3_sub(pts[c], pts[a]);
        let mut n = v3_norm(v3_cross(ab, ac));
        // Flip if normal points toward interior
        if v3_dot(n, v3_sub(interior, pts[a])) > 0.0 {
            n = v3_neg(n);
        }
        let d = v3_dot(n, pts[a]);
        // Ensure vertices are stored with outward-consistent winding:
        // if we flipped the normal we must also flip the vertex order.
        let verts = if v3_dot(v3_cross(ab, ac), n) >= 0.0 {
            [a, b, c]
        } else {
            [a, c, b]
        };
        HullFace { verts, normal: n, d }
    }

    /// Signed distance from plane (positive = outside).
    #[inline]
    fn signed_dist(&self, p: [f32; 3]) -> f32 {
        v3_dot(self.normal, p) - self.d
    }

    /// Is point p strictly outside this face?
    #[inline]
    fn visible_from(&self, p: [f32; 3]) -> bool {
        self.signed_dist(p) > 1e-7
    }

    /// The three directed edges of this face (consistent with outward winding).
    fn edges(&self) -> [(usize, usize); 3] {
        let [a, b, c] = self.verts;
        [(a, b), (b, c), (c, a)]
    }
}

/// Incremental 3-D convex hull.
/// Returns triangulated hull faces as (0-based) index triples.
/// Falls back to a fan triangulation for degenerate / near-planar clouds.
pub fn incremental_hull_3d(pts: &[[f32; 3]]) -> Vec<[usize; 3]> {
    let n = pts.len();
    if n < 3 { return vec![]; }
    if n == 3 { return vec![[0, 1, 2]]; }

    // ── Seed tetrahedron ──────────────────────────────────────────────────────
    // p0: min-x extreme point
    let p0 = (0..n).min_by(|&a, &b| pts[a][0].partial_cmp(&pts[b][0]).unwrap()).unwrap();
    // p1: farthest from p0
    let p1 = (0..n).filter(|&i| i != p0)
        .max_by(|&a, &b| {
            v3_len(v3_sub(pts[a], pts[p0]))
                .partial_cmp(&v3_len(v3_sub(pts[b], pts[p0]))).unwrap()
        }).unwrap();
    // p2: farthest from line p0-p1
    let p2 = (0..n).filter(|&i| i != p0 && i != p1)
        .max_by(|&a, &b| {
            dist_to_line(pts[a], pts[p0], pts[p1])
                .partial_cmp(&dist_to_line(pts[b], pts[p0], pts[p1])).unwrap()
        }).unwrap();
    // p3: farthest from plane p0-p1-p2
    let plane_n = v3_cross(v3_sub(pts[p1], pts[p0]), v3_sub(pts[p2], pts[p0]));
    let p3 = match (0..n).filter(|&i| i != p0 && i != p1 && i != p2)
        .max_by(|&a, &b| {
            v3_dot(plane_n, v3_sub(pts[a], pts[p0])).abs()
                .partial_cmp(&v3_dot(plane_n, v3_sub(pts[b], pts[p0])).abs()).unwrap()
        }) {
        Some(i) => i,
        None => return fan_triangulate(n),
    };

    // Bail if nearly coplanar
    if v3_dot(plane_n, v3_sub(pts[p3], pts[p0])).abs() < 1e-8 {
        return fan_triangulate(n);
    }

    // Interior centroid of the seed tetrahedron (used to orient all face normals)
    let interior = centroid(&[pts[p0], pts[p1], pts[p2], pts[p3]]);
    let seed = [p0, p1, p2, p3];

    // Build 4 faces of the initial tetrahedron
    let mut hull: Vec<HullFace> = vec![
        HullFace::new(pts, p0, p1, p2, interior),
        HullFace::new(pts, p0, p1, p3, interior),
        HullFace::new(pts, p0, p2, p3, interior),
        HullFace::new(pts, p1, p2, p3, interior),
    ];

    // ── Incremental expansion ─────────────────────────────────────────────────
    for p in 0..n {
        if seed.contains(&p) { continue; }
        let pt = pts[p];

        // Find faces visible from p
        let visible: Vec<usize> = (0..hull.len())
            .filter(|&i| hull[i].visible_from(pt))
            .collect();
        if visible.is_empty() { continue; } // p is inside current hull

        // Horizon: directed edges shared by one visible and one non-visible face.
        // A directed edge (a,b) in a visible face has its reverse (b,a) in the
        // adjacent face.  If that adjacent face is NOT visible, (a,b) is on the horizon.
        let vis_set: std::collections::HashSet<usize> = visible.iter().cloned().collect();
        let mut horizon: Vec<(usize, usize)> = Vec::new();

        for &fi in &visible {
            for (a, b) in hull[fi].edges() {
                // Check if the reverse edge (b, a) belongs to a non-visible face
                let has_nonvis_neighbour = (0..hull.len())
                    .filter(|i| !vis_set.contains(i))
                    .any(|i| {
                        hull[i].edges().iter().any(|&(ea, eb)| ea == b && eb == a)
                    });
                if has_nonvis_neighbour {
                    horizon.push((a, b));
                }
            }
        }

        if horizon.is_empty() { continue; }

        // New faces: one per horizon edge, pointing to p
        let new_faces: Vec<HullFace> = horizon.iter()
            .map(|&(a, b)| HullFace::new(pts, a, b, p, interior))
            .collect();

        // Remove visible faces (reverse order to keep indices valid)
        let mut vis_sorted = visible.clone();
        vis_sorted.sort_unstable_by(|a, b| b.cmp(a));
        for i in vis_sorted {
            hull.swap_remove(i);
        }

        hull.extend(new_faces);
    }

    hull.iter().map(|f| f.verts).collect()
}

/// Fallback: fan triangulation from vertex 0 (used for coplanar/degenerate clouds).
fn fan_triangulate(n: usize) -> Vec<[usize; 3]> {
    (1..n.saturating_sub(1)).map(|i| [0usize, i, i + 1]).collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cube_pts() -> Vec<[f32; 3]> {
        vec![
            [0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[1.0,1.0,0.0],
            [0.0,0.0,1.0],[1.0,0.0,1.0],[0.0,1.0,1.0],[1.0,1.0,1.0],
        ]
    }

    #[test]
    fn cube_hull_has_12_triangles() {
        let pts = cube_pts();
        let faces = incremental_hull_3d(&pts);
        // A cube has 6 faces × 2 triangles each = 12
        assert_eq!(faces.len(), 12, "got {}", faces.len());
    }

    #[test]
    fn synthesizer_produces_valid_mesh() {
        let points: Vec<MeshPoint> = cube_pts().into_iter().enumerate()
            .map(|(i, [x, y, z])| MeshPoint {
                label: format!("test_{i}"),
                x, y, z,
                activation: 0.8,
                color: [0.5, 0.3, 0.1],
            })
            .collect();
        let mesh = MeshSynthesizer::default().synthesize("test_cube", &points).unwrap();
        assert_eq!(mesh.vertex_count(), 8);
        assert_eq!(mesh.face_count(), 12);
        let obj = mesh.to_obj(Some("test_cube"));
        assert!(obj.contains("o test_cube"));
        assert!(obj.contains("mtllib test_cube.mtl"));
    }

    #[test]
    fn obj_output_is_valid() {
        let points: Vec<MeshPoint> = cube_pts().into_iter().enumerate()
            .map(|(i, [x, y, z])| MeshPoint {
                label: format!("v{i}"), x, y, z, activation: 1.0, color: [1.0, 0.5, 0.0],
            })
            .collect();
        let mesh = MeshSynthesizer::default().synthesize("obj_test", &points).unwrap();
        let obj = mesh.to_obj(None);
        // All face indices must be in 1..=vertex_count
        for line in obj.lines().filter(|l| l.starts_with("f ")) {
            for part in line[2..].split_whitespace() {
                let idx: u32 = part.split('/').next().unwrap().parse().unwrap();
                assert!(idx >= 1 && idx <= mesh.vertex_count() as u32);
            }
        }
    }

    #[test]
    fn sparse_cloud_returns_none() {
        let points = vec![
            MeshPoint { label: "a".into(), x: 0.0, y: 0.0, z: 0.0, activation: 0.9, color: [1.0;3] },
            MeshPoint { label: "b".into(), x: 1.0, y: 0.0, z: 0.0, activation: 0.9, color: [1.0;3] },
        ];
        assert!(MeshSynthesizer::default().synthesize("two_pts", &points).is_none());
    }
}
