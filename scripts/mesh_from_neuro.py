#!/usr/bin/env python3
# coding: utf-8
"""
mesh_from_neuro.py -- Developer template: W1z4rD neural mesh synthesis.

Shows three patterns for pulling 3D geometry out of the neural fabric:

  Pattern A -- Query-driven mesh (text query activates neural memory)
  Pattern B -- Image-driven mesh (image activates visual memory -> mesh)
  Pattern C -- Raw world3d centroid cloud (bypass synthesis, use directly)

The node does the heavy lifting -- this script just calls the HTTP API and
saves the resulting OBJ + MTL files ready for Three.js or any 3D viewer.

Usage:
  python scripts/mesh_from_neuro.py --pattern A --query "cow grazing field"
  python scripts/mesh_from_neuro.py --pattern B --image path/to/image.jpg
  python scripts/mesh_from_neuro.py --pattern C --out world3d_cloud.json
  python scripts/mesh_from_neuro.py --pattern A --query "horse" --format json
"""

import argparse, base64, json, sys, urllib.request, urllib.error
from pathlib import Path

DEFAULT_NODE = "localhost:8090"


# -- HTTP helpers --------------------------------------------------------------

def _post(node: str, path: str, body: dict) -> dict:
    url = f"http://{node}{path}"
    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _get(node: str, path: str) -> dict:
    url = f"http://{node}{path}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


# -- Pattern A: text query -> mesh ----------------------------------------------

def query_mesh(
    node: str,
    query: str,
    *,
    hops: int = 2,
    min_activation: float = 0.05,
    categories: list[str] | None = None,
    fmt: str = "obj",
    out_dir: str = ".",
) -> dict:
    """
    POST /mesh/synthesize -- activate the fabric with `query`, synthesize mesh.

    Returns the API response dict; also saves .obj + .mtl if format='obj'.
    """
    body = {
        "query":          query,
        "hops":           hops,
        "min_activation": min_activation,
        "format":         fmt,
    }
    if categories:
        body["categories"] = categories

    print(f"[mesh] POST /mesh/synthesize -- query: {query!r}")
    result = _post(node, "/mesh/synthesize", body)

    if result.get("mesh") is None and "reason" in result:
        print(f"  [!] No mesh: {result['reason']}")
        return result

    if fmt == "obj" and "obj" in result:
        name = query.replace(" ", "_")[:40]
        _save_obj(result, out_dir, name)
    elif fmt == "json":
        print(f"  vertices: {result.get('vertex_count')}, faces: {result.get('face_count')}")

    return result


# -- Pattern B: image -> mesh ---------------------------------------------------

def image_mesh(
    node: str,
    image_path: str,
    *,
    hops: int = 2,
    min_activation: float = 0.05,
    categories: list[str] | None = None,
    out_dir: str = ".",
) -> dict:
    """
    POST /mesh/from_image -- encode image through ImageBitsEncoder,
    propagate labels through the fabric, synthesize mesh.

    The resulting mesh reflects the spatial structure of what the node
    'sees' in the image, shaped by its Hebbian training history.
    """
    img_bytes = Path(image_path).read_bytes()
    img_b64   = base64.b64encode(img_bytes).decode("ascii")

    body = {
        "image_b64":      img_b64,
        "hops":           hops,
        "min_activation": min_activation,
    }
    if categories:
        body["categories"] = categories

    print(f"[mesh] POST /mesh/from_image -- {Path(image_path).name}")
    result = _post(node, "/mesh/from_image", body)

    if result.get("mesh") is None and "reason" in result:
        print(f"  [!] No mesh: {result['reason']}")
        print(f"       image_labels={result.get('image_labels')}, activated={result.get('activated')}")
        return result

    print(f"  image_labels={result.get('image_labels')}, activated={result.get('activated')}")
    name = Path(image_path).stem
    _save_obj(result, out_dir, name)
    return result


# -- Pattern C: raw world3d centroid cloud -------------------------------------

def world3d_cloud(node: str, *, out: str = "world3d_cloud.json") -> dict:
    """
    GET /neuro/world3d -- dump all centroid positions + colours as JSON.

    Use this to drive a custom Three.js point cloud or as input to your
    own meshing algorithm without going through the node's synthesizer.

    Response shape:
      {
        "objects": [
          { "id": "cow_back",
            "position": {"x": 1.2, "y": -0.4, "z": 0.8},
            "category": "cow_body",
            "color_rgb": [0.6, 0.4, 0.2],
            "confidence": 0.85,
            "active": true,
            "visual_labels": ["img:h3", "img:h5"] },
          ...
        ],
        "total_centroids": 312,
        "active_count": 47
      }
    """
    print(f"[mesh] GET /neuro/world3d")
    result = _get(node, "/neuro/world3d")
    total  = result.get("total_centroids", 0)
    active = result.get("active_count", 0)
    print(f"  total_centroids={total}, active_count={active}")

    Path(out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"  saved -> {out}")
    return result


# -- Three.js loader snippet ---------------------------------------------------

THREEJS_SNIPPET = """
// -- Three.js integration (copy this into your Vue/React/vanilla component) --

import * as THREE from 'three';
import { OBJLoader }  from 'three/addons/loaders/OBJLoader.js';
import { MTLLoader }  from 'three/addons/loaders/MTLLoader.js';

const NODE_URL = 'http://localhost:8090';

/**
 * Fetch a neural mesh for `query` and add it to `scene`.
 * Returns the Three.js Object3D.
 */
async function addNeuroMesh(scene, query, { hops=2, minActivation=0.05, categories=[] } = {}) {
  const res = await fetch(`${NODE_URL}/mesh/synthesize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, hops, min_activation: minActivation, categories }),
  });
  const data = await res.json();
  if (!data.obj) { console.warn('[neuro mesh] no mesh returned:', data.reason); return null; }

  // Build blob URLs for the loaders
  const mtlBlob = URL.createObjectURL(new Blob([data.mtl], { type: 'text/plain' }));
  const objBlob = URL.createObjectURL(new Blob([data.obj], { type: 'text/plain' }));

  const materials = await new MTLLoader().loadAsync(mtlBlob);
  materials.preload();
  const mesh = await new OBJLoader().setMaterials(materials).loadAsync(objBlob);

  // Centre the mesh on the origin
  const box = new THREE.Box3().setFromObject(mesh);
  const center = box.getCenter(new THREE.Vector3());
  mesh.position.sub(center);

  scene.add(mesh);
  URL.revokeObjectURL(mtlBlob);
  URL.revokeObjectURL(objBlob);
  console.log(`[neuro mesh] loaded -- ${data.vertex_count} verts, ${data.face_count} faces`);
  return mesh;
}

/**
 * Render the raw centroid cloud as a Three.js Points object.
 * Lower overhead than full mesh -- good for exploration / debug.
 */
async function addCentroidCloud(scene, { minActivation=0.0 } = {}) {
  const res = await fetch(`${NODE_URL}/neuro/world3d`);
  const { objects } = await res.json();

  const positions = [], colors = [];
  for (const obj of objects) {
    if (obj.active || minActivation === 0.0) {
      positions.push(obj.position.x, obj.position.y, obj.position.z);
      const [r, g, b] = obj.color_rgb;
      colors.push(r, g, b);
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('color',    new THREE.Float32BufferAttribute(colors, 3));

  const mat   = new THREE.PointsMaterial({ size: 0.04, vertexColors: true });
  const cloud = new THREE.Points(geo, mat);
  scene.add(cloud);
  return cloud;
}

export { addNeuroMesh, addCentroidCloud };
"""


# -- Developer tips ------------------------------------------------------------

def print_tips(node: str):
    """Fetch /mesh/template and print developer guidance."""
    try:
        data = _get(node, "/mesh/template")
        print("\n-- /mesh/template ----------------------------------------------")
        for step in data.get("pipeline", []):
            print(f"  {step}")
        print()
        print("  Architecture note:")
        print(f"  {data.get('architecture_note', '')}")
        print("----------------------------------------------------------------\n")
    except Exception as e:
        print(f"  [WARN] Could not fetch /mesh/template: {e}")


# -- OBJ/MTL save helper -------------------------------------------------------

def _save_obj(result: dict, out_dir: str, name: str):
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    obj_path = d / f"{name}.obj"
    mtl_path = d / f"{name}.mtl"
    obj_path.write_text(result["obj"], encoding="utf-8")
    mtl_path.write_text(result.get("mtl", ""), encoding="utf-8")
    v = result.get("vertex_count", "?")
    f = result.get("face_count",   "?")
    print(f"  saved -> {obj_path}  ({v} verts, {f} faces)")
    print(f"  saved -> {mtl_path}")


# -- CLI -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="W1z4rD neural mesh synthesis -- developer template"
    )
    ap.add_argument("--node",           default=DEFAULT_NODE, help="node host:port")
    ap.add_argument("--pattern",        default="A", choices=["A", "B", "C", "tips"],
                    help="A=query mesh, B=image mesh, C=world3d dump, tips=print guide")
    # Pattern A
    ap.add_argument("--query",          default="cow grazing field", help="text query (pattern A)")
    ap.add_argument("--hops",           type=int,   default=2)
    ap.add_argument("--min-activation", type=float, default=0.05, dest="min_activation")
    ap.add_argument("--categories",     nargs="*",  default=[],
                    help="optional category filter: cow_body visual_zone env visual_feature other")
    ap.add_argument("--format",         default="obj", choices=["obj", "json"])
    # Pattern B
    ap.add_argument("--image",          default=None, help="image path (pattern B)")
    # Pattern C / output
    ap.add_argument("--out",            default=".", help="output directory (A/B) or JSON path (C)")

    args = ap.parse_args()

    print(f"\nW1z4rD Mesh Dev Template -- node: http://{args.node}\n")

    if args.pattern == "tips":
        print_tips(args.node)
        print(THREEJS_SNIPPET)
        return

    if args.pattern == "A":
        result = query_mesh(
            args.node, args.query,
            hops=args.hops,
            min_activation=args.min_activation,
            categories=args.categories or None,
            fmt=args.format,
            out_dir=args.out,
        )
        if args.format == "json" and result.get("verts"):
            print(json.dumps(result, indent=2)[:2000])

    elif args.pattern == "B":
        if not args.image:
            print("ERROR: --image is required for pattern B")
            sys.exit(1)
        image_mesh(
            args.node, args.image,
            hops=args.hops,
            min_activation=args.min_activation,
            categories=args.categories or None,
            out_dir=args.out,
        )

    elif args.pattern == "C":
        cloud_out = args.out if args.out != "." else "world3d_cloud.json"
        world3d_cloud(args.node, out=cloud_out)


if __name__ == "__main__":
    main()
