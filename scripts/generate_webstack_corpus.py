#!/usr/bin/env python3
"""Generate a web-stack corpus: Django, Vue and three.js single units.

The programming brain is trained on CodeSearchNet Python, Jupyter notebooks,
MathInstruct and MetaMathQA. Measured 2026-08-20, that leaves it unable to
build a Django + Vue + three.js application: of ten decomposed single-class
tasks, eight returned nothing and the two that answered returned unrelated
Python (a Postgres migration for "Django model", `def square(n)` for "Vue
component"). The words django, vue and three.js produce ZERO intent labels, so
the extractor does not recognise the domain at all.

Every record here is authored for this corpus and published under CC0-1.0, so
the corpus carries no third-party licence obligations. Nothing is scraped.

Each record is ONE unit -- a class, a component, a module -- because that is
the granularity the agent decomposes into and the granularity the brain
composes from. Records follow the existing corpus schema exactly:
prompt, response, ctx{lang,intent,source}, license, source, source_hash,
script_id.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from webstack_units_extra import DJANGO_EXTRA, THREEJS_EXTRA, VUE_EXTRA

LICENSE = "CC0-1.0"
SCRIPT_ID = "programming_webstack_001"


def record(
    prompt: str,
    response: str,
    lang: str,
    intent: str,
    unit: str,
    variant: int | None = None,
) -> dict:
    """One training episode, hashed on its own content for provenance.

    Mirrors the CodeSearchNet convention: paraphrase rows carry a
    ``|gen=paraphrase:N`` suffix on ``source`` and hash over prompt+response so
    every row has a distinct ``source_hash`` (hashing the response alone would
    collide across all phrasings of a unit).
    """
    text = prompt.strip()
    body = response.strip() + "\n"
    origin = f"authored:webstack:{lang}:{unit}"
    if variant is not None:
        origin = f"{origin}|gen=paraphrase:{variant}"
    digest = hashlib.sha256(f"{text}\n{body}".encode("utf-8")).hexdigest()
    return {
        "prompt": text,
        "response": body,
        "ctx": {"lang": lang, "intent": intent, "source": "authored"},
        "license": LICENSE,
        "source": origin,
        "source_hash": digest,
        "script_id": SCRIPT_ID,
    }


DJANGO = [
    (
        "Write a Python Django model class storing a saved calculator "
        "expression with its variable bindings.",
        '''from django.db import models


class SavedExpression(models.Model):
    """A calculator expression and the variable bindings it was evaluated with."""

    name = models.CharField(max_length=120, unique=True)
    expression = models.TextField()
    bindings = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.name
''',
        "model",
    ),
    (
        "Write a Python Django REST serializer class for a calculator "
        "expression model.",
        '''from rest_framework import serializers

from .models import SavedExpression


class SavedExpressionSerializer(serializers.ModelSerializer):
    """Validate and represent a saved calculator expression."""

    class Meta:
        model = SavedExpression
        fields = ["id", "name", "expression", "bindings", "created_at"]
        read_only_fields = ["id", "created_at"]

    def validate_expression(self, value):
        if not value.strip():
            raise serializers.ValidationError("expression must not be empty")
        return value

    def validate_bindings(self, value):
        if not isinstance(value, dict):
            raise serializers.ValidationError("bindings must be an object")
        for name, bound in value.items():
            if not isinstance(bound, (int, float)):
                raise serializers.ValidationError(
                    f"binding {name} must be a number"
                )
        return value
''',
        "serializer",
    ),
    (
        "Write a Python Django REST view class that evaluates a submitted math "
        "expression and returns the result as JSON.",
        '''from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .evaluator import ExpressionError, evaluate_expression


class EvaluateExpressionView(APIView):
    """Evaluate a posted expression and return its numeric result."""

    def post(self, request):
        expression = request.data.get("expression", "")
        bindings = request.data.get("bindings", {})
        if not isinstance(expression, str) or not expression.strip():
            return Response(
                {"error": "expression is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if not isinstance(bindings, dict):
            return Response(
                {"error": "bindings must be an object"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            value = evaluate_expression(expression, bindings)
        except ExpressionError as error:
            return Response(
                {"error": str(error)}, status=status.HTTP_400_BAD_REQUEST
            )
        return Response({"expression": expression, "result": value})
''',
        "view",
    ),
    (
        "Write a Python Django urls module routing a calculator evaluate "
        "endpoint to its view.",
        '''from django.urls import path

from .views import EvaluateExpressionView, SurfaceSampleView

app_name = "calculator"

urlpatterns = [
    path("evaluate/", EvaluateExpressionView.as_view(), name="evaluate"),
    path("surface/", SurfaceSampleView.as_view(), name="surface"),
]
''',
        "urls",
    ),
    (
        "Write a Python Django REST view class that samples a two-variable "
        "expression over a grid and returns the surface points as JSON.",
        '''from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .evaluator import ExpressionError
from .sampling import sample_surface


class SurfaceSampleView(APIView):
    """Return a grid of z values for plotting a two-variable expression."""

    def post(self, request):
        expression = request.data.get("expression", "")
        steps = request.data.get("steps", 40)
        try:
            x_range = tuple(request.data["x_range"])
            y_range = tuple(request.data["y_range"])
        except (KeyError, TypeError, ValueError):
            return Response(
                {"error": "x_range and y_range are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            grid = sample_surface(expression, x_range, y_range, int(steps))
        except (ExpressionError, ValueError) as error:
            return Response(
                {"error": str(error)}, status=status.HTTP_400_BAD_REQUEST
            )
        return Response(grid)
''',
        "surface_view",
    ),
]


PYTHON_CORE = [
    (
        "Write a Python function that safely evaluates a math expression using "
        "the ast module without eval.",
        '''import ast
import math
import operator


class ExpressionError(ValueError):
    """The expression is not a safe, evaluable arithmetic expression."""


_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
    "abs": abs,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}


def evaluate_expression(expression, bindings=None):
    """Evaluate an arithmetic expression without executing arbitrary code.

    Only literals, named bindings, the operators in _BINARY/_UNARY and the
    whitelisted functions are reachable. Anything else raises ExpressionError,
    so a hostile expression cannot call into the interpreter.
    """
    bindings = dict(bindings or {})
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ExpressionError(f"could not parse expression: {error}") from error

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ExpressionError("only numeric literals are allowed")
        if isinstance(node, ast.Name):
            if node.id in bindings:
                return bindings[node.id]
            if node.id in _CONSTANTS:
                return _CONSTANTS[node.id]
            raise ExpressionError(f"unknown name: {node.id}")
        if isinstance(node, ast.BinOp):
            handler = _BINARY.get(type(node.op))
            if handler is None:
                raise ExpressionError("unsupported operator")
            return handler(visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp):
            handler = _UNARY.get(type(node.op))
            if handler is None:
                raise ExpressionError("unsupported unary operator")
            return handler(visit(node.operand))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ExpressionError("only named functions are allowed")
            handler = _FUNCTIONS.get(node.func.id)
            if handler is None:
                raise ExpressionError(f"unknown function: {node.func.id}")
            if node.keywords:
                raise ExpressionError("keyword arguments are not allowed")
            return handler(*[visit(argument) for argument in node.args])
        raise ExpressionError("unsupported expression")

    try:
        return visit(tree)
    except ZeroDivisionError as error:
        raise ExpressionError("division by zero") from error
''',
        "evaluator",
    ),
    (
        "Write a Python function that samples a two-variable math expression "
        "over a grid and returns z values.",
        '''from .evaluator import evaluate_expression


def sample_surface(expression, x_range, y_range, steps=40):
    """Evaluate expression over a steps x steps grid of (x, y).

    Returns the axes alongside the z rows so a client can build a surface
    without recomputing the sample points. A point that cannot be evaluated
    becomes None rather than aborting the whole grid: a surface with a
    singularity is still worth plotting.
    """
    if steps < 2:
        raise ValueError("steps must be at least 2")
    x_min, x_max = float(x_range[0]), float(x_range[1])
    y_min, y_max = float(y_range[0]), float(y_range[1])
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("range minimum must be below its maximum")

    x_step = (x_max - x_min) / (steps - 1)
    y_step = (y_max - y_min) / (steps - 1)
    xs = [x_min + index * x_step for index in range(steps)]
    ys = [y_min + index * y_step for index in range(steps)]

    rows = []
    for y in ys:
        row = []
        for x in xs:
            try:
                value = evaluate_expression(expression, {"x": x, "y": y})
            except Exception:
                value = None
            row.append(value if value is None else float(value))
        rows.append(row)
    return {"x": xs, "y": ys, "z": rows}
''',
        "sampling",
    ),
]


VUE = [
    (
        "Write a Vue single file component with a calculator keypad that emits "
        "the typed expression.",
        '''<template>
  <div class="keypad">
    <input v-model="expression" class="display" aria-label="expression" />
    <div class="keys">
      <button v-for="key in keys" :key="key" @click="press(key)">
        {{ key }}
      </button>
      <button class="wide" @click="clear">C</button>
      <button class="wide" @click="submit">=</button>
    </div>
  </div>
</template>

<script>
export default {
  name: "CalculatorKeypad",
  data() {
    return {
      expression: "",
      keys: [
        "7", "8", "9", "/",
        "4", "5", "6", "*",
        "1", "2", "3", "-",
        "0", ".", "x", "+",
        "(", ")", "y", "^",
      ],
    };
  },
  methods: {
    press(key) {
      this.expression += key;
    },
    clear() {
      this.expression = "";
      this.$emit("cleared");
    },
    submit() {
      const trimmed = this.expression.trim();
      if (trimmed.length > 0) {
        this.$emit("evaluate", trimmed);
      }
    },
  },
};
</script>
''',
        "keypad",
    ),
    (
        "Write a JavaScript module that posts a math expression to a REST "
        "endpoint and returns the parsed JSON result.",
        '''const DEFAULT_ENDPOINT = "/api/calculator/evaluate/";

export async function evaluateExpression(expression, bindings = {}, options = {}) {
  const endpoint = options.endpoint || DEFAULT_ENDPOINT;
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ expression, bindings }),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `request failed: ${response.status}`);
  }
  return payload;
}

export async function sampleSurface(expression, xRange, yRange, steps = 40) {
  const response = await fetch("/api/calculator/surface/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      expression,
      x_range: xRange,
      y_range: yRange,
      steps,
    }),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `request failed: ${response.status}`);
  }
  return payload;
}
''',
        "api_client",
    ),
]

THREEJS = [
    (
        "Write a JavaScript class that creates a three.js scene with a "
        "perspective camera and orbit controls.",
        '''import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

export class SceneHost {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x101317);

    const { clientWidth: width, clientHeight: height } = container;
    this.camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    this.camera.position.set(6, 6, 6);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const key = new THREE.DirectionalLight(0xffffff, 0.8);
    key.position.set(5, 10, 7);
    this.scene.add(key);

    this.onResize = this.onResize.bind(this);
    window.addEventListener("resize", this.onResize);
  }

  onResize() {
    const { clientWidth: width, clientHeight: height } = this.container;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  start() {
    const tick = () => {
      this.frame = requestAnimationFrame(tick);
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    };
    tick();
  }

  dispose() {
    cancelAnimationFrame(this.frame);
    window.removeEventListener("resize", this.onResize);
    this.controls.dispose();
    this.renderer.dispose();
  }
}
''',
        "scene",
    ),
    (
        "Write a JavaScript function that builds a three.js surface mesh from a "
        "grid of z values.",
        '''import * as THREE from "three";

export function buildSurfaceMesh(grid, options = {}) {
  const { x: xs, y: ys, z: rows } = grid;
  const geometry = new THREE.PlaneGeometry(
    1,
    1,
    xs.length - 1,
    ys.length - 1,
  );

  const position = geometry.attributes.position;
  let finite = [];
  for (let row = 0; row < ys.length; row += 1) {
    for (let column = 0; column < xs.length; column += 1) {
      const index = row * xs.length + column;
      const value = rows[row][column];
      const height = Number.isFinite(value) ? value : 0;
      position.setX(index, xs[column]);
      position.setY(index, ys[row]);
      position.setZ(index, height);
      if (Number.isFinite(value)) {
        finite.push(value);
      }
    }
  }
  position.needsUpdate = true;
  geometry.computeVertexNormals();

  const low = finite.length ? Math.min(...finite) : 0;
  const high = finite.length ? Math.max(...finite) : 1;
  const colors = new Float32Array(position.count * 3);
  const color = new THREE.Color();
  for (let index = 0; index < position.count; index += 1) {
    const span = high - low || 1;
    const ratio = (position.getZ(index) - low) / span;
    color.setHSL(0.65 - 0.65 * ratio, 0.7, 0.5);
    colors[index * 3] = color.r;
    colors[index * 3 + 1] = color.g;
    colors[index * 3 + 2] = color.b;
  }
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

  const material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    side: THREE.DoubleSide,
    wireframe: Boolean(options.wireframe),
    flatShading: false,
  });
  return new THREE.Mesh(geometry, material);
}
''',
        "surface",
    ),
]


#: Each unit is presented several ways. The suites score PARAPHRASE recall, and
#: measured 2026-08-20 the brain reaches a manifest only when the request shares
#: vocabulary with it, so a single phrasing per unit teaches a single phrasing.
PARAPHRASES = {
    "model": [
        "Create a Django model for storing calculator expressions and their variables.",
        "Define a Django ORM class that saves an expression with named variable values.",
    ],
    "serializer": [
        "Create a Django REST framework serializer validating a calculator expression.",
        "Write a DRF ModelSerializer that checks an expression is present and its bindings are numbers.",
    ],
    "view": [
        "Write a Django REST APIView that computes a posted expression and answers with JSON.",
        "Create a Django endpoint class that takes an expression plus bindings and returns the calculated value.",
    ],
    "urls": [
        "Write the Django URL configuration wiring calculator endpoints to their views.",
        "Define urlpatterns for a Django calculator app exposing evaluate and surface routes.",
    ],
    "surface_view": [
        "Write a Django REST view returning a grid of z values for a two-variable expression.",
        "Create a Django endpoint that samples an expression across x and y ranges for 3D plotting.",
    ],
    "evaluator": [
        "Write a Python expression evaluator using ast that refuses arbitrary code.",
        "Implement safe arithmetic evaluation in Python with an allow-list of functions, no eval.",
    ],
    "sampling": [
        "Write a Python routine that evaluates an expression across a 2D grid for surface plotting.",
        "Implement grid sampling of a two-variable function returning x, y axes and z rows.",
    ],
    "keypad": [
        "Create a Vue component rendering calculator buttons that emits the entered expression.",
        "Write a Vue SFC for a calculator keypad with a display and an equals button.",
    ],
    "api_client": [
        "Write a JavaScript client that sends an expression to the calculator API and parses the reply.",
        "Create a fetch-based JavaScript module for evaluating expressions and sampling surfaces.",
    ],
    "scene": [
        "Write a three.js class setting up a scene, perspective camera, renderer and orbit controls.",
        "Create a JavaScript 3D viewport class using three.js with damped orbit controls and lighting.",
    ],
    "surface": [
        "Write a three.js helper that turns a z-value grid into a coloured surface mesh.",
        "Create a JavaScript function building a three.js mesh from sampled heights with vertex colours.",
    ],
    "settings": [
        "Configure Django settings for a REST API consumed by a Vue development server.",
        "Show the INSTALLED_APPS, middleware and CORS settings a Django plus Vue project needs.",
    ],
    "appconfig": [
        "Write the apps.py AppConfig for a Django calculator app.",
        "Define a Django application configuration class with a ready hook.",
    ],
    "tests": [
        "Write Django TestCase tests covering a calculator evaluate endpoint.",
        "Write tests for a Django JSON API that computes expressions and rejects unsafe input.",
    ],
    "entry": [
        "Write main.js for a Vue 3 application that mounts the root component.",
        "Create the Vue bootstrap file that installs Pinia and mounts the app.",
    ],
    "store": [
        "Write a Vue state store tracking the expression, computed value and surface grid.",
        "Create a Pinia store with actions that call the calculator API and record history.",
    ],
    "plot_controls": [
        "Write a Vue form component for choosing plot ranges and resolution.",
        "Create a Vue component that emits x range, y range and step count for plotting.",
    ],
    "viewport": [
        "Write a Vue component wrapping a three.js canvas that rerenders on data change.",
        "Create a Vue wrapper component that mounts a three.js scene and disposes it on unmount.",
    ],
    "axes": [
        "Write a three.js helper drawing labelled coordinate axes and a grid.",
        "Create a JavaScript function adding x y z axis labels to a three.js scene.",
    ],
    "frame": [
        "Write a three.js function that fits the camera to an object's bounding box.",
        "Create a JavaScript helper repositioning a camera so a mesh fills the view.",
    ],
}


def build_records() -> list[dict]:
    """Every unit, with each of its phrasings, as its own episode."""
    groups = (
        (DJANGO, "python", "implement"),
        (DJANGO_EXTRA, "python", "implement"),
        (PYTHON_CORE, "python", "implement"),
        (VUE, "javascript", "implement"),
        (VUE_EXTRA, "javascript", "implement"),
        (THREEJS, "javascript", "implement"),
        (THREEJS_EXTRA, "javascript", "implement"),
    )
    records: list[dict] = []
    for units, lang, intent in groups:
        for prompt, response, unit in units:
            records.append(record(prompt, response, lang, intent, unit))
            for index, alternate in enumerate(PARAPHRASES.get(unit, [])):
                records.append(
                    record(alternate, response, lang, intent, unit, variant=index)
                )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="emit each episode this many times; the trainer reinforces by "
             "presentation count, so a small corpus needs repetition",
    )
    args = parser.parse_args()

    records = build_records()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for _ in range(max(1, args.repeats)):
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    units = {row["source"].split("|", 1)[0] for row in records}
    print(
        f"wrote {len(records) * max(1, args.repeats)} rows "
        f"({len(records)} unique, {len(units)} units) to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
