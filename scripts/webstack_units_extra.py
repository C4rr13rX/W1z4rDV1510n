"""Additional web-stack units: the scaffolding a Django+Vue+three.js app needs.

Split from generate_webstack_corpus.py to keep each module readable. Same
authored-CC0 provenance; imported and emitted by the generator.
"""
from __future__ import annotations

DJANGO_EXTRA = [
    (
        "Write the Django settings additions needed to enable Django REST "
        "framework and CORS for a Vue frontend.",
        '''INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.staticfiles",
    "corsheaders",
    "rest_framework",
    "calculator",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
]

REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ],
    "DEFAULT_THROTTLE_CLASSES": [
        "rest_framework.throttling.AnonRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {
        "anon": "120/minute",
    },
}

CORS_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
''',
        "settings",
    ),
    (
        "Write a Django AppConfig class for a calculator application.",
        '''from django.apps import AppConfig


class CalculatorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "calculator"
    verbose_name = "Scientific calculator"

    def ready(self):
        from . import signals  # noqa: F401
''',
        "appconfig",
    ),
    (
        "Write Django tests for a calculator API endpoint that evaluates an "
        "expression.",
        '''import json

from django.test import TestCase
from django.urls import reverse


class EvaluateEndpointTests(TestCase):
    def post(self, payload):
        return self.client.post(
            reverse("calculator:evaluate"),
            data=json.dumps(payload),
            content_type="application/json",
        )

    def test_evaluates_arithmetic(self):
        response = self.post({"expression": "2 + 3 * 4"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["value"], 14)

    def test_substitutes_bindings(self):
        response = self.post({"expression": "x ** 2", "bindings": {"x": 5}})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["value"], 25)

    def test_rejects_unknown_name(self):
        response = self.post({"expression": "__import__('os').listdir('.')"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())

    def test_rejects_empty_expression(self):
        response = self.post({"expression": "   "})
        self.assertEqual(response.status_code, 400)
''',
        "tests",
    ),
]

VUE_EXTRA = [
    (
        "Write the Vue application entry point that mounts the root component.",
        '''import { createApp } from "vue";
import App from "./App.vue";
import { createPinia } from "pinia";
import "./assets/main.css";

const app = createApp(App);
app.use(createPinia());
app.config.errorHandler = (error, instance, info) => {
  console.error("unhandled vue error", error, info);
};
app.mount("#app");
''',
        "entry",
    ),
    (
        "Write a Pinia store that holds the current expression, its result and "
        "the sampled surface.",
        '''import { defineStore } from "pinia";
import { evaluateExpression, sampleSurface } from "../api/calculator";

export const useCalculatorStore = defineStore("calculator", {
  state: () => ({
    expression: "",
    value: null,
    grid: null,
    error: null,
    pending: false,
    history: [],
  }),
  getters: {
    hasSurface: (state) => state.grid !== null,
    lastEntries: (state) => state.history.slice(-10).reverse(),
  },
  actions: {
    async evaluate(expression, bindings = {}) {
      this.pending = true;
      this.error = null;
      try {
        const payload = await evaluateExpression(expression, bindings);
        this.expression = expression;
        this.value = payload.value;
        this.history.push({ expression, value: payload.value });
      } catch (error) {
        this.error = error.message;
        this.value = null;
      } finally {
        this.pending = false;
      }
    },
    async plot(expression, xRange, yRange, steps = 40) {
      this.pending = true;
      this.error = null;
      try {
        this.grid = await sampleSurface(expression, xRange, yRange, steps);
        this.expression = expression;
      } catch (error) {
        this.error = error.message;
        this.grid = null;
      } finally {
        this.pending = false;
      }
    },
    reset() {
      this.expression = "";
      this.value = null;
      this.grid = null;
      this.error = null;
    },
  },
});
''',
        "store",
    ),
    (
        "Write a Vue component with inputs for the x and y plot ranges and the "
        "sample resolution.",
        '''<template>
  <form class="plot-controls" @submit.prevent="emitPlot">
    <label>
      x from
      <input v-model.number="xMin" type="number" step="any" />
      to
      <input v-model.number="xMax" type="number" step="any" />
    </label>
    <label>
      y from
      <input v-model.number="yMin" type="number" step="any" />
      to
      <input v-model.number="yMax" type="number" step="any" />
    </label>
    <label>
      resolution
      <input v-model.number="steps" type="range" min="8" max="120" />
      <span>{{ steps }}</span>
    </label>
    <p v-if="invalid" class="error">Each minimum must be below its maximum.</p>
    <button type="submit" :disabled="invalid">Plot</button>
  </form>
</template>

<script>
export default {
  name: "PlotControls",
  data() {
    return { xMin: -5, xMax: 5, yMin: -5, yMax: 5, steps: 40 };
  },
  computed: {
    invalid() {
      return this.xMin >= this.xMax || this.yMin >= this.yMax;
    },
  },
  methods: {
    emitPlot() {
      if (this.invalid) {
        return;
      }
      this.$emit("plot", {
        xRange: [this.xMin, this.xMax],
        yRange: [this.yMin, this.yMax],
        steps: this.steps,
      });
    },
  },
};
</script>
''',
        "plot_controls",
    ),
    (
        "Write a Vue component that hosts a three.js canvas and redraws when "
        "the surface data changes.",
        '''<template>
  <div ref="host" class="viewport"></div>
</template>

<script>
import { SceneHost } from "../three/SceneHost";
import { buildSurfaceMesh } from "../three/buildSurfaceMesh";

export default {
  name: "SurfaceViewport",
  props: {
    grid: { type: Object, default: null },
  },
  mounted() {
    this.host = new SceneHost(this.$refs.host);
    this.host.start();
    this.draw();
  },
  beforeUnmount() {
    this.host.dispose();
  },
  watch: {
    grid: "draw",
  },
  methods: {
    draw() {
      if (this.mesh) {
        this.host.scene.remove(this.mesh);
        this.mesh.geometry.dispose();
        this.mesh.material.dispose();
        this.mesh = null;
      }
      if (!this.grid) {
        return;
      }
      this.mesh = buildSurfaceMesh(this.grid);
      this.host.scene.add(this.mesh);
    },
  },
};
</script>
''',
        "viewport",
    ),
]

THREEJS_EXTRA = [
    (
        "Write a JavaScript function that adds labelled x, y and z axes to a "
        "three.js scene.",
        '''import * as THREE from "three";

export function addAxes(scene, size = 5) {
  const group = new THREE.Group();
  group.add(new THREE.AxesHelper(size));

  const grid = new THREE.GridHelper(size * 2, 20, 0x445566, 0x223344);
  grid.rotation.x = Math.PI / 2;
  group.add(grid);

  const axes = [
    { axis: "x", position: [size, 0, 0], color: 0xff5555 },
    { axis: "y", position: [0, size, 0], color: 0x55ff55 },
    { axis: "z", position: [0, 0, size], color: 0x5599ff },
  ];
  for (const { axis, position, color } of axes) {
    const canvas = document.createElement("canvas");
    canvas.width = 64;
    canvas.height = 64;
    const context = canvas.getContext("2d");
    context.fillStyle = "#ffffff";
    context.font = "48px sans-serif";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText(axis, 32, 32);

    const texture = new THREE.CanvasTexture(canvas);
    const sprite = new THREE.Sprite(
      new THREE.SpriteMaterial({ map: texture, color }),
    );
    sprite.position.set(...position);
    sprite.scale.set(0.5, 0.5, 0.5);
    group.add(sprite);
  }

  scene.add(group);
  return group;
}
''',
        "axes",
    ),
    (
        "Write a JavaScript function that moves a three.js camera to frame an "
        "object completely.",
        '''import * as THREE from "three";

export function frameObject(camera, controls, object, padding = 1.25) {
  const box = new THREE.Box3().setFromObject(object);
  if (box.isEmpty()) {
    return;
  }

  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const extent = Math.max(size.x, size.y, size.z);

  const fov = THREE.MathUtils.degToRad(camera.fov);
  let distance = (extent / 2) / Math.tan(fov / 2);
  distance *= padding;

  const direction = camera.position
    .clone()
    .sub(controls.target)
    .normalize();
  if (direction.lengthSq() === 0) {
    direction.set(1, 1, 1).normalize();
  }

  camera.position.copy(center).addScaledVector(direction, distance);
  camera.near = distance / 100;
  camera.far = distance * 100;
  camera.updateProjectionMatrix();

  controls.target.copy(center);
  controls.update();
}
''',
        "frame",
    ),
]
