/**
 * microcortex — lightweight CPU-first neural fabric building block.
 *
 * Provides a compact feedforward perceptron and activation utilities used by
 * the textbook segmentation pipeline and any other JS/Node component that
 * needs fast, no-dependency inference on pre-trained weights.
 *
 * Design notes
 * ─────────────
 * • No GPU, no framework. Runs entirely on CPU with typed arrays (Float32Array)
 *   for cache-friendly arithmetic.
 * • The weight layout is row-major: hiddenWeights[i*inputSize + j] is the
 *   weight from input neuron j to hidden neuron i.
 * • Hidden layer uses ReLU; output layer is linear (caller applies softmax or
 *   sigmoid as needed for the task).
 * • buildPerceptron() is pure — it closes over the config and returns a
 *   stateless { forward } object so the same weights can be shared across many
 *   concurrent classify calls without mutation.
 */

// ─── Activation functions ────────────────────────────────────────────────────

/**
 * Rectified Linear Unit.  Clamped to [0, x] element-wise.
 * @param {number} x
 * @returns {number}
 */
function relu(x) {
  return x > 0 ? x : 0;
}

/**
 * Leaky ReLU — lets a small gradient through for x < 0, which keeps dormant
 * neurons from going permanently dark.  alpha defaults to 0.01.
 * @param {number} x
 * @param {number} [alpha=0.01]
 * @returns {number}
 */
function leakyRelu(x, alpha = 0.01) {
  return x >= 0 ? x : alpha * x;
}

/**
 * Sigmoid — useful for binary classification output layers.
 * @param {number} x
 * @returns {number}
 */
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Numerically stable softmax over a plain number array.
 *
 * Subtracts max before exponentiation to avoid overflow.
 * @param {number[]} logits
 * @returns {number[]}
 */
function softmax(logits) {
  if (!logits.length) return [];
  let max = logits[0];
  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > max) max = logits[i];
  }
  const exps = new Float64Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    exps[i] = Math.exp(logits[i] - max);
    sum += exps[i];
  }
  const probs = new Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    probs[i] = exps[i] / sum;
  }
  return probs;
}

// ─── Perceptron builder ───────────────────────────────────────────────────────

/**
 * Build a 2-layer (input → hidden → output) feedforward perceptron from
 * explicit weight arrays.  The returned object is stateless and immutable —
 * weights are captured at construction time and never mutated.
 *
 * Weight layout (row-major):
 *   hiddenWeights[i * inputSize + j]  = weight from input j  → hidden i
 *   outputWeights[i * hiddenSize + j] = weight from hidden j → output i
 *
 * @param {{
 *   inputSize:     number,
 *   hiddenSize:    number,
 *   outputSize:    number,
 *   hiddenWeights: number[] | Float32Array,
 *   hiddenBiases:  number[] | Float32Array,
 *   outputWeights: number[] | Float32Array,
 *   outputBiases:  number[] | Float32Array,
 *   hiddenActivation?: 'relu' | 'leaky_relu' | 'sigmoid',
 * }} config
 *
 * @returns {{ forward(input: Float32Array | number[]): Float32Array }}
 */
function buildPerceptron(config) {
  const {
    inputSize,
    hiddenSize,
    outputSize,
    hiddenActivation = 'relu',
  } = config;

  // Copy weights into typed arrays for speed.
  const hw = new Float32Array(config.hiddenWeights);
  const hb = new Float32Array(config.hiddenBiases);
  const ow = new Float32Array(config.outputWeights);
  const ob = new Float32Array(config.outputBiases);

  if (hw.length !== hiddenSize * inputSize) {
    throw new Error(
      `hiddenWeights length ${hw.length} must equal hiddenSize(${hiddenSize}) * inputSize(${inputSize}) = ${hiddenSize * inputSize}`
    );
  }
  if (hb.length !== hiddenSize) {
    throw new Error(`hiddenBiases length ${hb.length} must equal hiddenSize ${hiddenSize}`);
  }
  if (ow.length !== outputSize * hiddenSize) {
    throw new Error(
      `outputWeights length ${ow.length} must equal outputSize(${outputSize}) * hiddenSize(${hiddenSize}) = ${outputSize * hiddenSize}`
    );
  }
  if (ob.length !== outputSize) {
    throw new Error(`outputBiases length ${ob.length} must equal outputSize ${outputSize}`);
  }

  // Pre-allocate output buffers so forward() doesn't GC on every call.
  const hiddenBuf = new Float32Array(hiddenSize);
  const outputBuf = new Float32Array(outputSize);

  // Pick the hidden activation function once.
  let hiddenAct;
  switch (hiddenActivation) {
    case 'leaky_relu': hiddenAct = leakyRelu; break;
    case 'sigmoid':    hiddenAct = sigmoid;    break;
    default:           hiddenAct = relu;        break;
  }

  /**
   * Run a forward pass.  Returns a Float32Array of raw output logits.
   * The caller is responsible for applying softmax / argmax as needed.
   *
   * @param {Float32Array | number[]} input  — length must equal inputSize
   * @returns {Float32Array}
   */
  function forward(input) {
    // ── Hidden layer ────────────────────────────────────────────────────────
    for (let i = 0; i < hiddenSize; i++) {
      let acc = hb[i];
      const rowOffset = i * inputSize;
      for (let j = 0; j < inputSize; j++) {
        acc += hw[rowOffset + j] * input[j];
      }
      hiddenBuf[i] = hiddenAct(acc);
    }

    // ── Output layer (linear) ────────────────────────────────────────────────
    for (let i = 0; i < outputSize; i++) {
      let acc = ob[i];
      const rowOffset = i * hiddenSize;
      for (let j = 0; j < hiddenSize; j++) {
        acc += ow[rowOffset + j] * hiddenBuf[j];
      }
      outputBuf[i] = acc;
    }

    return outputBuf;
  }

  return { forward };
}

// ─── Hebbian weight updater ───────────────────────────────────────────────────

/**
 * Apply a single in-place Hebbian update to a flat weight matrix.
 *
 * Hebb's rule: Δw[i][j] ∝ pre[j] * post[i]
 *
 * This is the same rule the Rust NeuronPool uses (hebbian_lr × a_pre × a_post),
 * exposed here so JS components can perform associative learning on the same
 * principles without reaching across the FFI boundary.
 *
 * @param {Float32Array} weights    — flat [outputSize × inputSize] matrix (row-major, mutated in place)
 * @param {number}       inputSize
 * @param {Float32Array} pre        — pre-synaptic activations  [inputSize]
 * @param {Float32Array} post       — post-synaptic activations [outputSize]
 * @param {number}       lr         — learning rate (e.g. 0.01)
 * @param {number}       [maxW=10]  — weight clamp to prevent runaway growth
 */
function hebbianUpdate(weights, inputSize, pre, post, lr, maxW = 10) {
  const outputSize = post.length;
  for (let i = 0; i < outputSize; i++) {
    const rowOffset = i * inputSize;
    const postVal = post[i];
    if (postVal === 0) continue;
    for (let j = 0; j < inputSize; j++) {
      const delta = lr * pre[j] * postVal;
      const w = weights[rowOffset + j] + delta;
      weights[rowOffset + j] = w > maxW ? maxW : w < -maxW ? -maxW : w;
    }
  }
}

/**
 * Winner-take-all sparsification — zero out all but the top-k activations.
 * Keeps network activity sparse so that distinct concepts don't bleed into
 * each other (same mechanism as wta_k_per_zone in the Rust fabric).
 *
 * @param {Float32Array} activations — mutated in place
 * @param {number}       k
 */
function winnerTakeAll(activations, k) {
  if (k <= 0 || k >= activations.length) return;
  // Find the k-th largest threshold.
  const copy = Array.from(activations).sort((a, b) => b - a);
  const threshold = copy[k - 1];
  for (let i = 0; i < activations.length; i++) {
    if (activations[i] < threshold) activations[i] = 0;
  }
}

// ─── Exports ──────────────────────────────────────────────────────────────────

export {
  buildPerceptron,
  softmax,
  relu,
  leakyRelu,
  sigmoid,
  hebbianUpdate,
  winnerTakeAll,
};
