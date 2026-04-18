#!/usr/bin/env python3
"""build_math_corpus.py — Stage 34: Comprehensive Mathematics Corpus

Trains the neural model on mathematics from arithmetic through advanced
topology, with deep emphasis on mathematical notation, symbolism, and
how symbols relate to real-world quantities and physical meaning.

Sources:
  - Curated mathematical symbol dictionary (70+ symbols) — hardcoded
  - Mathematical language and expression-reading guide — hardcoded
  - Wikipedia mathematics articles (~190 topics, all branches)

Stage 34: Complete mathematics — all branches, notation mastery

Usage:
  python scripts/build_math_corpus.py --stages 34 --node localhost:8090
"""

import argparse
import json
import re
import time
import urllib.parse
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Mathematical Symbol Dictionary
# Each entry trains: name, meaning, how to read, real-world interpretation,
# dimensional analysis, context-dependent meanings, and common confusions.
# ---------------------------------------------------------------------------

MATH_SYMBOLS = [
    {
        'symbol': '+', 'latex': '+', 'name': 'Plus / Addition',
        'domains': ['Arithmetic', 'Algebra', 'All mathematics'],
        'meaning': 'Binary operation that combines two quantities into their sum. Commutative and associative.',
        'how_to_read': '"a plus b" | "a added to b" | "the sum of a and b"',
        'real_world': 'Combining quantities of the same type: 3 kg + 5 kg = 8 kg. Superposition of forces: F_total = F₁ + F₂. Combining probabilities of mutually exclusive events: P(A∪B) = P(A) + P(B). Series resistors: R = R₁ + R₂.',
        'dimensional_note': 'Both terms MUST have the same units. You cannot add 3 meters + 5 seconds — dimensional homogeneity is required.',
        'context_note': 'In vector spaces: component-wise addition. In group theory: the group operation for abelian groups. In type theory: disjoint union.',
        'example': 'KE + PE = E_total → (½mv²) + (mgh) = constant [all Joules]',
    },
    {
        'symbol': '−', 'latex': '-', 'name': 'Minus / Subtraction',
        'domains': ['Arithmetic', 'Algebra', 'All mathematics'],
        'meaning': 'Binary: difference between two quantities. Unary: negation/opposite.',
        'how_to_read': '"a minus b" | "a subtract b" | "negative a" (unary)',
        'real_world': 'Change in quantity: Δx = x_final − x_initial. Net profit = revenue − cost. Temperature drop = T₁ − T₂. Displacement (can be negative, showing direction).',
        'context_note': 'As unary operator: −x means the additive inverse of x. In set theory: A − B or A \\ B means set difference (elements in A not in B).',
    },
    {
        'symbol': '×', 'latex': r'\times', 'name': 'Multiplication / Cross',
        'domains': ['Arithmetic', 'Algebra', 'Vector Calculus'],
        'meaning': 'Binary: product of two quantities. In vectors: cross product producing a perpendicular vector.',
        'how_to_read': '"a times b" | "a cross b" (vectors) | "a by b"',
        'real_world': 'Scaling: 3 m × 4 m = 12 m² (area). Rate × time = distance. In physics: torque τ = r × F (cross product, units N·m). Power = V × I (Watts).',
        'dimensional_note': 'Unlike addition, multiplication COMBINES units: [m/s] × [s] = [m]. The result has the product of the input units.',
        'context_note': 'In vector calculus: a × b produces a vector perpendicular to both a and b with magnitude |a||b|sin(θ). In set theory: A × B is the Cartesian product (set of ordered pairs).',
        'common_confusion': '× (cross product) vs · (dot product): cross product gives a vector, dot product gives a scalar.',
    },
    {
        'symbol': '÷ or /', 'latex': r'\div', 'name': 'Division',
        'domains': ['Arithmetic', 'Algebra'],
        'meaning': 'Inverse of multiplication; how many times one quantity fits into another; ratio.',
        'how_to_read': '"a divided by b" | "a over b" | "the ratio of a to b"',
        'real_world': 'Rate calculation: 120 km ÷ 2 h = 60 km/h. Density = mass ÷ volume (kg/m³). Concentration = moles ÷ liters (mol/L). Efficiency = output ÷ input.',
        'dimensional_note': 'Division divides units: [kg] ÷ [m³] = [kg/m³]. This is how derived units are formed.',
    },
    {
        'symbol': '=', 'latex': '=', 'name': 'Equals',
        'domains': ['All mathematics'],
        'meaning': 'Two expressions represent the same mathematical object or quantity.',
        'how_to_read': '"equals" | "is equal to" | "is"',
        'real_world': 'E = mc² states energy and mass×c² are the same quantity measured in different ways. An equation is a constraint — it restricts what values variables can take.',
        'context_note': 'In programming: = is assignment (left gets value of right). In math: = asserts equality. In logic: ≡ means logical equivalence.',
        'common_confusion': '= (equals) vs ≡ (identically equal/congruent) vs := (defined as). x² − 1 = 0 is an equation (true for specific x). (x+1)(x−1) ≡ x² − 1 is an identity (true for all x).',
    },
    {
        'symbol': '≠', 'latex': r'\neq', 'name': 'Not Equal',
        'domains': ['All mathematics'],
        'meaning': 'Two expressions are not equal; they represent different values.',
        'how_to_read': '"not equal to" | "is not"',
        'real_world': 'Used in constraints: x ≠ 0 means "x cannot be zero" (e.g., in a denominator). In proofs: establishing that two objects differ.',
    },
    {
        'symbol': '<, >, ≤, ≥', 'latex': r'<, >, \leq, \geq', 'name': 'Inequality Relations',
        'domains': ['Arithmetic', 'Algebra', 'Analysis', 'Optimization'],
        'meaning': 'Ordering relationship between quantities. < strictly less, ≤ less than or equal.',
        'how_to_read': '"a is less than b" | "a is at most b" (≤) | "a does not exceed b" (≤)',
        'real_world': 'Physical constraints: temperature T ≥ 0 K (absolute zero). Engineering tolerances: |x − target| ≤ 0.001 mm. Signal-to-noise: SNR ≥ 20 dB. Optimization: minimize cost subject to x ≥ 0.',
        'dimensional_note': 'Inequalities require same units on both sides, just like equalities.',
    },
    {
        'symbol': '≈', 'latex': r'\approx', 'name': 'Approximately Equal',
        'domains': ['Numerical Analysis', 'Physics', 'Engineering'],
        'meaning': 'Values are close but not exactly equal; used for estimates and approximations.',
        'how_to_read': '"approximately equal to" | "roughly" | "about"',
        'real_world': 'π ≈ 3.14159. g ≈ 9.8 m/s². For small angles: sin(θ) ≈ θ (in radians) when θ < 0.1 rad. Taylor series truncation: e^x ≈ 1 + x for small x.',
        'context_note': 'Related: ∼ (asymptotically equal, same rate of growth), ∝ (proportional to).',
    },
    {
        'symbol': '∝', 'latex': r'\propto', 'name': 'Proportional To',
        'domains': ['Physics', 'Algebra', 'Science'],
        'meaning': 'One quantity scales as a constant multiple of another.',
        'how_to_read': '"is proportional to" | "varies as"',
        'real_world': 'F ∝ ma (Newton: force proportional to mass × acceleration). Ohm\'s law: V ∝ I (voltage proportional to current, constant = resistance). Gravitational force ∝ 1/r² (inverse square law). When you say "y ∝ x", you mean y = kx for some constant k.',
        'context_note': 'To find k: measure y for a known x, then k = y/x. The proportionality constant often has deep physical meaning (e.g., k = R in V = IR).',
    },
    {
        'symbol': '²  ³  ⁿ or x^n', 'latex': 'x^n', 'name': 'Exponentiation / Power',
        'domains': ['Algebra', 'Calculus', 'All mathematics'],
        'meaning': 'x^n means x multiplied by itself n times (for positive integer n). Extended to all real and complex n.',
        'how_to_read': '"x to the power of n" | "x squared" (n=2) | "x cubed" (n=3)',
        'real_world': 'Area = L² (square of length). Volume = L³. Energy in springs: E = ½kx². Compound interest: A = P(1+r)^n. Population growth: N(t) = N₀ × 2^(t/T).',
        'dimensional_note': '[x^n] = [x]^n. If x is in meters, x³ is in cubic meters (m³).',
        'context_note': 'Negative exponent: x^(−n) = 1/x^n. Fractional exponent: x^(1/n) = ⁿ√x. Zero: x^0 = 1 (for x≠0). In quantum mechanics: |ψ|² = probability density (complex squaring).',
    },
    {
        'symbol': '√ and ⁿ√', 'latex': r'\sqrt{x}, \sqrt[n]{x}', 'name': 'Square Root / nth Root',
        'domains': ['Arithmetic', 'Algebra', 'Complex Analysis'],
        'meaning': '√x is the non-negative number whose square is x. ⁿ√x = x^(1/n).',
        'how_to_read': '"square root of x" | "radical x" | "nth root of x"',
        'real_world': 'RMS voltage: V_rms = √(V₁² + V₂²)/√2. Standard deviation: σ = √(variance). Speed from kinetic energy: v = √(2KE/m). Hypotenuse: c = √(a² + b²).',
        'common_confusion': '√x² = |x|, not x (because √ always returns non-negative). For x = −3: √(x²) = √9 = 3 = |−3|.',
    },
    {
        'symbol': '|x|', 'latex': '|x|', 'name': 'Absolute Value / Modulus / Determinant',
        'domains': ['Algebra', 'Analysis', 'Complex Analysis', 'Linear Algebra'],
        'meaning': 'Distance from zero on the number line; always non-negative.',
        'how_to_read': '"absolute value of x" | "modulus of x" | "determinant of matrix A" (|A|)',
        'real_world': '|x − target| < ε means x is within ε of the target. |v| = speed (magnitude of velocity vector). |z| for complex z = a+bi: modulus = √(a²+b²) = distance from origin.',
        'context_note': 'For matrices: |A| = det(A) = scalar. For vectors: |v| = ‖v‖ = length. For complex: |z| = √(a²+b²). For real: |x| = x if x≥0, −x if x<0.',
    },
    {
        'symbol': '!', 'latex': 'n!', 'name': 'Factorial',
        'domains': ['Combinatorics', 'Analysis', 'Number Theory'],
        'meaning': 'n! = n × (n−1) × (n−2) × ⋯ × 2 × 1. Counts the number of ways to arrange n distinct objects.',
        'how_to_read': '"n factorial"',
        'real_world': '5! = 120 ways to arrange 5 books. In probability: n! appears in permutation formulas P(n,k) = n!/(n−k)!. In Taylor series: e^x = Σ xⁿ/n!. In quantum mechanics: normalization factors.',
        'context_note': '0! = 1 by convention (empty product). n! grows faster than any exponential: 100! ≈ 9.3 × 10¹⁵⁷.',
    },
    {
        'symbol': 'C(n,k) or ₙCₖ or (n choose k)', 'latex': r'\binom{n}{k}', 'name': 'Binomial Coefficient',
        'domains': ['Combinatorics', 'Probability', 'Algebra'],
        'meaning': 'Number of ways to choose k items from n items without regard to order.',
        'how_to_read': '"n choose k" | "C of n k"',
        'real_world': 'C(52,5) = 2,598,960 possible poker hands. Probability of exactly k successes in n Bernoulli trials: P(X=k) = C(n,k)pᵏ(1−p)^(n−k). Pascal\'s triangle entries.',
        'example': 'C(6,2) = 6!/(2!4!) = 15 ways to pick 2 teammates from 6 people.',
    },
    {
        'symbol': 'Σ', 'latex': r'\sum', 'name': 'Summation (Sigma)',
        'domains': ['Algebra', 'Calculus', 'Statistics', 'All mathematics'],
        'meaning': 'Sum of a sequence of terms. Σᵢ₌₁ⁿ aᵢ = a₁ + a₂ + ⋯ + aₙ.',
        'how_to_read': '"the sum from i equals 1 to n of a-sub-i"',
        'real_world': 'Total distance = Σ(speed_i × time_i). Sample mean: x̄ = (1/n)Σxᵢ. Taylor series: sin(x) = Σ(−1)ⁿx^(2n+1)/(2n+1)!. Expected value: E[X] = Σ xᵢ P(xᵢ).',
        'context_note': 'The variable below Σ (called the index) is a dummy variable — its name doesn\'t matter. Σᵢaᵢ = Σⱼaⱼ. Related: Π (product), ∫ (continuous sum/integral).',
    },
    {
        'symbol': 'Π', 'latex': r'\prod', 'name': 'Product (Pi)',
        'domains': ['Algebra', 'Number Theory', 'Analysis'],
        'meaning': 'Product of a sequence of terms. Πᵢ₌₁ⁿ aᵢ = a₁ × a₂ × ⋯ × aₙ.',
        'how_to_read': '"the product from i equals 1 to n of a-sub-i"',
        'real_world': 'Compound probability of independent events: P(A₁ ∩ A₂ ∩ ⋯ ∩ Aₙ) = ΠP(Aᵢ). n! = Πᵢ₌₁ⁿ i. In number theory: any integer = Π p_i^(e_i) over its prime factors.',
        'common_confusion': 'Π (capital pi, product notation) vs π (lowercase pi, circle constant 3.14159…).',
    },
    {
        'symbol': 'lim', 'latex': r'\lim_{x \to a}', 'name': 'Limit',
        'domains': ['Calculus', 'Real Analysis', 'Complex Analysis'],
        'meaning': 'The value a function approaches as the input approaches some value, without necessarily reaching it.',
        'how_to_read': '"the limit as x approaches a of f of x"',
        'real_world': 'Instantaneous velocity: v = lim_{Δt→0} Δx/Δt. Defines derivatives, integrals, and continuity. In physics: idealized point charges, infinitely sharp edges, instantaneous processes.',
        'example': 'lim_{x→0} sin(x)/x = 1  (foundational in calculus and wave physics)',
        'context_note': 'One-sided limits: lim_{x→a⁺} (from right), lim_{x→a⁻} (from left). A limit exists iff both one-sided limits exist and are equal.',
    },
    {
        'symbol': "f'(x) or dy/dx or df/dx", 'latex': r"f'(x),\; \frac{dy}{dx}", 'name': 'Derivative',
        'domains': ['Calculus', 'Analysis', 'Physics', 'Engineering'],
        'meaning': 'Instantaneous rate of change of f with respect to x. Slope of the tangent line at x.',
        'how_to_read': '"f prime of x" | "dy by dx" | "the derivative of y with respect to x"',
        'real_world': 'v(t) = dx/dt (velocity = rate of change of position). a(t) = dv/dt (acceleration = rate of change of velocity). dT/dx (temperature gradient). dP/dt (rate of pressure change). In economics: marginal cost = dC/dq.',
        'dimensional_note': '[dy/dx] = [y]/[x]. If y is in meters and x in seconds: dy/dx has units m/s.',
        'context_note': 'Notations: f\'(x) (Lagrange), dy/dx (Leibniz), ẋ (Newton dot, for time derivatives), Df (operator). Higher derivatives: f\'\'(x) = d²y/dx².',
        'example': 'If s(t) = ½gt² then s\'(t) = gt (velocity increases linearly with time)',
    },
    {
        'symbol': '∂/∂x', 'latex': r'\frac{\partial f}{\partial x}', 'name': 'Partial Derivative',
        'domains': ['Multivariable Calculus', 'Physics', 'Engineering', 'Thermodynamics'],
        'meaning': 'Rate of change of a function of multiple variables with respect to one variable, holding all others constant.',
        'how_to_read': '"partial f partial x" | "the partial derivative of f with respect to x"',
        'real_world': 'In thermodynamics: (∂U/∂T)_V = heat capacity at constant volume. Weather: ∂T/∂x (temperature gradient east-west), ∂T/∂z (lapse rate vertical). In fluid mechanics: ∂v/∂x (shear rate).',
        'context_note': 'The ∂ symbol (curly d) distinguishes from ordinary d. In total differential: df = (∂f/∂x)dx + (∂f/∂y)dy + ⋯. The subscript notation (∂U/∂T)_V explicitly shows what is held constant.',
        'common_confusion': 'df/dx is used when f depends on only one variable. ∂f/∂x is used when f depends on multiple variables.',
    },
    {
        'symbol': '∫', 'latex': r'\int', 'name': 'Integral',
        'domains': ['Calculus', 'Analysis', 'Physics', 'Engineering', 'Probability'],
        'meaning': 'Continuous summation of infinitesimally thin slices. Inverse operation of differentiation. Represents accumulation.',
        'how_to_read': '"the integral of f of x, dee x" | "the antiderivative of f"',
        'real_world': 'Total distance = ∫ v(t) dt (sum of all velocity×time slices). Total charge = ∫ I(t) dt. Probability: P(a≤X≤b) = ∫ₐᵇ f(x)dx. Work = ∫ F·dx. Mass of rod with variable density = ∫ ρ(x) dx.',
        'dimensional_note': '[∫f(x)dx] = [f(x)]×[x]. Integrating force (N) over distance (m) gives energy (N·m = J).',
        'context_note': 'The dx part is not just notation — it represents an infinitesimal width. ∫ₐᵇ f(x)dx = lim_{n→∞} Σ f(xᵢ)Δx. The elongated S (∫) is from Latin "summa" (sum).',
        'example': '∫₀ᵀ v(t)dt = total displacement [meters] from t=0 to t=T',
    },
    {
        'symbol': '∬, ∭, ∮', 'latex': r'\iint, \iiint, \oint', 'name': 'Multiple / Line Integral',
        'domains': ['Multivariable Calculus', 'Physics', 'Electromagnetism'],
        'meaning': '∬ integrates over 2D area, ∭ over 3D volume, ∮ over a closed curve.',
        'how_to_read': '"double integral over R" | "line integral around C" | "surface integral"',
        'real_world': '∬ ρ(x,y) dA = total mass of a 2D object with density ρ. ∭ dV = volume of a 3D region. ∮ E·dl = EMF in Faraday\'s law. ∮ F·dr = work done by force F along closed path.',
    },
    {
        'symbol': '∇', 'latex': r'\nabla', 'name': 'Nabla / Del Operator',
        'domains': ['Vector Calculus', 'Physics', 'Electromagnetism', 'Fluid Mechanics'],
        'meaning': 'Vector differential operator. Applied to scalar gives gradient; dot product gives divergence; cross product gives curl.',
        'how_to_read': '"nabla f" | "del f" | "gradient of f"',
        'real_world': '∇T = temperature gradient (points toward hottest direction, tells heat flow direction). ∇·E = ρ/ε₀ (Gauss\'s law: divergence of E-field = charge density). ∇×B = μ₀J (Ampere\'s law: curl of B-field = current density). ∇² = Laplacian (used in wave equations, heat equation).',
        'context_note': 'In 3D Cartesian: ∇ = (∂/∂x, ∂/∂y, ∂/∂z). Operations: ∇f = gradient (scalar→vector), ∇·F = divergence (vector→scalar), ∇×F = curl (vector→vector), ∇²f = Laplacian (scalar→scalar).',
    },
    {
        'symbol': 'd/dt and ẋ', 'latex': r'\dot{x}, \ddot{x}', 'name': "Newton's Dot Notation (Time Derivative)",
        'domains': ['Classical Mechanics', 'Dynamics', 'Control Theory'],
        'meaning': 'One dot over variable = first time derivative. Two dots = second time derivative.',
        'how_to_read': '"x-dot" | "x-double-dot"',
        'real_world': 'ẋ = velocity (rate of change of position). ẍ = acceleration. θ̇ = angular velocity. In Lagrangian mechanics: L(q, q̇, t) where q are generalized coordinates. ẍ = F/m (Newton\'s second law in compact form).',
    },
    {
        'symbol': '∞', 'latex': r'\infty', 'name': 'Infinity',
        'domains': ['Analysis', 'Calculus', 'Set Theory', 'All mathematics'],
        'meaning': 'Not a real number but a concept representing unbounded growth or limitless extent.',
        'how_to_read': '"infinity"',
        'real_world': 'lim_{n→∞} (1 + 1/n)^n = e. ∫₀^∞ e^(−x²) dx = √π/2 (Gaussian integral, fundamental in probability). Infinite series: Σ_{n=1}^∞ 1/n² = π²/6 (Basel problem).',
        'context_note': 'Different sizes: ℕ, ℤ, ℚ all have countable infinity (ℵ₀). ℝ has uncountable infinity (ℵ₁ = 2^ℵ₀). ∞ + 1 = ∞ in the extended real line but ∞ − ∞ is indeterminate.',
        'common_confusion': 'Infinity is not a number — you cannot divide by it or subtract it without care. lim rules must be applied.',
    },
    {
        'symbol': 'Δ', 'latex': r'\Delta', 'name': 'Delta (Finite Change)',
        'domains': ['Physics', 'Calculus', 'Engineering'],
        'meaning': 'Capital Delta: finite difference or change in a quantity. Δx = x_final − x_initial.',
        'how_to_read': '"delta x" | "change in x"',
        'real_world': 'ΔT = temperature change. ΔE = energy change (conservation: ΔE = 0 in isolated system). Δv = velocity change (impulse = mΔv). In calculus: as Δx→0 we get the derivative dx.',
        'context_note': 'Lowercase δ: infinitesimally small variation (in calculus of variations). In Dirac delta: δ(x) is a generalized function, zero everywhere except x=0 with integral=1. In Kronecker delta: δᵢⱼ = 1 if i=j, else 0.',
    },
    {
        'symbol': 'ε and δ (epsilon-delta)', 'latex': r'\varepsilon,\; \delta', 'name': 'Epsilon-Delta (Limit Definition)',
        'domains': ['Real Analysis', 'Calculus'],
        'meaning': 'Formal language for limits. ε (epsilon) = desired closeness of output. δ (delta) = required closeness of input.',
        'how_to_read': '"for all epsilon greater than zero, there exists delta greater than zero..."',
        'real_world': 'Engineering tolerances: "within ε of target" means |output − target| < ε. lim_{x→a} f(x) = L means: for any error bound ε you specify, I can find δ such that if x is within δ of a, f(x) is within ε of L.',
        'context_note': 'ε (epsilon) also used for: permittivity in electromagnetics (ε₀), eccentricity of orbits, strain in materials. Context determines meaning.',
    },
    {
        'symbol': 'e', 'latex': 'e', 'name': "Euler's Number",
        'domains': ['Calculus', 'Analysis', 'Probability', 'Physics'],
        'meaning': 'The base of natural logarithm. e ≈ 2.71828... Unique property: d/dx(eˣ) = eˣ (the only function equal to its own derivative).',
        'how_to_read': '"e" | "Euler\'s number" | "the base of the natural logarithm"',
        'real_world': 'Continuous growth/decay: N(t) = N₀ eᵏᵗ. Radioactive decay: N(t) = N₀ e^(−λt). RC circuits: V(t) = V₀ e^(−t/RC). Normal distribution: e^(−x²/2). Fourier analysis: e^(iωt) = cos(ωt) + i·sin(ωt).',
        'context_note': 'e = lim_{n→∞}(1 + 1/n)^n = Σ_{n=0}^∞ 1/n!. In complex analysis: e^(iπ) + 1 = 0 (Euler\'s identity, linking e, i, π, 1, 0).',
    },
    {
        'symbol': 'π', 'latex': r'\pi', 'name': 'Pi',
        'domains': ['Geometry', 'Trigonometry', 'Analysis', 'Physics'],
        'meaning': 'Ratio of circle\'s circumference to its diameter. π ≈ 3.14159265...',
        'how_to_read': '"pi"',
        'real_world': 'Circumference = 2πr. Area = πr². Volume of sphere = (4/3)πr³. Period of pendulum: T = 2π√(L/g). In signal processing: angular frequency ω = 2πf. Heisenberg uncertainty: ΔxΔp ≥ ħ/2 where ħ = h/(2π).',
        'context_note': 'π appears in many non-circular contexts because it is fundamentally tied to the geometry of the Euclidean plane, Fourier analysis, and eigenvalues of differential operators.',
    },
    {
        'symbol': 'i', 'latex': 'i', 'name': 'Imaginary Unit',
        'domains': ['Complex Analysis', 'Algebra', 'Electrical Engineering', 'Quantum Mechanics'],
        'meaning': 'i = √(−1). The imaginary unit extends real numbers to complex numbers ℂ = {a + bi : a,b ∈ ℝ}.',
        'how_to_read': '"i" | "the imaginary unit" (note: engineers use j to avoid confusion with current)',
        'real_world': 'AC circuit analysis: impedance Z = R + jX (j used for imaginary to avoid confusion with current i). Quantum mechanics: wave functions ψ are complex-valued. Fourier transform: F(ω) = ∫ f(t)e^(−iωt)dt. Rotations in 2D correspond to multiplication by complex numbers.',
        'context_note': 'i² = −1, i³ = −i, i⁴ = 1. |a + bi| = √(a² + b²). Argument: arg(a+bi) = arctan(b/a). Polar form: re^(iθ) = r(cos θ + i sin θ).',
    },
    {
        'symbol': 'ln and log', 'latex': r'\ln, \log', 'name': 'Natural Logarithm / Logarithm',
        'domains': ['Algebra', 'Calculus', 'Physics', 'Information Theory'],
        'meaning': 'ln(x): logarithm base e. log(x): logarithm (base 10 in applied sciences, base e in pure math, base 2 in computer science — context dependent!).',
        'how_to_read': '"natural log of x" | "log of x"',
        'real_world': 'pH = −log₁₀[H⁺]. Decibels: dB = 10 log₁₀(P/P₀). Information entropy: H = −Σ p·log₂(p). RC circuit: t = −RC·ln(V/V₀). Growth rate: if N = N₀eᵏᵗ then k = ln(N/N₀)/t.',
        'dimensional_note': 'Logarithm arguments must be dimensionless. ln(5 kg) is undefined. Write ln(m/m₀) where m₀ is a reference mass.',
        'common_confusion': 'In pure mathematics, log typically means ln (base e). In physics/engineering, log typically means log₁₀. Always check context.',
    },
    {
        'symbol': 'sin, cos, tan', 'latex': r'\sin, \cos, \tan', 'name': 'Trigonometric Functions',
        'domains': ['Trigonometry', 'Calculus', 'Physics', 'Signal Processing'],
        'meaning': 'sin = opposite/hypotenuse, cos = adjacent/hypotenuse, tan = sin/cos = opposite/adjacent in a right triangle. Generalized: coordinates on unit circle.',
        'how_to_read': '"sine of theta" | "cosine of theta" | "tangent of theta"',
        'real_world': 'Wave: y(x,t) = A·sin(kx − ωt + φ). Projectile: x = v₀·cos(θ)·t, y = v₀·sin(θ)·t − ½gt². AC voltage: V(t) = V₀·cos(ωt). Power in 3-phase: P = √3·V·I·cos(φ).',
        'dimensional_note': 'The argument of trig functions must be dimensionless (radians are dimensionless). sin(30°) requires conversion: sin(π/6). Result is always between −1 and 1 (for sin and cos).',
    },
    {
        'symbol': 'arcsin, arccos, arctan', 'latex': r'\arcsin, \arccos, \arctan', 'name': 'Inverse Trigonometric Functions',
        'domains': ['Trigonometry', 'Geometry', 'Physics'],
        'meaning': 'Inverse functions: arcsin(y) = the angle whose sine is y.',
        'how_to_read': '"arc sine of x" | "inverse sine of x" | "sine inverse of x"',
        'real_world': 'Finding launch angle: θ = arctan(vy/vx). Snell\'s law angle: θ₂ = arcsin(n₁·sin(θ₁)/n₂). Phase angle in circuits: φ = arctan(X/R).',
        'context_note': 'Output range: arcsin → [−π/2, π/2], arccos → [0, π], arctan → (−π/2, π/2). For full-quadrant angles use atan2(y, x).',
    },
    {
        'symbol': '∈ and ∉', 'latex': r'\in, \notin', 'name': 'Element Of',
        'domains': ['Set Theory', 'Logic', 'All mathematics'],
        'meaning': '∈: an object belongs to a set. ∉: does not belong.',
        'how_to_read': '"x is in S" | "x is an element of S" | "x belongs to S"',
        'real_world': 'x ∈ ℝ means x is a real number. n ∈ ℕ means n is a positive integer. In code: validating that a value belongs to an allowed set of options.',
        'example': '√2 ∈ ℝ but √2 ∉ ℚ (irrational — not a ratio of integers)',
    },
    {
        'symbol': '⊂ and ⊆', 'latex': r'\subset, \subseteq', 'name': 'Subset',
        'domains': ['Set Theory', 'Logic', 'Topology'],
        'meaning': 'A ⊆ B: every element of A is also in B (B contains A). A ⊂ B: strict subset (A ⊆ B and A ≠ B).',
        'how_to_read': '"A is a subset of B" | "A is contained in B"',
        'real_world': 'ℕ ⊂ ℤ ⊂ ℚ ⊂ ℝ ⊂ ℂ (natural numbers are subset of integers, which are subset of rationals, etc.). All squares ⊂ All rectangles ⊂ All parallelograms.',
    },
    {
        'symbol': '∩ and ∪', 'latex': r'\cap, \cup', 'name': 'Intersection and Union',
        'domains': ['Set Theory', 'Logic', 'Probability'],
        'meaning': 'A ∩ B: elements in both A and B. A ∪ B: elements in A or B (or both).',
        'how_to_read': '"A intersect B" | "A union B" | "A and B" (∩) | "A or B" (∪)',
        'real_world': 'P(A ∩ B): probability both events occur. P(A ∪ B) = P(A) + P(B) − P(A ∩ B). In Venn diagrams: ∩ = overlap region, ∪ = entire shaded area.',
        'common_confusion': '∩ (intersection, like AND in logic) vs ∪ (union, like OR in logic). Memory trick: ∩ looks like an upside-down U for "and".',
    },
    {
        'symbol': '∅', 'latex': r'\emptyset', 'name': 'Empty Set',
        'domains': ['Set Theory', 'Logic'],
        'meaning': 'The set with no elements. Subset of every set.',
        'how_to_read': '"the empty set" | "the null set"',
        'real_world': 'A ∩ B = ∅ means events A and B are mutually exclusive (cannot both occur). Solution set of x² = −1 over ℝ is ∅.',
    },
    {
        'symbol': '∀', 'latex': r'\forall', 'name': 'Universal Quantifier (For All)',
        'domains': ['Logic', 'Set Theory', 'Proof Theory'],
        'meaning': 'A statement is true for every element in the domain.',
        'how_to_read': '"for all x" | "for every x" | "for each x"',
        'real_world': '∀ x ∈ ℝ: x² ≥ 0 (every real number squared is non-negative). ∀ ε > 0, ∃ δ > 0 such that... (formal limit definition). Universal laws in physics are written as ∀-statements.',
        'context_note': 'Read ∀x P(x) as: no matter what x you pick from the domain, P(x) holds. To disprove: find ONE counterexample.',
    },
    {
        'symbol': '∃ and ∄', 'latex': r'\exists, \nexists', 'name': 'Existential Quantifier',
        'domains': ['Logic', 'Set Theory', 'Proof Theory'],
        'meaning': '∃: there is at least one element for which the statement holds. ∄: no such element exists.',
        'how_to_read': '"there exists x such that" | "there is an x where"',
        'real_world': '∃ x ∈ ℝ: x² = 2 (√2 is a real number). ∄ x ∈ ℚ: x² = 2 (√2 is irrational). In optimization: prove ∃ a minimum before trying to find it.',
        'context_note': 'To prove ∃ x: exhibit the x. To disprove ∃ x: show ∀ x the property fails. ∃! means "there exists exactly one."',
    },
    {
        'symbol': '∧ ∨ ¬', 'latex': r'\wedge, \vee, \neg', 'name': 'Logical AND, OR, NOT',
        'domains': ['Logic', 'Set Theory', 'Boolean Algebra', 'Computer Science'],
        'meaning': '∧: true when both are true (AND). ∨: true when at least one is true (OR). ¬: flips truth value (NOT).',
        'how_to_read': '"P and Q" | "P or Q" | "not P"',
        'real_world': 'Boolean circuits use these as logic gates. In probability: P(A ∩ B) corresponds to ∧, P(A ∪ B) corresponds to ∨. SQL WHERE clauses use AND/OR/NOT.',
        'context_note': '∧ vs ∩: ∧ operates on propositions (TRUE/FALSE), ∩ operates on sets. Related by: x ∈ A ∩ B ↔ (x ∈ A) ∧ (x ∈ B).',
    },
    {
        'symbol': '→ and ↔', 'latex': r'\rightarrow, \leftrightarrow', 'name': 'Implication and Biconditional',
        'domains': ['Logic', 'Proof Theory'],
        'meaning': 'P → Q: if P then Q. Q → P would be the converse. P ↔ Q: P if and only if Q (P iff Q).',
        'how_to_read': '"P implies Q" | "if P then Q" | "P if and only if Q"',
        'real_world': 'Physical laws: (energy conserved) → (momentum conserved) [in isolated system]. Theorems: P ↔ Q means P and Q are equivalent characterizations of the same thing.',
        'common_confusion': 'P → Q is NOT the same as Q → P (converse). "If it rains, the street is wet" ≠ "If the street is wet, it rained."',
    },
    {
        'symbol': '∴ and ∵', 'latex': r'\therefore, \because', 'name': 'Therefore and Because',
        'domains': ['Logic', 'Proof Theory'],
        'meaning': '∴ marks a conclusion. ∵ introduces a reason.',
        'how_to_read': '"therefore" | "because" | "since"',
        'example': 'All mammals are warm-blooded. ∵ Dogs are mammals. ∴ Dogs are warm-blooded.',
    },
    {
        'symbol': 'ℕ ℤ ℚ ℝ ℂ', 'latex': r'\mathbb{N}, \mathbb{Z}, \mathbb{Q}, \mathbb{R}, \mathbb{C}',
        'name': 'Number Sets (Blackboard Bold)',
        'domains': ['Set Theory', 'Algebra', 'Analysis'],
        'meaning': 'ℕ: natural numbers {1,2,3,…} or {0,1,2,…}. ℤ: integers. ℚ: rationals (a/b form). ℝ: reals. ℂ: complex numbers.',
        'how_to_read': '"the natural numbers" | "the integers" | "the rationals" | "the reals" | "the complex numbers"',
        'real_world': 'Counting objects: n ∈ ℕ. Signed quantities (debt, temperature below zero): ℤ. Measurements with fractions: ℚ. Continuous quantities (length, time, probability): ℝ. Signal frequencies, quantum states: ℂ.',
        'context_note': 'Containment: ℕ ⊂ ℤ ⊂ ℚ ⊂ ℝ ⊂ ℂ. "Blackboard bold" font (ℝ) distinguishes number sets from ordinary variables. Some authors include 0 in ℕ, others don\'t — check convention.',
    },
    {
        'symbol': 'λ (lambda)', 'latex': r'\lambda', 'name': 'Lambda — Eigenvalue / Rate / Wavelength',
        'domains': ['Linear Algebra', 'Physics', 'Quantum Mechanics', 'Statistics'],
        'meaning': 'Context-dependent: eigenvalue in linear algebra, wavelength in physics, decay/rate parameter in Poisson process, lambda calculus in CS.',
        'how_to_read': '"lambda"',
        'real_world': 'Eigenvalue: Av = λv (matrix A scales vector v by factor λ). Wavelength: λ = c/f (speed of light / frequency, in meters). Decay constant: N(t) = N₀e^(−λt) [units: 1/time]. Poisson rate: λ events per unit time. Half-life: t½ = ln(2)/λ.',
        'context_note': 'This is a prime example of context-dependent symbols. The domain/field immediately tells you which meaning applies.',
    },
    {
        'symbol': 'μ (mu)', 'latex': r'\mu', 'name': 'Mu — Mean / Micro- / Friction / Permeability',
        'domains': ['Statistics', 'Physics', 'Engineering'],
        'meaning': 'Mean (expected value) in statistics. Prefix micro- (10⁻⁶) in SI units. Coefficient of friction in mechanics. Magnetic permeability in electromagnetism.',
        'how_to_read': '"mu"',
        'real_world': 'μ = 5.2 kg → mean mass is 5.2 kg. 5 μm = 5 micrometers (wavelength of infrared). μₖ = 0.3 → kinetic friction coefficient. μ₀ = 4π×10⁻⁷ H/m (permeability of free space in Ampere\'s law).',
    },
    {
        'symbol': 'σ (sigma)', 'latex': r'\sigma', 'name': 'Sigma — Standard Deviation / Stress / Conductivity',
        'domains': ['Statistics', 'Materials Science', 'Electromagnetism'],
        'meaning': 'Lowercase sigma: standard deviation (spread of data). Also: stress in materials, electrical conductivity, surface charge density.',
        'how_to_read': '"sigma"',
        'real_world': 'σ = 2.5 kg → data spreads ±2.5 kg from mean. 68% of data within 1σ of mean (normal distribution). σ = 50 MPa → material under 50 megapascal stress. σ = 5.8×10⁷ S/m → electrical conductivity of copper.',
        'context_note': 'Capital Σ = summation. Lowercase σ = standard deviation or stress. Context is everything.',
    },
    {
        'symbol': 'θ (theta)', 'latex': r'\theta', 'name': 'Theta — Angle',
        'domains': ['Geometry', 'Trigonometry', 'Physics', 'Statistics'],
        'meaning': 'Conventionally represents an angle, especially in polar coordinates and trigonometry.',
        'how_to_read': '"theta"',
        'real_world': 'θ = 45° (or π/4 radians) in projectile launch. θ in polar coordinates (r, θ). Phase angle in AC circuits. In statistics: generic parameter to be estimated.',
    },
    {
        'symbol': 'φ or ϕ (phi)', 'latex': r'\phi', 'name': 'Phi — Angle / Golden Ratio / Phase / Field',
        'domains': ['Geometry', 'Physics', 'Number Theory', 'Field Theory'],
        'meaning': 'Context: azimuthal angle in spherical coordinates; golden ratio φ=(1+√5)/2≈1.618; phase angle in waves; scalar field in physics; Euler totient φ(n).',
        'how_to_read': '"phi"',
        'real_world': 'Wave phase: y = A·sin(kx − ωt + φ₀). Golden ratio: Fibonacci ratios approach φ. Euler totient: φ(12) = 4 (four integers < 12 coprime to 12).',
    },
    {
        'symbol': 'ω (omega)', 'latex': r'\omega', 'name': 'Omega — Angular Frequency / Vorticity',
        'domains': ['Physics', 'Signal Processing', 'Mechanics'],
        'meaning': 'Angular frequency ω = 2πf (radians per second). Angular velocity of rotation.',
        'how_to_read': '"omega"',
        'real_world': 'ω = 2π × 60 Hz = 376.99 rad/s for 60 Hz AC. Period T = 2π/ω. In LC circuit: resonant frequency ω₀ = 1/√(LC). Capital Ω = ohms (resistance unit) in electronics.',
    },
    {
        'symbol': 'ρ (rho)', 'latex': r'\rho', 'name': 'Rho — Density / Resistivity / Radius',
        'domains': ['Physics', 'Chemistry', 'Engineering', 'Fluid Mechanics'],
        'meaning': 'Mass density (kg/m³), charge density, resistivity of material, radial coordinate in cylindrical coordinates.',
        'how_to_read': '"rho"',
        'real_world': 'ρ_water = 1000 kg/m³. ρ_copper = 1.72×10⁻⁸ Ω·m (resistivity). In fluid mechanics: ρ appears in Navier-Stokes equations. Pressure: P = ρgh (hydrostatic).',
    },
    {
        'symbol': 'τ (tau)', 'latex': r'\tau', 'name': 'Tau — Time Constant / Torque / Proper Time',
        'domains': ['Physics', 'Engineering', 'Relativity'],
        'meaning': 'Time constant of exponential decay/growth. Torque in mechanics. Proper time in special relativity.',
        'how_to_read': '"tau"',
        'real_world': 'RC circuit: τ = RC [seconds]. After time τ, signal decays to 1/e ≈ 37% of original. Torque: τ = r × F [N·m]. After 5τ, a system is considered settled (≈ 99.3% of final value).',
    },
    {
        'symbol': 'α β γ', 'latex': r'\alpha, \beta, \gamma', 'name': 'Greek Letters — General Parameters',
        'domains': ['Physics', 'Statistics', 'Mathematics generally'],
        'meaning': 'α, β, γ are extremely common general-purpose parameter symbols. Meaning entirely context-dependent.',
        'how_to_read': '"alpha", "beta", "gamma"',
        'real_world': 'α: significance level in statistics (α=0.05), angular acceleration, fine-structure constant. β: regression coefficient, beta decay, velocity as fraction of c. γ: Lorentz factor γ=1/√(1−v²/c²), heat capacity ratio Cp/Cv.',
    },
    {
        'symbol': 'A^T', 'latex': 'A^T', 'name': 'Matrix Transpose',
        'domains': ['Linear Algebra', 'Matrix Theory'],
        'meaning': 'Reflect matrix across its main diagonal. Rows become columns.',
        'how_to_read': '"A transpose"',
        'real_world': '(AB)^T = B^T A^T. For rotation matrix R: R^T = R^(−1) (orthogonal matrix property). In least squares: optimal fit β = (X^T X)^(−1) X^T y.',
    },
    {
        'symbol': 'A⁻¹', 'latex': 'A^{-1}', 'name': 'Matrix Inverse',
        'domains': ['Linear Algebra'],
        'meaning': 'Matrix that when multiplied by A gives identity: A·A⁻¹ = I.',
        'how_to_read': '"A inverse"',
        'real_world': 'Solving Ax = b: x = A⁻¹b. Exists only when det(A) ≠ 0. Computationally expensive for large matrices — prefer LU decomposition or iterative solvers.',
    },
    {
        'symbol': 'det(A) or |A|', 'latex': r'\det(A)', 'name': 'Determinant',
        'domains': ['Linear Algebra'],
        'meaning': 'Scalar value encoding the volume scaling factor of the linear transformation described by A.',
        'how_to_read': '"determinant of A"',
        'real_world': 'det(A) = 0: matrix is singular (collapses space to lower dimension; system has no unique solution). det(A) > 0: transformation preserves orientation. |det(A)| = scale factor for volumes. In Jacobian of coordinate transformations.',
    },
    {
        'symbol': 'λ in Av = λv', 'latex': r'A\mathbf{v} = \lambda\mathbf{v}', 'name': 'Eigenvalue Equation',
        'domains': ['Linear Algebra', 'Quantum Mechanics', 'Differential Equations'],
        'meaning': 'v is an eigenvector of A if A only scales v (by eigenvalue λ) without changing its direction.',
        'how_to_read': '"A times v equals lambda times v" | "v is an eigenvector of A with eigenvalue lambda"',
        'real_world': 'Quantum mechanics: Ĥψ = Eψ (energy E is eigenvalue; ψ is energy eigenstate). PCA: eigenvectors of covariance matrix are principal components. Vibration modes: natural frequencies are eigenvalues of the stiffness matrix.',
    },
    {
        'symbol': '‖v‖ or |v|', 'latex': r'\|v\|', 'name': 'Vector Norm',
        'domains': ['Linear Algebra', 'Analysis', 'Physics'],
        'meaning': 'Length/magnitude of a vector.',
        'how_to_read': '"the norm of v" | "the magnitude of v" | "the length of v"',
        'real_world': '‖v‖ = √(v₁² + v₂² + ⋯ + vₙ²) (Euclidean norm). Unit vector: v̂ = v/‖v‖. Speed = ‖velocity vector‖. Distance = ‖position₂ − position₁‖.',
    },
    {
        'symbol': '⟨u, v⟩ or u · v', 'latex': r'\langle u, v \rangle,\; u \cdot v', 'name': 'Inner Product / Dot Product',
        'domains': ['Linear Algebra', 'Analysis', 'Physics'],
        'meaning': 'Scalar measure of alignment between two vectors. u · v = ‖u‖‖v‖cos(θ) where θ is angle between them.',
        'how_to_read': '"u dot v" | "inner product of u and v"',
        'real_world': 'Work = F · d (force dot displacement). Power = F · v. Projection: (u·v̂)v̂. u · v = 0 means vectors are perpendicular (orthogonal). In quantum: ⟨ψ|φ⟩ = probability amplitude.',
    },
    {
        'symbol': 'O(f(n))', 'latex': 'O(f(n))', 'name': 'Big-O Notation',
        'domains': ['Analysis', 'Computer Science', 'Asymptotics'],
        'meaning': 'f(n) = O(g(n)) means f grows no faster than g asymptotically (up to constant factor).',
        'how_to_read': '"O of f of n" | "order f of n" | "big-O of f"',
        'real_world': 'Bubble sort: O(n²) operations. Binary search: O(log n). Describes how algorithm time/space grows with input size. Physics: Taylor series remainder term O(x³) means error is bounded by Cx³ for small x.',
        'context_note': 'o(f) (little-o): grows strictly slower than f. Θ(f): grows at same rate. Ω(f): grows at least as fast.',
    },
    {
        'symbol': '≡ (mod n)', 'latex': r'a \equiv b \pmod{n}', 'name': 'Modular Congruence',
        'domains': ['Number Theory', 'Abstract Algebra', 'Cryptography'],
        'meaning': 'a ≡ b (mod n): a and b have the same remainder when divided by n.',
        'how_to_read': '"a is congruent to b modulo n" | "a mod n equals b mod n"',
        'real_world': '17 ≡ 2 (mod 5) because 17 = 3×5 + 2 and 2 = 0×5 + 2. Clock arithmetic: 14:00 ≡ 2 (mod 12). Cryptography (RSA): based on modular exponentiation.',
    },
    {
        'symbol': '⌊x⌋ and ⌈x⌉', 'latex': r'\lfloor x \rfloor, \lceil x \rceil', 'name': 'Floor and Ceiling',
        'domains': ['Number Theory', 'Computer Science', 'Analysis'],
        'meaning': '⌊x⌋: largest integer ≤ x. ⌈x⌉: smallest integer ≥ x.',
        'how_to_read': '"floor of x" | "ceiling of x"',
        'real_world': 'Pages needed for n items in groups of k: ⌈n/k⌉. Integer part of a decimal: ⌊3.7⌋ = 3. Rounding: ⌊x + 0.5⌋.',
    },
    {
        'symbol': 'gcd and lcm', 'latex': r'\gcd(a,b), \text{lcm}(a,b)', 'name': 'GCD and LCM',
        'domains': ['Number Theory', 'Arithmetic'],
        'meaning': 'gcd: largest integer dividing both a and b. lcm: smallest positive integer divisible by both.',
        'how_to_read': '"greatest common divisor of a and b" | "least common multiple of a and b"',
        'real_world': 'Simplifying fractions: 12/18 = 2/3 because gcd(12,18) = 6. Finding a common clock period: lcm(12, 15) = 60 (events at 12s and 15s intervals coincide every 60s).',
    },
    {
        'symbol': 'P(A), P(A|B)', 'latex': r'P(A),\; P(A \mid B)', 'name': 'Probability / Conditional Probability',
        'domains': ['Probability', 'Statistics', 'Bayesian Inference'],
        'meaning': 'P(A): probability event A occurs. P(A|B): probability A occurs given B has occurred.',
        'how_to_read': '"probability of A" | "probability of A given B"',
        'real_world': 'Medical test: P(disease|positive test) vs P(positive test|disease) — very different! Bayes: P(A|B) = P(B|A)P(A)/P(B). Weather: P(rain|cloudy) = 0.7.',
        'common_confusion': 'P(A|B) ≠ P(B|A). This confusion ("prosecutor\'s fallacy") has led to wrongful convictions.',
    },
    {
        'symbol': 'E[X], Var(X)', 'latex': r'E[X], \text{Var}(X)', 'name': 'Expected Value and Variance',
        'domains': ['Probability', 'Statistics'],
        'meaning': 'E[X]: average value of random variable X over many trials. Var(X) = E[(X−μ)²]: average squared deviation from mean.',
        'how_to_read': '"expected value of X" | "variance of X"',
        'real_world': 'E[X] = μ = mean. Var(X) = σ². Insurance: E[payout] determines fair premium. E[X+Y] = E[X] + E[Y] always. Var(X+Y) = Var(X) + Var(Y) only if X,Y independent.',
    },
    {
        'symbol': '~ (distributed as)', 'latex': r'X \sim N(\mu, \sigma^2)', 'name': 'Distributed As',
        'domains': ['Probability', 'Statistics'],
        'meaning': 'Specifies the probability distribution that a random variable follows.',
        'how_to_read': '"X is distributed as" | "X follows a normal distribution with mean mu and variance sigma-squared"',
        'real_world': 'X ~ N(μ, σ²): X is normally distributed. X ~ Poisson(λ): X counts rare events per interval. X ~ Bernoulli(p): X is 1 with probability p, 0 otherwise.',
    },
    {
        'symbol': 'ħ (h-bar)', 'latex': r'\hbar', 'name': 'Reduced Planck Constant',
        'domains': ['Quantum Mechanics', 'Physics'],
        'meaning': 'ħ = h/(2π) ≈ 1.055×10⁻³⁴ J·s. Fundamental quantum of angular momentum.',
        'how_to_read': '"h-bar"',
        'real_world': 'Heisenberg uncertainty: ΔxΔp ≥ ħ/2. Energy of photon: E = ħω. Schrödinger equation: iħ ∂ψ/∂t = Ĥψ. Spin angular momentum: s = ½ħ for electrons.',
    },
    {
        'symbol': '⟨ψ|Â|ψ⟩', 'latex': r'\langle \psi | \hat{A} | \psi \rangle', 'name': 'Dirac Bra-Ket Notation',
        'domains': ['Quantum Mechanics'],
        'meaning': 'Inner product in quantum state space. |ψ⟩: "ket" (state vector). ⟨ψ|: "bra" (dual vector). ⟨ψ|Â|ψ⟩: expectation value of observable Â in state |ψ⟩.',
        'how_to_read': '"bra psi | A-hat | ket psi" | "expectation value of A in state psi"',
        'real_world': '⟨ψ|Ĥ|ψ⟩ = expected energy. ⟨ψ|x̂|ψ⟩ = expected position. |⟨φ|ψ⟩|² = probability of measuring state φ when system is in state ψ.',
    },
]


# ---------------------------------------------------------------------------
# "Reading Mathematics" — Language Guide
# This hardcoded guide teaches the meta-skill of interpreting mathematical
# notation and connecting symbols to real-world meaning.
# ---------------------------------------------------------------------------

READING_MATH_GUIDE = """
=== READING AND INTERPRETING MATHEMATICAL LANGUAGE ===

MATHEMATICS AS A LANGUAGE
Mathematics is a precise, universal language — not just a collection of formulas. Like any
language, it has vocabulary (symbols), grammar (rules of combination), syntax (how expressions
are structured), and semantics (what expressions mean). Fluency means reading an equation like
a sentence, immediately extracting its meaning.

SYMBOL-TO-QUANTITY MAPPING
Every symbol in a mathematical expression stands for a quantity, a relationship, or an operation.
When reading a physical equation, always ask: "What does this symbol represent? What are its units?
What range of values makes physical sense?"

Example: F = ma
  F → force [Newtons = kg·m/s²]
  m → mass [kilograms]
  a → acceleration [m/s²]
This isn't just algebra — it says: "the net force acting on an object equals the product of its
mass and its acceleration." Every variable has a physical referent.

THE DIMENSIONAL HOMOGENEITY PRINCIPLE
Every term added or subtracted must have the same units. This is one of the most powerful
error-checking tools in science.
  E = ½mv² + mgh
  Check: ½mv² → kg × (m/s)² = kg·m²/s² = J ✓
         mgh  → kg × m/s² × m = kg·m²/s² = J ✓
  Both terms are in Joules — dimensionally consistent. ✓

If you derive an equation and two terms being added have different units, there's an error.

READING AN EQUATION LEFT TO RIGHT
Standard mathematical reading:
  1. Identify the main equality/relation (=, ≤, →)
  2. Identify left-hand side (usually "what we want to know")
  3. Parse right-hand side using operator precedence
  4. Identify which symbols are variables vs constants vs operators

Operator precedence (order of operations):
  1. Parentheses (innermost first)
  2. Exponents and roots
  3. Multiplication and division (left to right)
  4. Addition and subtraction (left to right)

VARIABLE NAMING CONVENTIONS (informal but widely followed)
  x, y, z       → unknown variables or spatial coordinates
  t             → time
  n, k, m, i, j → integers (counting, indices)
  a, b, c       → constants or coefficients
  f, g, h       → function names
  ε, δ          → small positive quantities (analysis)
  α, β, γ       → angles or generic parameters
  λ             → eigenvalue, wavelength, or rate constant
  μ             → mean, coefficient, or micro- prefix
  σ             → standard deviation or stress
  ρ             → density
  ω             → angular frequency
  Capital letters (A, B, M) → matrices or sets or major constants

Subscripts often denote: specific instances (x₁, x₂), initial/final values (x_i, x_f),
partial components (F_x = x-component of F), or labels (T_air, T_water).

SUPERSCRIPTS
  xⁿ → n-th power of x
  x⁻¹ → reciprocal of x (or inverse in group theory/linear algebra)
  x* → complex conjugate
  x̂ → unit vector or operator (hat notation)
  x̄ → sample mean or complex conjugate (in some notations)
  x' (prime) → derivative, or a transformed/related quantity

FUNCTION NOTATION
  f(x) means: evaluate function f at input x. The output depends on x.
  f: A → B means f is a function from set A to set B.
  f(x,y): function of TWO variables — output depends on both x and y.
  f ∘ g (composition): apply g first, then f. (f ∘ g)(x) = f(g(x)).
  f⁻¹(x): the INVERSE function (not 1/f(x) — context matters!).
    In trig: sin⁻¹(x) = arcsin(x), not 1/sin(x).

READING A CALCULUS EXPRESSION
  dy/dx: instantaneous rate of change of y with respect to x.
    → If y is in meters and x is in seconds: dy/dx is in m/s.
    → "How fast does y change when x increases by a tiny bit?"

  ∫ₐᵇ f(x) dx: sum of f(x)·dx for all infinitesimal strips from a to b.
    → Area under the curve y=f(x) between x=a and x=b.
    → If f(x) is velocity [m/s] and x is time [s]: integral gives displacement [m].
    → The dx tells you WHICH variable you're integrating over.

  ∂f/∂x: partial derivative — rate of change of f with respect to x,
    holding all other variables fixed. Used when f depends on multiple variables.

READING A SUM OR SERIES
  Σ_{n=0}^{∞} xⁿ/n! = eˣ
  Read: "The sum, for n going from 0 to infinity, of x-to-the-n divided by n-factorial, equals e-to-the-x."

  The index n is a dummy variable — it's just counting through the terms:
    n=0: x⁰/0! = 1
    n=1: x¹/1! = x
    n=2: x²/2! = x²/2
    n=3: x³/3! = x³/6   ... and so on forever.

  The physical meaning: e^x can be computed to any precision by summing enough terms.

CONTEXT-DEPENDENT SYMBOLS
The SAME symbol can mean different things in different fields:
  |x|  → absolute value (analysis) vs cardinality of set (set theory) vs determinant (linear algebra)
  ∗    → convolution (signal processing) vs complex conjugate vs multiplication vs pointer (C)
  ∇    → gradient/divergence/curl (vector calculus) vs "gradient descent" in ML
  ×    → multiplication vs cross product vs Cartesian product
  ≡    → identically equal vs congruent vs defined as (≜)
  ~    → similar to vs distributed as vs asymptotically equivalent
  |    → absolute value vs divisibility (a|b means a divides b) vs conditional (P(A|B))
  log  → log₁₀ (engineering) vs ln (pure math) vs log₂ (computer science)

READING PHYSICS EQUATIONS AS PHYSICAL STORIES
  Ohm's Law: V = IR
  Story: voltage (electrical pressure) equals current (flow rate) times resistance (opposition).
  Units: [V] = [A][Ω] = [A][V/A] = [V] ✓

  Newton's Second Law: F = ma → a = F/m
  Story: the harder you push (F↑), the faster it accelerates. The heavier it is (m↑),
  the less it accelerates. The acceleration is directly proportional to force and
  inversely proportional to mass.

  Ideal Gas Law: PV = nRT
  Story: pressure times volume equals moles times gas constant times temperature.
  If you compress gas (V↓) at constant T and n, pressure must rise (P↑) proportionally.

UNDERSTANDING "IFF" AND EQUIVALENCE
  "A if and only if B" (A ↔ B): A and B are equivalent — whenever one is true, the other is true.
  This is much stronger than "if A then B."
  Example: "A function is differentiable if and only if it is locally linear."

  "Necessary vs sufficient conditions":
  A is sufficient for B: A → B (A guarantees B)
  A is necessary for B: B → A (B can't happen without A)
  A is necessary AND sufficient for B: A ↔ B

DIMENSIONAL ANALYSIS AS PROBLEM-SOLVING
  To find the formula for the period of a pendulum T, we know:
    T depends on: length L [m], gravity g [m/s²], mass m [kg]
    T has units: [s]

    Try T = k × Lᵃ × gᵇ × mᶜ
    [s] = [m]ᵃ × [m/s²]ᵇ × [kg]ᶜ
    [s¹] = [m^(a+b)] × [s^(−2b)] × [kg^c]

    Matching: −2b = 1 → b = −½, a+b = 0 → a = ½, c = 0.
    Therefore: T = k√(L/g)  (k = 2π from full derivation)

  This is dimensional analysis: you can derive formula structure from units alone.
"""


# ---------------------------------------------------------------------------
# Wikipedia math articles organized by branch (~190 topics)
# ---------------------------------------------------------------------------

MATH_WIKI_TOPICS = {
    'Foundations & Logic': [
        'Set theory', 'Axiom', 'Mathematical proof', 'Mathematical logic', 'Formal language',
        'Propositional calculus', 'Predicate logic', 'First-order logic', 'Boolean algebra',
        'Model theory', 'Proof theory', 'Gödel\'s incompleteness theorems',
        'Axiom of choice', 'Zermelo–Fraenkel set theory', 'Cardinality', 'Ordinal number',
        'Cardinal number', 'Cantor\'s diagonal argument',
    ],
    'Number Theory': [
        'Number theory', 'Prime number', 'Fundamental theorem of arithmetic',
        'Divisibility', 'Modular arithmetic', 'Euler\'s totient function',
        'Fermat\'s little theorem', 'Chinese remainder theorem', 'Quadratic reciprocity',
        'Diophantine equation', 'Integer', 'Rational number', 'Irrational number',
        'Transcendental number', 'Prime number theorem', 'Riemann hypothesis',
        'Greatest common divisor', 'Least common multiple',
    ],
    'Arithmetic & Pre-Algebra': [
        'Arithmetic', 'Fraction', 'Decimal', 'Percentage', 'Ratio', 'Proportion',
        'Order of operations', 'Exponentiation', 'Logarithm', 'Scientific notation',
        'Significant figures', 'Rounding', 'Absolute value',
    ],
    'Elementary Algebra': [
        'Algebra', 'Variable (mathematics)', 'Equation', 'Linear equation',
        'Quadratic equation', 'Polynomial', 'Factorization', 'System of equations',
        'Inequality (mathematics)', 'Function (mathematics)', 'Inverse function',
        'Composite function', 'Domain of a function', 'Range of a function',
        'Completing the square', 'Quadratic formula', 'Vieta\'s formulas',
    ],
    'Linear Algebra': [
        'Linear algebra', 'Vector space', 'Matrix (mathematics)', 'Matrix multiplication',
        'Determinant', 'Eigenvalues and eigenvectors', 'Linear map', 'Basis (linear algebra)',
        'Dimension (vector space)', 'Rank (linear algebra)', 'Null space',
        'Inner product space', 'Orthogonality', 'Gram–Schmidt process',
        'Singular value decomposition', 'LU decomposition', 'Gaussian elimination',
        'Dot product', 'Cross product', 'Transpose', 'Trace (linear algebra)',
    ],
    'Abstract Algebra': [
        'Abstract algebra', 'Group (mathematics)', 'Ring (mathematics)', 'Field (mathematics)',
        'Group homomorphism', 'Subgroup', 'Normal subgroup', 'Quotient group',
        'Symmetric group', 'Cyclic group', 'Abelian group', 'Ring homomorphism',
        'Ideal (ring theory)', 'Polynomial ring', 'Galois theory', 'Isomorphism theorem',
    ],
    'Geometry': [
        'Euclidean geometry', 'Pythagorean theorem', 'Trigonometry', 'Circle', 'Polygon',
        'Congruence (geometry)', 'Similarity (geometry)', 'Triangle', 'Coordinate geometry',
        'Conic section', 'Ellipse', 'Parabola', 'Hyperbola', 'Area', 'Volume',
        'Non-Euclidean geometry', 'Hyperbolic geometry', 'Elliptic geometry',
        'Projective geometry', 'Affine geometry',
    ],
    'Trigonometry': [
        'Sine and cosine', 'Tangent (trigonometry)', 'Unit circle', 'Radian',
        'Trigonometric identities', 'Pythagorean trigonometric identity',
        'Sum and difference formulas', 'Double angle formula', 'Law of sines',
        'Law of cosines', 'Inverse trigonometric functions', 'Hyperbolic functions',
    ],
    'Calculus': [
        'Calculus', 'Limit (mathematics)', 'Derivative', 'Differentiation rules',
        'Product rule', 'Chain rule', 'Implicit differentiation', 'Mean value theorem',
        'L\'Hôpital\'s rule', 'Taylor series', 'Maclaurin series', 'Integration by parts',
        'Integration by substitution', 'Fundamental theorem of calculus',
        'Riemann integral', 'Improper integral', 'Multivariable calculus',
        'Partial derivative', 'Gradient', 'Directional derivative', 'Total derivative',
        'Jacobian matrix and determinant', 'Hessian matrix', 'Lagrange multiplier',
        'Multiple integral', 'Line integral', 'Surface integral',
        'Green\'s theorem', 'Stokes\' theorem', 'Divergence theorem',
    ],
    'Real Analysis': [
        'Real analysis', 'Real number', 'Completeness of the real numbers', 'Supremum',
        'Infimum', 'Cauchy sequence', 'Metric space', 'Open set', 'Closed set',
        'Compact space', 'Continuous function', 'Uniform continuity',
        'Intermediate value theorem', 'Extreme value theorem', 'Bolzano–Weierstrass theorem',
        'Uniform convergence', 'Power series', 'Radius of convergence',
        'Lebesgue integration', 'Measure theory',
    ],
    'Complex Analysis': [
        'Complex analysis', 'Complex number', 'Euler\'s formula', 'Holomorphic function',
        'Cauchy–Riemann equations', 'Contour integration', 'Residue theorem',
        'Laurent series', 'Analytic continuation', 'Conformal map', 'Riemann surface',
    ],
    'Differential Equations': [
        'Differential equation', 'Ordinary differential equation',
        'Partial differential equation', 'Separable differential equation',
        'Linear differential equation', 'Homogeneous differential equation',
        'Initial value problem', 'Boundary value problem', 'Laplace transform',
        'Fourier series', 'Fourier transform', 'Wave equation', 'Heat equation',
        'Laplace\'s equation', 'Schrödinger equation',
    ],
    'Topology': [
        'Topology', 'Topological space', 'Homeomorphism', 'Continuous function',
        'Connected space', 'Simply connected space', 'Homotopy', 'Fundamental group',
        'Manifold', 'Differential topology', 'Algebraic topology',
        'Euler characteristic', 'Möbius strip', 'Klein bottle', 'Torus',
    ],
    'Differential Geometry': [
        'Differential geometry', 'Curve', 'Curvature', 'Torsion of a curve',
        'Surface (mathematics)', 'Gaussian curvature', 'Riemannian geometry',
        'Metric tensor', 'Geodesic', 'Parallel transport', 'Lie group',
    ],
    'Discrete Mathematics': [
        'Discrete mathematics', 'Graph theory', 'Tree (graph theory)',
        'Planar graph', 'Graph coloring', 'Eulerian path', 'Hamiltonian path',
        'Network flow', 'Combinatorics', 'Permutation', 'Combination',
        'Binomial theorem', 'Pascal\'s triangle', 'Pigeonhole principle',
        'Inclusion–exclusion principle', 'Generating function', 'Recurrence relation',
    ],
    'Probability & Statistics': [
        'Probability theory', 'Random variable', 'Probability distribution',
        'Normal distribution', 'Central limit theorem', 'Law of large numbers',
        'Bayes\' theorem', 'Conditional probability', 'Variance', 'Standard deviation',
        'Covariance', 'Correlation', 'Hypothesis testing', 'Confidence interval',
        'P-value', 'Regression analysis', 'Markov chain', 'Poisson distribution',
        'Binomial distribution', 'Exponential distribution',
    ],
    'Mathematical Physics & Applied': [
        'Dimensional analysis', 'Fourier analysis', 'Laplace transform',
        'Z-transform', 'Numerical analysis', 'Newton\'s method',
        'Runge–Kutta methods', 'Finite element method', 'Optimization (mathematics)',
        'Convex optimization', 'Linear programming', 'Game theory',
        'Information theory', 'Shannon entropy', 'Cryptography',
    ],
    'History & Philosophy': [
        'History of mathematics', 'History of calculus', 'Mathematical notation',
        'Greek mathematics', 'Islamic mathematics', 'Mathematical induction',
        'Proof by contradiction', 'Constructive proof', 'Philosophy of mathematics',
    ],
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    s.mount('https://', HTTPAdapter(max_retries=retry))
    s.mount('http://', HTTPAdapter(max_retries=retry))
    return s


def _train(node: str, text: str, session: requests.Session) -> bool:
    try:
        r = session.post(f'http://{node}/train', json={'text': text}, timeout=30)
        return r.status_code == 200
    except Exception as e:
        print(f'  [WARN] train error: {e}')
        return False


def _wiki_summary(title: str, session: requests.Session) -> str:
    url = f'https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}'
    try:
        r = session.get(url, timeout=15)
        if r.status_code == 200:
            d = r.json()
            return d.get('extract', '')
    except Exception:
        pass
    return ''


def _wiki_content(title: str, session: requests.Session) -> str:
    """Fetch Wikipedia article content via parse API."""
    url = 'https://en.wikipedia.org/w/api.php'
    params = {
        'action': 'query', 'prop': 'extracts', 'exintro': False,
        'explaintext': True, 'titles': title, 'format': 'json',
    }
    try:
        r = session.get(url, params=params, timeout=20)
        if r.status_code == 200:
            pages = r.json().get('query', {}).get('pages', {})
            for page in pages.values():
                text = page.get('extract', '')
                return text[:6000]  # cap per article
    except Exception:
        pass
    return ''


def _load_checkpoint(path: Path) -> set:
    if path.exists():
        return set(json.loads(path.read_text()))
    return set()


def _save_checkpoint(path: Path, done: set):
    path.write_text(json.dumps(sorted(done)))


def _format_symbol(sym: dict) -> str:
    lines = [
        f"MATHEMATICAL SYMBOL: {sym['symbol']}",
        f"LaTeX: {sym.get('latex', '')}",
        f"Name: {sym['name']}",
        f"Domains: {', '.join(sym['domains'])}",
        '',
        f"MEANING: {sym['meaning']}",
        '',
        f"HOW TO READ IT: {sym['how_to_read']}",
        '',
        f"REAL-WORLD INTERPRETATION:",
        f"  {sym['real_world']}",
    ]
    if 'dimensional_note' in sym:
        lines += ['', f"DIMENSIONAL ANALYSIS: {sym['dimensional_note']}"]
    if 'context_note' in sym:
        lines += ['', f"CONTEXT AND VARIANTS: {sym['context_note']}"]
    if 'common_confusion' in sym:
        lines += ['', f"COMMON CONFUSION: {sym['common_confusion']}"]
    if 'example' in sym:
        lines += ['', f"EXAMPLE: {sym['example']}"]
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Stage 34 runner
# ---------------------------------------------------------------------------

def train_stage34(node: str, data_dir: str, args):
    out = Path(data_dir) / 'training' / 'stage34'
    out.mkdir(parents=True, exist_ok=True)
    ckpt_symbols = out / 'checkpoint_symbols.json'
    ckpt_wiki = out / 'checkpoint_wiki.json'
    session = _session()
    trained_symbols = _load_checkpoint(ckpt_symbols)
    trained_wiki = _load_checkpoint(ckpt_wiki)

    # --- Part 1: Mathematical symbol dictionary ---
    print('[Stage 34] Training mathematical symbol dictionary...')
    for sym in MATH_SYMBOLS:
        key = sym['name']
        if key in trained_symbols:
            continue
        text = _format_symbol(sym)
        if _train(node, text, session):
            trained_symbols.add(key)
            print(f'  ✓ Symbol: {sym["symbol"]} ({sym["name"]})')
        time.sleep(0.05)
    _save_checkpoint(ckpt_symbols, trained_symbols)

    # --- Part 2: Reading Mathematics guide ---
    guide_key = '__reading_math_guide__'
    if guide_key not in trained_wiki:
        print('[Stage 34] Training mathematical language guide...')
        if _train(node, READING_MATH_GUIDE, session):
            trained_wiki.add(guide_key)
            print('  ✓ Reading mathematics guide trained.')
        _save_checkpoint(ckpt_wiki, trained_wiki)

    # --- Part 3: Wikipedia mathematics articles ---
    print('[Stage 34] Training Wikipedia mathematics articles...')
    total_topics = sum(len(v) for v in MATH_WIKI_TOPICS.values())
    done = 0
    for branch, topics in MATH_WIKI_TOPICS.items():
        for topic in topics:
            key = f'wiki:{topic}'
            if key in trained_wiki:
                done += 1
                continue
            text = _wiki_content(topic, session)
            if not text:
                text = _wiki_summary(topic, session)
            if text:
                full = f'MATHEMATICS — {branch.upper()}\nTopic: {topic}\n\n{text}'
                if _train(node, full, session):
                    trained_wiki.add(key)
                    done += 1
                    print(f'  [{done}/{total_topics}] ✓ {branch}: {topic}')
            time.sleep(0.15)
        _save_checkpoint(ckpt_wiki, trained_wiki)

    print(f'[Stage 34] Complete. Symbols: {len(trained_symbols)}, Articles: {done}')


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description='Build comprehensive mathematics corpus (Stage 34)')
    p.add_argument('--stages', default='34')
    p.add_argument('--node', default='localhost:8090')
    p.add_argument('--data-dir', default='D:/w1z4rdv1510n-data')
    return p.parse_args()


def main():
    args = _parse_args()
    stages = [int(s.strip()) for s in args.stages.split(',')]
    if 34 in stages:
        train_stage34(args.node, args.data_dir, args)


if __name__ == '__main__':
    main()
