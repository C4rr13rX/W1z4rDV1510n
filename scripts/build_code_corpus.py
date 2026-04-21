#!/usr/bin/env python3
"""
build_code_corpus.py -- Comprehensive multi-language code, terminal, and agent corpus.

Covers (with version metadata and platform awareness):
  JavaScript ES3->ES2024 | TypeScript 1->5 | CSS1->CSS2024 | SCSS
  HTML 1->Living Standard | Angular AngularJS->17 | Ionic v1->v7
  Python 2.7->3.12 | PHP 5->8.3 | C C89->C17 | C++ C++98->C++23
  C# .NET Framework->.NET 8 | Rust 1.0->1.7x | Perl 5 modern
  JSON/XML/XSLT | Applied Math in Code | Statistics | Geometry
  Blender CLI (bpy) | Electronics CLIs (KiCad, ngspice, OpenSCAD, Verilator)
  Linux/Windows Terminal | Git | Docker | Agent decision-making

All answers include: correct imports, version requirements, platform notes, gotchas.

Usage:
    python scripts/build_code_corpus.py --node localhost:8090 [--repeats 25]

Dependencies:
  pip install httpx
"""

import argparse
import asyncio
import sys
import time
import uuid
from pathlib import Path

try:
    import httpx
except ImportError:
    sys.exit("Missing: pip install httpx")

_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir not in __import__("sys").path:
    __import__("sys").path.insert(0, _scripts_dir)
from neuro_client import NeuroClient

DEFAULT_NODE    = "localhost:8090"
DEFAULT_REPEATS = 25

# ---------------------------------------------------------------------------
# Corpus -- (group_title, discipline, [(question, answer)])
# Answers include: exact version where syntax applies, imports, gotchas.
# ---------------------------------------------------------------------------

CORPUS: list[tuple[str, str, list[tuple[str, str]]]] = [

    # ========================================================================
    # JAVASCRIPT -- ES3 through ES2024
    # Runtime: Node.js, browser. No compile step for plain JS.
    # ========================================================================
    ("JavaScript ES3/ES5 Foundations (1999-2009)", "programming", [
        ("JavaScript ES3: How do I declare a function and call it? Runs in any browser or Node.js.",
         "ES3 (1999+) -- works everywhere including IE6+. "
         "Function declaration:\nfunction add(a, b) { return a + b; }\nvar result = add(2, 3); // 5 "
         "Function expression: var multiply = function(a, b) { return a * b; }; "
         "Gotcha: function declarations are hoisted -- you can call them before the definition line. "
         "var is function-scoped, not block-scoped."),
        ("JavaScript ES5: What is 'use strict' and why use it? Platform: any browser or Node.js.",
         "ES5 (2009) -- 'use strict' enables strict mode, which catches common bugs. "
         "Add at top of file or function: 'use strict'; "
         "Strict mode: prevents using undeclared variables, disallows duplicate params, "
         "makes this undefined in non-method functions (not window/global). "
         "In ES6 modules and classes, strict mode is always on -- no need to add it."),
        ("JavaScript ES5: How do I use Array methods forEach, map, filter, reduce?",
         "ES5 (2009) -- all browsers IE9+ and Node.js. No imports needed. "
         "var nums = [1, 2, 3, 4, 5]; "
         "nums.forEach(function(n) { console.log(n); }); // iterate, no return "
         "var doubled = nums.map(function(n) { return n * 2; }); // [2,4,6,8,10] "
         "var evens = nums.filter(function(n) { return n % 2 === 0; }); // [2,4] "
         "var sum = nums.reduce(function(acc, n) { return acc + n; }, 0); // 15 "
         "Gotcha: reduce requires an initial value (second arg) to avoid errors on empty arrays."),
    ]),

    ("JavaScript ES6/ES2015 Features (2015)", "programming", [
        ("JavaScript ES6: How do I use let and const? What is the difference from var?",
         "ES6 (2015) -- Node.js 6+, Chrome 49+, Firefox 44+. "
         "let and const are block-scoped (inside {} braces), not function-scoped like var. "
         "const x = 10; // cannot be reassigned -- but object properties can still change "
         "let y = 5; y = 6; // allowed "
         "Gotcha: const does not make objects immutable -- const obj = {}; obj.key = 'val'; is valid. "
         "Use const by default. Use let when you need to reassign. Avoid var in modern code."),
        ("JavaScript ES6: How do I use arrow functions? What is different about 'this'?",
         "ES6 (2015) -- Node.js 4+, Chrome 45+. "
         "const add = (a, b) => a + b; // single expression, implicit return "
         "const greet = name => `Hello ${name}`; // single param, no parens needed "
         "const log = () => { console.log('hi'); }; // no params, use () "
         "KEY DIFFERENCE: Arrow functions do NOT have their own 'this'. "
         "They inherit 'this' from the enclosing scope. Use regular functions as methods on objects "
         "if you need 'this' to refer to the object. "
         "Gotcha: arrow functions cannot be used as constructors (no new ArrowFn())."),
        ("JavaScript ES6: How do I use template literals (template strings)?",
         "ES6 (2015) -- Use backticks instead of quotes. "
         "const name = 'Alice'; const age = 30; "
         "const msg = `Hello ${name}, you are ${age} years old`; "
         "Multi-line: const html = `<div>\n  <p>${name}</p>\n</div>`; "
         "Expressions: `${2 + 2}` -> '4'. Any JS expression works inside ${}. "
         "Gotcha: backticks are the character ` (below Escape key), not single quotes."),
        ("JavaScript ES6: How do I use destructuring assignment for arrays and objects?",
         "ES6 (2015) -- Node.js 6+. "
         "Array destructuring: const [a, b, c] = [1, 2, 3]; // a=1, b=2, c=3 "
         "Skip elements: const [,, third] = [1, 2, 3]; // third=3 "
         "Object destructuring: const { name, age } = { name: 'Alice', age: 30 }; "
         "Rename: const { name: userName } = obj; // userName = obj.name "
         "Default values: const { x = 10 } = {}; // x=10 if undefined "
         "In function params: function greet({ name, age = 0 }) { return `${name} is ${age}`; }"),
        ("JavaScript ES6: How do I use the spread operator and rest parameters?",
         "ES6 (2015) -- "
         "Spread in array: const combined = [...arr1, ...arr2]; // merge arrays "
         "Spread in function call: Math.max(...nums); // spread array as args "
         "Spread in object (ES2018): const copy = { ...original, newKey: 'val' }; "
         "Rest params: function sum(...nums) { return nums.reduce((a, b) => a + b, 0); } "
         "sum(1, 2, 3, 4) -> 10. Rest must be the last parameter."),
        ("JavaScript ES6: How do I use Promises for async code?",
         "ES6 (2015) -- Node.js 4+. "
         "const p = new Promise((resolve, reject) => { "
         "  setTimeout(() => resolve('done'), 1000); "
         "}); "
         "p.then(result => console.log(result)) "
         " .catch(err => console.error(err)) "
         " .finally(() => console.log('always runs')); "
         "Promise.all([p1, p2]) -- waits for ALL, rejects if any fails. "
         "Promise.race([p1, p2]) -- resolves/rejects with the first to finish. "
         "Gotcha: always add .catch() or unhandled rejections will crash Node.js."),
        ("JavaScript ES6: How do I use ES6 modules (import/export)?",
         "ES6 (2015) -- Node.js 12+ with .mjs or 'type':'module' in package.json. Browsers with type='module'. "
         "Export: export const PI = 3.14159; export function add(a,b) { return a+b; } "
         "Default export: export default class MyClass {} "
         "Import: import { PI, add } from './math.js'; "
         "Import default: import MyClass from './myclass.js'; "
         "Import all: import * as math from './math.js'; math.PI "
         "Gotcha: must use .js extension in import paths for Node.js. Cannot use require() in ES modules."),
    ]),

    ("JavaScript ES2017-ES2024 Modern Features", "programming", [
        ("JavaScript ES2017: How do I use async/await? Requires Promise-based functions.",
         "ES2017 -- Node.js 7.6+, Chrome 55+. "
         "async function fetchData(url) { "
         "  try { "
         "    const response = await fetch(url); "
         "    const data = await response.json(); "
         "    return data; "
         "  } catch (err) { "
         "    console.error('Failed:', err); "
         "    throw err; "
         "  } "
         "} "
         "fetchData('https://api.example.com').then(d => console.log(d)); "
         "Gotcha: async functions always return a Promise. "
         "await only works inside async functions (except top-level await in ES2022 modules). "
         "Always wrap in try/catch or .catch() -- unhandled rejections crash Node.js."),
        ("JavaScript ES2020: How do I use optional chaining (?.) and nullish coalescing (??)?",
         "ES2020 -- Node.js 14+, Chrome 80+. "
         "Optional chaining: const city = user?.address?.city; // undefined if any part is null/undefined "
         "Without it you'd write: user && user.address && user.address.city "
         "Method call: arr?.find(x => x.id === 1) -- safe even if arr is null "
         "Nullish coalescing: const name = input ?? 'default'; "
         "?? returns right side only if left is null or undefined (NOT 0, '', false). "
         "Compare: || returns right side for ANY falsy value -- gotcha for 0 and empty string."),
        ("JavaScript ES2022: How do I use class fields and private properties (#)?",
         "ES2022 -- Node.js 12+, Chrome 74+ (public fields), Chrome 84+ (private). "
         "class Counter { "
         "  #count = 0; // private field -- only accessible inside class "
         "  increment() { this.#count++; } "
         "  get value() { return this.#count; } "
         "} "
         "const c = new Counter(); c.increment(); c.value // 1 "
         "c.#count // SyntaxError -- truly private, not just convention "
         "Static private: static #instances = 0; "
         "Gotcha: # syntax is not the same as the old _ convention. It is enforced by the engine."),
        ("JavaScript ES2024: How do I use Object.groupBy to group array elements?",
         "ES2024 -- Node.js 21+, Chrome 117+. No polyfill needed in modern environments. "
         "const people = [{name:'Alice',dept:'eng'},{name:'Bob',dept:'eng'},{name:'Carol',dept:'hr'}]; "
         "const byDept = Object.groupBy(people, p => p.dept); "
         "// { eng: [{name:'Alice',...},{name:'Bob',...}], hr: [{name:'Carol',...}] } "
         "Also: Map.groupBy(iterable, keyFn) returns a Map instead of plain object. "
         "Polyfill for older environments: use Array.prototype.reduce to build the grouped object manually."),
    ]),

    # ========================================================================
    # TYPESCRIPT -- 1.x through 5.x
    # ========================================================================
    ("TypeScript Basics and Types (TS 2.x+)", "programming", [
        ("TypeScript: How do I set up a TypeScript project from scratch? Node.js environment.",
         "Requires: Node.js 16+, npm. Install: npm install -D typescript "
         "Initialize: npx tsc --init -- creates tsconfig.json "
         "Key tsconfig.json settings: "
         "{ 'target': 'ES2020', 'module': 'commonjs', 'strict': true, 'outDir': './dist' } "
         "Compile: npx tsc -- outputs .js files to dist/. Or: npx tsc --watch for continuous compilation. "
         "Run compiled: node dist/index.js "
         "For modern projects use tsx or ts-node: npm install -D ts-node && npx ts-node index.ts"),
        ("TypeScript: How do I define types, interfaces, and type aliases?",
         "TypeScript 1.0+. "
         "Type alias: type Point = { x: number; y: number }; "
         "Interface: interface User { id: number; name: string; email?: string; } // ? = optional "
         "Difference: interfaces can be extended and merged; type aliases cannot be re-opened. "
         "Use interface for object shapes that may be extended. Use type for unions/intersections. "
         "Union type: type Status = 'active' | 'inactive' | 'banned'; "
         "Intersection: type AdminUser = User & { adminLevel: number };"),
        ("TypeScript: How do I use generics? TS 2.0+.",
         "Generics let functions and classes work with any type while staying type-safe. "
         "function identity<T>(value: T): T { return value; } "
         "const n = identity<number>(42); // n is number "
         "const s = identity('hello'); // TS infers T = string "
         "Generic interface: interface ApiResponse<T> { data: T; status: number; } "
         "Constraints: function first<T extends { length: number }>(arr: T): T { return arr; } "
         "Multiple: function zip<A, B>(a: A[], b: B[]): [A, B][] { return a.map((v,i)=>[v,b[i]]); }"),
        ("TypeScript 4.1+: How do I use template literal types?",
         "TypeScript 4.1 (2020). "
         "type EventName = 'click' | 'focus' | 'blur'; "
         "type Handler = `on${Capitalize<EventName>}`; // 'onClick' | 'onFocus' | 'onBlur' "
         "type CSSProperty = `--${string}`; // any CSS custom property "
         "Useful for strongly typed event names, CSS variables, API paths. "
         "Combined with mapped types for full type-safe routing or i18n key systems."),
        ("TypeScript 4.9+: How do I use the satisfies operator?",
         "TypeScript 4.9 (2022). "
         "satisfies validates a value against a type WITHOUT widening it. "
         "const palette = { red: [255,0,0], green: '#00ff00' } satisfies Record<string, string|number[]>; "
         "Now palette.red is typed as number[] (not string|number[]) -- better inference. "
         "Compare to as: as overrides type checks and loses precision. "
         "satisfies is safer because it still reports type errors."),
        ("TypeScript: What are utility types and how do I use Partial, Required, Pick, Omit, Record?",
         "TypeScript 2.1+. Built-in, no imports needed. "
         "interface User { id: number; name: string; email: string; } "
         "Partial<User> -- all fields optional: { id?: number; name?: string; email?: string } "
         "Required<User> -- all fields required (reverses Partial) "
         "Pick<User, 'id'|'name'> -- subset: { id: number; name: string } "
         "Omit<User, 'email'> -- exclude: { id: number; name: string } "
         "Record<string, number> -- dictionary: { [key: string]: number } "
         "ReadonlyArray<T> -- immutable array. Readonly<User> -- all props readonly."),
    ]),

    # ========================================================================
    # CSS -- CSS1 through CSS2024
    # ========================================================================
    ("CSS Fundamentals and Box Model (CSS1-CSS2.1)", "programming", [
        ("CSS1/CSS2: What is the CSS box model? How do margin, border, padding, and content relate?",
         "Every HTML element is a rectangular box with four layers from inside out: "
         "Content (actual text/image), Padding (space inside border), Border, Margin (space outside). "
         "Default box-sizing is content-box: width/height applies to content only. "
         "padding and border ADD to the rendered size. "
         "Best practice (CSS3): * { box-sizing: border-box; } -- width/height INCLUDES padding and border. "
         "This prevents layout surprises when adding padding to sized elements."),
        ("CSS2: How do I position elements? What are the values of the position property?",
         "CSS2 (1998). "
         "static (default): normal flow. top/left/right/bottom have no effect. "
         "relative: offset from its normal position. Other elements still see it in original place. "
         "absolute: removed from flow. Positioned relative to nearest non-static ancestor. "
         "fixed: removed from flow. Positioned relative to viewport. Stays during scroll. "
         "sticky (CSS3): stays in flow but sticks when scrolled past threshold. "
         "Use z-index to control stack order on positioned elements (position != static)."),
        ("CSS: How do I center an element horizontally and vertically? Modern CSS approach.",
         "For block elements in a container -- CSS Flexbox (Chrome 29+, Firefox 28+, IE11+): "
         ".container { display: flex; justify-content: center; align-items: center; } "
         "For CSS Grid (Chrome 57+, Firefox 52+): "
         ".container { display: grid; place-items: center; } "
         "Old approach (works everywhere): position:absolute; top:50%; left:50%; "
         "transform:translate(-50%,-50%); -- works but requires positioned parent. "
         "For text only: text-align: center; line-height: [container height]; (single line only)"),
    ]),

    ("CSS3 and Modern CSS (Flexbox, Grid, Custom Properties)", "programming", [
        ("CSS Flexbox: How do I create a horizontal navbar with items spaced evenly?",
         "Flexbox (CSS3, 2012 -- Chrome 29+, Firefox 28+, IE11 partial). "
         "nav { display: flex; justify-content: space-between; align-items: center; "
         "  padding: 0 20px; height: 60px; } "
         "justify-content values: flex-start | center | flex-end | space-between | space-around | space-evenly "
         "align-items (cross axis): stretch | center | flex-start | flex-end | baseline "
         "flex-wrap: wrap; -- allows items to wrap to next line. "
         "flex: 1; on a child makes it grow to fill available space."),
        ("CSS Grid: How do I create a responsive 3-column layout that collapses on mobile?",
         "CSS Grid (Chrome 57+, Firefox 52+, Safari 10.1+, IE11 with -ms- prefix partial). "
         ".grid { display: grid; "
         "  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); "
         "  gap: 20px; } "
         "repeat(auto-fit, minmax(250px, 1fr)) creates as many columns as fit, each minimum 250px. "
         "gap sets space between cells (replaces grid-gap which is deprecated). "
         "Named areas: grid-template-areas: 'header header' 'sidebar main' 'footer footer'; "
         "On a child: grid-area: header;"),
        ("CSS Custom Properties (variables): How do I define and use them? Browser support?",
         "CSS Custom Properties / CSS Variables -- Chrome 49+, Firefox 31+, Safari 9.1+. IE11 NOT supported. "
         ":root { --primary-color: #3498db; --font-size-base: 16px; --spacing: 8px; } "
         ".button { background: var(--primary-color); padding: var(--spacing); } "
         "With fallback: color: var(--text, #333); -- uses #333 if --text is undefined. "
         "Change in JS: document.documentElement.style.setProperty('--primary-color', '#e74c3c'); "
         "Override in component scope: .dark-theme { --primary-color: #2c3e50; }"),
        ("CSS: How do I write responsive styles with media queries?",
         "CSS2 introduced media types; CSS3 adds media features. All modern browsers. "
         "@media (max-width: 768px) { .sidebar { display: none; } } "
         "Mobile-first approach (recommended): start with mobile styles, add desktop with min-width: "
         ".container { width: 100%; } "
         "@media (min-width: 768px) { .container { max-width: 1200px; margin: 0 auto; } } "
         "Modern: container queries (Chrome 105+, Firefox 110+): "
         "@container sidebar (min-width: 400px) { .card { font-size: 1.2rem; } }"),
        ("CSS: How do I use CSS animations and transitions?",
         "CSS3 Transitions (Chrome 26+, Firefox 16+, all modern): "
         ".button { background: blue; transition: background 0.3s ease, transform 0.2s; } "
         ".button:hover { background: darkblue; transform: scale(1.05); } "
         "CSS Animations: @keyframes slideIn { from { opacity:0; transform:translateY(-20px); } "
         "to { opacity:1; transform:translateY(0); } } "
         ".element { animation: slideIn 0.4s ease forwards; } "
         "animation shorthand: name duration timing-function delay iteration-count direction fill-mode. "
         "Gotcha: use transform and opacity for smooth 60fps -- avoid animating width/height/top/left."),
        ("CSS Nesting: How do I use native CSS nesting? When is it supported?",
         "Native CSS Nesting -- Chrome 112+, Firefox 117+, Safari 16.5+ (2023). "
         ".card { color: #333; background: white; "
         "  & .title { font-size: 1.5rem; } /* & references parent (.card) */ "
         "  &:hover { background: #f5f5f5; } "
         "  @media (max-width: 600px) { padding: 10px; } "
         "} "
         "The & is required when nesting element selectors directly inside class selectors. "
         "For older browsers, use SCSS/PostCSS nesting plugin to compile to flat CSS."),
    ]),

    # ========================================================================
    # SCSS -- requires sass compiler
    # ========================================================================
    ("SCSS: Sass Preprocessing", "programming", [
        ("SCSS: How do I set up and compile SCSS? What tools do I need?",
         "Install Dart Sass (the official, maintained implementation -- LibSass is deprecated): "
         "npm install -D sass "
         "Compile once: npx sass input.scss output.css "
         "Watch mode: npx sass --watch src/styles:dist/styles "
         "With source maps (for DevTools debugging): npx sass --source-map input.scss output.css "
         "Gotcha: do NOT use node-sass (deprecated, uses LibSass). Use 'sass' package (Dart Sass)."),
        ("SCSS: How do I use variables, nesting, and mixins?",
         "SCSS (Sass 3+ syntax, .scss files -- CSS-compatible syntax). "
         "$primary: #3498db; $spacing: 8px; // variables "
         ".card { padding: $spacing * 2; // arithmetic "
         "  .title { color: $primary; font-size: 1.5rem; } // nesting "
         "  &:hover { background: lighten($primary, 10%); } // & = parent "
         "} "
         "@mixin flex-center { display:flex; justify-content:center; align-items:center; } "
         ".hero { @include flex-center; height: 100vh; } // use mixin "
         "@mixin respond-to($bp) { @if $bp == 'mobile' { @media (max-width:768px) { @content; } } }"),
        ("SCSS: How do I use @use and @forward instead of @import?",
         "Dart Sass 1.23+ (2019). @import is deprecated -- use @use and @forward. "
         "// _variables.scss (partials start with _) "
         "$color: #333; "
         "// main.scss: "
         "@use './variables' as vars; // namespace: vars.$color "
         "@use './variables'; // namespace: variables.$color "
         "@use './variables' as *; // no namespace -- use $color directly (careful of conflicts) "
         "@forward './variables'; // re-export for downstream @use "
         "Gotcha: each file can only @use a module once. Variables are scoped to the module."),
    ]),

    # ========================================================================
    # HTML -- since inception through HTML5 Living Standard
    # ========================================================================
    ("HTML: History and Fundamentals", "programming", [
        ("HTML: What is the correct DOCTYPE for HTML5? How does it differ from HTML4?",
         "HTML5 (2014) DOCTYPE: <!DOCTYPE html> -- simple, case-insensitive. "
         "HTML 4.01 Strict DOCTYPE was complex: "
         "<!DOCTYPE HTML PUBLIC '-//W3C//DTD HTML 4.01//EN' 'http://www.w3.org/TR/html4/strict.dtd'> "
         "XHTML 1.0 Strict: <!DOCTYPE html PUBLIC '-//W3C//DTD XHTML 1.0 Strict//EN' '...'> "
         "The HTML5 DOCTYPE is just a browser trigger to enable standards mode -- it doesn't reference a DTD. "
         "Always use <!DOCTYPE html> for new projects."),
        ("HTML5: What are semantic elements and why use them?",
         "HTML5 (2014) -- semantic elements describe meaning, not appearance. "
         "<header> -- page or section header "
         "<nav> -- navigation links "
         "<main> -- main content (only one per page) "
         "<article> -- independent, self-contained content "
         "<section> -- thematic grouping with heading "
         "<aside> -- sidebar / tangentially related content "
         "<footer> -- footer for page or section "
         "<figure> and <figcaption> -- media with caption "
         "Benefits: better accessibility (screen readers), SEO, maintainability. "
         "Replace generic <div>/<span> with semantic elements where meaning applies."),
        ("HTML5: How do I use the data- attribute for custom data?",
         "HTML5 -- all modern browsers. "
         "<div id='user' data-user-id='42' data-role='admin'>Alice</div> "
         "Access in JS: "
         "const el = document.getElementById('user'); "
         "el.dataset.userId // '42' (camelCase: data-user-id -> userId) "
         "el.dataset.role // 'admin' "
         "Set: el.dataset.newProp = 'value'; adds data-new-prop attribute. "
         "Useful for passing server-rendered data to JavaScript without hidden inputs."),
        ("HTML5: How do I use the canvas element for 2D drawing?",
         "HTML5 -- Chrome 4+, Firefox 3.6+, Safari 3.1+. "
         "<canvas id='c' width='400' height='300'></canvas> "
         "JS: const ctx = document.getElementById('c').getContext('2d'); "
         "ctx.fillStyle = '#3498db'; ctx.fillRect(10, 10, 100, 50); // x,y,w,h "
         "ctx.strokeStyle = 'black'; ctx.lineWidth = 2; "
         "ctx.beginPath(); ctx.arc(200, 150, 40, 0, Math.PI*2); ctx.stroke(); // circle "
         "ctx.clearRect(0, 0, 400, 300); // clear entire canvas "
         "Gotcha: canvas width/height attributes (NOT CSS width/height) set the pixel resolution."),
        ("HTML: How do I make forms accessible with proper labels and ARIA?",
         "HTML5 + WAI-ARIA. "
         "<form> "
         "  <label for='email'>Email address</label> "
         "  <input type='email' id='email' name='email' required "
         "         aria-describedby='email-hint' autocomplete='email'> "
         "  <span id='email-hint'>We'll never share your email.</span> "
         "</form> "
         "for= on <label> must match id= on <input>. This links them for screen readers. "
         "Use type='email', type='tel', type='number' for proper mobile keyboards. "
         "aria-required='true' for screen readers. required attribute for browser validation."),
    ]),

    # ========================================================================
    # ANGULAR -- AngularJS 1.x through Angular 17
    # ========================================================================
    ("Angular: AngularJS 1.x vs Modern Angular", "programming", [
        ("AngularJS 1.x vs Angular 2+: What are the major differences?",
         "AngularJS 1.x (2010-2021, EOL): MVC architecture, $scope, two-way binding with ng-model, "
         "$http service, directives with link functions, JavaScript only. "
         "Angular 2+ (2016-present): Component-based, TypeScript required, no $scope, "
         "RxJS Observables for HTTP (HttpClient), decorators (@Component, @Injectable), "
         "NgModules (Angular 2-16) or Standalone Components (Angular 14+). "
         "AngularJS apps CANNOT be incrementally upgraded to Angular -- requires rewrite. "
         "AngularJS reached EOL December 2021. Do not start new projects with AngularJS."),
        ("Angular 14+: How do I create a standalone component? No NgModule needed.",
         "Angular 14+ (standalone preview), Angular 15+ (stable). "
         "ng new myapp --standalone (Angular 17+ default) "
         "import { Component } from '@angular/core'; "
         "import { NgIf, NgFor } from '@angular/common'; "
         "@Component({ "
         "  selector: 'app-hello', "
         "  standalone: true, "
         "  imports: [NgIf, NgFor], // import directly, not via NgModule "
         "  template: `<h1 *ngIf=\"show\">Hello {{name}}</h1>`, "
         "}) "
         "export class HelloComponent { name = 'World'; show = true; }"),
        ("Angular 17: How do I use the new @if and @for template control flow?",
         "Angular 17 (2023) -- new built-in control flow replaces *ngIf and *ngFor directives. "
         "@if (isLoggedIn) { <app-dashboard /> } @else { <app-login /> } "
         "@for (item of items; track item.id) { "
         "  <div>{{ item.name }}</div> "
         "} @empty { <p>No items found</p> } "
         "track is required and must be a unique identifier -- replaces trackBy. "
         "The old *ngIf and *ngFor still work but are considered legacy. "
         "No import needed -- built into the template compiler."),
        ("Angular: How do I make HTTP requests with HttpClient?",
         "Angular 4.3+ (HttpClient replaces old Http). "
         "In standalone app, provide in main.ts: provideHttpClient() in providers. "
         "In NgModule: import HttpClientModule in AppModule imports. "
         "import { HttpClient } from '@angular/common/http'; "
         "@Injectable({ providedIn: 'root' }) "
         "export class DataService { "
         "  constructor(private http: HttpClient) {} "
         "  getData(): Observable<any[]> { return this.http.get<any[]>('/api/items'); } "
         "} "
         "Subscribe: this.dataService.getData().subscribe(items => this.items = items); "
         "Gotcha: HttpClient returns cold Observables -- must subscribe or use async pipe."),
        ("Angular: How do I use reactive forms for validation?",
         "Angular 2+ Reactive Forms. Import ReactiveFormsModule (NgModule) or ReactiveFormsModule (standalone). "
         "import { FormBuilder, Validators } from '@angular/forms'; "
         "constructor(private fb: FormBuilder) {} "
         "form = this.fb.group({ "
         "  email: ['', [Validators.required, Validators.email]], "
         "  password: ['', [Validators.required, Validators.minLength(8)]], "
         "}); "
         "In template: <form [formGroup]='form' (ngSubmit)='onSubmit()'> "
         "<input formControlName='email'> "
         "<span *ngIf='form.get(\"email\")?.invalid && form.get(\"email\")?.touched'>Invalid email</span> "
         "onSubmit() { if (this.form.valid) { console.log(this.form.value); } }"),
    ]),

    # ========================================================================
    # IONIC -- v1 through v7
    # ========================================================================
    ("Ionic Framework: v1 through v7", "programming", [
        ("Ionic 1 vs Ionic 2+: What changed? Should I upgrade?",
         "Ionic 1 (2013-2017): AngularJS-based, Cordova only, ng-controller, "
         "$ionicModal, $ionicPopup. Uses bower, gulp. "
         "Ionic 2+ (2017-present): Angular/React/Vue or framework-agnostic web components, "
         "Capacitor (recommended) or Cordova. TypeScript default. "
         "Ionic 4+ (2019): Framework-agnostic -- works with Angular, React, Vue, or plain JS. "
         "Ionic 7 (2023): Angular 16+, Capacitor 5, improved theming. "
         "Ionic 1 reached EOL. Do not start new projects with Ionic 1."),
        ("Ionic + Angular: How do I create a new Ionic Angular project?",
         "Requires: Node.js 16+, npm. "
         "npm install -g @ionic/cli "
         "ionic start myapp tabs --type=angular --capacitor "
         "cd myapp && ionic serve // run in browser "
         "To build for iOS: ionic cap add ios && ionic cap sync && ionic cap open ios "
         "To build for Android: ionic cap add android && ionic cap sync && ionic cap open android "
         "Gotcha: ionic serve uses Vite under the hood (Ionic 7). First install is slow."),
        ("Ionic 4+: How do I use ion-list, ion-item, and ion-card components?",
         "Ionic 4+ web components -- same markup for all frameworks. "
         "<ion-list> "
         "  <ion-item button (click)='onClick(item)' *ngFor='let item of items'> "
         "    <ion-label>{{ item.title }}</ion-label> "
         "    <ion-badge slot='end'>{{ item.count }}</ion-badge> "
         "  </ion-item> "
         "</ion-list> "
         "<ion-card> "
         "  <ion-card-header><ion-card-title>Title</ion-card-title></ion-card-header> "
         "  <ion-card-content>Content here</ion-card-content> "
         "</ion-card> "
         "slot='end' positions content at the trailing edge of ion-item."),
    ]),

    # ========================================================================
    # PYTHON -- 2.7 vs 3.x, Python 3.6 through 3.12
    # ========================================================================
    ("Python: Version Differences 2.7 vs 3.x", "programming", [
        ("Python 2.7 vs Python 3: What are the breaking changes I must know?",
         "Python 3 (released 2008, 2.7 EOL January 2020). Do not use Python 2 for new code. "
         "print: Python 2: print 'hello' -> Python 3: print('hello') "
         "Integer division: Python 2: 7/2 = 3 -> Python 3: 7/2 = 3.5, use 7//2 for floor div "
         "Unicode: Python 2: str is bytes, unicode is text -> Python 3: str is always Unicode "
         "xrange: Python 2: xrange(n) -> Python 3: range(n) (range is lazy in Python 3) "
         "dict.keys()/values()/items(): Python 2 returns lists -> Python 3 returns views "
         "Exception syntax: except Exception, e -> except Exception as e "
         "Check version in script: import sys; assert sys.version_info >= (3, 8), 'Python 3.8+ required'"),
    ]),

    ("Python Modern Features (3.6-3.12)", "programming", [
        ("Python 3.7+: How do I use dataclasses? What do they replace?",
         "Python 3.7+ -- from dataclasses import dataclass, field "
         "@dataclass "
         "class Point: "
         "    x: float "
         "    y: float "
         "    label: str = 'origin' // default value "
         "    history: list = field(default_factory=list) // mutable default -- MUST use field() "
         "p = Point(1.0, 2.0) // __init__ auto-generated "
         "print(p) // __repr__ auto-generated: Point(x=1.0, y=2.0, label='origin', history=[]) "
         "@dataclass(frozen=True) makes instances immutable (hashable). "
         "Gotcha: NEVER use mutable defaults like list or dict directly -- use field(default_factory=list)."),
        ("Python 3.8+: How do I use the walrus operator (:=)?",
         "Python 3.8+ -- assignment expression. "
         "while chunk := f.read(8192): process(chunk) // read and check in one line "
         "if m := re.search(r'(\\d+)', text): print(m.group(1)) // match and use "
         "Useful in list comprehensions: "
         "results = [y for x in data if (y := transform(x)) is not None] "
         "Gotcha: walrus assigns to the enclosing scope, not just the expression. "
         "Use sparingly -- it can make code harder to read."),
        ("Python 3.10+: How do I use structural pattern matching (match/case)?",
         "Python 3.10+ -- match statement, similar to switch but more powerful. "
         "match command.split(): "
         "    case ['quit']: sys.exit() "
         "    case ['go', direction]: move(direction) "
         "    case ['get', item, *rest]: pick_up(item, rest) "
         "    case _: print('Unknown command') "
         "Match on type: match event: case MouseClick(x=x, y=y): handle_click(x,y) "
         "Match with guard: case n if n > 0: positive() "
         "Python match is NOT a jump table -- it does structural pattern matching, not equality."),
        ("Python 3.11+: What are ExceptionGroup and except*?",
         "Python 3.11+ -- handle multiple concurrent exceptions (used with asyncio.TaskGroup). "
         "try: "
         "    async with asyncio.TaskGroup() as tg: "
         "        tg.create_task(failing_coro_1()) "
         "        tg.create_task(failing_coro_2()) "
         "except* ValueError as eg: "
         "    for exc in eg.exceptions: print(f'ValueError: {exc}') "
         "except* TypeError as eg: "
         "    for exc in eg.exceptions: print(f'TypeError: {exc}') "
         "ExceptionGroup wraps multiple exceptions raised together. except* handles by type."),
    ]),

    # ========================================================================
    # PHP -- 5.x, 7.x, 8.x
    # ========================================================================
    ("PHP: Version-Specific Features (PHP 7-8.3)", "programming", [
        ("PHP 7 vs PHP 8: What are the major improvements to know?",
         "PHP 7 (2015): scalar type hints (int, string, float, bool), "
         "return type declarations, null coalescing ?? operator, "
         "spaceship operator <=>, anonymous classes, PDO/MySQLi only (mysql_ removed). "
         "PHP 8.0 (2020): named arguments, union types (int|string), match expression, "
         "nullsafe operator ?->, constructor property promotion, JIT compilation. "
         "PHP 8.1 (2021): enums, fibers, readonly properties, intersection types (A&B), "
         "never return type. "
         "PHP 8.2 (2022): readonly classes, true/false/null standalone types. "
         "PHP 8.3 (2023): typed class constants, json_validate() function."),
        ("PHP 8.0+: How do I use named arguments and constructor property promotion?",
         "PHP 8.0+. "
         "Named arguments: array_slice(array: $arr, offset: 2, length: 3, preserve_keys: true); "
         "Order doesn't matter. Skip optional params: htmlspecialchars(string: $s, double_encode: false); "
         "Constructor property promotion: "
         "class User { "
         "  public function __construct( "
         "    public readonly int $id, "
         "    public string $name, "
         "    private string $email = '', "
         "  ) {} // params become properties automatically "
         "} "
         "$u = new User(id: 1, name: 'Alice', email: 'a@b.com');"),
        ("PHP 8.1+: How do I use enums?",
         "PHP 8.1+. "
         "enum Status { case Active; case Inactive; case Banned; } "
         "$s = Status::Active; "
         "$s === Status::Active // true "
         "Backed enum (with values): "
         "enum Color: string { case Red = 'red'; case Blue = 'blue'; } "
         "Color::Red->value // 'red' "
         "Color::from('red') // Color::Red "
         "Color::tryFrom('invalid') // null (doesn't throw) "
         "Enums can implement interfaces and have methods. Cannot be instantiated with new."),
        ("PHP: How do I connect to a database safely using PDO?",
         "PHP 5.1+ (PDO), PHP 7+ (mysql_ functions removed). Always use PDO or MySQLi. "
         "$pdo = new PDO('mysql:host=localhost;dbname=mydb;charset=utf8mb4', "
         "               'user', 'password', "
         "               [PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION, "
         "                PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC]); "
         "NEVER interpolate user input into SQL. Use prepared statements: "
         "$stmt = $pdo->prepare('SELECT * FROM users WHERE email = :email'); "
         "$stmt->execute([':email' => $email]); "
         "$user = $stmt->fetch(); // returns associative array or false "
         "Gotcha: charset=utf8mb4 in DSN (NOT utf8) to support emoji and all Unicode."),
        ("PHP: How do I handle errors and exceptions in PHP 8?",
         "PHP 8 -- use try/catch/finally. "
         "try { "
         "  $result = riskyOperation(); "
         "} catch (InvalidArgumentException $e) { "
         "  error_log($e->getMessage()); "
         "  throw new RuntimeException('Failed: ' . $e->getMessage(), 0, $e); "
         "} catch (Throwable $e) { // catches ALL errors and exceptions "
         "  error_log('Unexpected: ' . $e->getMessage()); "
         "} finally { "
         "  cleanUp(); // always runs "
         "} "
         "Set error handler: set_error_handler(function($errno, $errstr) { throw new ErrorException($errstr, $errno); }); "
         "In PHP 8, most built-in errors throw exceptions. Throwable catches both Error and Exception."),
    ]),

    # ========================================================================
    # C -- C89 through C17
    # Platform: Linux gcc/clang, Windows MSVC/MinGW
    # ========================================================================
    ("C Programming: C89 through C17", "systems_programming", [
        ("C: How do I compile a C program on Linux? On Windows? Specify C standard.",
         "Linux with gcc (install: sudo apt install build-essential): "
         "gcc -std=c11 -Wall -Wextra -o myprogram main.c "
         "-std=c11 (or c99, c89, c17). -Wall enables most warnings. -Wextra adds more. "
         "Linux with clang: clang -std=c11 -Wall -o myprogram main.c "
         "Windows with MSVC (Visual Studio Developer Command Prompt): "
         "cl /std:c11 /W4 main.c /Fe:myprogram.exe "
         "Windows with MinGW/GCC: gcc -std=c11 -Wall -o myprogram.exe main.c "
         "Gotcha: MSVC supports C99 and C11 features but NOT C11 threads or some C99 VLAs."),
        ("C99: What new features does C99 add over C89/ANSI C?",
         "C99 (1999) adds over C89: "
         "1. // single-line comments (was C++ only in C89) "
         "2. Variable-length arrays (VLAs): int arr[n]; where n is runtime value "
         "3. Designated initializers: int arr[5] = {[2]=10, [4]=20}; "
         "4. Compound literals: (struct Point){.x=1, .y=2} "
         "5. for loop variable declaration: for (int i=0; i<n; i++) "
         "6. stdint.h: int32_t, uint8_t, int64_t (exact-width integers) "
         "7. stdbool.h: bool, true, false "
         "8. Flexible array members in structs "
         "Compile: gcc -std=c99"),
        ("C: How do I allocate and free heap memory safely?",
         "#include <stdlib.h>  // malloc, calloc, realloc, free "
         "#include <string.h>  // memset, memcpy "
         "int *arr = malloc(100 * sizeof(int)); "
         "if (!arr) { fprintf(stderr, 'malloc failed\\n'); exit(1); } // ALWAYS check NULL "
         "memset(arr, 0, 100 * sizeof(int)); // zero-initialize (or use calloc) "
         "int *arr2 = calloc(100, sizeof(int)); // allocate and zero-initialize "
         "arr = realloc(arr, 200 * sizeof(int)); // resize -- returns NULL on failure "
         "// Gotcha: if realloc returns NULL, original pointer is NOT freed "
         "free(arr); arr = NULL; // free and null the pointer to prevent double-free "
         "On Windows, same code works -- malloc/free are standard C."),
        ("C11: What is _Generic and how do I use it for type-generic macros?",
         "C11 (2011). _Generic is a compile-time type selection. "
         "#define ABS(x) _Generic((x), \\ "
         "  int: abs, \\ "
         "  long: labs, \\ "
         "  float: fabsf, \\ "
         "  double: fabs)(x) "
         "#include <stdlib.h> // abs, labs "
         "#include <math.h>   // fabs, fabsf "
         "ABS(-3)   // calls abs() "
         "ABS(-3.0) // calls fabs() "
         "C11 also adds _Atomic, _Static_assert, aligned_alloc. "
         "Compile: gcc -std=c11"),
    ]),

    # ========================================================================
    # C++ -- C++98 through C++23 -- PLATFORM AWARE
    # ========================================================================
    ("C++: Compilation and Platform Differences", "systems_programming", [
        ("C++: How do I compile a C++ program on Linux, Mac, and Windows? What compiler to use?",
         "Linux -- gcc or clang (install: sudo apt install build-essential): "
         "g++ -std=c++17 -Wall -Wextra -O2 -o myapp main.cpp "
         "macOS -- Xcode Command Line Tools (xcode-select --install): "
         "clang++ -std=c++17 -Wall -O2 -o myapp main.cpp "
         "Note: macOS 'g++' is actually clang. True GCC available via Homebrew: brew install gcc "
         "Windows MSVC (Visual Studio 2022, latest supports C++23): "
         "cl /std:c++17 /W4 /O2 main.cpp /Fe:myapp.exe "
         "Windows MinGW-w64 (like GCC on Linux): g++ -std=c++17 -Wall -O2 -o myapp.exe main.cpp "
         "Windows clang (via LLVM installer): clang++ -std=c++17 -o myapp.exe main.cpp"),
        ("C++: Which headers are platform-specific and which are portable?",
         "Portable (ISO C++ standard library -- works on all platforms): "
         "#include <iostream>, <vector>, <string>, <map>, <unordered_map>, "
         "#include <algorithm>, <memory>, <thread>, <mutex>, <filesystem> (C++17) "
         "Linux/Mac ONLY (POSIX): "
         "#include <unistd.h> // fork, exec, pipe, read, write, close "
         "#include <sys/socket.h> // POSIX sockets "
         "#include <pthread.h> // pthreads (though C++11 std::thread is portable) "
         "Windows ONLY: "
         "#include <windows.h> // Win32 API, HANDLE, CreateThread, WaitForSingleObject "
         "#include <winsock2.h> // Windows sockets "
         "GOTCHA: <windows.h> defines min() and max() macros that conflict with std::min/max. "
         "Fix: #define NOMINMAX before including <windows.h>, or call (std::min)(a,b)."),
        ("C++: How does filesystem differ on Windows vs Linux/Mac?",
         "Path separator: Linux/Mac use /, Windows uses \\ (but / also works in most Win APIs). "
         "C++17 std::filesystem is portable: "
         "#include <filesystem> // g++ -std=c++17 or MSVC /std:c++17 "
         "namespace fs = std::filesystem; "
         "fs::path p = fs::current_path() / 'data' / 'file.txt'; // portable "
         "fs::exists(p), fs::create_directories(p), fs::remove(p) "
         "On Windows, std::filesystem paths use wchar_t internally. "
         "Gotcha: on macOS 10.14 (Mojave), std::filesystem requires macOS 10.15+ at runtime. "
         "Use MACOSX_DEPLOYMENT_TARGET=10.15 when building."),
    ]),

    ("C++11 through C++17 Features", "systems_programming", [
        ("C++11: How do I use auto, range-for, and lambda expressions?",
         "C++11 (gcc 4.8+, clang 3.3+, MSVC 2015+). "
         "auto x = 42; // compiler deduces type: int "
         "auto v = std::vector<int>{1,2,3}; "
         "Range-for: for (auto& item : container) { item *= 2; } "
         "Use & to avoid copies. Use const auto& for read-only. "
         "Lambda: auto add = [](int a, int b) { return a + b; }; add(2,3) // 5 "
         "Capture by reference: [&] captures all local vars by reference. "
         "[=] captures by copy. [x, &y] captures x by copy, y by reference."),
        ("C++11: How do I use unique_ptr and shared_ptr for memory management?",
         "C++11 -- #include <memory> "
         "unique_ptr: sole owner, zero overhead. "
         "auto p = std::make_unique<MyClass>(args); // prefer make_unique over new "
         "p->method(); // access like raw pointer "
         "// Automatically freed when p goes out of scope -- no delete needed "
         "std::unique_ptr<MyClass> q = std::move(p); // transfer ownership -- p is now null "
         "shared_ptr: reference-counted, multiple owners. "
         "auto sp = std::make_shared<MyClass>(args); // prefer make_shared "
         "auto sp2 = sp; // both own the object; freed when last shared_ptr is destroyed "
         "Gotcha: don't create two shared_ptrs from the same raw pointer -- use make_shared."),
        ("C++17: How do I use structured bindings and std::optional?",
         "C++17 (gcc 7+, clang 5+, MSVC 2017+). "
         "Structured bindings: "
         "auto [x, y] = std::make_pair(1.0, 2.0); "
         "for (auto& [key, val] : mymap) { std::cout << key << ':' << val; } "
         "std::optional<T>: "
         "#include <optional> "
         "std::optional<int> find_user(int id) { "
         "  if (id == 1) return 42; "
         "  return std::nullopt; // no value "
         "} "
         "auto result = find_user(1); "
         "if (result) std::cout << *result; // dereference "
         "result.value_or(0) // returns 0 if nullopt"),
        ("C++20: How do I use concepts to constrain templates?",
         "C++20 (gcc 10+, clang 10+, MSVC 2019 16.3+). "
         "#include <concepts> "
         "template<typename T> "
         "concept Numeric = std::integral<T> || std::floating_point<T>; "
         "template<Numeric T> "
         "T square(T x) { return x * x; } "
         "square(3); // ok "
         "square(\"hi\"); // compile error: 'hi' doesn't satisfy Numeric "
         "Using requires clause: template<typename T> requires std::copyable<T> "
         "Also: abbreviated templates: auto add(std::integral auto a, std::integral auto b) { return a+b; }"),
        ("C++20: How do I use std::format for string formatting?",
         "C++20 (gcc 13+, clang 14+, MSVC 2019 16.10+). "
         "#include <format> "
         "std::string s = std::format('Hello, {}! You are {} years old.', name, age); "
         "With spec: std::format('{:.2f}', 3.14159) -> '3.14' "
         "Padding: std::format('{:>10}', 'hi') -> '        hi' "
         "Hex: std::format('{:#x}', 255) -> '0xff' "
         "If C++20 format not available: use fmt library (same API): "
         "#include <fmt/format.h> and link -lfmt. "
         "Gotcha: not all MSVC versions support <format> -- check with _MSVC_LANG >= 202002L."),
        ("C++23: How do I use std::expected for error handling without exceptions?",
         "C++23 (gcc 12+, clang 16+, MSVC 2022 17.3+). "
         "#include <expected> "
         "std::expected<int, std::string> parse_int(std::string_view s) { "
         "  try { return std::stoi(std::string(s)); } "
         "  catch (...) { return std::unexpected('Not a number: ' + std::string(s)); } "
         "} "
         "auto result = parse_int('42'); "
         "if (result) std::cout << *result; // 42 "
         "else std::cerr << result.error(); // error string "
         "result.value_or(0) -- returns default on error. "
         "Preferred over exceptions for performance-sensitive or no-exception code."),
    ]),

    # ========================================================================
    # C# -- .NET Framework through .NET 8
    # ========================================================================
    ("C#: .NET Framework to .NET 8", "programming", [
        ("C#: .NET Framework vs .NET Core vs .NET 5+: Which should I use?",
         ".NET Framework (2002-present): Windows-only, built into Windows. "
         "Use for maintaining existing Windows desktop apps (WinForms, WPF) or ASP.NET Web Forms. "
         ".NET Core (2016-2020): cross-platform, open source. .NET Core 3.1 is LTS (EOL 2022). "
         ".NET 5+ (2020-present): unified platform, cross-platform, LTS versions: .NET 6, .NET 8. "
         "Current recommendation: .NET 8 (LTS, 2023) for new projects. "
         "Install: https://dotnet.microsoft.com or winget install Microsoft.DotNet.SDK.8 "
         "Create project: dotnet new console -n MyApp && cd MyApp && dotnet run"),
        ("C# 9+: How do I use records for immutable data types?",
         ".NET 5+ (C# 9, 2020). "
         "record Point(double X, double Y); // positional record -- immutable "
         "var p = new Point(1.0, 2.0); "
         "p.X // 1.0 -- getter-only "
         "var p2 = p with { X = 3.0 }; // non-destructive mutation -- creates new record "
         "p == p2 // false -- structural equality built in "
         "record class vs record struct (C# 10): "
         "record struct Point(double X, double Y); // value type, mutable by default "
         "readonly record struct Point(double X, double Y); // immutable value type"),
        ("C# 8+: How do I use nullable reference types?",
         "C# 8+ (.NET Core 3.0+). Enable in .csproj: <Nullable>enable</Nullable> "
         "With enabled, string is non-nullable by default: "
         "string name = null; // Warning: cannot assign null to non-nullable string "
         "string? name = null; // OK -- explicitly nullable "
         "void Greet(string? name) { "
         "  if (name is not null) Console.WriteLine(name.ToUpper()); // safe "
         "  Console.WriteLine(name!.ToUpper()); // ! suppresses warning -- only if sure "
         "} "
         "Gotcha: enabling nullable on existing code generates many warnings. "
         "Use #nullable enable/disable to enable incrementally per file."),
        ("C# 10+: How do I use global usings and file-scoped namespaces?",
         "C# 10+ (.NET 6+). "
         "File-scoped namespace (no braces): "
         "namespace MyApp.Models; // replaces namespace MyApp.Models { ... } "
         "Applies to entire file -- no extra indentation needed. "
         "Global usings (in a single file, e.g., GlobalUsings.cs): "
         "global using System; "
         "global using System.Collections.Generic; "
         "global using Microsoft.AspNetCore.Mvc; "
         "These apply to the entire project -- no need to repeat using statements in every file. "
         ".NET 6+ templates include global usings by default in the .csproj with <ImplicitUsings>enable</ImplicitUsings>."),
        ("C# 12 (.NET 8): How do I use primary constructors for classes?",
         "C# 12 (.NET 8, 2023). Primary constructors for non-record classes and structs. "
         "class Service(ILogger logger, string connectionString) { "
         "  public void Connect() => logger.Log($'Connecting to {connectionString}'); "
         "} "
         "Parameters are in scope throughout the class body. "
         "Gotcha: unlike record parameters, primary constructor params are NOT automatically properties. "
         "They are captured variables. To make a property: "
         "class User(string name) { public string Name { get; } = name; }"),
    ]),

    # ========================================================================
    # RUST -- 1.0 through 1.7x (2024)
    # ========================================================================
    ("Rust: Ownership, Borrowing, and Lifetimes", "systems_programming", [
        ("Rust: How do ownership and borrowing work? Why does the borrow checker refuse my code?",
         "Rust 1.0+. Every value has one owner. When owner goes out of scope, value is dropped. "
         "Move semantics: let s2 = s1; // s1 is MOVED -- no longer valid "
         "Borrow (reference): let r = &s1; // immutable borrow -- s1 still valid "
         "Mutable borrow: let r = &mut s1; // ONE mutable borrow at a time "
         "Rule: cannot have mutable AND immutable borrows simultaneously. "
         "Clone to avoid move: let s2 = s1.clone(); // deep copy -- both valid "
         "Copy types (integers, bools, chars): let x = 5; let y = x; // both valid -- copied "
         "If borrow checker refuses: 1) restructure to avoid overlapping borrows "
         "2) clone the data 3) use Arc<Mutex<T>> for shared mutable state across threads"),
        ("Rust: How do I handle errors idiomatically with Result and the ? operator?",
         "Rust 1.0+. "
         "use std::fs; use std::io; "
         "fn read_file(path: &str) -> Result<String, io::Error> { "
         "    let content = fs::read_to_string(path)?; // ? returns Err early if fails "
         "    Ok(content.trim().to_string()) "
         "} "
         "fn main() { "
         "    match read_file('config.txt') { "
         "        Ok(content) => println!('Got: {content}'), "
         "        Err(e) => eprintln!('Error: {e}'), "
         "    } "
         "} "
         "For multiple error types: use anyhow crate: fn run() -> anyhow::Result<()> "
         "Then ? works across any error type. Add to Cargo.toml: anyhow = '1'"),
        ("Rust 1.39+: How do I use async/await with tokio?",
         "Rust async/await stable since 1.39 (2019). Most async code uses tokio runtime. "
         "Cargo.toml: tokio = { version = '1', features = ['full'] } "
         "use tokio::time::{sleep, Duration}; "
         "#[tokio::main] "
         "async fn main() { "
         "    let result = fetch_data().await; "
         "    println!('{result}'); "
         "} "
         "async fn fetch_data() -> String { "
         "    sleep(Duration::from_millis(100)).await; "
         "    'data'.to_string() "
         "} "
         "For HTTP: reqwest = { version = '0.12', features = ['json'] } "
         "let resp: MyType = reqwest::get(url).await?.json().await?;"),
        ("Rust: How do I use traits for polymorphism?",
         "Rust 1.0+. Traits are like interfaces. "
         "trait Animal { fn sound(&self) -> &str; fn name(&self) -> &str; } "
         "struct Dog; "
         "impl Animal for Dog { fn sound(&self) -> &str { 'woof' } fn name(&self) -> &str { 'Dog' } } "
         "Static dispatch (zero cost): fn speak<T: Animal>(a: &T) { println!('{}', a.sound()); } "
         "Dynamic dispatch (runtime): fn speak_dyn(a: &dyn Animal) { println!('{}', a.sound()); } "
         "Box<dyn Animal> stores a trait object on the heap. "
         "Use static dispatch (generics) for performance, dyn for heterogeneous collections."),
    ]),

    # ========================================================================
    # PERL -- Perl 5 modern (5.10+)
    # ========================================================================
    ("Perl 5: Modern Perl (5.10-5.38)", "programming", [
        ("Perl 5.10+: How do I write modern Perl? What pragmas should I always use?",
         "Always start scripts with: "
         "#!/usr/bin/perl "
         "use strict;   # require variable declarations with my/our "
         "use warnings; # enable common warnings "
         "use 5.010;    # require Perl 5.10+ features (say, given/when, // operator) "
         "my $x = 10;   # strict requires 'my' for variable declaration "
         "say 'Hello';  # say adds newline, like print with \\n (5.10+) "
         "my $val = $hash{key} // 'default'; # defined-or operator (not || which fails on '0') "
         "Install modern Perl on Linux: sudo apt install perl "
         "Check version: perl --version. CPAN modules: cpan Module::Name"),
        ("Perl: How do I use arrays, hashes, and references?",
         "Perl 5 data types: "
         "Scalar: my $x = 42; my $name = 'Alice'; "
         "Array: my @nums = (1,2,3); $nums[0] # 1; push @nums, 4; "
         "Hash: my %h = (name=>'Alice', age=>30); $h{name} # 'Alice'; "
         "Reference to array: my $aref = [1,2,3]; $aref->[0] # 1; @{$aref} # dereference "
         "Reference to hash: my $href = {a=>1}; $href->{a} # 1; %{$href} # dereference "
         "Anonymous hash ref: my $obj = { name => 'Bob', scores => [90, 85, 95] }; "
         "$obj->{scores}[1] # 85"),
        ("Perl: How do I use regular expressions?",
         "Perl 5 has the most powerful regex engine. "
         "Match: if ($str =~ /pattern/) { ... } "
         "Capture: if ($str =~ /^(\\w+)\\s+(\\d+)$/) { my ($word, $num) = ($1, $2); } "
         "Named captures (5.10+): if ($str =~ /(?<year>\\d{4})-(?<month>\\d{2})/) "
         "  { say $+{year}; say $+{month}; } "
         "Substitution: $str =~ s/foo/bar/g; # replace all occurrences "
         "Modifiers: /i case-insensitive, /g global, /m multiline, /x extended (whitespace ignored) "
         "Split: my @parts = split /,/, $csv_line;"),
        ("Perl: How do I read and write files?",
         "use strict; use warnings; "
         "open(my $fh, '<', 'input.txt') or die 'Cannot open: $!'; // read "
         "while (my $line = <$fh>) { chomp $line; say $line; } "
         "close $fh; "
         "open(my $out, '>', 'output.txt') or die 'Cannot write: $!'; // write (overwrite) "
         "open(my $app, '>>', 'log.txt') or die $!; // append "
         "print $out 'Hello\\n'; "
         "close $out; "
         "chomp removes trailing newline. die prints message and exits. $! is the OS error message."),
    ]),

    # ========================================================================
    # JSON and XML
    # ========================================================================
    ("JSON: Parsing, Generation, and Schema", "programming", [
        ("JSON: How do I parse and generate JSON in Python, JavaScript, and Rust?",
         "Python: import json "
         "data = json.loads('{\"name\":\"Alice\",\"age\":30}') # parse string "
         "obj = json.load(open('file.json'))  # parse file "
         "json_str = json.dumps(data, indent=2) # generate with pretty-print "
         "json.dump(data, open('out.json','w'), indent=2) # write to file "
         "JavaScript (built-in, no import): "
         "const data = JSON.parse(jsonString); "
         "const str = JSON.stringify(data, null, 2); // null=no replacer, 2=indent "
         "Rust: serde_json crate -- Cargo.toml: serde = {version='1',features=['derive']}; serde_json = '1' "
         "use serde::{Serialize,Deserialize}; "
         "#[derive(Serialize,Deserialize)] struct User { name: String, age: u32 } "
         "let u: User = serde_json::from_str(json_str)?; "
         "let s = serde_json::to_string_pretty(&u)?;"),
        ("JSON: What are common JSON gotchas and mistakes?",
         "1. JSON strings must use double quotes. Single quotes are NOT valid JSON. "
         "2. Trailing commas are NOT allowed: {\"a\":1,} is invalid. "
         "3. JSON has no comments. JSONC (used by VS Code) is a non-standard extension. "
         "4. All keys must be quoted strings: {name:'Alice'} is invalid. "
         "5. JSON numbers have precision limits -- very large integers may lose precision in JS. "
         "   Use strings for IDs > 2^53. "
         "6. JSON does not support undefined, functions, Date objects, or NaN/Infinity. "
         "   JSON.stringify(undefined) -> undefined (not a string). "
         "7. Circular references crash JSON.stringify() -- use a replacer or a library."),
    ]),

    ("XML: Parsing, XPath, and Namespaces", "programming", [
        ("XML: How do I parse XML in Python? When to use ElementTree vs lxml?",
         "Python stdlib -- xml.etree.ElementTree (safe for trusted XML): "
         "import xml.etree.ElementTree as ET "
         "tree = ET.parse('data.xml') "
         "root = tree.getroot() "
         "for child in root: print(child.tag, child.attrib, child.text) "
         "Find with XPath (limited subset): root.findall('.//item[@id]') "
         "lxml (pip install lxml) for: full XPath/XSLT, namespaces, huge files, better errors. "
         "from lxml import etree "
         "tree = etree.parse('data.xml') "
         "nodes = tree.xpath('//ns:item', namespaces={'ns':'http://example.com/ns'}) "
         "GOTCHA: xml.etree is NOT safe for untrusted XML (billion laughs attack). "
         "Use defusedxml: pip install defusedxml for untrusted input."),
        ("XML: How do I handle XML namespaces in Python ElementTree?",
         "XML namespaces appear as {uri}localname in ElementTree. "
         "Example: <root xmlns='http://example.com'><item/></root> "
         "In ElementTree: root.find('{http://example.com}item') "
         "Register prefix for cleaner output: ET.register_namespace('', 'http://example.com') "
         "In lxml with XPath: use namespaces dict: tree.xpath('//ex:item', namespaces={'ex':'http://example.com'}) "
         "Parse namespace map: root.nsmap (lxml only) "
         "For documents with xs:schema prefix, the URI is http://www.w3.org/2001/XMLSchema"),
    ]),

    # ========================================================================
    # APPLIED MATHEMATICS IN CODE
    # ========================================================================
    ("Applied Mathematics in Code: Arithmetic to Calculus", "quantum_mechanics", [
        ("Python: How do I avoid floating-point precision errors in arithmetic?",
         "Floating point (IEEE 754) cannot represent all decimals exactly. "
         "0.1 + 0.2 == 0.3 is False in Python/JS/C. "
         "Fix 1 -- use decimal.Decimal for exact decimal arithmetic (financial): "
         "from decimal import Decimal, getcontext "
         "getcontext().prec = 28 "
         "Decimal('0.1') + Decimal('0.2') == Decimal('0.3') # True "
         "Fix 2 -- use math.isclose for comparisons: "
         "import math; math.isclose(0.1+0.2, 0.3) # True "
         "Fix 3 -- use fractions.Fraction for exact rational arithmetic: "
         "from fractions import Fraction; Fraction(1,3) + Fraction(1,6) == Fraction(1,2) # True"),
        ("Python: How do I do numerical integration and differentiation with scipy?",
         "pip install scipy numpy "
         "Numerical integration (quad): "
         "from scipy import integrate "
         "result, error = integrate.quad(lambda x: x**2, 0, 1) # integral of x^2 from 0 to 1 "
         "# result ~= 0.3333 (exact: 1/3) "
         "Numerical differentiation: "
         "from scipy.misc import derivative "
         "dfdx = derivative(lambda x: x**3, x0=2.0, dx=1e-6) # f'(2) for f(x)=x^3 ~= 12.0 "
         "Solving equations: "
         "from scipy.optimize import fsolve "
         "root = fsolve(lambda x: x**2 - 4, x0=1.0) # solves x^2=4, returns [2.0]"),
        ("Python: How do I do linear algebra operations with numpy?",
         "pip install numpy -- import numpy as np "
         "Matrix creation: A = np.array([[1,2],[3,4]]); B = np.eye(3) # identity "
         "Matrix multiply: C = A @ B  # @ operator (Python 3.5+, numpy 1.10+) "
         "NOT A * B which is element-wise. "
         "Transpose: A.T "
         "Inverse: np.linalg.inv(A) -- only for square non-singular matrices "
         "Eigenvalues: eigenvalues, eigenvectors = np.linalg.eig(A) "
         "Solve Ax=b: x = np.linalg.solve(A, b) -- more numerically stable than inv(A)@b "
         "Dot product: np.dot(v1, v2) or v1 @ v2 for 1D arrays "
         "Norm: np.linalg.norm(v) -- Euclidean length of vector"),
        ("How do I implement Newton's method to find roots numerically?",
         "Newton's method: x_{n+1} = x_n - f(x_n)/f'(x_n). Converges quadratically. "
         "Python implementation: "
         "def newtons_method(f, df, x0, tol=1e-10, max_iter=100): "
         "    x = x0 "
         "    for _ in range(max_iter): "
         "        fx = f(x) "
         "        if abs(fx) < tol: return x "
         "        x = x - fx / df(x) "
         "    raise ValueError('Did not converge') "
         "# Find sqrt(2): f(x)=x^2-2, f'(x)=2x "
         "root = newtons_method(lambda x: x**2-2, lambda x: 2*x, x0=1.0) # ~= 1.41421356 "
         "Gotcha: diverges if df(x)=0 or starting point is far from root. Check convergence."),
    ]),

    # ========================================================================
    # STATISTICS IN CODE
    # ========================================================================
    ("Statistics in Code: Python and NumPy", "information_theory", [
        ("Python: How do I compute descriptive statistics with numpy and scipy?",
         "pip install numpy scipy "
         "import numpy as np; from scipy import stats "
         "data = np.array([2,4,4,4,5,5,7,9]) "
         "np.mean(data)   # 5.0 -- arithmetic mean "
         "np.median(data) # 4.5 "
         "np.std(data)    # population std (ddof=0) "
         "np.std(data, ddof=1) # sample std (Bessel's correction) "
         "np.var(data)    # variance "
         "np.percentile(data, 25) # 25th percentile (Q1) "
         "stats.mode(data).mode[0] # mode "
         "np.min(data), np.max(data), np.ptp(data) # range"),
        ("Python: How do I test if two groups are significantly different (t-test)?",
         "from scipy import stats "
         "group_a = [23, 25, 28, 22, 27] "
         "group_b = [30, 32, 28, 35, 31] "
         "t_stat, p_value = stats.ttest_ind(group_a, group_b) "
         "print(f't={t_stat:.3f}, p={p_value:.3f}') "
         "If p < 0.05: reject null hypothesis (groups are significantly different at 5% level). "
         "Paired t-test: stats.ttest_rel(before, after) "
         "One-sample: stats.ttest_1samp(data, popmean=0) "
         "Mann-Whitney U (non-parametric): stats.mannwhitneyu(group_a, group_b)"),
        ("Python: How do I do linear regression with numpy or scikit-learn?",
         "NumPy (simple): "
         "import numpy as np "
         "x = np.array([1,2,3,4,5]); y = np.array([2.1,3.9,6.0,8.1,9.8]) "
         "coeffs = np.polyfit(x, y, deg=1) # [slope, intercept] "
         "y_pred = np.polyval(coeffs, x) "
         "Scikit-learn (general): pip install scikit-learn "
         "from sklearn.linear_model import LinearRegression "
         "import numpy as np "
         "model = LinearRegression() "
         "model.fit(x.reshape(-1,1), y) # must be 2D for sklearn "
         "model.coef_[0] # slope; model.intercept_ # intercept "
         "model.score(x.reshape(-1,1), y) # R^2 score"),
    ]),

    # ========================================================================
    # GEOMETRY IN CODE
    # ========================================================================
    ("Geometry in Code: 2D and 3D", "quantum_mechanics", [
        ("Python: How do I work with 2D vectors and compute distance, angle, dot product?",
         "import math "
         "def dist(p1, p2): return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2) "
         "def dot2d(v1, v2): return v1[0]*v2[0] + v1[1]*v2[1] "
         "def magnitude(v): return math.sqrt(v[0]**2 + v[1]**2) "
         "def normalize(v): m=magnitude(v); return (v[0]/m, v[1]/m) "
         "def angle_between(v1,v2): return math.acos(dot2d(v1,v2)/(magnitude(v1)*magnitude(v2))) "
         "Using numpy (more practical for 3D too): "
         "import numpy as np "
         "v1, v2 = np.array([1,0]), np.array([0,1]) "
         "np.linalg.norm(v1-v2) # distance; np.dot(v1,v2) # dot product "
         "np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))) # angle in radians"),
        ("Python: How do I do 3D rotation with rotation matrices and quaternions?",
         "Using numpy for rotation matrices: "
         "import numpy as np "
         "def rot_x(angle): c,s = np.cos(angle),np.sin(angle) "
         "  return np.array([[1,0,0],[0,c,-s],[0,s,c]]) "
         "def rot_y(angle): c,s = np.cos(angle),np.sin(angle) "
         "  return np.array([[c,0,s],[0,1,0],[-s,0,c]]) "
         "Rotate point: p = np.array([1,0,0]); rotated = rot_y(np.pi/4) @ p "
         "For quaternions: pip install pyquaternion or scipy.spatial.transform "
         "from scipy.spatial.transform import Rotation "
         "r = Rotation.from_euler('xyz', [30,45,0], degrees=True) "
         "r.apply(np.array([1,0,0])) # rotate a vector"),
    ]),

    # ========================================================================
    # BLENDER CLI (bpy)
    # ========================================================================
    ("Blender CLI: Headless Rendering and bpy Scripting", "programming", [
        ("Blender CLI: How do I run a Python script in Blender headlessly?",
         "Blender 2.80+ (all platforms: Windows, macOS, Linux). Blender is free, open source (GPL). "
         "Run a script headlessly (no GUI): "
         "blender --background --python myscript.py "
         "blender --background scene.blend --python render.py "
         "blender --background --python-expr 'import bpy; bpy.ops.render.render(write_still=True)' "
         "On Windows: 'C:\\Program Files\\Blender Foundation\\Blender 4.x\\blender.exe' --background --python script.py "
         "On macOS: /Applications/Blender.app/Contents/MacOS/Blender --background --python script.py "
         "The bpy module is ONLY available inside Blender -- cannot pip install it for regular Python."),
        ("Blender bpy: How do I create a mesh object from vertices and faces?",
         "Blender 2.80+ bpy API. Run inside Blender (--python or Blender's scripting workspace). "
         "import bpy "
         "verts = [(0,0,0),(1,0,0),(1,1,0),(0,1,0)] # 4 vertices "
         "faces = [(0,1,2,3)] # one quad face "
         "mesh = bpy.data.meshes.new('MyMesh') "
         "mesh.from_pydata(verts, [], faces) "
         "mesh.update() "
         "obj = bpy.data.objects.new('MyObject', mesh) "
         "bpy.context.collection.objects.link(obj) # add to scene "
         "Gotcha: from_pydata takes (vertices, edges, faces). Use [] for edges when defining faces."),
        ("Blender bpy: How do I render a scene to an image file headlessly?",
         "Blender 2.80+ headless rendering. "
         "import bpy "
         "scene = bpy.context.scene "
         "scene.render.engine = 'CYCLES' # or 'BLENDER_EEVEE' "
         "scene.render.filepath = '/tmp/render_output.png' "
         "scene.render.image_settings.file_format = 'PNG' "
         "scene.render.resolution_x = 1920 "
         "scene.render.resolution_y = 1080 "
         "scene.cycles.samples = 128 # cycles samples "
         "bpy.ops.render.render(write_still=True) "
         "Command: blender --background scene.blend --python render_script.py "
         "Gotcha: CYCLES needs GPU or CPU. Set scene.cycles.device='GPU' and configure GPU in Preferences."),
        ("Blender bpy: How do I export a mesh to GLTF/GLB format?",
         "Blender 2.80+ (GLTF exporter built-in since 2.80). "
         "import bpy "
         "bpy.ops.export_scene.gltf( "
         "    filepath='/tmp/model.glb', "
         "    export_format='GLB', # GLB = binary, GLTF = text+bin "
         "    use_selection=True, # only export selected objects "
         "    export_apply=True,  # apply modifiers "
         "    export_materials='EXPORT', "
         ") "
         "For OBJ: bpy.ops.wm.obj_export(filepath='/tmp/model.obj') # Blender 3.3+ "
         "For FBX: bpy.ops.export_scene.fbx(filepath='/tmp/model.fbx') "
         "Gotcha: bpy.ops require a context with correct mode. Switch to OBJECT mode first: "
         "bpy.ops.object.mode_set(mode='OBJECT')"),
    ]),

    # ========================================================================
    # ELECTRONICS ENGINEERING CLIs (all MIT / open source)
    # ========================================================================
    ("Electronics Engineering CLIs: KiCad, ngspice, OpenSCAD, Verilator", "systems_programming", [
        ("KiCad CLI: How do I export Gerber files and run DRC headlessly?",
         "KiCad 7.0+ includes kicad-cli (2023). Free, open source (GPL). "
         "Export Gerbers from a PCB file: "
         "kicad-cli pcb export gerbers --output ./gerbers myboard.kicad_pcb "
         "Export drill files: kicad-cli pcb export drill --output ./gerbers myboard.kicad_pcb "
         "Run DRC: kicad-cli pcb drc --output report.rpt myboard.kicad_pcb "
         "Export schematic to PDF: kicad-cli sch export pdf --output schematic.pdf myschematic.kicad_sch "
         "Install on Ubuntu: sudo apt install kicad (Ubuntu 22.04+ has KiCad 6) "
         "For KiCad 7+: sudo add-apt-repository ppa:kicad/kicad-7.0-releases && sudo apt install kicad"),
        ("ngspice: How do I run a SPICE simulation from the command line?",
         "ngspice -- open source SPICE simulator (BSD license). "
         "Install: sudo apt install ngspice (Linux) or https://ngspice.sourceforge.io "
         "Run batch simulation: ngspice -b -o output.log circuit.sp "
         "-b = batch mode (no GUI). -o writes output log. "
         "Example RC circuit netlist (circuit.sp): "
         "RC Low-Pass Filter "
         "R1 in out 1k "
         "C1 out 0 1u "
         "Vin in 0 AC 1 "
         ".AC DEC 100 1 1MEG "
         ".PRINT AC V(out) "
         ".END "
         "Parse output with Python or gnuplot. "
         "ngspice can also be driven programmatically via PySpice: pip install PySpice"),
        ("OpenSCAD: How do I generate a 3D model from a script and export to STL?",
         "OpenSCAD -- free, open source (GPL) parametric CAD. Script-based 3D modeling. "
         "Install: sudo apt install openscad or https://openscad.org "
         "Create model.scad: "
         "difference() { "
         "  cube([20, 20, 10]); "
         "  translate([5, 5, -1]) cylinder(h=12, r=4, $fn=32); "
         "} "
         "Export to STL: openscad -o model.stl model.scad "
         "Export to DXF (2D): openscad -o model.dxf model.scad "
         "Pass parameters: openscad -o model.stl -D 'width=30' -D 'height=15' model.scad "
         "Useful for automated PCB standoffs, enclosures, jigs."),
        ("Verilator: How do I simulate Verilog with Verilator from the command line?",
         "Verilator -- open source Verilog/SystemVerilog simulator (LGPL). "
         "Install: sudo apt install verilator (Ubuntu 20.04+) "
         "Compile Verilog to C++: verilator --cc --exe --build top.v sim_main.cpp "
         "sim_main.cpp drives the simulation: "
         "#include 'verilated.h' "
         "#include 'Vtop.h' // generated from top.v "
         "int main(int argc, char** argv) { "
         "  VerilatedContext* ctx = new VerilatedContext; "
         "  Vtop* top = new Vtop{ctx}; "
         "  while (!ctx->gotFinish()) { ctx->timeInc(1); top->eval(); } "
         "  delete top; delete ctx; return 0; "
         "} "
         "Run: make -C obj_dir -f Vtop.mk && ./obj_dir/Vtop "
         "For GTKWave waveforms: verilator --cc --trace --exe --build top.v sim_main.cpp"),
        ("iverilog: How do I simulate Verilog with Icarus Verilog?",
         "Icarus Verilog (iverilog) -- open source Verilog simulator (GPL). "
         "Install: sudo apt install iverilog "
         "Compile: iverilog -o sim top.v testbench.v "
         "Run: vvp sim "
         "View waveforms (if VCD dumped): gtkwave output.vcd "
         "In testbench.v to dump waveforms: "
         "initial begin $dumpfile('output.vcd'); $dumpvars(0, top_module); end "
         "For SystemVerilog: iverilog -g2012 -o sim design.sv tb.sv "
         "-g2012 enables IEEE 1800-2012 (SystemVerilog). "
         "Icarus is simpler than Verilator; Verilator is faster for large designs."),
    ]),

    # ========================================================================
    # LINUX / WINDOWS TERMINAL -- PLATFORM-AWARE
    # ========================================================================
    ("Terminal: Linux vs Windows Platform Differences", "systems_programming", [
        ("What are the key differences between Linux bash and Windows PowerShell for scripting?",
         "Path separators: Linux: /home/user/file, Windows: C:\\Users\\user\\file "
         "(PowerShell accepts both / and \\). "
         "Line endings: Linux: LF (\\n), Windows: CRLF (\\r\\n). "
         "Scripts: Linux: .sh (bash), Windows: .ps1 (PowerShell) or .bat (cmd). "
         "Variables: bash: $VAR or ${VAR}, PowerShell: $env:VAR or $var. "
         "Conditional: bash: if [ -f file ]; then, PowerShell: if (Test-Path file) { "
         "Package manager: Linux: apt/yum/dnf, Windows: winget/choco/scoop. "
         "Case sensitivity: Linux paths are case-sensitive. Windows paths are NOT."),
        ("How do I write a script that works on both Linux and Windows?",
         "Option 1: Python script -- runs on both with python script.py or python3 script.py. "
         "Use pathlib.Path (not string concatenation) for paths: Path('dir') / 'file.txt'. "
         "Use os.environ.get('HOME') or Path.home() for home directory. "
         "Option 2: Separate scripts -- script.sh for Linux/Mac, script.ps1 for Windows. "
         "Option 3: Use WSL2 on Windows to run bash scripts natively. "
         "Option 4: Makefile -- works on Linux/Mac natively, on Windows via Git Bash or WSL. "
         "Always use os.path.sep or pathlib for path joins. Never hardcode / or \\."),
        ("Linux: How do I find and kill a process by name or port?",
         "By name: pkill python or killall python (sends SIGTERM). "
         "pkill -9 python -- force kill. "
         "By port: lsof -ti :8090 -- prints PID. "
         "kill $(lsof -ti :8090) -- kill all processes on port 8090. "
         "Windows equivalent: "
         "netstat -ano | findstr :8090 -- find PID. "
         "taskkill /PID 1234 /F -- force kill. "
         "Stop-Process -Name python -Force (PowerShell)"),
    ]),

    # ========================================================================
    # GIT -- complete workflows
    # ========================================================================
    ("Git: Workflows and Common Scenarios", "programming", [
        ("Git: How do I set up a new repo, make a commit, and push to GitHub?",
         "git init && git add . && git commit -m 'Initial commit' "
         "Create repo on GitHub (no README). "
         "git remote add origin https://github.com/user/repo.git "
         "git branch -M main && git push -u origin main "
         "After that: git add . && git commit -m 'msg' && git push "
         "Use SSH (no password): git remote add origin git@github.com:user/repo.git "
         "For SSH: generate key: ssh-keygen -t ed25519 -C 'email@example.com' "
         "Add public key to GitHub: Settings -> SSH keys -> New SSH key."),
        ("Git: How do I resolve a merge conflict?",
         "Conflicts appear when two branches modify the same lines. "
         "After git merge or git pull with conflicts: git status shows conflicting files. "
         "Open each conflicting file. Look for markers: "
         "<<<<<<< HEAD (your changes) "
         "======= "
         ">>>>>>> branch-name (incoming changes) "
         "Edit the file to keep what you want. Remove ALL marker lines (<<<<, ====, >>>>). "
         "git add conflicted-file.txt -- mark as resolved. "
         "git commit -- completes the merge. "
         "Or abort: git merge --abort to go back to pre-merge state."),
        ("Git: How do I work with branches for a feature workflow?",
         "Create feature branch: git checkout -b feature/new-button (or git switch -c feature/new-button) "
         "Make commits on the branch. "
         "Push: git push -u origin feature/new-button "
         "Create Pull Request on GitHub/GitLab. "
         "After PR merged: git checkout main && git pull "
         "Delete local branch: git branch -d feature/new-button "
         "Delete remote: git push origin --delete feature/new-button "
         "Rebase to keep clean history: git rebase main (while on feature branch) "
         "then git push --force-with-lease (safe force push after rebase)."),
    ]),

    # ========================================================================
    # AGENT MODE -- Decision trees and project completion
    # ========================================================================
    ("Agent Mode: Code Generation and Project Completion", "programming", [
        ("As an AI agent, I need to write working code. What must I include in every code response?",
         "Every code response must include: "
         "1. Language version and minimum requirement (e.g., Python 3.10+, C++17, Node.js 18+). "
         "2. All necessary imports/includes at the top. "
         "3. Complete, runnable code -- not pseudocode or fragments. "
         "4. Install command for any non-standard dependencies (pip install, cargo add, npm install). "
         "5. The exact command to run the code. "
         "6. Expected output for the example input. "
         "7. Platform notes if the code behaves differently on Windows vs Linux/Mac."),
        ("As an AI agent, I ran the code I generated and got an import error. What do I do?",
         "An import error means a module is missing or the wrong Python/Node/etc. version is active. "
         "Steps: 1) Read the exact error -- it names the missing module. "
         "2) Install it: pip install module-name (Python), npm install package (Node), cargo add crate (Rust). "
         "3) If installed but still failing: check which python/node/cargo is active (which python3). "
         "4) In Python: if in a virtual environment, activate it first. "
         "5) Re-run the code. If still failing, check if the package name differs from the import name "
         "   (e.g., pip install Pillow but import PIL, pip install opencv-python but import cv2)."),
        ("As an AI agent building a project, how do I know what file to create or modify next?",
         "Process: 1) Re-read the original goal -- what is the final deliverable? "
         "2) List what exists: ls or find . -name '*.py' to see current files. "
         "3) List what's missing: what files does the goal require that don't exist yet? "
         "4) Check dependencies: what does the next file depend on that isn't written yet? "
         "5) Work bottom-up: write helper/utility code before code that uses it. "
         "6) After each file: run it or test it immediately. Don't write 5 files then test. "
         "7) Commit after each working unit: git add . && git commit -m 'add: description'."),
        ("As an AI agent, how do I verify that the code I wrote actually works?",
         "1) Run it: actually execute the code with test input and observe output. "
         "2) Check output matches spec: compare actual output to the expected output in the task. "
         "3) Test edge cases: empty input, maximum values, invalid input. "
         "4) Check error handling: deliberately pass bad input -- does it fail gracefully? "
         "5) For web servers: curl the endpoints and check responses. "
         "6) For file operations: check that files were created/modified correctly. "
         "7) Do NOT assume it works based on no error output -- always verify the actual result."),
    ]),
]


async def train_corpus(node: str, repeats: int) -> None:
    async with httpx.AsyncClient(timeout=60) as client:
        nc = NeuroClient(f"http://{node}", client)

        try:
            r = await client.get(f"http://{node}/health", timeout=5)
            info = r.json()
            print(f"Node: {info.get('node_id','?')}  status={info.get('status')}")
        except Exception as e:
            sys.exit(f"Node not reachable at {node}: {e}")

        total_pairs = sum(len(pairs) for _, _, pairs in CORPUS)
        print(f"Comprehensive code corpus -- {total_pairs} pairs in {len(CORPUS)} groups")
        print(f"Repeats: {repeats}\n")

        t0 = time.time()
        stats = {"qa": 0, "ep": 0, "eq": 0, "know": 0, "seq": 0}

        for rep in range(1, repeats + 1):
            for group_title, discipline, pairs in CORPUS:
                candidates = [
                    {
                        "qa_id":      str(uuid.uuid4()),
                        "question":   q,
                        "answer":     a,
                        "book_id":    "code_corpus",
                        "page_index": 1,
                        "confidence": 0.97,
                    }
                    for q, a in pairs
                ]

                n = await nc.ingest_qa_full(
                    candidates, pool="knowledge", record_episodes=True)
                stats["qa"] += n
                stats["seq"] += len(candidates)
                stats["ep"] += len(candidates)

                combined = " ".join(a for _, a in pairs)
                n_eq = await nc.ingest_equations(combined, discipline=discipline)
                stats["eq"] += n_eq

                if rep == 1:
                    body = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in pairs)
                    ok = await nc.ingest_knowledge(
                        title=group_title,
                        body=body,
                        source="code_corpus",
                        tags=["code", "programming", discipline],
                    )
                    if ok:
                        stats["know"] += 1

            if rep % 5 == 0 or rep == repeats:
                elapsed = time.time() - t0
                print(f"  [{rep}/{repeats}]  qa={stats['qa']}  ep={stats['ep']}  "
                      f"eq={stats['eq']}  know={stats['know']}  seq={stats['seq']}  "
                      f"({elapsed:.0f}s)")

        await nc.checkpoint()
        elapsed = time.time() - t0
        print(f"\nDone in {elapsed:.0f}s  |  pairs={total_pairs}  repeats={repeats}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node",    default=DEFAULT_NODE)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    args = parser.parse_args()
    asyncio.run(train_corpus(args.node, args.repeats))


if __name__ == "__main__":
    main()
