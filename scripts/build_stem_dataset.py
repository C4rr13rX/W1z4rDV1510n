#!/usr/bin/env python3
# coding: utf-8
"""
build_stem_dataset.py -- Universal STEM + Engineering + Coding knowledge builder
for the W1z4rD V1510n neural fabric.

Goal: train the fabric to pass any STEM exam, answer any science/engineering/
coding question correctly, and understand CAD/3D printing/manufacturing.

Stages:
  10 -- LibreTexts textbooks (math, physics, chem, bio, eng, cs, stats)
  11 -- Wikipedia STEM article corpus (10K+ articles)
  12 -- Programming languages: official docs, manuals, historical examples
       (BASIC/QBasic 1980->, C, C++, Java, Python, Rust, Go, JS, etc.)
  13 -- CAD/3D printing/manufacturing/Blender corpus
  14 -- Q&A pairs from all above (for /qa/ingest)
  15 -- arXiv preprint abstracts (CS, math, physics, engineering)

Usage:
  python scripts/build_stem_dataset.py [--stages 10,11,12,13,14,15]
                                        [--node localhost:8090]
                                        [--data-dir D:/w1z4rdv1510n-data]

Requirements: pip install requests tqdm
"""

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    def tqdm(it, **kw): return it
    HAS_TQDM = False

# -- Defaults ------------------------------------------------------------------
DEFAULT_NODE     = 'localhost:8090'
DEFAULT_DATA_DIR = 'D:/w1z4rdv1510n-data'
UA = 'W1z4rDV1510n-STEMBuilder/1.0 (https://github.com/C4rr13rX/W1z4rDV1510n; adamedsall@gmail.com)'

STAGES = {
    10: 'LibreTexts textbooks      -- math, physics, chem, bio, eng, CS, stats',
    11: 'Wikipedia STEM corpus     -- 10K+ science/engineering articles',
    12: 'Programming language docs -- official manuals 1980-present, all langs',
    13: 'CAD/3D/Manufacturing      -- Blender, OpenSCAD, 3D printing, automotive',
    14: 'STEM Q&A pairs            -- exam questions across all domains',
    15: 'arXiv abstracts           -- CS, math, physics, engineering preprints',
    16: 'Electronics & Logic       -- Boolean algebra, gates, circuits, PCB, ICs',
    17: 'Embedded/Firmware/Systems -- Arduino, ARM, AVR, BIOS/UEFI, RTOS, CAN',
}

# -- Stage 10: LibreTexts -------------------------------------------------------
# LibreTexts batch-print PDF URL: https://batch.libretexts.org/print/Letter/Finished/{printId}/Full.pdf
# printId = "{subdomain}-{data-page-id}" obtained by crawling bookshelves HTML

LIBRETEXTS_ROOTS = [
    ('math',       'https://math.libretexts.org/Bookshelves'),
    ('phys',       'https://phys.libretexts.org/Bookshelves'),
    ('chem',       'https://chem.libretexts.org/Bookshelves'),
    ('bio',        'https://bio.libretexts.org/Bookshelves'),
    ('eng',        'https://eng.libretexts.org/Bookshelves'),
    ('stats',      'https://stats.libretexts.org/Bookshelves'),
    ('geo',        'https://geo.libretexts.org/Bookshelves'),
    ('med',        'https://med.libretexts.org/Bookshelves'),
    ('socialsci',  'https://socialsci.libretexts.org/Bookshelves'),
    ('workforce',  'https://workforce.libretexts.org/Bookshelves'),
    ('k12',        'https://k12.libretexts.org/Bookshelves'),
]

# -- Stage 11: Wikipedia STEM articles -----------------------------------------
WIKI_STEM_CATEGORIES = [
    # Mathematics
    'Calculus', 'Linear_algebra', 'Abstract_algebra', 'Number_theory',
    'Topology', 'Differential_equations', 'Probability_theory', 'Statistics',
    'Discrete_mathematics', 'Numerical_analysis', 'Combinatorics',
    'Graph_theory', 'Set_theory', 'Logic', 'Mathematical_optimization',
    # Physics
    'Classical_mechanics', 'Quantum_mechanics', 'Electromagnetism',
    'Thermodynamics', 'Special_relativity', 'General_relativity',
    'Particle_physics', 'Nuclear_physics', 'Condensed_matter_physics',
    'Optics', 'Acoustics', 'Fluid_mechanics', 'Statistical_mechanics',
    # Chemistry
    'Organic_chemistry', 'Inorganic_chemistry', 'Physical_chemistry',
    'Biochemistry', 'Analytical_chemistry', 'Electrochemistry',
    'Polymer_chemistry', 'Materials_science', 'Chemical_engineering',
    # Biology
    'Molecular_biology', 'Cell_biology', 'Genetics', 'Evolutionary_biology',
    'Ecology', 'Microbiology', 'Neuroscience', 'Biochemistry',
    'Bioinformatics', 'Developmental_biology',
    # Computer Science
    'Algorithm', 'Data_structure', 'Computer_architecture',
    'Operating_system', 'Computer_network', 'Database', 'Compiler',
    'Machine_learning', 'Artificial_intelligence', 'Cryptography',
    'Computer_graphics', 'Distributed_computing', 'Software_engineering',
    'Programming_language', 'Automata_theory', 'Computational_complexity_theory',
    # Engineering
    'Mechanical_engineering', 'Electrical_engineering', 'Civil_engineering',
    'Chemical_engineering', 'Aerospace_engineering', 'Biomedical_engineering',
    'Control_theory', 'Signal_processing', 'Robotics',
    'Structural_engineering', 'Thermodynamics', 'Fluid_dynamics',
    # Project Management
    'Project_management', 'Agile_software_development', 'Scrum_(software_development)',
    'Kanban_(development)', 'Critical_path_method', 'Gantt_chart',
    'Risk_management', 'Systems_engineering',
    # CAD / Manufacturing
    'Computer-aided_design', 'Computer-aided_manufacturing',
    '3D_printing', 'Fused_deposition_modeling', 'Stereolithography',
    'CNC_router', 'Automotive_engineering', 'Manufacturing_engineering',
    'Industrial_design', 'Finite_element_method',
]

# Specific high-value articles to always include
WIKI_STEM_ARTICLES = [
    # Math fundamentals
    'Derivative', 'Integral', 'Fourier_transform', 'Laplace_transform',
    'Taylor_series', 'Matrix_(mathematics)', 'Eigenvalues_and_eigenvectors',
    'Gradient', 'Divergence', 'Curl_(mathematics)', 'Stokes_theorem',
    'Bayes_theorem', 'Normal_distribution', 'Central_limit_theorem',
    # Physics
    "Newton's_laws_of_motion", 'Maxwell_equations', 'Schrodinger_equation',
    "Ohm's_law", 'Kirchhoff_circuit_laws', 'Bernoulli_principle',
    'Second_law_of_thermodynamics', 'Special_relativity',
    # CS / Programming
    'Big_O_notation', 'Sorting_algorithm', 'Binary_search_algorithm',
    'Hash_table', 'Binary_tree', 'Graph_(abstract_data_type)',
    'Dynamic_programming', 'Recursion_(computer_science)',
    'Object-oriented_programming', 'Functional_programming',
    'Turing_machine', 'P_versus_NP_problem', 'Von_Neumann_architecture',
    'TCP/IP', 'HTTP', 'SQL', 'Regular_expression',
    # 3D / CAD
    'Blender_(software)', 'FreeCAD', 'OpenSCAD', 'AutoCAD',
    'G-code', 'Stereolithography_file_format', 'STEP_file',
    'Polygon_mesh', 'NURBS', 'Solid_modeling', 'Parametric_design',
    # Quantum mechanics
    'Quantum_mechanics', 'Quantum_entanglement', 'Quantum_superposition',
    'Wave_function', 'Quantum_field_theory', 'Quantum_electrodynamics',
    'Standard_Model', 'Higgs_boson', 'Quantum_computing',
    "Schrodinger_equation", 'Double-slit_experiment', 'Bell_theorem',
    'Uncertainty_principle', 'Copenhagen_interpretation',
    'Quantum_tunnelling', 'Spin_(physics)', 'Photon',
    # Astrophysics / Cosmology
    'Astrophysics', 'Cosmology', 'Big_Bang', 'Dark_matter', 'Dark_energy',
    'Black_hole', 'Neutron_star', 'Pulsar', 'Supernova', 'White_dwarf',
    'Stellar_evolution', 'Hertzsprung-Russell_diagram',
    'Galaxy', 'Milky_Way', 'Cosmic_microwave_background', 'Hubble_constant',
    'Gravitational_wave', 'Event_horizon', 'Hawking_radiation',
    'Inflation_(cosmology)', 'Lambda-CDM_model', 'Exoplanet',
    'Solar_System', 'Planetary_science', 'General_relativity',
]

# -- Stage 12: Programming languages -------------------------------------------
# Official documentation, language specs, GitHub repos, historical manuals

PROG_LANG_SOURCES = [
    # -- 1980s era --
    {
        'lang': 'BASIC', 'era': '1975-1990', 'compiler': 'Microsoft BASIC / GW-BASIC / QBasic',
        'wiki': 'BASIC', 'docs_url': 'https://en.wikipedia.org/wiki/BASIC',
        'spec_urls': [
            'https://en.wikipedia.org/wiki/QBasic',
            'https://en.wikipedia.org/wiki/GW-BASIC',
            'https://en.wikipedia.org/wiki/Altair_BASIC',
        ],
        'context': 'Line-numbered BASIC. RUN, LIST, GOTO, GOSUB, DIM, PRINT, INPUT. '
                   'Runs on: MS-DOS, IBM PC, Apple II, Commodore 64, TRS-80. '
                   'Interpreter-based. Variable types: integer%, single!, double#, string$.',
    },
    {
        'lang': 'Pascal', 'era': '1970-1995', 'compiler': 'Turbo Pascal / Free Pascal',
        'wiki': 'Pascal_(programming_language)',
        'spec_urls': ['https://en.wikipedia.org/wiki/Turbo_Pascal',
                      'https://en.wikipedia.org/wiki/Pascal_(programming_language)'],
        'context': 'Strongly typed, structured. BEGIN/END blocks, PROGRAM, PROCEDURE, '
                   'FUNCTION, RECORD, ARRAY. Used in CS education 1970s-1990s.',
    },
    {
        'lang': 'C', 'era': '1972-present', 'compiler': 'GCC / Clang / MSVC',
        'wiki': 'C_(programming_language)',
        'spec_urls': ['https://en.wikipedia.org/wiki/C_(programming_language)',
                      'https://en.wikipedia.org/wiki/C17_(C_standard_revision)',
                      'https://en.wikipedia.org/wiki/ANSI_C'],
        'context': 'Systems programming language. K&R C (1978), ANSI C89, C99, C11, C17, C23. '
                   'Pointers, manual memory management, undefined behavior, preprocessor. '
                   'Compilers: gcc -std=c17, clang, MSVC. Platform: UNIX, Windows, embedded.',
    },
    {
        'lang': 'C++', 'era': '1985-present', 'compiler': 'GCC / Clang / MSVC',
        'wiki': 'C%2B%2B',
        'spec_urls': ['https://en.wikipedia.org/wiki/C%2B%2B',
                      'https://en.wikipedia.org/wiki/C%2B%2B17',
                      'https://en.wikipedia.org/wiki/C%2B%2B20'],
        'context': 'C++98, C++03, C++11 (lambdas, auto, move semantics), C++14, C++17, C++20 '
                   '(concepts, coroutines, ranges), C++23. RAII, templates, STL. '
                   'Compiler: g++ -std=c++20, clang++, cl /std:c++20.',
    },
    {
        'lang': 'Fortran', 'era': '1957-present', 'compiler': 'gfortran / Intel Fortran',
        'wiki': 'Fortran',
        'spec_urls': ['https://en.wikipedia.org/wiki/Fortran'],
        'context': 'First high-level language. FORTRAN 66, 77, 90, 95, 2003, 2008, 2018. '
                   'Scientific computing, HPC. Arrays first-class. Column-oriented I/O.',
    },
    {
        'lang': 'Ada', 'era': '1983-present', 'compiler': 'GNAT',
        'wiki': 'Ada_(programming_language)',
        'spec_urls': ['https://en.wikipedia.org/wiki/Ada_(programming_language)'],
        'context': 'Ada 83, 95, 2005, 2012. Strong typing, design by contract, tasking. '
                   'Used in safety-critical: avionics, military. Compiler: gnat.',
    },
    {
        'lang': 'Lisp', 'era': '1958-present', 'compiler': 'Common Lisp / Scheme / Clojure',
        'wiki': 'Lisp_(programming_language)',
        'spec_urls': ['https://en.wikipedia.org/wiki/Common_Lisp',
                      'https://en.wikipedia.org/wiki/Scheme_(programming_language)'],
        'context': 'Homoiconicity, S-expressions, CAR/CDR, CONS, macros, REPL. '
                   'Dialects: Common Lisp, Scheme (R5RS, R7RS), Clojure. '
                   'Runs in: SBCL, CLISP, Racket, LispWorks.',
    },
    {
        'lang': 'Prolog', 'era': '1972-present', 'compiler': 'SWI-Prolog',
        'wiki': 'Prolog',
        'spec_urls': ['https://en.wikipedia.org/wiki/Prolog'],
        'context': 'Logic programming. Facts, rules, queries. Unification, backtracking. '
                   'Runs in: SWI-Prolog 9.x. Used in: AI, NLP, constraint solving.',
    },
    {
        'lang': 'Smalltalk', 'era': '1972-present', 'compiler': 'Squeak / Pharo',
        'wiki': 'Smalltalk',
        'spec_urls': ['https://en.wikipedia.org/wiki/Smalltalk'],
        'context': 'Pure OOP. Everything is an object. Message passing. Blocks, metaclasses. '
                   'Inspired Python, Ruby, Objective-C. Runs in: Pharo, Squeak image.',
    },
    # -- 1990s --
    {
        'lang': 'Java', 'era': '1995-present', 'compiler': 'JDK (OpenJDK / Oracle)',
        'wiki': 'Java_(programming_language)',
        'spec_urls': ['https://en.wikipedia.org/wiki/Java_(programming_language)',
                      'https://en.wikipedia.org/wiki/Java_version_history'],
        'context': 'Write once run anywhere. JVM. Java 1.0->1.4, J2SE 5 (generics), '
                   'Java 6-8 (lambdas, streams), Java 11 (LTS), Java 17 (LTS), Java 21 (LTS). '
                   'Compiler: javac. Runtime: java -jar. JVM flags, GC tuning.',
    },
    {
        'lang': 'Python', 'era': '1991-present', 'compiler': 'CPython / PyPy',
        'wiki': 'Python_(programming_language)',
        'spec_urls': ['https://en.wikipedia.org/wiki/Python_(programming_language)',
                      'https://en.wikipedia.org/wiki/History_of_Python'],
        'context': 'Python 1.x, 2.x (2000-2020 EOL), 3.x (3.6 f-strings, 3.8 walrus, '
                   '3.10 match, 3.12 current). PEP 8 style. GIL. CPython interpreter. '
                   'pip, venv, type hints (3.5+). Runtime: python3 --version.',
    },
    {
        'lang': 'Ruby', 'era': '1995-present', 'compiler': 'MRI / JRuby',
        'wiki': 'Ruby_(programming_language)',
        'spec_urls': ['https://en.wikipedia.org/wiki/Ruby_(programming_language)'],
        'context': 'Dynamic, OOP, duck typing. Blocks, procs, lambdas. Rails framework. '
                   'Ruby 1.8, 1.9 (encoding), 2.x, 3.x (RBS, Ractor). Runtime: ruby --version.',
    },
    {
        'lang': 'JavaScript', 'era': '1995-present', 'compiler': 'V8 / SpiderMonkey / Node.js',
        'wiki': 'JavaScript',
        'spec_urls': ['https://en.wikipedia.org/wiki/JavaScript',
                      'https://en.wikipedia.org/wiki/ECMAScript'],
        'context': 'ECMAScript 3 (1999), ES5 (2009), ES6/ES2015 (arrows, classes, promises, '
                   'modules), ES2017 (async/await), ES2020+. Browser DOM API. Node.js runtime. '
                   'V8 engine. npm/yarn. TypeScript superset.',
    },
    {
        'lang': 'PHP', 'era': '1994-present', 'compiler': 'Zend Engine',
        'wiki': 'PHP',
        'spec_urls': ['https://en.wikipedia.org/wiki/PHP'],
        'context': 'PHP 3, 4, 5 (OOP), 7 (2x speedup, types), 8 (JIT, named args, match). '
                   'Server-side web. LAMP stack. Composer. Runtime: php --version.',
    },
    {
        'lang': 'Perl', 'era': '1987-present', 'compiler': 'Perl interpreter',
        'wiki': 'Perl',
        'spec_urls': ['https://en.wikipedia.org/wiki/Perl'],
        'context': 'Perl 4, 5 (1994, OOP, CPAN), Perl 6/Raku. Regex, CGI, sysadmin. '
                   'TIMTOWTDI. use strict; use warnings. Runtime: perl --version.',
    },
    # -- 2000s --
    {
        'lang': 'C#', 'era': '2000-present', 'compiler': '.NET / Roslyn',
        'wiki': 'C_Sharp_(programming_language)',
        'spec_urls': ['https://en.wikipedia.org/wiki/C_Sharp_(programming_language)',
                      'https://en.wikipedia.org/wiki/C_Sharp_version_history'],
        'context': 'C# 1.0->12.0. .NET Framework->.NET Core->.NET 6/8 (LTS). '
                   'LINQ, async/await, generics, records, nullable ref types. '
                   'Compiler: dotnet build. Runtime: dotnet run.',
    },
    {
        'lang': 'Swift', 'era': '2014-present', 'compiler': 'Swift compiler (LLVM)',
        'wiki': 'Swift_(programming_language)',
        'spec_urls': ['https://en.wikipedia.org/wiki/Swift_(programming_language)'],
        'context': 'Apple Swift 1.0->5.x. Optionals, protocols, ARC, generics. '
                   'iOS/macOS development. Xcode. Swift Package Manager.',
    },
    {
        'lang': 'Kotlin', 'era': '2011-present', 'compiler': 'Kotlin compiler (kotlinc)',
        'wiki': 'Kotlin_(programming_language)',
        'spec_urls': ['https://en.wikipedia.org/wiki/Kotlin_(programming_language)'],
        'context': 'JVM + Native + JS targets. Null safety, data classes, coroutines. '
                   'Android primary language. Interop with Java. kotlinc / gradle.',
    },
    {
        'lang': 'Go', 'era': '2009-present', 'compiler': 'gc (Google Go compiler)',
        'wiki': 'Go_(programming_language)',
        'spec_urls': ['https://en.wikipedia.org/wiki/Go_(programming_language)'],
        'context': 'Go 1.0->1.22. Goroutines, channels, interfaces, defer, panic/recover. '
                   'Static binaries. No inheritance. go build, go test, go mod. '
                   'Runtime: go version. GOPATH vs modules.',
    },
    {
        'lang': 'Rust', 'era': '2015-present', 'compiler': 'rustc (LLVM)',
        'wiki': 'Rust_(programming_language)',
        'spec_urls': ['https://en.wikipedia.org/wiki/Rust_(programming_language)'],
        'context': 'Ownership, borrowing, lifetimes. No GC, no data races. '
                   'Edition 2015, 2018, 2021. cargo build/test/run. '
                   'unsafe blocks, traits, generics, async/await (tokio).',
    },
    {
        'lang': 'TypeScript', 'era': '2012-present', 'compiler': 'tsc (TypeScript compiler)',
        'wiki': 'TypeScript',
        'spec_urls': ['https://en.wikipedia.org/wiki/TypeScript'],
        'context': 'Superset of JavaScript. Static types, interfaces, enums, generics. '
                   'tsc --target ES2020. tsconfig.json. Strict mode. Declaration files .d.ts.',
    },
    {
        'lang': 'Scala', 'era': '2004-present', 'compiler': 'scalac / Scala 3',
        'wiki': 'Scala_(programming_language)',
        'spec_urls': ['https://en.wikipedia.org/wiki/Scala_(programming_language)'],
        'context': 'JVM. Scala 2 vs Scala 3 (Dotty). Functional + OOP. '
                   'Case classes, pattern matching, implicits/givens, for-comprehension.',
    },
    {
        'lang': 'Haskell', 'era': '1990-present', 'compiler': 'GHC',
        'wiki': 'Haskell',
        'spec_urls': ['https://en.wikipedia.org/wiki/Haskell'],
        'context': 'Pure functional, lazy evaluation. Monads, typeclasses, GADTs. '
                   'GHC 9.x. Haskell 98, Haskell 2010. cabal / stack build system.',
    },
    {
        'lang': 'R', 'era': '1993-present', 'compiler': 'R interpreter / GNU R',
        'wiki': 'R_(programming_language)',
        'spec_urls': ['https://en.wikipedia.org/wiki/R_(programming_language)'],
        'context': 'Statistical computing. Vectors, data frames, factors. '
                   'CRAN packages. ggplot2, dplyr, tidyverse. R 4.x. RStudio.',
    },
    {
        'lang': 'MATLAB', 'era': '1984-present', 'compiler': 'MathWorks MATLAB / Octave',
        'wiki': 'MATLAB',
        'spec_urls': ['https://en.wikipedia.org/wiki/MATLAB'],
        'context': 'Matrix-oriented. Toolboxes: Signal Processing, Control, Image Processing. '
                   'MATLAB vs GNU Octave (open source). .m files, mex files.',
    },
    {
        'lang': 'Assembly', 'era': '1949-present', 'compiler': 'NASM / MASM / GAS',
        'wiki': 'Assembly_language',
        'spec_urls': ['https://en.wikipedia.org/wiki/Assembly_language',
                      'https://en.wikipedia.org/wiki/X86_assembly_language'],
        'context': 'x86 (8086->x86-64), ARM, RISC-V. MOV, ADD, SUB, JMP, CALL, RET. '
                   'Registers: AX/EAX/RAX, etc. Segments, interrupts, calling conventions. '
                   'Assemblers: NASM (nasm -f elf64), GAS (as), MASM.',
    },
    {
        'lang': 'SQL', 'era': '1974-present', 'compiler': 'PostgreSQL / MySQL / SQLite / MSSQL',
        'wiki': 'SQL',
        'spec_urls': ['https://en.wikipedia.org/wiki/SQL',
                      'https://en.wikipedia.org/wiki/SQL:2023'],
        'context': 'SQL:1986, SQL:1992, SQL:1999, SQL:2003, SQL:2011, SQL:2016, SQL:2023. '
                   'SELECT, INSERT, UPDATE, DELETE, JOIN, GROUP BY, HAVING, subqueries, CTEs, '
                   'window functions. Dialects: PostgreSQL, MySQL, SQLite, T-SQL (MSSQL), Oracle.',
    },
    {
        'lang': 'Bash/Shell', 'era': '1989-present', 'compiler': 'bash / sh / zsh',
        'wiki': 'Bash_(Unix_shell)',
        'spec_urls': ['https://en.wikipedia.org/wiki/Bash_(Unix_shell)',
                      'https://en.wikipedia.org/wiki/Shell_script'],
        'context': 'Bash 3.x->5.x. POSIX sh compatibility. Variables, arrays, functions, '
                   'pipes, redirects, here-docs, process substitution. '
                   'set -euo pipefail. Runs on: Linux, macOS, WSL, Git Bash.',
    },
    {
        'lang': 'PowerShell', 'era': '2006-present', 'compiler': 'PowerShell / pwsh',
        'wiki': 'PowerShell',
        'spec_urls': ['https://en.wikipedia.org/wiki/PowerShell'],
        'context': 'PowerShell 1.0->7.x (cross-platform). Objects not text. '
                   'Cmdlets, pipelines, modules, .ps1 scripts. '
                   'Windows PowerShell 5.1 vs PowerShell Core 7.x.',
    },
    {
        'lang': 'Lua', 'era': '1993-present', 'compiler': 'Lua interpreter / LuaJIT',
        'wiki': 'Lua_(programming_language)',
        'spec_urls': ['https://en.wikipedia.org/wiki/Lua_(programming_language)'],
        'context': 'Lightweight embeddable. Tables as everything. Metatables, coroutines. '
                   'Lua 5.1 (LuaJIT), 5.2, 5.3, 5.4. Used in: games (Roblox), Neovim, Redis.',
    },
    {
        'lang': 'Dart', 'era': '2011-present', 'compiler': 'Dart VM / dart2js / AOT',
        'wiki': 'Dart_(programming_language)',
        'spec_urls': ['https://en.wikipedia.org/wiki/Dart_(programming_language)'],
        'context': 'Flutter framework. Sound null safety. async/await, streams, isolates. '
                   'Dart 2.x, 3.x. dart compile, flutter build.',
    },
    {
        'lang': 'Julia', 'era': '2012-present', 'compiler': 'Julia JIT (LLVM)',
        'wiki': 'Julia_(programming_language)',
        'spec_urls': ['https://en.wikipedia.org/wiki/Julia_(programming_language)'],
        'context': 'Scientific computing. Multiple dispatch, type system, macros. '
                   'Julia 1.x. REPL. Pkg manager. NumPy-class performance. '
                   'Used in: HPC, differential equations, ML.',
    },
    {
        'lang': 'VHDL', 'era': '1983-present', 'compiler': 'GHDL / ModelSim / Vivado',
        'wiki': 'VHDL',
        'spec_urls': ['https://en.wikipedia.org/wiki/VHDL'],
        'context': 'Hardware description language. VHDL-87, 93, 2002, 2008. '
                   'Entity/architecture structure. std_logic, std_logic_vector. '
                   'Concurrent and sequential processes. Synthesis vs simulation. '
                   'Used for: FPGA (Xilinx, Intel/Altera) and ASIC design. '
                   'Testbenches, assert statements. ghdl -a/-e/-r for sim. '
                   'Vivado/Quartus for synthesis to bitstream.',
    },
    {
        'lang': 'Verilog / SystemVerilog', 'era': '1985-present',
        'compiler': 'Icarus Verilog / Verilator / ModelSim / Vivado',
        'wiki': 'Verilog',
        'spec_urls': ['https://en.wikipedia.org/wiki/Verilog',
                      'https://en.wikipedia.org/wiki/SystemVerilog'],
        'context': 'IEEE 1364 Verilog, IEEE 1800 SystemVerilog. '
                   'module/endmodule, wire, reg, always, assign. '
                   'Blocking (=) vs non-blocking (<=) assignments. '
                   'always @(posedge clk) for sequential logic. '
                   'Testbenches: $display, $finish, #delay. '
                   'Synthesis: Xilinx/Intel FPGAs. iverilog for simulation.',
    },
    {
        'lang': 'MicroPython', 'era': '2013-present', 'compiler': 'MicroPython firmware',
        'wiki': 'MicroPython',
        'spec_urls': ['https://en.wikipedia.org/wiki/MicroPython'],
        'context': 'Subset of Python 3 for microcontrollers. '
                   'Runs on: ESP8266, ESP32, RP2040 (Pi Pico), STM32, BBC micro:bit. '
                   'machine module: Pin, ADC, PWM, I2C, SPI, UART, Timer. '
                   'Flash filesystem (littlefs). REPL over serial/WebREPL. '
                   'uasyncio for async. Network/WiFi socket on ESP32.',
    },
    {
        'lang': 'Arduino C++', 'era': '2005-present',
        'compiler': 'avr-gcc / arm-none-eabi-gcc (Arduino IDE / PlatformIO)',
        'wiki': 'Arduino',
        'spec_urls': ['https://en.wikipedia.org/wiki/Arduino'],
        'context': 'C++ for Arduino microcontrollers (AVR, SAMD, RP2040, ESP32). '
                   'setup() runs once at power-on; loop() runs forever. '
                   'pinMode(pin, INPUT/OUTPUT/INPUT_PULLUP). '
                   'digitalWrite(pin, HIGH/LOW). digitalRead(pin). '
                   'analogRead(pin) returns 0-1023 (10-bit ADC, 0-5V on Uno). '
                   'analogWrite(pin, 0-255) for PWM. delay(ms). millis(). '
                   'Serial.begin(9600). Serial.println(val). '
                   'Wire.h (I2C), SPI.h, Servo.h, Ethernet.h libraries. '
                   'Boards: Uno (ATmega328P @16MHz), Nano, Mega (ATmega2560), '
                   'Due (ARM Cortex-M3), Leonardo, Pro Mini.',
    },
    {
        'lang': 'C (Embedded/Bare-metal)', 'era': '1972-present',
        'compiler': 'avr-gcc / arm-none-eabi-gcc / sdcc',
        'wiki': 'Embedded_C',
        'spec_urls': ['https://en.wikipedia.org/wiki/Embedded_C',
                      'https://en.wikipedia.org/wiki/AVR_microcontrollers'],
        'context': 'Bare-metal C for microcontrollers without OS. '
                   'avr-gcc: -mmcu=atmega328p -DF_CPU=16000000UL -O2. '
                   'AVR GPIO: DDRB (direction), PORTB (output), PINB (input). '
                   'Bit manipulation: PORTB |= (1<<PB5); PORTB &= ~(1<<PB5). '
                   'arm-none-eabi-gcc: -mcpu=cortex-m4 -mthumb -mfloat-abi=hard. '
                   'Linker scripts (.ld) define FLASH/RAM regions. '
                   'volatile for memory-mapped I/O registers. '
                   'ISR(TIMER1_OVF_vect) on AVR. NVIC for ARM Cortex-M. '
                   'avrdude flash: -p m328p -c arduino -P /dev/ttyUSB0. '
                   'OpenOCD + GDB for ARM JTAG/SWD debugging.',
    },
    {
        'lang': 'COBOL', 'era': '1959-present',
        'compiler': 'GnuCOBOL / IBM Enterprise COBOL',
        'wiki': 'COBOL',
        'spec_urls': ['https://en.wikipedia.org/wiki/COBOL'],
        'context': 'Business-oriented language. COBOL-60, 68, 74, 85, 2002, 2014, 2023. '
                   'Four divisions: IDENTIFICATION, ENVIRONMENT, DATA, PROCEDURE. '
                   'WORKING-STORAGE SECTION for variables. '
                   'PERFORM, MOVE, COMPUTE, DISPLAY, ACCEPT, OPEN/READ/WRITE/CLOSE. '
                   'Still handles 95% of ATM transactions. Mainframe (IBM z/OS). '
                   'GnuCOBOL open source. Fixed-format: cols 7-72 code area.',
    },
    {
        'lang': 'Visual Basic / VBA', 'era': '1991-present',
        'compiler': 'VB6 / VBScript / VB.NET / VBA (Excel)',
        'wiki': 'Visual_Basic_(classic)',
        'spec_urls': ['https://en.wikipedia.org/wiki/Visual_Basic_(classic)',
                      'https://en.wikipedia.org/wiki/Visual_Basic_.NET'],
        'context': 'VB1-VB6 (1991-1998): event-driven GUI, COM/ActiveX, forms/controls. '
                   'VB.NET (2002+): .NET CLR, full OOP. '
                   'VBA: macro scripting in Office (Excel, Word, Access). '
                   'Dim x As Integer. Sub MySub() / End Sub. '
                   'Function MyFunc() As String / End Function. '
                   'MsgBox, InputBox, For/Next, Do While/Loop, With/End With.',
    },
    {
        'lang': 'ALGOL', 'era': '1958-1970', 'compiler': 'ALGOL 60 / ALGOL 68',
        'wiki': 'ALGOL',
        'spec_urls': ['https://en.wikipedia.org/wiki/ALGOL'],
        'context': 'Ancestor of Pascal, C, Java, and most procedural languages. '
                   'ALGOL 58, 60, 68. Block structure with begin/end, scope rules. '
                   'First language specified with BNF grammar. '
                   'Introduced if/then/else, for loops, recursive procedures. '
                   'ALGOL 68 added orthogonality: any type anywhere.',
    },
]

# -- Stage 12b: Code examples (plain-English -> code, versioned) ----------------
# Each example: description, year, platform, lang, compiler, code, notes
# These are trained as rich structured text for English->code association.

PROG_CODE_EXAMPLES = [
    {
        'description': 'Print Hello World to screen',
        'year': '1981', 'platform': 'MS-DOS / IBM PC 5150',
        'lang': 'GW-BASIC', 'compiler': 'GW-BASIC 3.22 (Microsoft)',
        'code': '10 PRINT "Hello, World!"\n20 END',
        'notes': 'Line numbers required. RUN executes. LIST shows program. '
                 'SAVE "HELLO.BAS" to cassette or disk. '
                 'Gotcha: GOTO/GOSUB use line numbers, not labels.',
    },
    {
        'description': 'Print Hello World to screen',
        'year': '1991', 'platform': 'MS-DOS 5 / Windows 3.1',
        'lang': 'QBasic', 'compiler': 'QBasic 1.1 (included with MS-DOS 5+)',
        'code': 'PRINT "Hello, World!"',
        'notes': 'No line numbers needed. Press F5 to run in IDE. '
                 'SUB/FUNCTION for structure. No GOTO required. '
                 'Saved as .BAS text file.',
    },
    {
        'description': 'Print Hello World -- original from K&R book',
        'year': '1978', 'platform': 'PDP-11 UNIX',
        'lang': 'C (K&R pre-ANSI)', 'compiler': 'cc (Bell Labs cc)',
        'code': '#include <stdio.h>\nmain()\n{\n    printf("Hello, World!\\n");\n}',
        'notes': 'No return type on main in K&R C. No int, no return 0. '
                 'Directly from Kernighan & Ritchie "The C Programming Language" (1978) p.6. '
                 'Gotcha: implicit int was allowed; ANSI C89 made int explicit.',
    },
    {
        'description': 'Print Hello World -- ANSI standard C',
        'year': '1989', 'platform': 'UNIX / DOS',
        'lang': 'C (ANSI C89 / ISO C90)', 'compiler': 'gcc -std=c89 or turbo c 2.0',
        'code': '#include <stdio.h>\nint main(void) {\n    printf("Hello, World!\\n");\n    return 0;\n}',
        'notes': 'ANSI C89 (1989) / ISO C90. int main(void). return 0. '
                 'Compile: gcc -std=c89 -Wall hello.c -o hello. '
                 'Gotcha: void in parameter list means no args; main() without void allows any args in C.',
    },
    {
        'description': 'Print Hello World',
        'year': '1991', 'platform': 'DOS (Turbo C++ 3.0)',
        'lang': 'C++ (pre-standard Borland)', 'compiler': 'Turbo C++ 3.0',
        'code': '#include <iostream.h>\nvoid main() {\n    cout << "Hello, World!" << endl;\n}',
        'notes': '<iostream.h> (old header, no std:: namespace). void main() accepted by Borland. '
                 'Standard C++98 requires <iostream> and std::cout and int main().',
    },
    {
        'description': 'Print Hello World -- ISO C++98',
        'year': '1998', 'platform': 'Linux / Windows',
        'lang': 'C++ (ISO C++98)', 'compiler': 'g++ -std=c++98',
        'code': '#include <iostream>\nint main() {\n    std::cout << "Hello, World!" << std::endl;\n    return 0;\n}',
        'notes': '<iostream> without .h. std:: namespace required. '
                 'Compile: g++ -std=c++98 hello.cpp -o hello. '
                 'Gotcha: endl flushes buffer (slower); "\\n" is faster.',
    },
    {
        'description': 'Print Hello World',
        'year': '1995', 'platform': 'Any JVM',
        'lang': 'Java 1.0', 'compiler': 'javac (JDK 1.0, Sun)',
        'code': 'public class Hello {\n    public static void main(String[] args) {\n        System.out.println("Hello, World!");\n    }\n}',
        'notes': 'File must be Hello.java (matches class name). '
                 'Compile: javac Hello.java  Run: java Hello. '
                 'Gotcha: class name must match filename exactly, case-sensitive.',
    },
    {
        'description': 'Print Hello World',
        'year': '1994', 'platform': 'Python 1.x on Linux',
        'lang': 'Python 1.x', 'compiler': 'python 1.0.1',
        'code': 'print "Hello, World!"',
        'notes': 'print is a statement in Python 1.x and 2.x. '
                 'Python 3.0 (2008) changed it to a function: print("Hello, World!"). '
                 'Gotcha: mixing Python 2 print statement in Python 3 is SyntaxError.',
    },
    {
        'description': 'Print Hello World',
        'year': '2008', 'platform': 'Python 3.x on any OS',
        'lang': 'Python 3.0+', 'compiler': 'python3 / CPython 3.12',
        'code': 'print("Hello, World!")',
        'notes': 'print() is a function in Python 3. '
                 'Run: python3 hello.py. '
                 'Gotcha: python may point to Python 2 on older systems; use python3 explicitly.',
    },
    {
        'description': 'Blink LED connected to pin 13 (Arduino Uno built-in LED)',
        'year': '2005', 'platform': 'Arduino Uno (ATmega328P @16MHz)',
        'lang': 'Arduino C++', 'compiler': 'avr-gcc 7.3.0 via Arduino IDE 2.3',
        'code': 'void setup() {\n  pinMode(13, OUTPUT);\n}\nvoid loop() {\n  digitalWrite(13, HIGH);\n  delay(1000);\n  digitalWrite(13, LOW);\n  delay(1000);\n}',
        'notes': 'setup() called once. loop() runs forever. delay(1000) = 1 second. '
                 'Pin 13 = PB5 on ATmega328P = built-in LED on Uno. '
                 'Upload: Tools -> Board -> Arduino Uno, then Ctrl+U. '
                 'Gotcha: delay() blocks all other code; use millis() for non-blocking timing.',
    },
    {
        'description': 'Read analog voltage from sensor on pin A0',
        'year': '2005', 'platform': 'Arduino Uno (ATmega328P)',
        'lang': 'Arduino C++', 'compiler': 'avr-gcc via Arduino IDE',
        'code': 'void setup() {\n  Serial.begin(9600);\n}\nvoid loop() {\n  int raw = analogRead(A0);\n  float volts = raw * (5.0 / 1023.0);\n  Serial.print("ADC="); Serial.print(raw);\n  Serial.print("  V="); Serial.println(volts);\n  delay(500);\n}',
        'notes': 'analogRead() returns 0-1023 (10-bit, 0-5V reference). '
                 'Serial.begin(9600) sets baud rate for USB serial monitor. '
                 'Gotcha: analogRead on 3.3V boards (Due, Zero) uses 3.3V reference; adjust formula.',
    },
    {
        'description': 'Blink LED bare-metal AVR (no Arduino framework)',
        'year': '1997', 'platform': 'ATmega328P / Arduino Uno hardware only',
        'lang': 'C (bare-metal avr-gcc)', 'compiler': 'avr-gcc 7.x -mmcu=atmega328p',
        'code': '#include <avr/io.h>\n#include <util/delay.h>\nint main(void) {\n    DDRB |= (1 << PB5);   /* PB5 = output */\n    for (;;) {\n        PORTB |= (1 << PB5);  /* HIGH */\n        _delay_ms(1000);\n        PORTB &= ~(1 << PB5); /* LOW */\n        _delay_ms(1000);\n    }\n}',
        'notes': 'DDRB = Data Direction Register B. 1=output, 0=input. '
                 'PORTB = output latch. PINB = input register. '
                 'Compile: avr-gcc -mmcu=atmega328p -DF_CPU=16000000UL -O2 -o blink.elf blink.c\n'
                 'avr-objcopy -O ihex blink.elf blink.hex\n'
                 'avrdude -p m328p -c arduino -P COM3 -b 115200 -U flash:w:blink.hex',
    },
    {
        'description': 'Toggle GPIO on ARM Cortex-M4 (STM32F4) bare-metal',
        'year': '2011', 'platform': 'STM32F4 Discovery (ARM Cortex-M4 @168MHz)',
        'lang': 'C (bare-metal arm-none-eabi-gcc)', 'compiler': 'arm-none-eabi-gcc 10.x',
        'code': '#include "stm32f4xx.h"\nint main(void) {\n    RCC->AHB1ENR |= RCC_AHB1ENR_GPIODEN;  /* enable GPIOD clock */\n    GPIOD->MODER |= (1 << 26);             /* PD13 = output */\n    while (1) {\n        GPIOD->ODR ^= (1 << 13);           /* toggle */\n        for (int i=0; i<1000000; i++) {}   /* crude delay */\n    }\n}',
        'notes': 'STM32 uses CMSIS headers. Must enable peripheral clock via RCC first. '
                 'MODER: 00=input, 01=output, 10=alternate function, 11=analog. '
                 'Compile: arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16\n'
                 'Flash via STLink: openocd -f interface/stlink.cfg -f target/stm32f4x.cfg',
    },
    {
        'description': 'Send data over I2C to MPU-6050 gyroscope on Arduino',
        'year': '2005', 'platform': 'Arduino Uno, Wire.h I2C library',
        'lang': 'Arduino C++', 'compiler': 'avr-gcc via Arduino IDE',
        'code': '#include <Wire.h>\nvoid setup() {\n  Wire.begin();\n  Wire.beginTransmission(0x68); /* MPU-6050 I2C addr */\n  Wire.write(0x6B);              /* PWR_MGMT_1 register */\n  Wire.write(0);                 /* wake up (clear sleep bit) */\n  Wire.endTransmission();\n  Serial.begin(9600);\n}\nvoid loop() {\n  Wire.beginTransmission(0x68);\n  Wire.write(0x3B);  /* ACCEL_XOUT_H */\n  Wire.endTransmission(false);\n  Wire.requestFrom(0x68, 6);\n  int16_t ax = (Wire.read() << 8) | Wire.read();\n  Serial.println(ax);\n  delay(100);\n}',
        'notes': 'I2C on Uno: SDA=A4, SCL=A5. Wire.begin() = master. '
                 'beginTransmission/write/endTransmission = write sequence. '
                 'requestFrom() reads N bytes. MPU-6050 default addr=0x68 (AD0 low) or 0x69 (AD0 high).',
    },
    {
        'description': 'Allocate and free heap memory in C -- avoid memory leak',
        'year': '1989', 'platform': 'UNIX / Windows',
        'lang': 'C (ANSI C89)', 'compiler': 'gcc',
        'code': '#include <stdlib.h>\n#include <string.h>\nint main(void) {\n    int *arr = (int *)malloc(10 * sizeof(int));\n    if (arr == NULL) return 1;  /* always check */\n    memset(arr, 0, 10 * sizeof(int));\n    arr[5] = 42;\n    free(arr);\n    arr = NULL;  /* prevent dangling pointer use */\n    return 0;\n}',
        'notes': 'malloc returns NULL on failure. sizeof ensures portability. '
                 'Set pointer NULL after free to prevent accidental use. '
                 'Gotcha: double-free is undefined behavior. Use valgrind to detect leaks.',
    },
    {
        'description': 'Fibonacci sequence iterative -- early JavaScript',
        'year': '1999', 'platform': 'Browser (IE/Netscape) or Node.js',
        'lang': 'JavaScript ES3 (1999)', 'compiler': 'Any JS engine (V8/SpiderMonkey)',
        'code': 'function fib(n) {\n    var a = 0, b = 1, temp;\n    for (var i = 0; i < n; i++) {\n        temp = a; a = b; b = temp + b;\n    }\n    return a;\n}\nconsole.log(fib(10)); // 55',
        'notes': 'ES3 (1999): var has function scope, not block scope. '
                 'ES6 (2015) equivalent: let, const, destructuring: [a,b]=[b,a+b]. '
                 'Gotcha: var declarations are hoisted to function top.',
    },
    {
        'description': 'Python generator for infinite Fibonacci sequence',
        'year': '2001', 'platform': 'Python 2.2+ / Python 3.x',
        'lang': 'Python 2.2+', 'compiler': 'python / python3',
        'code': 'def fib_gen():\n    a, b = 0, 1\n    while True:\n        yield a\n        a, b = b, a + b\n\ng = fib_gen()\nfor _ in range(10):\n    print(next(g))',
        'notes': 'yield makes a generator (lazy iterator, PEP 255, Python 2.2). '
                 'next() pulls the next value. Infinite but memory-constant. '
                 'Gotcha: Python 2 uses g.next(); Python 3 uses next(g).',
    },
    {
        'description': 'Go HTTP server',
        'year': '2012', 'platform': 'Linux / Windows / macOS (Go 1.x)',
        'lang': 'Go 1.x', 'compiler': 'go build (gc compiler)',
        'code': 'package main\nimport (\n    "fmt"\n    "net/http"\n)\nfunc handler(w http.ResponseWriter, r *http.Request) {\n    fmt.Fprintf(w, "Hello from %s", r.URL.Path)\n}\nfunc main() {\n    http.HandleFunc("/", handler)\n    http.ListenAndServe(":8080", nil)\n}',
        'notes': 'Standard library net/http -- no frameworks needed. '
                 'ListenAndServe blocks. Run: go run main.go. '
                 'Gotcha: error return from ListenAndServe is ignored here; check in production.',
    },
    {
        'description': 'Rust ownership: move vs clone',
        'year': '2015', 'platform': 'Any (Rust 1.0+)',
        'lang': 'Rust 1.0+', 'compiler': 'rustc / cargo',
        'code': 'fn main() {\n    let s1 = String::from("hello");\n    let s2 = s1;  // s1 MOVED -- no longer valid\n    // println!("{}", s1);  // compile error: value borrowed after move\n    println!("{}", s2);\n\n    let s3 = String::from("world");\n    let s4 = s3.clone();  // deep copy -- both valid\n    println!("{} {}", s3, s4);\n}',
        'notes': 'Assignment of heap types moves ownership in Rust (no GC, no copy). '
                 'clone() deep-copies. Copy trait types (i32, bool, f64) copy implicitly. '
                 'Compile: cargo run. Borrow checker enforces this at compile time.',
    },
    {
        'description': 'VHDL: 8-bit adder entity',
        'year': '1987', 'platform': 'FPGA synthesis / simulation',
        'lang': 'VHDL-87/93', 'compiler': 'GHDL 3.x / Vivado 2023',
        'code': 'library IEEE;\nuse IEEE.STD_LOGIC_1164.ALL;\nuse IEEE.NUMERIC_STD.ALL;\nentity adder8 is\n  Port (a : in  STD_LOGIC_VECTOR(7 downto 0);\n        b : in  STD_LOGIC_VECTOR(7 downto 0);\n        s : out STD_LOGIC_VECTOR(8 downto 0));\nend adder8;\narchitecture Behavioral of adder8 is\nbegin\n  s <= STD_LOGIC_VECTOR(\n    unsigned(\'0\' & a) + unsigned(\'0\' & b));\nend Behavioral;',
        'notes': 'STD_LOGIC_VECTOR is a bit array. downto = MSB at left. '
                 'unsigned() for arithmetic. \'0\' & a pads 1 bit for carry-out. '
                 'Simulate: ghdl -a adder8.vhd && ghdl -e adder8 && ghdl -r adder8',
    },
    {
        'description': 'Verilog: synchronous 4-bit counter',
        'year': '1995', 'platform': 'FPGA / simulation',
        'lang': 'Verilog 2001 (IEEE 1364-2001)', 'compiler': 'iverilog 11.x',
        'code': 'module counter4(\n  input  wire       clk,\n  input  wire       rst,\n  output reg  [3:0] count\n);\nalways @(posedge clk or posedge rst) begin\n  if (rst)\n    count <= 4\'b0000;\n  else\n    count <= count + 1\'b1;\nend\nendmodule',
        'notes': 'Non-blocking (<=) for sequential logic (registers). '
                 'Blocking (=) for combinational always blocks. '
                 '4\'b0000 = 4-bit binary literal. 1\'b1 = 1-bit literal 1. '
                 'Simulate: iverilog -o cnt counter4.v tb_counter4.v && ./cnt',
    },
    {
        'description': 'MicroPython: blink LED on Raspberry Pi Pico (RP2040)',
        'year': '2021', 'platform': 'Raspberry Pi Pico (RP2040)',
        'lang': 'MicroPython 1.20+', 'compiler': 'MicroPython firmware (RP2040)',
        'code': 'from machine import Pin\nimport time\n\nled = Pin(25, Pin.OUT)  # GP25 = onboard LED on Pico\nwhile True:\n    led.toggle()\n    time.sleep(0.5)',
        'notes': 'Pin(25, Pin.OUT) configures GP25 as output. '
                 'toggle() flips state. time.sleep() in seconds (float). '
                 'Flash firmware: hold BOOTSEL, plug USB, copy .uf2 file. '
                 'Then upload code via Thonny IDE or rshell.',
    },
    {
        'description': 'x86-64 assembly: print Hello World on Linux via syscall',
        'year': '2003', 'platform': 'Linux x86-64',
        'lang': 'x86-64 Assembly (NASM syntax)', 'compiler': 'nasm 2.15 + ld',
        'code': 'section .data\n    msg db "Hello, World!", 10  ; 10 = newline\n    len equ $ - msg\nsection .text\n    global _start\n_start:\n    mov rax, 1       ; syscall: sys_write\n    mov rdi, 1       ; fd: stdout\n    mov rsi, msg     ; buffer\n    mov rdx, len     ; length\n    syscall\n    mov rax, 60      ; syscall: sys_exit\n    xor rdi, rdi     ; exit code 0\n    syscall',
        'notes': 'Linux x86-64 syscall: rax=number, rdi/rsi/rdx/r10/r8/r9=args. '
                 'sys_write=1, sys_exit=60. syscall instruction (not int 0x80 which is 32-bit). '
                 'Build: nasm -f elf64 hello.asm -o hello.o && ld hello.o -o hello',
    },
    {
        'description': '8086 assembly: add two numbers and exit (MS-DOS COM program)',
        'year': '1981', 'platform': 'MS-DOS (IBM PC, 8086/8088)',
        'lang': 'x86 16-bit Assembly (MASM/TASM syntax)', 'compiler': 'MASM 5.0 / TASM 3.0',
        'code': '; MS-DOS .COM program -- runs at CS:100h\nORG 100h\nMOV AX, 5     ; AX = 5\nMOV BX, 3     ; BX = 3\nADD AX, BX    ; AX = 8\nMOV AH, 4Ch   ; DOS service: exit\nXOR AL, AL    ; exit code 0\nINT 21h       ; call DOS\n',
        'notes': 'ORG 100h: .COM files load at offset 100h in segment. '
                 'INT 21h / AH=4Ch = DOS terminate with exit code in AL. '
                 'Real mode: 16-bit registers AX BX CX DX SI DI SP BP. '
                 'Segmented: CS:IP (code), SS:SP (stack), DS: (data), ES: (extra). '
                 'Build: masm hello.asm; link hello.obj; exe2bin hello.exe hello.com',
    },
]

# -- Stage 13: CAD / 3D printing / Manufacturing / Blender ---------------------
CAD_3D_WIKI_ARTICLES = [
    # Blender
    'Blender_(software)', 'Blender_Game_Engine', 'Blender_Foundation',
    # CAD tools
    'AutoCAD', 'FreeCAD', 'OpenSCAD', 'SolidWorks', 'CATIA',
    'Fusion_360', 'Inventor_(software)', 'Rhino3D',
    # 3D printing / additive manufacturing
    '3D_printing', 'Fused_deposition_modeling', 'Stereolithography',
    'Selective_laser_sintering', 'Binder_jetting', 'Digital_light_processing',
    'Stereolithography_file_format', 'G-code',
    'RepRap_project', 'Ultimaker', 'Prusa_i3',
    # File formats
    'STEP_file', 'Initial_Graphics_Exchange_Specification', 'DXF',
    'Wavefront_.obj_file', 'STL_(file_format)',
    'GL_Transmission_Format', 'Collada',
    # Engineering / manufacturing
    'Computer-aided_design', 'Computer-aided_manufacturing',
    'Computer-aided_engineering', 'Finite_element_method',
    'Computational_fluid_dynamics', 'Numerical_control',
    'Machining', 'Injection_moulding', 'Die_casting',
    'Tolerances_in_engineering', 'Geometric_dimensioning_and_tolerancing',
    # Automotive
    'Automotive_engineering', 'Internal_combustion_engine',
    'Transmission_(mechanics)', 'Suspension_(vehicle)',
    'Chassis', 'Monocoque', 'Body-on-frame',
    'Powertrain', 'Drivetrain', 'Electric_vehicle',
    'Automotive_design', 'Crashworthiness',
    # Robotics / automation
    'Robotics', 'Robot_kinematics', 'Degrees_of_freedom_(mechanics)',
    'Servo_motor', 'Stepper_motor', 'PID_controller',
    'Programmable_logic_controller', 'Industrial_robot',
]

CAD_3D_TEXT_CORPUS = [
    # Blender workflow
    ('blender_basics', 'Blender 3D software workflow: Object mode vs Edit mode. '
     'Mesh editing: vertices (V), edges (E), faces (F). '
     'Loop cut (Ctrl+R), Extrude (E), Inset (I), Bevel (Ctrl+B). '
     'Modifiers: Subdivision Surface, Boolean, Mirror, Array, Solidify. '
     'Rendering: Cycles (ray trace) vs EEVEE (rasterize). '
     'Rigging: armatures, weight painting, inverse kinematics. '
     'Python scripting: bpy module. bpy.ops.mesh, bpy.data, bpy.context. '
     'File formats: .blend (native), export to .obj .fbx .gltf .stl .ply.'),

    ('blender_modeling', 'Blender mesh modeling techniques: '
     'Polygon modeling from primitives (UV sphere, cube, cylinder, plane). '
     'Box modeling: start with cube, extrude faces, refine with loops. '
     'NURBS curves and surfaces for smooth organic shapes. '
     'Metaballs for organic blobby forms. '
     'Sculpting mode: Draw, Inflate, Smooth, Crease, Grab brushes. '
     'Retopology: drawing clean quads over high-poly sculpt. '
     'UV unwrapping: Smart UV Project, seams, island packing. '
     'Texture painting: vertex paint, image textures, normal maps.'),

    ('openscad_basics', 'OpenSCAD parametric CAD language: '
     'CSG (Constructive Solid Geometry): union(), difference(), intersection(). '
     'Primitives: cube([x,y,z]), cylinder(h,r), sphere(r). '
     'Transforms: translate([x,y,z]), rotate([x,y,z]), scale([x,y,z]). '
     'Variables and modules (functions). for() loops. '
     'Export to STL for 3D printing. F5=preview, F6=render, F7=export. '
     'Used for: mechanical parts, enclosures, prosthetics. '
     'Parametric design: change one value updates whole model.'),

    ('fdm_3d_printing', '3D printing FDM (Fused Deposition Modeling): '
     'Layer height: 0.1-0.3mm typical. Infill: 10-100% (gyroid, honeycomb, grid). '
     'Print speeds: 40-150mm/s. Bed temp: PLA 60C, ABS 100C, PETG 80C. '
     'Nozzle temp: PLA 190-220C, ABS 230-250C, PETG 220-240C. '
     'Slicers: Cura, PrusaSlicer, Bambu Studio, SuperSlicer. '
     'Supports: tree supports vs normal. Brim vs raft for adhesion. '
     'Retraction settings to prevent stringing. '
     'Post-processing: sanding, acetone smoothing (ABS), priming, painting. '
     'Printer types: Cartesian (Prusa), CoreXY (Bambu, Voron), Delta.'),

    ('gcode_basics', 'G-code for CNC / 3D printing: '
     'G0/G1: rapid/linear move. G28: home all axes. G29: bed leveling. '
     'G92: set position. M104/M109: set/wait extruder temp. '
     'M140/M190: bed temp. M82/M83: absolute/relative extrusion. '
     'F: feed rate mm/min. E: extruder position. '
     'Coordinate systems: G17 XY-plane, G20 inches, G21 mm. '
     'Spindle: M3 (CW), M4 (CCW), M5 (stop). Coolant: M7, M8, M9. '
     'CNC mills use G41/G42 cutter radius compensation.'),

    ('automotive_engineering', 'Automotive engineering fundamentals: '
     'Powertrain: engine, transmission, driveshaft, differential, wheels. '
     'Engine types: inline-4, V6, V8, boxer, Wankel, electric motor. '
     'Transmission: manual (clutch, gears), automatic (planetary gearsets, torque converter), '
     'CVT, DCT (dual-clutch), single-speed (EV). '
     'Suspension: MacPherson strut (front), multi-link, double wishbone, solid axle. '
     'Brakes: disc (caliper, rotor, pads), drum, ABS (anti-lock), EBD, ESC. '
     'Steering: rack-and-pinion, EPS (electric power steering). '
     'Body: monocoque unibody vs body-on-frame. CRUMPLE ZONES, side-impact beams. '
     'EV: battery pack (kWh), motor (kW/Nm), regenerative braking, BMS, CAN bus.'),

    ('cad_tolerances', 'Engineering tolerances and GD&T: '
     'Tolerance: allowable deviation from nominal dimension. '
     'Fit types: clearance (shaft smaller than hole), interference (press fit), transition. '
     'ISO limits and fits: H7/h6 (clearance), H7/p6 (interference). '
     'GD&T symbols: flatness, straightness, circularity, cylindricity, '
     'parallelism, perpendicularity, angularity, position, concentricity, runout. '
     'Datum references: primary, secondary, tertiary. '
     'MMC (maximum material condition), LMC, RFS. '
     'Surface finish: Ra (arithmetic mean), Rz (max height). '
     'Drawn on engineering drawings per ASME Y14.5 or ISO 1101.'),

    ('finite_element_method', 'Finite Element Analysis (FEA): '
     'Discretize geometry into elements (tetrahedra, hexahedra). '
     'Nodes: points where equations solved. DOF: degrees of freedom. '
     'Mesh: coarser far from stress concentrations, finer near them. '
     'Element types: shell (thin surfaces), solid (3D volumes), beam (1D). '
     'Loads: forces, pressures, moments. BCs: fixed, pinned, roller. '
     'Solve: [K]{u} = {F} (stiffness matrix x displacement = force vector). '
     'Results: displacement, von Mises stress, principal stresses, factor of safety. '
     'Software: ANSYS, Abaqus, SolidWorks Simulation, FEniCS (open source), CalculiX.'),
]

# -- Stage 16: Electronics, Digital Logic, Circuit Theory ----------------------

ELECTRONICS_TEXT_CORPUS = [
    ('boolean_algebra',
     'Boolean algebra: variables have values 0 or 1. '
     'Operations: AND (A.B or AB), OR (A+B), NOT (A\' or ~A). '
     'Laws: Identity (A+0=A, A.1=A), Null (A+1=1, A.0=0), '
     'Idempotent (A+A=A, A.A=A), Complement (A+A\'=1, A.A\'=0), '
     'Double negation (~~A=A), De Morgan\'s: ~(A.B)=~A+~B, ~(A+B)=~A.~B. '
     'Absorb: A+AB=A, A(A+B)=A. Consensus: AB+A\'C+BC=AB+A\'C. '
     'XOR: A⊕B = AB\'+A\'B. XNOR: A⊙B = AB+A\'B\'. '
     'Used in: digital circuit design, switching theory, logic simplification.'),

    ('logic_gates',
     'Digital logic gates -- physical implementation of Boolean operations. '
     'AND gate: output 1 only when ALL inputs are 1. Truth table: 00->0,01->0,10->0,11->1. '
     'OR gate: output 1 when ANY input is 1. 00->0,01->1,10->1,11->1. '
     'NOT gate (inverter): output is complement of input. 0->1, 1->0. '
     'NAND gate: NOT AND. Universal gate -- can implement any Boolean function. '
     'NOR gate: NOT OR. Also universal. '
     'XOR (Exclusive OR): output 1 when inputs DIFFER. 00->0,01->1,10->1,11->0. '
     'XNOR: output 1 when inputs SAME. 00->1,01->0,10->0,11->1. '
     'Buffer: output = input. Used for signal amplification/fanout. '
     'Tri-state buffer: enables/disables output (high-Z state for bus sharing). '
     'Physical: TTL (7400 series), CMOS (4000 series, 74HC, 74HCT). '
     'Gate propagation delay: typically 1-10 ns. Fanout: max inputs a gate can drive.'),

    ('karnaugh_maps',
     'Karnaugh map (K-map): graphical method to minimize Boolean expressions. '
     '2-variable: 2x2 grid. 3-variable: 2x4 grid. 4-variable: 4x4 grid. '
     'Gray code ordering: 00,01,11,10 (only one bit changes per step). '
     'Group 1s in powers of 2 (1,2,4,8,16). Groups wrap around edges. '
     'Larger groups = simpler expression. Overlapping groups allowed. '
     'Prime implicants: largest possible groups. '
     'Essential prime implicant: covers a 1 not covered by any other group. '
     'Example: F(A,B,C,D) with K-map -> minimal SOP (Sum of Products). '
     'Don\'t care (X): cells where output is irrelevant -- can be 0 or 1 to simplify. '
     'Equivalent to Quine-McCluskey algorithm (tabular minimization).'),

    ('number_systems',
     'Number systems in computing. '
     'Binary (base-2): digits 0,1. 1010₂ = 10₁₀. '
     'Octal (base-8): digits 0-7. 012₈ = 10₁₀. Prefix 0 in C (0755). '
     'Hexadecimal (base-16): digits 0-9,A-F. 0xA = 10₁₀. Prefix 0x in C. '
     'Conversion: binary->hex: group 4 bits from right (1010 1111 = 0xAF). '
     'Signed integers: Two\'s complement. -1 = all 1s. '
     'To negate: invert all bits then add 1. '
     '8-bit range: -128 to +127. 16-bit: -32768 to +32767. 32-bit: -2147483648 to +2147483647. '
     'Unsigned 8-bit: 0 to 255. '
     'Float: IEEE 754. 32-bit: 1 sign, 8 exp (bias 127), 23 mantissa. '
     '64-bit: 1 sign, 11 exp (bias 1023), 52 mantissa. '
     'Special: +inf, -inf, NaN (0/0), -0. '
     'Gotcha: 0.1 + 0.2 != 0.3 in IEEE 754 floating point.'),

    ('ttl_cmos_families',
     'Logic IC families: TTL (Transistor-Transistor Logic) vs CMOS (Complementary MOS). '
     'TTL (7400 series, 1964): bipolar transistors. VCC=5V. '
     'Logic levels: HIGH >=2.4V, LOW <=0.4V (output); HIGH >=2.0V, LOW <=0.8V (input). '
     'Propagation delay ~10ns. Power: ~10mW/gate. Fanout: 10 standard TTL loads. '
     'Sub-families: 74LS (low-power Schottky), 74S (Schottky), 74F (fast), 74AS, 74ALS. '
     'CMOS (4000 series, 1968): MOSFETs. Wide supply: 3-18V. Near-zero static power. '
     '74HC: CMOS speed comparable to LS-TTL. 74HCT: CMOS with TTL-compatible inputs (5V). '
     '74LVC: 3.3V logic, 5V tolerant inputs. '
     'Modern: 74AHC, 74ALVC for high-speed low-voltage. '
     'Mixing TTL and CMOS: 74HCT as interface (CMOS output drives HCT input at 5V). '
     'Unused CMOS inputs must be tied to VCC or GND (floating -> oscillation, damage).'),

    ('circuit_theory',
     'Circuit theory fundamentals. '
     'Ohm\'s Law: V = I.R. V=voltage (V), I=current (A), R=resistance (Ω). '
     'Power: P = V.I = I^2.R = V^2/R. '
     'Kirchhoff\'s Voltage Law (KVL): sum of voltages around any loop = 0. '
     'Kirchhoff\'s Current Law (KCL): sum of currents entering a node = 0. '
     'Series resistors: R_total = R1+R2+...  Voltage divides. '
     'Parallel resistors: 1/R_total = 1/R1+1/R2+...  Current divides. '
     'Capacitor: I = C.dV/dt. Energy = ½CV^2. Impedance Xc = 1/(2pifC). '
     'Inductor: V = L.dI/dt. Energy = ½LI^2. Impedance XL = 2pifL. '
     'Thevenin: any linear network = Vth (open-circuit voltage) + Rth (Thevenin resistance). '
     'Norton: equivalent = Isc (short-circuit current) + Rth in parallel. '
     'Superposition: response of linear network = sum of responses to each source alone. '
     'RC time constant: τ = R.C. 5τ to charge/discharge fully. '
     'RLC resonance: f0 = 1/(2pi√(LC)). Q factor = (1/R)√(L/C).'),

    ('semiconductors_transistors',
     'Semiconductor devices fundamentals. '
     'Diode: PN junction. Forward bias (Vf ~0.7V silicon): conducts. Reverse: blocks. '
     'Zener diode: conducts in reverse at breakdown voltage Vz (regulated). '
     'LED: emits photons when forward biased. '
     'BJT (Bipolar Junction Transistor): NPN or PNP. '
     'Regions: Emitter, Base, Collector. '
     'Active: VCE > VCE_sat, IC = beta.IB (current amplifier, beta typically 50-300). '
     'Saturation: both junctions forward biased, VCE_sat ~0.2V (switch ON). '
     'Cutoff: both junctions reverse biased (switch OFF). '
     'MOSFET: Voltage-controlled. N-channel: gate voltage Vgs > Vth -> conducts drain-source. '
     'Enhancement mode (normally off): digital switches, CMOS. '
     'Depletion mode (normally on): RF amplifiers. '
     'N-MOSFET: fast, lower Rdson. P-MOSFET: for high-side switching. '
     'CMOS inverter: N-MOSFET (pull-down) + P-MOSFET (pull-up) in series. '
     'When input HIGH: N on, P off -> output LOW. When input LOW: N off, P on -> output HIGH. '
     'Near-zero static power because only one transistor conducts at a time.'),

    ('operational_amplifiers',
     'Operational amplifier (op-amp): differential amplifier with very high gain. '
     'Ideal op-amp: infinite open-loop gain A, infinite input impedance, zero output impedance. '
     'Virtual short: with negative feedback, V+ ~= V-. '
     'Inverting amplifier: Vout = -(Rf/Rin).Vin. '
     'Non-inverting amplifier: Vout = (1+Rf/R1).Vin. '
     'Voltage follower (buffer): Vout = Vin. Rf=0, R1=inf. '
     'Summing amplifier: Vout = -(Rf/R1.V1 + Rf/R2.V2 + ...). '
     'Difference amplifier: Vout = (R2/R1).(V2-V1) (if matched resistors). '
     'Integrator: Vout = -(1/RC).∫Vin.dt. '
     'Differentiator: Vout = -RC.dVin/dt. '
     'Comparator: no feedback, output rails HIGH or LOW based on V+>V-. '
     'Common op-amps: LM741, LM358, TL071, LM324, NE5532, OPA2134. '
     'Slew rate: maximum dVout/dt (V/µs). Gain-bandwidth product (GBW). '
     'Single-supply: output limited to ~1.5V above GND. Rail-to-rail op-amps go to supply rails.'),

    ('pcb_design',
     'PCB (Printed Circuit Board) design. '
     'Layers: 2-layer (top copper, bottom copper), 4-layer (add power/ground planes), 6+. '
     'Trace width: determines current capacity. ~1mm per amp (copper 1oz=35µm). '
     'Clearance: minimum spacing between traces (6 mil = 0.15mm typical). '
     'Via: plated hole connecting layers. Through-hole vs blind vs buried. '
     'Footprints: SMD (surface mount) vs THT (through-hole). '
     'Design rules: DRC (Design Rule Check) in EDA tool. '
     'Ground plane on inner layer: low impedance return path, reduces EMI. '
     'Decoupling capacitors: 100nF ceramic close to each IC VCC pin. '
     'Differential pairs: route impedance-controlled (50Ω, 100Ω) for high-speed signals. '
     'EDA tools: KiCad (free), Eagle, Altium Designer, EasyEDA. '
     'Gerber files: industry standard for manufacturing (copper, silkscreen, drill). '
     'Assembly: reflow soldering (SMD paste + oven), wave soldering (THT), hand soldering. '
     'PCB manufacturers: JLCPCB, PCBWay, OSHPark (typical 2-layer ~$2/5pcs).'),
]

ELECTRONICS_WIKI_ARTICLES = [
    # Boolean / digital logic
    'Boolean_algebra', 'Logic_gate', 'Karnaugh_map',
    'Flip-flop_(electronics)', 'Multiplexer', 'Decoder_(digital)',
    'Combinational_logic', 'Sequential_logic', 'Finite-state_machine',
    'Adder_(electronics)', 'Arithmetic_logic_unit', 'Register_(digital)',
    # Number representation
    'Binary_number', 'Hexadecimal', "Two's_complement", 'IEEE_754',
    'Signed_number_representations', 'Binary-coded_decimal',
    # Circuit theory
    "Kirchhoff's_circuit_laws", "Ohm's_law", "Thevenin's_theorem",
    "Norton's_theorem", 'Superposition_theorem',
    'RC_circuit', 'RL_circuit', 'RLC_circuit',
    'Electrical_impedance', 'Capacitor', 'Inductor', 'Resistor',
    'Transformer_(electrical)', 'Voltage_divider',
    # Semiconductors / devices
    'Semiconductor_device', 'P-n_junction', 'Diode',
    'Bipolar_junction_transistor', 'MOSFET',
    'Operational_amplifier', 'Comparator_(electronics)',
    'Voltage_regulator', 'Zener_diode', 'Light-emitting_diode',
    'Thyristor', 'Photodiode',
    # IC families / digital hardware
    'Transistor%E2%80%93transistor_logic', 'CMOS',
    'Integrated_circuit', 'Application-specific_integrated_circuit',
    'Field-programmable_gate_array', 'Complex_programmable_logic_device',
    # PCB / EDA
    'Printed_circuit_board', 'Surface-mount_technology',
    'Through-hole_technology', 'Soldering', 'KiCad', 'Gerber_format',
    # Power electronics
    'Power_electronics', 'Switched-mode_power_supply',
    'Buck_converter', 'Boost_converter', 'H-bridge',
    # Signals
    'Signal_processing', 'Analog-to-digital_converter',
    'Digital-to-analog_converter', 'Pulse-width_modulation',
    'Oscilloscope', 'Multimeter',
]

# -- Stage 17: Embedded Systems, Firmware, BIOS, Systems Engineering ------------

EMBEDDED_TEXT_CORPUS = [
    ('microcontroller_architecture',
     'Microcontroller unit (MCU): CPU + RAM + Flash + peripherals in one chip. '
     'Harvard architecture: separate instruction and data buses (AVR, PIC). '
     'Von Neumann: shared bus (ARM Cortex-M0). '
     'Clock sources: internal RC oscillator (±1%), external crystal (±50ppm). '
     'PLL: multiply crystal frequency (e.g., 8MHz crystal -> 168MHz on STM32F4). '
     'Memory map: Flash (code, 0x08000000 on STM32), SRAM (data, 0x20000000), '
     'Peripheral registers (0x40000000+). '
     'Watchdog timer: resets MCU if not "kicked" within timeout (prevent hangs). '
     'DMA: Direct Memory Access -- transfers without CPU intervention. '
     'ADC: analog-to-digital converter. DAC: digital-to-analog. '
     'Timers: PWM generation, input capture, output compare. '
     'UART/USART: asynchronous serial (baud rate, 8N1 framing). '
     'SPI: synchronous serial, 4-wire (MOSI, MISO, SCK, CS), full-duplex. '
     'I2C: 2-wire (SDA, SCL), multi-master/slave, 7-bit address, 100kHz/400kHz/1MHz. '
     'CAN bus: automotive/industrial, differential, 1Mbit/s. '
     'USB: complex protocol, often handled by USB stack library.'),

    ('avr_architecture',
     'Atmel AVR (now Microchip) -- 8-bit RISC microcontrollers. '
     'ATmega328P (Arduino Uno): 32KB Flash, 2KB SRAM, 1KB EEPROM, '
     '23 GPIO, 6 ADC, 3 timers, UART/SPI/I2C, 16MHz max. '
     'ATmega2560 (Arduino Mega): 256KB Flash, 8KB SRAM, 86 GPIO. '
     'Registers: 32 general-purpose 8-bit (R0-R31). R26-R31 = X,Y,Z pointer pairs. '
     'I/O: DDRx (direction), PORTx (output/pullup), PINx (input read). '
     'Interrupts: global enable I-bit in SREG. sei()/cli() in C (avr-libc). '
     'Timer0 (8-bit): millis()/micros() in Arduino. Timer1 (16-bit): precise timing. '
     'ADC: 10-bit, Vref=AVCC (5V) or internal 1.1V. ADCSRA, ADCL/ADCH registers. '
     'Bootloader: optiboot (512 bytes), runs at power-on, checks for serial upload. '
     'avr-libc: C library for AVR. pgmspace.h for Flash (PROGMEM) data. '
     'Fuse bytes: configure clock source, BOD, JTAG, SPI programming.'),

    ('arm_cortex_m',
     'ARM Cortex-M processor series -- 32-bit RISC for embedded. '
     'Cortex-M0: ultra-low power, 56 instructions, Thumb only. '
     'Cortex-M3: Thumb-2 (mix 16/32-bit), hardware multiply/divide, NVIC. '
     'Cortex-M4: M3 + optional FPU (single-precision), DSP instructions. '
     'Cortex-M7: dual-issue superscalar, double-precision FPU, cache. '
     'NVIC (Nested Vectored Interrupt Controller): priority-based interrupts, up to 240. '
     'SysTick: 24-bit down-counter, OS tick (FreeRTOS), HAL_Delay(). '
     'Memory map (standard): 0x00000000 code, 0x20000000 SRAM, 0x40000000 peripherals. '
     'Cortex-M exception model: HardFault, MemManage, BusFault, UsageFault. '
     'CMSIS: ARM Cortex Microcontroller Software Interface Standard. '
     'Vendor HAL: STM32 HAL (STMicroelectronics), nRF5 SDK (Nordic), SDK_2.x (NXP). '
     'Debug: SWD (Serial Wire Debug, 2-pin) or JTAG (4-pin). '
     'OpenOCD: open-source on-chip debugger. GDB for source-level debug. '
     'Popular MCUs: STM32F4 (Cortex-M4), nRF52840 (M4+BLE), RP2040 (dual M0+), '
     'SAMD21 (M0+, Arduino Zero), LPC1768 (M3), ESP32-S3 (dual M0 + Xtensa).'),

    ('esp32_wifi_ble',
     'ESP32 (Espressif): dual-core Xtensa LX6 @240MHz + WiFi + Bluetooth. '
     'Variants: ESP32, ESP32-S2 (single core, USB), ESP32-S3 (M0+AI), '
     'ESP32-C3 (RISC-V), ESP32-H2 (RISC-V, Zigbee/Thread). '
     'Flash: 4MB typical. PSRAM option for extra RAM. '
     'WiFi: 802.11 b/g/n 2.4GHz. BLE 4.2/5.0. '
     'Frameworks: ESP-IDF (Espressif IoT Development Framework, FreeRTOS-based), '
     'Arduino-ESP32 (Arduino API on top of ESP-IDF), MicroPython, CircuitPython. '
     'Peripherals: 34 GPIO, 18 ADC, 2 DAC, SPI/I2C/UART/I2S/CAN/SDMMC. '
     'Touch sensing, hall sensor, ULP (ultra-low-power coprocessor). '
     'Partition table: divides Flash into app, OTA, SPIFFS/LittleFS, NVS. '
     'OTA (Over-the-Air): update firmware over WiFi. '
     'Power modes: active (~240mA), modem sleep, light sleep (~0.8mA), deep sleep (~10µA). '
     'Wake from deep sleep: timer, GPIO, touch, ULP. '
     'Flash: esptool.py -p COM3 -b 460800 write_flash 0x0 firmware.bin'),

    ('bios_uefi_boot',
     'BIOS (Basic Input/Output System): firmware in ROM/Flash on motherboard. '
     'Legacy BIOS (INT 10h video, INT 13h disk, INT 16h keyboard): IBM PC 1981. '
     'POST (Power-On Self Test): RAM test, CPU check, peripheral init. '
     'MBR (Master Boot Record): first 512 bytes of disk, partition table, boot code. '
     'BIOS boot: load MBR to 0x7C00, jump to it. Bootloader loads OS kernel. '
     'UEFI (Unified Extensible Firmware Interface, 2005): replaces legacy BIOS. '
     'GPT (GUID Partition Table): replaces MBR, supports >2TB, 128 partitions. '
     'EFI System Partition (ESP): FAT32, contains .efi bootloader files. '
     'Secure Boot: only signed bootloaders allowed (Shim for Linux). '
     'UEFI Shell: scripting environment, drivers, diagnostics. '
     'Boot order: NVRAM stores boot entries. efibootmgr on Linux. '
     'Bootloaders: GRUB2 (Linux), Windows Boot Manager, rEFInd. '
     'ACPI (Advanced Configuration and Power Interface): power management, hardware description. '
     'SMM (System Management Mode): invisible to OS, used for power/thermal management. '
     'Intel ME / AMD PSP: embedded management processor, runs before main CPU.'),

    ('bootloaders_embedded',
     'Embedded bootloader: small program that runs before main application. '
     'Purpose: initialize hardware, check for new firmware, load/verify application. '
     'AVR optiboot: 512 bytes, STK500v1 protocol over UART. '
     'Checks for new firmware on UART; if none received within timeout, jumps to app. '
     'DFU (Device Firmware Update): USB-based update protocol (STM32 built-in DFU). '
     'U-Boot: universal bootloader for embedded Linux (BeagleBone, Raspberry Pi, etc.). '
     'SPL (Secondary Program Loader): U-Boot\'s first-stage loader for constrained environments. '
     'Chain-loading: ROM bootloader -> first-stage -> second-stage -> OS. '
     'Linker script (.ld): defines memory regions (FLASH, RAM), section placement '
     '(.text code in Flash, .data initialized vars copied to RAM, .bss zeroed). '
     'Startup code (crt0.S / startup.s): sets up stack pointer, copies .data, zeros .bss, '
     'calls SystemInit(), then calls main(). '
     'Reset vector: first address CPU reads at boot (ARM: 0x00000004 = initial PC). '
     'Vector table: array of function pointers for all exceptions/interrupts at 0x0.'),

    ('rtos_concepts',
     'RTOS (Real-Time Operating System) for embedded systems. '
     'Hard real-time: missed deadline = system failure (airbag, pacemaker). '
     'Soft real-time: missed deadline = degraded performance (video playback). '
     'Task/thread: independent execution context with own stack. '
     'Scheduler: determines which task runs. '
     'Preemptive: higher priority task can interrupt lower priority. '
     'Cooperative: tasks yield explicitly. '
     'Priority inversion: low-priority task holds resource needed by high-priority. '
     'Priority inheritance: solution to priority inversion. '
     'Semaphore: signaling mechanism (binary semaphore, counting semaphore). '
     'Mutex: mutual exclusion for shared resources (includes priority inheritance). '
     'Queue/mailbox: pass data between tasks. '
     'FreeRTOS: most popular embedded RTOS. MIT license. '
     'xTaskCreate(), vTaskDelay(), xQueueCreate(), xQueueSend(), xQueueReceive(). '
     'Tick rate: configTICK_RATE_HZ (typically 100-1000 Hz). '
     'Stack overflow detection: configCHECK_FOR_STACK_OVERFLOW. '
     'Zephyr: Linux Foundation RTOS, modern, extensive driver support. '
     'ThreadX (Azure RTOS): deterministic, safety certifiable. '
     'CMSIS-RTOS2: vendor-neutral API over FreeRTOS/RTX.'),

    ('systems_engineering',
     'Systems engineering: interdisciplinary approach to design complex systems. '
     'V-Model: requirements -> architecture -> detailed design -> code -> unit test '
     '-> integration test -> system test -> acceptance test. '
     'MBSE (Model-Based Systems Engineering): SysML diagrams instead of documents. '
     'SysML: Block Definition Diagram (BDD), Internal Block Diagram (IBD), '
     'Sequence Diagram, State Machine, Requirements Diagram. '
     'Requirements: shall statements. Functional (what) vs non-functional (how well). '
     'Traceability matrix: links requirements to design to tests. '
     'Interface Control Document (ICD): defines interfaces between subsystems. '
     'FMEA (Failure Mode and Effects Analysis): identify failure modes, severity, probability. '
     'FMEA Risk Priority Number (RPN) = Severity x Occurrence x Detectability. '
     'Configuration management: version control of all artifacts (code, docs, CAD). '
     'Verification vs Validation: V&V. Verification = "built it right" (tests). '
     'Validation = "built the right thing" (meets user need). '
     'Standards: ISO/IEC 15288, IEEE 1220, NASA-STD-7009, DO-178C (avionics SW).'),

    ('communication_protocols_embedded',
     'Embedded communication protocols. '
     'UART: asynchronous, 2-wire (TX, RX). Start bit, 8 data, stop bit (8N1). '
     'Baud = bits/second. Common: 9600, 115200, 921600. No clock wire. '
     'RS-232: ±12V levels. RS-485: differential, 1200m, multi-drop. '
     'SPI: synchronous, 4-wire (MOSI, MISO, SCLK, CS). Full-duplex. '
     'Modes 0-3: CPOL (clock polarity) x CPHA (clock phase). '
     'Up to 80 MHz on some MCUs. Multiple devices via separate CS lines. '
     'I2C: 2-wire (SDA, SCL), open-drain, pull-up resistors (4.7kΩ typical). '
     '7-bit address (112 devices max). ACK/NACK after each byte. '
     '100kHz (standard), 400kHz (fast), 1MHz (fast-plus), 3.4MHz (high-speed). '
     'I2S: audio serial, 3-wire (BCLK, LRCLK, DATA). Stereo interleaved. '
     'CAN (Controller Area Network): differential (CANH/CANL), up to 1Mbit/s. '
     'Message-based, collision detection, multi-master. ISO 11898. '
     'CAN FD: up to 8Mbit/s data phase, 64-byte frame. '
     'Modbus RTU: industrial serial protocol over RS-485. Coils, registers. '
     'Modbus TCP: same protocol over TCP/IP. '
     'MQTT: lightweight pub/sub over TCP for IoT (broker, topic, QoS 0/1/2).'),

    ('interrupt_handling',
     'Interrupt service routines (ISR) in embedded systems. '
     'Interrupt: hardware signal that pauses main code, runs ISR, then resumes. '
     'Sources: GPIO edge (button, encoder), timer overflow, UART receive, SPI, ADC, DMA. '
     'AVR ISR: ISR(INT0_vect) { ... } -- uses avr/interrupt.h. '
     'ARM Cortex-M: void EXTI0_IRQHandler(void) { ... } -- weak symbol in startup. '
     'ISR rules: MUST be fast. No blocking calls. No malloc. No float in some contexts. '
     'Use volatile for shared variables: volatile uint8_t flag;. '
     'Critical section: disable interrupts around shared data access. '
     'cli()/sei() on AVR. __disable_irq()/__enable_irq() on ARM. '
     'NVIC on Cortex-M: priority 0=highest. NVIC_SetPriority(IRQn, priority). '
     'Interrupt latency: ARM M3/M4 = 12 cycles minimum. '
     'Nested interrupts: higher-priority ISR can interrupt lower-priority ISR. '
     'Deferred processing: ISR sets flag, main loop handles work (avoid long ISRs). '
     'DMA + ISR: DMA transfer complete triggers ISR; CPU never touches data bytes.'),
]

EMBEDDED_WIKI_ARTICLES = [
    # Microcontrollers
    'Microcontroller', 'Arduino', 'AVR_microcontrollers',
    'ARM_Cortex-M', 'STM32', 'ESP32', 'Raspberry_Pi',
    'Raspberry_Pi_Pico', 'BBC_Micro_Bit', 'PIC_microcontrollers',
    'ATmega328', 'ATtiny_microcontrollers',
    # Firmware / OS
    'Firmware', 'Bootloader', 'BIOS', 'UEFI', 'Unified_Extensible_Firmware_Interface',
    'Master_boot_record', 'GUID_Partition_Table',
    'FreeRTOS', 'Zephyr_(operating_system)', 'Embedded_operating_system',
    'Real-time_operating_system', 'Interrupt',
    # Protocols
    'Universal_asynchronous_receiver-transmitter', 'Serial_Peripheral_Interface',
    'I%C2%B2C', 'Controller_area_network', 'USB', 'I%C2%B2S',
    'RS-232', 'RS-485', 'Modbus', 'MQTT',
    # Architecture
    'Harvard_architecture', 'Von_Neumann_architecture',
    'Reduced_instruction_set_computer', 'Complex_instruction_set_computer',
    'Memory-mapped_I/O', 'Direct_memory_access',
    'Watchdog_timer', 'Phase-locked_loop',
    # FPGA / HDL
    'Field-programmable_gate_array', 'Verilog', 'VHDL',
    'SystemVerilog', 'Hardware_description_language',
    'Application-specific_integrated_circuit',
    # Systems engineering
    'Systems_engineering', 'V-model', 'Requirements_engineering',
    'Failure_mode_and_effects_analysis', 'Model-based_systems_engineering',
    # Quantum / Astrophysics (add here for Stage 11 coverage)
    'Quantum_mechanics', 'Quantum_entanglement', 'Quantum_superposition',
    'Wave_function', 'Quantum_field_theory', 'Quantum_electrodynamics',
    'Quantum_chromodynamics', 'Standard_Model', 'Higgs_boson',
    'Quantum_computing', "Schrodinger's_cat", 'Double-slit_experiment',
    'Bell_theorem', 'Copenhagen_interpretation', 'Many-worlds_interpretation',
    'Astrophysics', 'Cosmology', 'Big_Bang', 'Dark_matter', 'Dark_energy',
    'Black_hole', 'Neutron_star', 'Pulsar', 'Supernova',
    'Hertzsprung-Russell_diagram', 'Stellar_evolution', 'Galaxy',
    'Milky_Way', 'Cosmic_microwave_background', 'Hubble_constant',
    'General_relativity', 'Gravitational_wave', 'Event_horizon',
    'Hawking_radiation', 'Inflation_(cosmology)', 'Lambda-CDM_model',
    'Exoplanet', 'Planetary_science', 'Solar_System',
]

# -- Stage 14: STEM Q&A pairs ---------------------------------------------------
STEM_QA_PAIRS = [
    # Mathematics
    ('What is the derivative of x^n?', 'The derivative of x^n is n*x^(n-1). This is the power rule.'),
    ('What is the integral of 1/x?', 'The integral of 1/x is ln|x| + C, where C is the constant of integration.'),
    ('State the Pythagorean theorem.', 'In a right triangle: a^2 + b^2 = c^2, where c is the hypotenuse.'),
    ('What is Euler\'s identity?', "Euler's identity: e^(i*pi) + 1 = 0. Connects e, i, pi, 1, and 0."),
    ('What is Big O notation O(n log n)?', 'O(n log n) describes an algorithm whose time scales by n*log(n). Typical of efficient sorting algorithms like merge sort and heap sort.'),
    ('Define a hash table.', 'A hash table stores key-value pairs. A hash function maps keys to array indices. Average O(1) lookup. Collisions handled by chaining or open addressing.'),
    ('What is dynamic programming?', 'Dynamic programming solves problems by breaking them into overlapping subproblems and caching results (memoization or tabulation) to avoid recomputation.'),
    ('Explain Big O of binary search.', 'Binary search is O(log n). Each step halves the search space by comparing with the middle element.'),
    ('What is a Fourier transform?', 'The Fourier transform decomposes a function into its frequency components: F(omega) = integral of f(t)*e^(-i*omega*t) dt.'),

    # Physics
    ("State Newton's second law.", "F = ma. Force equals mass times acceleration. Net force on an object equals its mass multiplied by its acceleration."),
    ('What is the speed of light?', 'c = 2.998 x 10^8 m/s in vacuum. The universal speed limit per special relativity.'),
    ('State Ohm\'s law.', "V = I*R. Voltage equals current times resistance. Applies to ohmic (linear) conductors at constant temperature."),
    ('What is kinetic energy?', 'KE = (1/2)*m*v^2. Kinetic energy equals half the mass times velocity squared.'),
    ('State the first law of thermodynamics.', 'Energy cannot be created or destroyed, only converted. Delta_U = Q - W (change in internal energy = heat added minus work done by system).'),
    ('What is Heisenberg\'s uncertainty principle?', 'Delta_x * Delta_p >= hbar/2. Position and momentum cannot both be known precisely simultaneously. Fundamental quantum limit.'),

    # Chemistry
    ('What is Avogadro\'s number?', "6.022 x 10^23 mol^-1. The number of atoms/molecules in one mole of a substance."),
    ('What is pH?', 'pH = -log10[H+]. Measures acidity: pH < 7 is acidic, pH = 7 is neutral, pH > 7 is basic.'),
    ('What is the ideal gas law?', 'PV = nRT. Pressure times volume equals moles times gas constant (R=8.314 J/mol/K) times temperature in Kelvin.'),

    # Computer Science
    ('What is the time complexity of quicksort (average)?', 'O(n log n) average case. O(n^2) worst case (sorted input without randomization). In-place, cache-friendly.'),
    ('Explain TCP vs UDP.', 'TCP: connection-oriented, reliable, ordered delivery, flow control, 3-way handshake. UDP: connectionless, unreliable, low latency. Use UDP for streaming/games, TCP for HTTP/email.'),
    ('What is a pointer in C?', 'A pointer stores a memory address. int *p = &x; stores address of x. *p dereferences (reads value at address). Used for dynamic memory, arrays, and pass-by-reference.'),
    ('What is a closure in programming?', 'A closure is a function that captures variables from its enclosing scope. The captured variables remain accessible even after the outer function returns.'),
    ('What is the difference between a process and a thread?', 'A process has its own memory space. Threads share memory within a process. Context switching between processes is heavier than between threads.'),
    ('What is REST?', 'REST (Representational State Transfer): stateless HTTP API architecture. Resources identified by URLs. Methods: GET (read), POST (create), PUT/PATCH (update), DELETE. Returns JSON/XML.'),
    ('What does SOLID stand for?', 'Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion. Five OOP design principles for maintainable code.'),

    # Engineering / 3D
    ('What is FDM 3D printing?', 'Fused Deposition Modeling: thermoplastic filament melted and extruded layer by layer. Materials: PLA, ABS, PETG, TPU. Layer heights 0.1-0.3mm. Sliced by software into G-code.'),
    ('What is G-code G1 command?', 'G1 is a linear interpolation move command. G1 X100 Y50 F3000 moves the tool to (100, 50) at 3000 mm/min feed rate. E parameter controls extrusion amount.'),
    ('What does union() do in OpenSCAD?', 'union() in OpenSCAD merges two or more solid objects into one. Equivalent to Boolean OR. Example: union(){cube(10); sphere(8);} creates a combined solid.'),
    ('What is von Mises stress?', 'Von Mises stress combines all stress components into a single scalar to predict yielding. Material yields when von Mises stress >= yield strength. Used in FEA.'),
    ('What is the difference between Cycles and EEVEE in Blender?', 'Cycles is a path-tracing renderer (physically accurate, slower). EEVEE is a real-time rasterizer (fast, approximations). Cycles for photorealism, EEVEE for previews and stylized renders.'),
    ('What is a MacPherson strut?', 'A MacPherson strut combines the shock absorber and coil spring into a single unit, acting also as the upper suspension pivot. Common in front-wheel-drive cars. Simple, compact, but less ideal for handling than double wishbone.'),

    # Project Management
    ('What is the critical path in project management?', 'The critical path is the longest sequence of dependent tasks determining minimum project duration. Any delay on the critical path delays the whole project. Found using CPM (Critical Path Method).'),
    ('What is a sprint in Scrum?', 'A sprint is a time-boxed iteration (1-4 weeks) in Scrum. The team pulls work from the backlog, develops and tests it, and delivers a potentially shippable increment by sprint end.'),
    ('What is a WBS?', 'Work Breakdown Structure: hierarchical decomposition of project deliverables into manageable work packages. Each level breaks work into smaller pieces until packages are estimable.'),

    # Coding history/gotchas
    ('What is the Year 2000 (Y2K) problem?', 'Y2K: programs stored years as 2 digits (e.g., 99 for 1999). Year 2000 would read as 1900. Affected COBOL, FORTRAN, and embedded systems. Massive remediation effort 1995-1999.'),
    ('Why does C use NULL-terminated strings?', 'C strings are char arrays ending with \\0 (null byte). strlen() counts until \\0. No length prefix. Leads to buffer overflows if bounds not checked. Historical from early UNIX.'),
    ('What is a segmentation fault?', 'Segfault: program accesses memory it should not (null pointer dereference, buffer overflow, stack overflow, use-after-free). OS sends SIGSEGV signal. In C/C++, typically caused by pointer errors.'),
    ('What is Python\'s GIL?', "Python's Global Interpreter Lock (GIL) allows only one thread to execute Python bytecode at a time. Prevents data races in CPython but limits CPU-bound parallelism. Use multiprocessing or async for concurrency."),

    # Electronics / Digital Logic
    ('What is De Morgan\'s theorem?', "De Morgan's laws: NOT(A AND B) = NOT_A OR NOT_B. NOT(A OR B) = NOT_A AND NOT_B. Used to convert between NAND/NOR implementations and simplify logic expressions."),
    ('What is a NAND gate and why is it universal?', 'NAND output = NOT(A AND B). It is universal because ANY Boolean function can be implemented using only NAND gates. NOT(A) = NAND(A,A). AND(A,B) = NAND(NAND(A,B), NAND(A,B)).'),
    ('What is two\'s complement?', "Two's complement represents negative integers in binary. To negate: invert all bits, then add 1. Example: +5 = 00000101, -5 = 11111010 + 1 = 11111011. Range for 8-bit: -128 to +127."),
    ('What is IEEE 754 floating point?', 'IEEE 754 is the standard for binary floating-point. 32-bit: 1 sign bit, 8 exponent bits (bias 127), 23 mantissa bits. Represents fractions as 1.mantissa x 2^(exp-127). Special values: ±infinity, NaN.'),
    ('What is Thevenin\'s theorem?', "Thevenin's theorem: any linear circuit can be simplified to a single voltage source (Vth = open-circuit voltage) in series with a resistance (Rth = resistance with sources zeroed). Useful for analyzing circuit behavior with different load resistances."),
    ('State Kirchhoff\'s voltage law.', 'KVL: The sum of all voltages around any closed loop in a circuit equals zero. Based on conservation of energy. Used for mesh analysis.'),
    ('What is the difference between TTL and CMOS logic?', 'TTL (7400 series) uses bipolar transistors, 5V supply, ~10mW/gate static power, ~10ns delay. CMOS (4000/74HC) uses MOSFETs, wide supply (3-18V), near-zero static power, slower at low voltage but faster at 5V in 74HC variants.'),
    ('What is an op-amp inverting amplifier?', 'Inverting amplifier: output = -(Rf/Rin) * Vin. Negative sign because input is at inverting terminal. Virtual short principle: V- ~= V+ = 0V (virtual ground). Gain magnitude = Rf/Rin.'),
    ('What is PWM?', 'Pulse-Width Modulation: digital output switched rapidly between HIGH and LOW. Duty cycle = ON_time/period. 50% duty = half average voltage. Used for motor speed control, LED dimming, DAC approximation. Arduino analogWrite(pin, 0-255) sets duty cycle.'),

    # Embedded systems
    ('What does setup() and loop() do in Arduino?', 'setup() runs once when the Arduino powers on or resets -- used for initialization (pinMode, Serial.begin). loop() runs continuously in a forever loop -- contains main program logic. This is the Arduino framework entry point built on top of main().'),
    ('What is the difference between analogRead and digitalRead on Arduino?', 'digitalRead(pin) reads HIGH(1) or LOW(0) -- boolean. analogRead(pin) reads 0-1023 from the ADC -- proportional to voltage (0V=0, 5V=1023 on Uno). Only pins A0-A5 support analogRead on Uno.'),
    ('What is UART communication?', 'UART (Universal Asynchronous Receiver-Transmitter): serial protocol with no clock wire. Data framed as: start bit, 8 data bits, stop bit (8N1). Baud rate must match on both sides (e.g., 9600 baud). TX of one device connects to RX of the other.'),
    ('What is the difference between I2C and SPI?', 'I2C: 2 wires (SDA+SCL), addressed (7-bit), slower (100kHz-1MHz), multi-master capable. SPI: 4 wires (MOSI/MISO/SCK/CS), no address, faster (up to 80MHz), full-duplex, requires one CS pin per device.'),
    ('What is a volatile variable in C for embedded?', 'volatile tells the compiler the variable can change outside normal program flow (e.g., modified by an ISR or hardware register). Prevents the compiler from caching it in a register. Required for ISR-shared variables and memory-mapped peripheral registers.'),
    ('What is an interrupt service routine?', 'An ISR is a function that runs automatically when a hardware interrupt occurs (button press, timer overflow, data received). The CPU pauses current code, saves state, executes the ISR, then resumes. ISRs must be very short -- no blocking, no malloc, no heavy computation.'),
    ('What is the purpose of a bootloader?', 'A bootloader is firmware that runs before the main application. It initializes hardware, checks for firmware updates (via UART/USB/SD), optionally verifies application integrity, then transfers execution to the main app. Arduino uses optiboot to receive new sketches over USB-serial.'),
    ('What is UEFI vs BIOS?', 'BIOS (1981): 16-bit real-mode firmware, MBR boot, 2TB disk limit, no GUI. UEFI (2005): 32/64-bit, GPT partition table (>2TB, 128 partitions), secure boot, EFI bootloader files on FAT32 ESP partition, faster POST, mouse-capable GUI.'),
    ('What is a Karnaugh map used for?', 'K-maps minimize Boolean expressions by grouping adjacent 1s in powers of 2 (1,2,4,8,16) on a Gray-code grid. Larger groups = fewer variables = simpler gate circuit. Eliminates algebraic manipulation for up to 4-6 variables.'),

    # Quantum mechanics
    ('What is quantum superposition?', "A quantum particle exists in multiple states simultaneously until measured. Schrodinger's cat analogy: cat is both alive and dead until box is opened. Mathematically: |psi> = alpha|0> + beta|1> where |alpha|^2 + |beta|^2 = 1."),
    ('What is quantum entanglement?', 'Entanglement: two particles share a quantum state such that measuring one instantly determines the state of the other, regardless of distance. No information travels faster than light -- measurement outcomes are correlated but random. Demonstrated by Bell inequality violations.'),
    ('What is the Schrodinger equation?', 'i*hbar * d|psi>/dt = H|psi>. The time-dependent Schrodinger equation governs how quantum state |psi> evolves. H is the Hamiltonian operator (total energy). For stationary states: H|psi> = E|psi> (time-independent form, eigenvalue equation).'),
    ('What is wave-particle duality?', 'Quantum objects behave as waves (interference, diffraction) AND particles (discrete detection events) depending on the experiment. Double-slit experiment: single electrons create interference pattern when not observed, but act as particles when which-path is detected.'),
    ('What is the uncertainty principle?', "Heisenberg uncertainty principle: Delta_x * Delta_p >= hbar/2. Cannot simultaneously know position and momentum with arbitrary precision. Similarly Delta_E * Delta_t >= hbar/2. This is fundamental -- not a measurement limitation but a property of quantum systems."),
    ('What is quantum tunneling?', 'Quantum tunneling: a particle can pass through a potential energy barrier even if it lacks sufficient classical energy. Wave function has nonzero amplitude on the other side of the barrier. Used in: tunnel diodes, flash memory (Fowler-Nordheim tunneling), nuclear fusion in stars, STM.'),
    ('What is the Standard Model of particle physics?', 'The Standard Model describes fundamental particles and forces. Fermions (matter): quarks (u,d,s,c,b,t) and leptons (e,mu,tau + neutrinos). Bosons (force carriers): photon (EM), W/Z (weak), gluon (strong), Higgs (mass). Gravity not included.'),

    # Astrophysics
    ('What is a black hole?', "A region of spacetime where gravity is so strong that nothing -- not even light -- can escape beyond the event horizon. Schwarzschild radius: rs = 2GM/c^2. Formed when massive stars collapse. Hawking radiation: quantum effect causing black holes to slowly evaporate."),
    ('What is the Big Bang?', 'The Big Bang (~13.8 billion years ago): the universe began as an extremely hot, dense state and has been expanding ever since. Evidence: cosmic microwave background (CMB), Hubble expansion (galaxies receding), abundance of light elements (BBN nucleosynthesis).'),
    ('What is dark matter?', 'Dark matter: non-luminous matter that does not interact electromagnetically. Makes up ~27% of universe energy-density. Evidence: galaxy rotation curves (flat where Keplerian would fall off), gravitational lensing, CMB power spectrum. Candidates: WIMPs, axions, sterile neutrinos. Not yet directly detected.'),
    ('What is a neutron star?', 'Extremely dense stellar remnant from supernova. Mass 1.4-2 solar masses in ~10km radius. Density ~10^17 kg/m^3. Surface gravity ~10^11 times Earth. Composed of degenerate neutron matter. Pulsars: rapidly rotating neutron stars emitting beamed radiation detected as pulses.'),
    ('What causes stellar fusion?', 'Stars fuse hydrogen to helium via the proton-proton chain (low-mass) or CNO cycle (high-mass) in their cores. Requires T > 10 million K and extreme pressure. Energy from mass defect: E=mc^2. Hydrostatic equilibrium: gravity inward balanced by radiation pressure outward.'),
    ('What is the Hertzsprung-Russell diagram?', 'H-R diagram plots stellar luminosity vs surface temperature (or spectral class). Main sequence: fusing hydrogen (dwarf stars including Sun). Giants/supergiants: upper right. White dwarfs: lower left. Stars spend most of life on main sequence; mass determines position and lifetime.'),
    ('What is a gravitational wave?', 'Ripples in spacetime caused by accelerating massive objects, predicted by general relativity. First detected by LIGO in 2015 from merging black holes. Strain h = delta_L/L ~ 10^-21. Travel at speed of light. Sources: binary black holes, neutron star mergers (also produce kilonova EM counterpart).'),
]

# -- Stage 15: arXiv abstracts --------------------------------------------------
ARXIV_CATEGORIES = [
    # CS
    'cs.AI', 'cs.LG', 'cs.DS', 'cs.PL', 'cs.SE', 'cs.NE',
    'cs.CV', 'cs.CL', 'cs.CR', 'cs.DB', 'cs.DC', 'cs.OS',
    # Math
    'math.CA', 'math.NA', 'math.ST', 'math.CO', 'math.OC',
    # Physics
    'physics.comp-ph', 'cond-mat', 'quant-ph', 'physics.flu-dyn',
    # Engineering
    'eess.SP', 'eess.SY', 'eess.IV',
    # Quantum / Astrophysics
    'quant-ph', 'astro-ph.GA', 'astro-ph.CO', 'astro-ph.HE',
    'astro-ph.SR', 'gr-qc', 'hep-ph', 'hep-th',
]


# -- Helpers --------------------------------------------------------------------

def _get(url, timeout=15):
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def _train_text(text, node):
    body = json.dumps({'modality': 'text', 'text': text[:6000]}).encode()
    req  = urllib.request.Request(f'http://{node}/media/train', data=body,
           headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())

def _qa_ingest(pairs, node):
    body = json.dumps({'pairs': pairs}).encode()
    req  = urllib.request.Request(f'http://{node}/qa/ingest', data=body,
           headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f'  [WARN] qa/ingest: {e}')
        return {}

def _wiki_extract(title):
    url = ('https://en.wikipedia.org/api/rest_v1/page/summary/'
           + urllib.parse.quote(title.replace(' ', '_')))
    try:
        d = json.loads(_get(url, timeout=10))
        return d.get('extract', '')[:2000]
    except Exception:
        return ''

def _wiki_full_text(title):
    """Fetch full Wikipedia article text via action API."""
    params = {
        'action': 'query', 'prop': 'extracts', 'explaintext': '1',
        'titles': title.replace('_', ' '), 'format': 'json', 'exlimit': '1',
    }
    url = 'https://en.wikipedia.org/w/api.php?' + urllib.parse.urlencode(params)
    try:
        d = json.loads(_get(url, timeout=12))
        pages = d.get('query', {}).get('pages', {})
        for p in pages.values():
            return (p.get('extract') or '')[:6000]
    except Exception:
        return ''


# -- Stage implementations ------------------------------------------------------

def build_libretexts(out_dir: Path, node: str,
                     max_per_domain: int = 30) -> list:
    """Stage 10: Crawl LibreTexts bookshelves HTML, extract printIds,
    download PDFs, extract text with pdfminer if available, train."""
    out_dir.mkdir(parents=True, exist_ok=True)
    items = []
    ok = 0

    try:
        from pdfminer.high_level import extract_text as pdf_extract
        HAS_PDFMINER = True
    except ImportError:
        HAS_PDFMINER = False
        print('  [INFO] pdfminer.six not installed -- will train PDF metadata only')
        print('         Install with: pip install pdfminer.six')

    for subdomain, root_url in tqdm(LIBRETEXTS_ROOTS, desc='LibreTexts domains'):
        domain_dir = out_dir / subdomain
        domain_dir.mkdir(exist_ok=True)
        fetched = 0

        try:
            html = _get(root_url, timeout=15).decode('utf-8', errors='replace')
        except Exception as e:
            print(f'\n  [WARN] {subdomain}: {e}')
            continue

        # Extract data-page-id from listing HTML
        page_ids = re.findall(r'data-page-id=["\'](\d+)["\']', html)
        page_ids = list(dict.fromkeys(page_ids))[:max_per_domain]

        for pid in page_ids:
            if fetched >= max_per_domain:
                break
            print_id  = f'{subdomain}-{pid}'
            pdf_url   = f'https://batch.libretexts.org/print/Letter/Finished/{print_id}/Full.pdf'
            pdf_path  = domain_dir / f'{print_id}.pdf'

            if not pdf_path.exists():
                try:
                    raw = _get(pdf_url, timeout=60)
                    pdf_path.write_bytes(raw)
                except Exception as e:
                    print(f'\n  [WARN] PDF {print_id}: {e}')
                    continue

            # Extract text
            text = ''
            if HAS_PDFMINER:
                try:
                    text = pdf_extract(str(pdf_path))[:6000]
                except Exception:
                    pass

            # Train: text content + source metadata
            train_text = (f'LibreTexts {subdomain} textbook (printId: {print_id}). '
                          f'Subject: {subdomain}. Open-access CC-BY textbook. '
                          f'URL: {pdf_url}. ')
            if text:
                train_text += text[:4000]

            try:
                _train_text(train_text, node)
                ok += 1
                fetched += 1
                items.append({'stage': 10, 'type': 'libretexts_pdf',
                               'subdomain': subdomain, 'print_id': print_id,
                               'modality': 'text', 'tags': ['textbook', subdomain]})
            except Exception as e:
                print(f'\n  [WARN] train {print_id}: {e}')
            time.sleep(0.5)

    print(f'  LibreTexts: {ok} textbooks trained')
    return items


def build_wikipedia_stem(node: str, max_articles: int = 5000) -> list:
    """Stage 11: Fetch full-text Wikipedia articles on STEM topics."""
    items = []
    ok = 0
    seen = set()

    all_titles = list(WIKI_STEM_ARTICLES)

    # Expand from categories
    for cat in tqdm(WIKI_STEM_CATEGORIES[:40], desc='Wiki categories'):
        params = {'action': 'query', 'list': 'categorymembers',
                  'cmtitle': f'Category:{cat}', 'cmlimit': '50',
                  'cmnamespace': '0', 'format': 'json'}
        url = 'https://en.wikipedia.org/w/api.php?' + urllib.parse.urlencode(params)
        try:
            d   = json.loads(_get(url, timeout=10))
            for m in d.get('query', {}).get('categorymembers', []):
                title = m.get('title', '')
                if title and title not in seen:
                    all_titles.append(title)
                    seen.add(title)
        except Exception:
            pass
        time.sleep(0.2)

    print(f'  {len(all_titles)} Wikipedia articles to fetch...')

    for title in tqdm(all_titles[:max_articles], desc='Wikipedia STEM'):
        extract = _wiki_full_text(title)
        if not extract:
            time.sleep(0.2)
            continue

        text = (f'Wikipedia: {title.replace("_"," ")}. '
                f'STEM reference article. {extract}')
        try:
            _train_text(text, node)
            ok += 1
            items.append({'stage': 11, 'type': 'wiki_stem', 'title': title,
                           'modality': 'text', 'tags': ['wikipedia', 'stem']})
        except Exception as e:
            print(f'\n  [WARN] wiki {title}: {e}')
        time.sleep(0.15)

    print(f'  Wikipedia STEM: {ok} articles trained')
    return items


def build_programming_corpus(node: str) -> list:
    """Stage 12: Official language docs, Wikipedia articles, structured context,
    and versioned code examples covering every major language from 1958 to present."""
    items = []
    ok = 0

    for lang_info in tqdm(PROG_LANG_SOURCES, desc='Programming languages'):
        lang   = lang_info['lang']
        era    = lang_info['era']
        comp   = lang_info['compiler']
        ctx    = lang_info['context']

        # Train the context text (rich metadata)
        base_text = (f'Programming language: {lang}. Era: {era}. '
                     f'Compiler/Runtime: {comp}. '
                     f'{ctx}')
        try:
            _train_text(base_text, node)
            ok += 1
            items.append({'stage': 12, 'type': 'lang_context', 'lang': lang,
                           'modality': 'text', 'tags': ['programming', lang]})
        except Exception as e:
            print(f'\n  [WARN] {lang} context: {e}')

        # Fetch Wikipedia article for full historical/technical detail
        for spec_url in lang_info.get('spec_urls', []):
            if 'wikipedia.org/wiki/' in spec_url:
                wiki_title = spec_url.split('/wiki/')[-1]
                extract = _wiki_full_text(wiki_title)
                if extract:
                    text = (f'Official documentation / history: {lang} programming language. '
                            f'Era: {era}. Compiler: {comp}. {extract}')
                    try:
                        _train_text(text, node)
                        ok += 1
                        items.append({'stage': 12, 'type': 'lang_wiki', 'lang': lang,
                                       'modality': 'text',
                                       'tags': ['programming', lang, 'docs']})
                    except Exception as e:
                        print(f'\n  [WARN] {lang} wiki: {e}')
                    time.sleep(0.2)

        time.sleep(0.15)

    # Train versioned code examples (plain-English -> code with platform/compiler metadata)
    print(f'  Training {len(PROG_CODE_EXAMPLES)} versioned code examples...')
    for ex in tqdm(PROG_CODE_EXAMPLES, desc='Code examples'):
        text = (
            f'Coding example -- plain English description: {ex["description"]}.\n'
            f'Language: {ex["lang"]}. Compiler/Runtime: {ex["compiler"]}. '
            f'Year: {ex["year"]}. Platform: {ex["platform"]}.\n'
            f'Code:\n{ex["code"]}\n'
            f'Notes: {ex["notes"]}'
        )
        try:
            _train_text(text, node)
            ok += 1
            items.append({'stage': 12, 'type': 'code_example',
                           'lang': ex['lang'], 'year': ex['year'],
                           'modality': 'text',
                           'tags': ['programming', 'code_example', ex['lang']]})
        except Exception as e:
            print(f'\n  [WARN] code example {ex["lang"]} {ex["year"]}: {e}')
        time.sleep(0.1)

    print(f'  Programming corpus: {ok} items trained')
    return items


def build_cad_3d_corpus(node: str) -> list:
    """Stage 13: CAD, 3D printing, Blender, manufacturing training corpus."""
    items = []
    ok = 0

    # A: Structured text corpus
    for name, text in tqdm(CAD_3D_TEXT_CORPUS, desc='CAD text corpus'):
        try:
            _train_text(text, node)
            ok += 1
            items.append({'stage': 13, 'type': 'cad_text', 'name': name,
                           'modality': 'text', 'tags': ['cad', '3d', 'manufacturing']})
        except Exception as e:
            print(f'\n  [WARN] {name}: {e}')
        time.sleep(0.05)

    # B: Wikipedia articles on CAD/manufacturing
    for title in tqdm(CAD_3D_WIKI_ARTICLES, desc='CAD Wikipedia'):
        extract = _wiki_full_text(title)
        if not extract:
            time.sleep(0.2)
            continue
        text = (f'3D/CAD/Manufacturing reference: {title.replace("_"," ")}. '
                f'{extract}')
        try:
            _train_text(text, node)
            ok += 1
            items.append({'stage': 13, 'type': 'cad_wiki', 'title': title,
                           'modality': 'text', 'tags': ['cad', '3d', 'manufacturing']})
        except Exception as e:
            print(f'\n  [WARN] {title}: {e}')
        time.sleep(0.15)

    print(f'  CAD/3D corpus: {ok} items trained')
    return items


def build_stem_qa(node: str) -> list:
    """Stage 14: Ingest STEM Q&A pairs via /qa/ingest."""
    items = []

    # Batch by 20
    batch_size = 20
    batches = [STEM_QA_PAIRS[i:i+batch_size]
               for i in range(0, len(STEM_QA_PAIRS), batch_size)]

    ok = 0
    for i, batch in enumerate(tqdm(batches, desc='STEM Q&A')):
        pairs = [{'question': q, 'answer': a} for q, a in batch]
        r = _qa_ingest(pairs, node)
        if r.get('ingested'):
            ok += len(batch)
        items.extend({'stage': 14, 'type': 'qa_pair', 'question': q,
                       'modality': 'qa', 'tags': ['stem', 'qa']}
                     for q, _ in batch)
        time.sleep(0.1)

    print(f'  STEM Q&A: {ok}/{len(STEM_QA_PAIRS)} pairs ingested')
    return items


def build_arxiv_abstracts(node: str, max_per_cat: int = 200) -> list:
    """Stage 15: Download arXiv abstracts via Atom API and train."""
    items = []
    ok = 0

    for cat in tqdm(ARXIV_CATEGORIES, desc='arXiv categories'):
        url = (f'https://export.arxiv.org/api/query?search_query=cat:{cat}'
               f'&max_results={max_per_cat}&sortBy=submittedDate&sortOrder=descending')
        try:
            xml = _get(url, timeout=20).decode('utf-8', errors='replace')
        except Exception as e:
            print(f'\n  [WARN] arXiv {cat}: {e}')
            continue

        # Parse title + summary from Atom XML (no lxml needed)
        entries = re.findall(r'<entry>(.*?)</entry>', xml, re.DOTALL)
        for entry in entries:
            title   = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
            summary = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
            authors = re.findall(r'<name>(.*?)</name>', entry)
            if not title or not summary:
                continue
            t = title.group(1).strip().replace('\n', ' ')
            s = summary.group(1).strip().replace('\n', ' ')[:2000]
            a = ', '.join(authors[:3])

            text = (f'arXiv preprint: {t}. Category: {cat}. Authors: {a}. '
                    f'Abstract: {s}')
            try:
                _train_text(text, node)
                ok += 1
                items.append({'stage': 15, 'type': 'arxiv_abstract',
                               'title': t, 'category': cat,
                               'modality': 'text', 'tags': ['arxiv', cat]})
            except Exception as e:
                print(f'\n  [WARN] arXiv entry: {e}')
            time.sleep(0.05)

        time.sleep(0.5)

    print(f'  arXiv: {ok} abstracts trained')
    return items


def build_electronics_corpus(node: str) -> list:
    """Stage 16: Boolean algebra, digital logic, circuit theory, semiconductors, PCB."""
    items = []
    ok = 0

    # A: Structured text corpus
    for name, text in tqdm(ELECTRONICS_TEXT_CORPUS, desc='Electronics text'):
        try:
            _train_text(text, node)
            ok += 1
            items.append({'stage': 16, 'type': 'electronics_text', 'name': name,
                           'modality': 'text',
                           'tags': ['electronics', 'digital_logic', 'circuits']})
        except Exception as e:
            print(f'\n  [WARN] {name}: {e}')
        time.sleep(0.05)

    # B: Wikipedia articles
    for title in tqdm(ELECTRONICS_WIKI_ARTICLES, desc='Electronics Wikipedia'):
        extract = _wiki_full_text(title)
        if not extract:
            time.sleep(0.2)
            continue
        text = (f'Electronics/Digital Logic reference: {title.replace("_", " ")}. '
                f'{extract}')
        try:
            _train_text(text, node)
            ok += 1
            items.append({'stage': 16, 'type': 'electronics_wiki', 'title': title,
                           'modality': 'text',
                           'tags': ['electronics', 'wikipedia']})
        except Exception as e:
            print(f'\n  [WARN] {title}: {e}')
        time.sleep(0.15)

    print(f'  Electronics corpus: {ok} items trained')
    return items


def build_embedded_corpus(node: str) -> list:
    """Stage 17: Embedded systems, firmware, BIOS/UEFI, RTOS, systems engineering."""
    items = []
    ok = 0

    # A: Structured text corpus
    for name, text in tqdm(EMBEDDED_TEXT_CORPUS, desc='Embedded text'):
        try:
            _train_text(text, node)
            ok += 1
            items.append({'stage': 17, 'type': 'embedded_text', 'name': name,
                           'modality': 'text',
                           'tags': ['embedded', 'firmware', 'microcontroller']})
        except Exception as e:
            print(f'\n  [WARN] {name}: {e}')
        time.sleep(0.05)

    # B: Wikipedia articles (includes quantum/astrophysics added to EMBEDDED_WIKI_ARTICLES)
    for title in tqdm(EMBEDDED_WIKI_ARTICLES, desc='Embedded/Quantum/Astro Wikipedia'):
        extract = _wiki_full_text(title)
        if not extract:
            time.sleep(0.2)
            continue
        text = (f'Embedded systems / firmware / science reference: '
                f'{title.replace("_", " ")}. {extract}')
        try:
            _train_text(text, node)
            ok += 1
            items.append({'stage': 17, 'type': 'embedded_wiki', 'title': title,
                           'modality': 'text',
                           'tags': ['embedded', 'wikipedia']})
        except Exception as e:
            print(f'\n  [WARN] {title}: {e}')
        time.sleep(0.15)

    print(f'  Embedded/Firmware corpus: {ok} items trained')
    return items


# -- Main ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Build STEM training dataset for W1z4rD V1510n')
    ap.add_argument('--stages',   default='10,11,12,13,14,15,16,17')
    ap.add_argument('--node',     default=DEFAULT_NODE)
    ap.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    ap.add_argument('--wiki-max', type=int, default=5000,
                    help='Max Wikipedia articles (stage 11)')
    ap.add_argument('--libretexts-max', type=int, default=30,
                    help='Max textbooks per LibreTexts domain (stage 10)')
    ap.add_argument('--arxiv-max', type=int, default=200,
                    help='Max arXiv abstracts per category (stage 15)')
    args = ap.parse_args()

    stages    = [int(s) for s in args.stages.split(',') if s.strip().isdigit()]
    data_dir  = Path(args.data_dir)
    train_dir = data_dir / 'training'
    train_dir.mkdir(parents=True, exist_ok=True)

    print('\n' + '='*72)
    print('  W1z4rD V1510n -- STEM + Engineering + Coding Dataset Builder')
    print('='*72)
    print(f'  Node:    http://{args.node}')
    print(f'  Stages:  {stages}')
    print()

    all_items: dict = {}

    if 10 in stages:
        print('\n[Stage 10] LibreTexts open textbooks')
        all_items[10] = build_libretexts(
            train_dir / 'stage10_libretexts', args.node,
            max_per_domain=args.libretexts_max)

    if 11 in stages:
        print('\n[Stage 11] Wikipedia STEM corpus')
        all_items[11] = build_wikipedia_stem(args.node, max_articles=args.wiki_max)

    if 12 in stages:
        print('\n[Stage 12] Programming language documentation corpus')
        all_items[12] = build_programming_corpus(args.node)

    if 13 in stages:
        print('\n[Stage 13] CAD / 3D printing / manufacturing corpus')
        all_items[13] = build_cad_3d_corpus(args.node)

    if 14 in stages:
        print('\n[Stage 14] STEM Q&A pairs')
        all_items[14] = build_stem_qa(args.node)

    if 15 in stages:
        print('\n[Stage 15] arXiv preprint abstracts')
        all_items[15] = build_arxiv_abstracts(args.node, max_per_cat=args.arxiv_max)

    if 16 in stages:
        print('\n[Stage 16] Electronics, digital logic, circuit theory, PCB')
        all_items[16] = build_electronics_corpus(args.node)

    if 17 in stages:
        print('\n[Stage 17] Embedded systems, firmware, BIOS/UEFI, RTOS, quantum, astrophysics')
        all_items[17] = build_embedded_corpus(args.node)

    print('\n' + '='*72)
    print('  STEM dataset build complete.')
    total = sum(len(v) for v in all_items.values())
    for sid, items in sorted(all_items.items()):
        print(f'    Stage {sid} ({STAGES[sid][:50]:<50}): {len(items):>6} items')
    print(f'    {"TOTAL":<55}: {total:>6} items')
    print('='*72 + '\n')


if __name__ == '__main__':
    main()
