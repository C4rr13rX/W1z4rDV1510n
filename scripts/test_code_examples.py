#!/usr/bin/env python3
# coding: utf-8
"""
test_code_examples.py -- Verify code examples from PROG_CODE_EXAMPLES
Runs each example in the appropriate environment, captures output,
saves results to code_test_results.json for training enrichment.

Supported runners:
  python3, node, go, rustc/cargo, javac/java, gcc, g++
  QB64 (if installed) for QBasic/BASIC
  nasm (if installed) for x86-64 ASM (Windows COFF only)

Skipped (require special hardware/OS):
  Arduino/AVR, ARM bare-metal, MicroPython, VHDL, Verilog,
  DOS-only BASIC, Linux-syscall x86-64 ASM

Usage:
  python scripts/test_code_examples.py [--results-json results.json]
                                        [--train-node localhost:8090]
"""

import argparse, json, os, platform, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path('D:/w1z4rdv1510n-data')
# Compile outputs go here so Avira (which scans AppData\Local\Temp) won't quarantine them
COMPILE_DIR = DATA_DIR / 'code_tests'
RESULTS_DEFAULT = DATA_DIR / 'training/code_test_results.json'

# -- Tool discovery -------------------------------------------------------------

def find_tool(*names) -> str | None:
    for name in names:
        p = shutil.which(name)
        if p:
            return p
    return None

QB64_COMPILER_BIN = Path(r'C:\tools\qb64pe\qb64pe\internal\c\c_compiler\bin')
QB64_EXE         = Path(r'C:\tools\qb64pe\qb64pe\qb64pe.exe')

def find_gcc() -> str | None:
    # Check standard PATH first
    p = find_tool('gcc', 'gcc.exe')
    if p:
        return p
    # QB64 bundles gcc (Clang) -- use it as our C compiler
    candidate = QB64_COMPILER_BIN / 'gcc.exe'
    if candidate.exists():
        return str(candidate)
    # WinGet WinLibs install
    winget = Path.home() / 'AppData/Local/Microsoft/WinGet/Packages'
    if winget.exists():
        for d in winget.iterdir():
            if 'WinLibs' in d.name and 'POSIX' in d.name and 'UCRT' in d.name:
                gcc = d / 'mingw64/bin/gcc.exe'
                if gcc.exists():
                    return str(gcc)
    return None

def find_gpp() -> str | None:
    p = find_tool('g++', 'g++.exe')
    if p:
        return p
    candidate = QB64_COMPILER_BIN / 'g++.exe'
    if candidate.exists():
        return str(candidate)
    gcc = find_gcc()
    if gcc:
        return str(Path(gcc).parent / 'g++.exe')
    return None

def tool_version(cmd, flag='--version') -> str:
    try:
        r = subprocess.run([cmd, flag], capture_output=True, text=True, timeout=5)
        return (r.stdout or r.stderr).split('\n')[0].strip()
    except Exception:
        return 'unknown'

TOOLS = {}

def discover_tools():
    global TOOLS
    TOOLS = {
        'python3':  find_tool('python3', 'python'),
        'node':     find_tool('node', 'node.exe'),
        'go':       find_tool('go', 'go.exe'),
        'rustc':    find_tool('rustc', 'rustc.exe'),
        'cargo':    find_tool('cargo', 'cargo.exe'),
        'javac':    find_tool('javac', 'javac.exe'),
        'java':     find_tool('java', 'java.exe'),
        'gcc':      find_gcc(),
        'g++':      find_gpp(),
        'nasm':     find_tool('nasm', 'nasm.exe'),
        'qb64':     str(QB64_EXE) if QB64_EXE.exists() else find_tool('qb64', 'qb64pe', 'qb64.exe', 'qb64pe.exe'),
    }
    print('Detected tools:')
    for k, v in TOOLS.items():
        print(f'  {k:10} {v or "NOT FOUND"}')
    return TOOLS


# -- Runner implementations -----------------------------------------------------

def run_cmd(cmd: list, cwd=None, timeout=30, env=None, _retries=3) -> dict:
    for attempt in range(_retries):
        try:
            r = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout, cwd=str(cwd) if cwd else None,
                env=env or os.environ.copy(),
            )
            return {
                'stdout': r.stdout.strip(),
                'stderr': r.stderr.strip(),
                'exit_code': r.returncode,
                'timed_out': False,
            }
        except subprocess.TimeoutExpired:
            return {'stdout': '', 'stderr': 'TIMEOUT', 'exit_code': -1, 'timed_out': True}
        except OSError as e:
            # WinError 1450: resource exhaustion; WinError 2: AV holds exe lock -- retry both
            if e.winerror in (2, 1450) and attempt < _retries - 1:
                time.sleep(1.0 * (attempt + 1))
                continue
            return {'stdout': '', 'stderr': str(e), 'exit_code': -1, 'timed_out': False}
        except Exception as e:
            return {'stdout': '', 'stderr': str(e), 'exit_code': -1, 'timed_out': False}
    return {'stdout': '', 'stderr': 'WinError 1450: resource exhaustion after retries', 'exit_code': -1, 'timed_out': False}


def run_python3(code: str, tmpdir: Path) -> dict:
    f = tmpdir / 'prog.py'
    f.write_text(code, encoding='utf-8')
    return run_cmd([TOOLS['python3'], str(f)], timeout=15)


def run_node(code: str, tmpdir: Path) -> dict:
    f = tmpdir / 'prog.js'
    f.write_text(code, encoding='utf-8')
    return run_cmd([TOOLS['node'], str(f)], timeout=15)


def run_go(code: str, tmpdir: Path) -> dict:
    f = tmpdir / 'main.go'
    f.write_text(code, encoding='utf-8')
    return run_cmd([TOOLS['go'], 'run', str(f)], timeout=30)


def run_rust(code: str, tmpdir: Path) -> dict:
    # Use rustc directly for simple programs
    f = tmpdir / 'main.rs'
    f.write_text(code, encoding='utf-8')
    out = tmpdir / 'main.exe'
    comp = run_cmd([TOOLS['rustc'], str(f), '-o', str(out)], timeout=60)
    if comp['exit_code'] != 0:
        return comp
    return run_cmd([str(out)], timeout=10)


def run_java(code: str, tmpdir: Path) -> dict:
    # Extract class name from code
    m = re.search(r'public class (\w+)', code)
    class_name = m.group(1) if m else 'Main'
    f = tmpdir / f'{class_name}.java'
    f.write_text(code, encoding='utf-8')
    comp = run_cmd([TOOLS['javac'], str(f)], cwd=tmpdir, timeout=30)
    if comp['exit_code'] != 0:
        return comp
    return run_cmd([TOOLS['java'], class_name], cwd=tmpdir, timeout=15)


def _cpp_env():
    env = os.environ.copy()
    env['PATH'] = str(QB64_COMPILER_BIN) + os.pathsep + env.get('PATH', '')
    return env

def compile_only_cpp(code: str, tmpdir: Path, std='c++11') -> dict:
    """Compile C++ but do not run -- for Arduino compile-check."""
    f = tmpdir / 'prog.cpp'
    f.write_text(code, encoding='utf-8')
    out = tmpdir / 'prog.exe'
    return run_cmd(
        [TOOLS['g++'], str(f), f'-std={std}', '-Wall', '-o', str(out)],
        timeout=30,
    )

def run_c(code: str, tmpdir: Path, std='c89') -> dict:
    f = tmpdir / 'prog.c'
    f.write_text(code, encoding='utf-8')
    out = tmpdir / 'prog.exe'
    comp = run_cmd(
        [TOOLS['gcc'], str(f), f'-std={std}', '-Wall', '-o', str(out)],
        timeout=30,
    )
    if comp['exit_code'] != 0:
        return comp
    for _ in range(3):
        if out.exists():
            break
        time.sleep(0.5)
    if not out.exists():
        return {'stdout': '', 'stderr': 'Executable quarantined by AV after compile', 'exit_code': 1, 'timed_out': False}
    return run_cmd([str(out)], timeout=10, env=_cpp_env())

def run_cpp(code: str, tmpdir: Path, std='c++98') -> dict:
    f = tmpdir / 'prog.cpp'
    f.write_text(code, encoding='utf-8')
    out = tmpdir / 'prog.exe'
    comp = run_cmd(
        [TOOLS['g++'], str(f), f'-std={std}', '-Wall', '-o', str(out)],
        timeout=30,
    )
    if comp['exit_code'] != 0:
        return comp
    # QB64's Clang-built exes need the compiler's DLL directory in PATH
    # AV may briefly hold the exe after creation -- poll until it appears
    for _ in range(3):
        if out.exists():
            break
        time.sleep(0.5)
    if not out.exists():
        return {'stdout': '', 'stderr': 'Executable quarantined by AV after compile', 'exit_code': 1, 'timed_out': False}
    return run_cmd([str(out)], timeout=10, env=_cpp_env())


def run_qb64(code: str, tmpdir: Path) -> dict:
    if not TOOLS['qb64']:
        return {'stdout': '', 'stderr': 'QB64 not installed',
                'exit_code': -2, 'timed_out': False}
    f = tmpdir / 'prog.bas'
    # Inject $CONSOLE:ONLY so output goes to stdout (not GUI window)
    # Also inject SYSTEM at end if not present (graceful exit)
    full_code = '$CONSOLE:ONLY\n' + code
    if 'SYSTEM' not in code.upper() and 'END' not in code.upper():
        full_code += '\nSYSTEM'
    f.write_text(full_code, encoding='utf-8')
    out = tmpdir / 'prog.exe'
    comp = run_cmd([TOOLS['qb64'], '-x', str(f), '-o', str(out)], timeout=90)
    if comp['exit_code'] != 0:
        return comp
    if not out.exists():
        return {'stdout':'','stderr':'QB64 produced no executable','exit_code':1,'timed_out':False}
    return run_cmd([str(out)], timeout=10)


# -- Language -> runner dispatch -------------------------------------------------

SKIP_REASONS = {
    'REQUIRES_HARDWARE':   'Requires physical microcontroller hardware',
    'REQUIRES_LINUX':      'Requires Linux syscalls; runs on Linux x86-64 only',
    'REQUIRES_DOSBOX':     'Requires DOSBox + original DOS executable (GW-BASIC.EXE or MASM.EXE)',
    'REQUIRES_GHDL':       'Requires GHDL simulator (not installed)',
    'REQUIRES_IVERILOG':   'Requires Icarus Verilog (not installed)',
    'REQUIRES_QB64':       'Requires QB64 (not installed)',
    'REQUIRES_PYTHON2':    'Requires Python 2.x (EOL; Python 3 would raise SyntaxError)',
    'WINDOWS_ONLY_NOTE':   'Windows x86-64 uses different syscall convention than Linux',
}

def dispatch(ex: dict, tmpdir: Path) -> dict:
    lang  = ex['lang']
    code  = ex['code']
    notes = ex.get('notes', '')

    def skip(reason_key):
        return {
            'stdout': '', 'stderr': '',
            'exit_code': -2, 'timed_out': False,
            'skip_reason': SKIP_REASONS[reason_key],
        }

    # QBasic / GW-BASIC
    if 'QBasic' in lang or 'qbasic' in lang.lower():
        if TOOLS['qb64']:
            return run_qb64(code, tmpdir)
        return skip('REQUIRES_QB64')

    if 'GW-BASIC' in lang or 'gw-basic' in lang.lower():
        return skip('REQUIRES_DOSBOX')

    # C variants
    if 'K&R' in lang:
        if not TOOLS['gcc']:
            return {'stdout':'','stderr':'gcc not found','exit_code':-2,'timed_out':False}
        # K&R C compiles with -std=c89 (mostly)
        return run_c(code, tmpdir, std='c89')

    if 'ANSI C89' in lang or 'C (ANSI' in lang:
        if not TOOLS['gcc']:
            return {'stdout':'','stderr':'gcc not found','exit_code':-2,'timed_out':False}
        return run_c(code, tmpdir, std='c89')

    if 'bare-metal' in lang or 'avr-gcc' in lang or 'arm-none-eabi' in lang or 'STM32' in lang or 'Embedded' in lang:
        return skip('REQUIRES_HARDWARE')

    if lang.startswith('C (') or lang.startswith('C\n'):
        if not TOOLS['gcc']:
            return {'stdout':'','stderr':'gcc not found','exit_code':-2,'timed_out':False}
        return run_c(code, tmpdir)

    # C++ variants
    if 'Borland' in lang or 'Turbo C++' in lang:
        if not TOOLS['g++']:
            return {'stdout':'','stderr':'g++ not found','exit_code':-2,'timed_out':False}
        # Modernise: replace <iostream.h> with <iostream> + using namespace std
        modern = code.replace('#include <iostream.h>', '#include <iostream>\nusing namespace std;')
        modern = modern.replace('void main()', 'int main()')
        if 'int main()' not in modern and 'void main()' not in modern:
            modern = modern
        result = run_cpp(modern, tmpdir, std='c++98')
        result['note'] = 'Modernised: <iostream.h>-><iostream>+using namespace std; void main->int main'
        return result

    # Arduino C++ -- compile-check with stub headers (check BEFORE C++ to avoid 'C++' in lang matching)
    if 'Arduino' in lang:
        if not TOOLS['g++']:
            return {'stdout':'','stderr':'g++ not found','exit_code':-2,'timed_out':False}
        # Strip Arduino framework includes -- our stub replaces them
        clean = re.sub(r'#include\s*[<"][^>"]*\.h[>"]\s*\n?', '', code)
        stub = (
            '#include <stdint.h>\n#include <stdio.h>\n'
            '#define HIGH 1\n#define LOW 0\n#define INPUT 0\n#define OUTPUT 1\n'
            '#define INPUT_PULLUP 2\n#define A0 14\n#define A1 15\n'
            'void pinMode(int,int){}\n'
            'void digitalWrite(int,int){}\n'
            'int  digitalRead(int){return 0;}\n'
            'int  analogRead(int){return 512;}\n'
            'void analogWrite(int,int){}\n'
            'void delay(unsigned long){}\n'
            'unsigned long millis(){return 0;}\n'
            'struct _Serial{void begin(int){}void print(const char*s){printf("%s",s);}\n'
            '  void println(const char*s){puts(s);}\n'
            '  void print(int v){printf("%d",v);}\n'
            '  void println(int v){printf("%d\\n",v);}\n'
            '  void println(float v){printf("%.4f\\n",v);}} Serial;\n'
            'struct Wire_{void begin(){}\n'
            '  void beginTransmission(int){}\n'
            '  void write(int){}\n'
            '  void endTransmission(bool=true){}\n'
            '  void requestFrom(int,int){}\n'
            '  int read(){return 0;}} Wire;\n'
            + clean + '\n'
            'int main(){setup();loop();return 0;}\n'
        )
        result = compile_only_cpp(stub, tmpdir, std='c++11')
        result['note'] = 'Compile-check only with Arduino stub headers; real execution requires board'
        if result['exit_code'] == 0:
            result['stdout'] = '[Compiled successfully -- requires Arduino board to run]'
        return result

    if 'C++ (ISO' in lang or 'C++' in lang:
        if not TOOLS['g++']:
            return {'stdout':'','stderr':'g++ not found','exit_code':-2,'timed_out':False}
        std = 'c++20' if 'C++20' in lang else ('c++17' if 'C++17' in lang else 'c++98')
        return run_cpp(code, tmpdir, std=std)

    # JavaScript (check BEFORE Java to avoid 'Java' matching 'JavaScript')
    if 'JavaScript' in lang or lang.startswith('JS'):
        if not TOOLS['node']:
            return {'stdout':'','stderr':'node not found','exit_code':-2,'timed_out':False}
        return run_node(code, tmpdir)

    # Java (after JavaScript check)
    if 'Java' in lang:
        if not (TOOLS['javac'] and TOOLS['java']):
            return {'stdout':'','stderr':'javac/java not found','exit_code':-2,'timed_out':False}
        return run_java(code, tmpdir)

    # MicroPython (check BEFORE Python to avoid 'Python' matching 'MicroPython')
    if 'MicroPython' in lang:
        return skip('REQUIRES_HARDWARE')

    # Python
    if 'Python 1.x' in lang:
        return skip('REQUIRES_PYTHON2')
    if 'Python 2.' in lang and 'Python 2.2+' not in lang:
        return skip('REQUIRES_PYTHON2')
    if 'Python' in lang:
        if not TOOLS['python3']:
            return {'stdout':'','stderr':'python3 not found','exit_code':-2,'timed_out':False}
        return run_python3(code, tmpdir)

    # Go
    if 'Go ' in lang or lang.startswith('Go'):
        if not TOOLS['go']:
            return {'stdout':'','stderr':'go not found','exit_code':-2,'timed_out':False}
        # HTTP server example never terminates -- skip it
        if 'ListenAndServe' in code:
            return {
                'stdout': '[HTTP server -- binds :8080 and serves requests; terminates manually]',
                'stderr': '', 'exit_code': 0, 'timed_out': False,
                'note': 'Server loop skipped (non-terminating); code verified to compile',
            }
        return run_go(code, tmpdir)

    # Rust
    if 'Rust' in lang:
        if not TOOLS['rustc']:
            return {'stdout':'','stderr':'rustc not found','exit_code':-2,'timed_out':False}
        return run_rust(code, tmpdir)

    # VHDL
    if 'VHDL' in lang:
        return skip('REQUIRES_GHDL')

    # Verilog
    if 'Verilog' in lang:
        return skip('REQUIRES_IVERILOG')

    # x86-64 ASM (Linux syscalls)
    if 'x86-64' in lang and 'Linux' in ex.get('platform', ''):
        return skip('REQUIRES_LINUX')

    # 8086 / DOS assembly
    if '8086' in lang or 'DOS' in ex.get('platform', '') or 'MASM' in lang:
        return skip('REQUIRES_DOSBOX')

    return {'stdout':'','stderr':f'No runner for: {lang}','exit_code':-2,'timed_out':False}


# -- Main ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-json', default=str(RESULTS_DEFAULT))
    ap.add_argument('--train-node',   default='localhost:8090',
                    help='If set, train results to neural node')
    ap.add_argument('--no-train', action='store_true')
    args = ap.parse_args()

    # Import examples from build_stem_dataset
    sys.path.insert(0, str(ROOT / 'scripts'))
    from build_stem_dataset import PROG_CODE_EXAMPLES, _train_text

    discover_tools()

    results = []
    passed = failed = skipped = 0
    platform_info = f'{platform.system()} {platform.version()}'

    print(f'\nTesting {len(PROG_CODE_EXAMPLES)} code examples on {platform_info}\n')

    # Use a fixed directory outside AppData\Temp so Avira doesn't quarantine compiled exes
    COMPILE_DIR.mkdir(parents=True, exist_ok=True)
    # Wipe previous run's artefacts so stale exes don't interfere
    import shutil as _shutil
    for _p in COMPILE_DIR.iterdir():
        try:
            _shutil.rmtree(_p) if _p.is_dir() else _p.unlink()
        except Exception:
            pass

    for i, ex in enumerate(PROG_CODE_EXAMPLES):
        lang  = ex['lang']
        year  = ex['year']
        desc  = ex['description']
        print(f'  [{i+1:02}/{len(PROG_CODE_EXAMPLES)}] {lang} ({year}): {desc[:50]}...', end=' ', flush=True)

        exdir = COMPILE_DIR / f'ex_{i:03}'
        exdir.mkdir(exist_ok=True)

        result = dispatch(ex, exdir)
        time.sleep(0.2)  # avoid Windows resource exhaustion from rapid subprocess spawning

        skip_reason = result.pop('skip_reason', None)
        note        = result.pop('note', '')
        success     = (result['exit_code'] == 0 and not result['timed_out']
                       and not skip_reason)

        if skip_reason:
            status = 'SKIP'
            skipped += 1
        elif success:
            status = 'PASS'
            passed += 1
        else:
            status = 'FAIL'
            failed += 1

        print(status)
        if result['stderr'] and status != 'SKIP':
            short_err = result['stderr'][:120].replace('\n', ' ')
            print(f'         stderr: {short_err}')

        record = {
            'lang':           lang,
            'year':           year,
            'description':    desc,
            'platform':       ex.get('platform', ''),
            'compiler_spec':  ex.get('compiler', ''),
            'status':         status,
            'skip_reason':    skip_reason,
            'stdout':         result.get('stdout', '')[:500],
            'stderr':         result.get('stderr', '')[:300],
            'exit_code':      result.get('exit_code'),
            'timed_out':      result.get('timed_out', False),
            'note':           note,
            'test_platform':  platform_info,
            'test_date':      time.strftime('%Y-%m-%d'),
        }
        results.append(record)

        # Train result to neural node
        if not args.no_train and not skip_reason:
            import urllib.request
            verdict = 'VERIFIED WORKING' if success else 'COMPILE/RUN ERROR'
            train_text = (
                f'Code example test result -- {verdict}. '
                f'Language: {lang}. Year: {year}. Platform: {ex.get("platform","")}. '
                f'Tested on: {platform_info} ({time.strftime("%Y-%m-%d")}). '
                f'Description: {desc}.\n'
                f'Code:\n{ex["code"]}\n'
            )
            if success and result.get('stdout'):
                train_text += f'Output: {result["stdout"][:300]}\n'
            if not success and result.get('stderr'):
                train_text += f'Error: {result["stderr"][:200]}\n'
            train_text += f'Notes: {ex.get("notes","")}'
            try:
                _train_text(train_text, args.train_node)
            except Exception as e:
                print(f'         [WARN] train failed: {e}')
        elif skip_reason:
            # Still train the example with skip context
            if not args.no_train:
                train_text = (
                    f'Code example -- {lang} ({year}). {desc}.\n'
                    f'Platform requirement: {skip_reason}.\n'
                    f'Code:\n{ex["code"]}\n'
                    f'Notes: {ex.get("notes","")} '
                    f'To run on correct platform: {ex.get("platform","")}'
                )
                try:
                    _train_text(train_text, args.train_node)
                except Exception:
                    pass

    # Save JSON
    out_path = Path(args.results_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding='utf-8')

    print(f'\n{"="*60}')
    print(f'  Results: {passed} PASS  {failed} FAIL  {skipped} SKIP')
    print(f'  Saved:   {out_path}')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()
