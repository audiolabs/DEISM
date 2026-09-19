"""Build the single-file DEISM playground (playground/demo.html).

Concatenates the engine modules, the UI modules and the worker into one
HTML page with directivity datasets loaded from adjacent data/ files. No
bundler or network access is needed: the ES-module files are joined in
dependency order with their import/export statements rewritten.

Usage (from the repository root)::

    python tools/playground_directivity.py
    python tools/playground_build.py [--out playground/demo.html]
"""

import argparse
import base64
import json
import mimetypes
import os
import re
import shutil
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PG = os.path.join(REPO_ROOT, "playground")

ENGINE_MODULES = [
    "special.js",
    "linalg.js",
    "fft.js",
    "geometry.js",
    "shoebox.js",
    "materials.js",
    "directivity.js",
    "kernels.js",
    "deism.js",
    "data.js",
]
APP_MODULES = ["scene.js", "plots.js", "datasets.js", "presets.js", "state.js", "python-export.js", "app.js"]


LICENSE_NOTICE_HTML = """<!--
  DEISM Playground
  Copyright (c) 2022-2026 Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V.
  Generated from playground/src and playground/engine; the engine is a JavaScript
  port of the DEISM Python package. Subject to the Fraunhofer Software Copyright
  License distributed with the DEISM package (LICENSE). Requires a separate
  license from Fraunhofer beyond internal, non-commercial use for evaluation,
  testing, and academic research.
  The convex-room geometry helpers derive from the MIT-licensed libroom core of
  pyroomacoustics, Copyright (C) 2019 Robin Scheibler, Cyril Cadoux; the MIT
  notice is reproduced in the engine source (playground/engine/geometry.js) and
  in thirdPartyLegalNotices/attributions.txt.
-->
"""


IMPORT_RE = re.compile(r"^import\s+(\{[^}]*\}|[\w$]+)\s+from\s+[\"']([^\"']+)[\"'];?\s*$", re.M | re.S)
EXPORT_LIST_RE = re.compile(r"^export\s*\{([^}]*)\};?\s*$", re.M)
EXPORT_DECL_RE = re.compile(r"^export\s+(?=(?:async\s+)?(?:function|class|const|let|var)\b)", re.M)
DECL_NAME_RE = re.compile(r"^(?:export\s+)?(?:async\s+)?(?:function|class|const|let|var)\s+([\w$]+)", re.M)


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def strip_module(src, engine_names, declared=None):
    """Rewrite one ES module into plain script text.

    Imports from the engine become destructuring from the DeismEngine global
    (names already pulled in by an earlier UI module, tracked in `declared`,
    are skipped so the shared scope has no duplicate declarations); imports
    between UI modules are dropped (they share one scope). Exports are turned
    into plain declarations; exported names are returned.
    """
    exported = set()
    for m in EXPORT_LIST_RE.finditer(src):
        for name in m.group(1).split(","):
            name = name.strip()
            if name:
                exported.add(name.split(" as ")[-1].strip())
    src = EXPORT_LIST_RE.sub("", src)
    for m in re.finditer(r"^export\s+(?:async\s+)?(?:function|class|const|let|var)\s+([\w$]+)", src, re.M):
        exported.add(m.group(1))
    src = EXPORT_DECL_RE.sub("", src)

    def repl(m):
        spec, target = m.group(1), m.group(2)
        if "/engine/" in target:
            names = [n.strip().split(" as ")[-1].strip() for n in spec.strip("{} \n").split(",") if n.strip()]
            if declared is not None:
                names = [n for n in names if n not in declared]
                declared.update(names)
            if not names:
                return ""
            return "const { " + ", ".join(names) + " } = DeismEngine;"
        return ""  # engine-internal or ui-internal import: same scope after concatenation

    src = IMPORT_RE.sub(repl, src)
    return src, exported


def bundle_engine():
    parts = []
    names = set()
    seen = {}
    for mod in ENGINE_MODULES:
        src = read(os.path.join(PG, "engine", mod))
        # engine-internal imports are dropped (engine_names=set() -> not "/engine/" paths are ./ relative)
        body, exported = strip_module(src, set())
        for decl in DECL_NAME_RE.findall(body):
            if decl in seen and seen[decl] != mod:
                raise SystemExit(f"duplicate top-level name {decl!r} in {mod} and {seen[decl]}")
            seen[decl] = mod
        names |= exported
        parts.append(f"// ---- engine/{mod} ----\n{body}")
    body = "\n".join(parts)
    return (
        "const DeismEngine = (() => {\n"
        + body
        + "\nreturn { "
        + ", ".join(sorted(names))
        + " };\n})();\n"
    )


def bundle_app():
    parts = []
    declared = set()
    for mod in APP_MODULES:
        src = read(os.path.join(PG, "src", mod))
        body, _ = strip_module(src, None, declared)
        parts.append(f"// ---- src/{mod} ----\n{body}")
    return "(() => {\n" + "\n".join(parts) + "\n})();\n"


def bundle_worker(engine_js):
    src = read(os.path.join(PG, "src", "worker.js"))
    body, _ = strip_module(src, None)
    return engine_js + "\n// ---- src/worker.js ----\n" + body


def replace_block(html, tag, content):
    pattern = re.compile(rf"<!-- BUILD:{tag} -->.*?<!-- /BUILD:{tag} -->", re.S)
    if not pattern.search(html):
        raise SystemExit(f"index.html has no BUILD:{tag} block")
    return pattern.sub(lambda _m: content, html)


def safe_script(text):
    return text.replace("</script", "<\\/script")


ASSET_SRC_RE = re.compile(r'src="assets/([^"]+)"')


def inline_assets(html):
    """Embed images referenced as src="assets/..." (the AudioLabs logo) as data
    URIs so the built pages keep loading no external resources."""

    def repl(m):
        path = os.path.join(PG, "src", "assets", m.group(1))
        if not os.path.isfile(path):
            raise SystemExit(f"missing asset {path}")
        mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        return f'src="data:{mime};base64,{data}"'

    return ASSET_SRC_RE.sub(repl, html)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(PG, "demo.html"))
    ap.add_argument("--native-out", default=os.path.join(PG, "native.html"))
    ap.add_argument("--data", default=os.path.join(PG, "data"), help="directory with dataset JSON files")
    args = ap.parse_args()

    html = read(os.path.join(PG, "src", "index.html"))
    if not html.startswith("<!DOCTYPE html>"):
        raise SystemExit("index.html must start with <!DOCTYPE html>")
    html = html.replace("<!DOCTYPE html>", "<!DOCTYPE html>\n" + LICENSE_NOTICE_HTML.rstrip("\n"), 1)
    css = read(os.path.join(PG, "src", "style.css"))
    engine_js = bundle_engine()
    app_js = bundle_app()
    worker_js = bundle_worker(engine_js)

    html = replace_block(html, "STYLE", "<style>\n" + css + "\n</style>")
    catalog_src = read(os.path.join(PG, "src", "datasets.js"))
    catalog = json.loads(catalog_src.split(" = ", 1)[1].strip().rstrip(";"))
    # Keep generated data outside the HTML; custom output locations remain usable.
    for name, info in catalog.items():
        if not info["supported"]:
            continue
        path = os.path.join(args.data, name + ".json")
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}; run tools/playground_directivity.py")
        for output in (args.out, args.native_out):
            destination = os.path.join(os.path.dirname(os.path.abspath(output)), "data", name + ".json")
            if os.path.abspath(path) != destination:
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                shutil.copyfile(path, destination)
    html = html.replace('data-directivity-base="../data/"', 'data-directivity-base="data/"')
    html = inline_assets(html)
    # The footer links to LICENSE.txt next to the page (the .txt suffix makes
    # plain static servers deliver it as text); the native server serves the
    # same packaged copy at /LICENSE.txt.
    for output in (args.out, args.native_out):
        destination = os.path.join(os.path.dirname(os.path.abspath(output)), "LICENSE.txt")
        shutil.copyfile(os.path.join(REPO_ROOT, "LICENSE"), destination)
    html = replace_block(html, "WORKER", '<script type="text/js-worker" id="worker-src">\n' + safe_script(worker_js) + "\n</script>")
    html = replace_block(html, "APP", "<script>\n" + engine_js + "\n" + app_js + "\n</script>")
    native_html = replace_block(html, "DATA", "")
    with open(args.native_out, "w", encoding="utf-8") as f:
        f.write(native_html)
    html = replace_block(html, "DATA", "")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
