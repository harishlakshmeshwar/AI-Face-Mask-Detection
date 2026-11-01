"""Static frontend consistency checks.

The dashboard runs in the browser, so pytest cannot execute it directly.
These tests catch the failure modes that previously broke Start detection:

* ``els.foo`` used but ``foo`` missing from the els map -> TypeError the
  moment Start is clicked (verified root cause of 'camera never opens').
* ``getElementById`` referencing an id that does not exist in the template
  -> null element and a silent crash at runtime.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JS = (ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")
HTML = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")


def test_every_els_reference_is_defined():
    start = JS.index("var els = {")
    els_block = JS[start : JS.index("};", start)]
    defined = set(re.findall(r"(\w+):\s*document\.getElementById", els_block))
    used = set(re.findall(r"els\.(\w+)", JS))
    assert used - defined == set(), (
        "app.js references undefined els entries (throws at runtime): "
        f"{sorted(used - defined)}"
    )


def test_every_dom_id_exists_in_template():
    ids_used = set(re.findall(r'getElementById\("([^"]+)"\)', JS))
    ids_in_html = set(re.findall(r'id="([^"]+)"', HTML))
    assert ids_used - ids_in_html == set(), (
        f"app.js looks up ids missing from index.html: {sorted(ids_used - ids_in_html)}"
    )


def test_template_ids_referenced_by_css_exist():
    css = (ROOT / "static" / "css" / "style.css").read_text(encoding="utf-8")
    # Id selectors only (ignore hex colours like #1f6feb).
    css_ids = set(re.findall(r'#([A-Za-z][\w-]*)\s*[{,:]', css))
    ids_in_html = set(re.findall(r'id="([^"]+)"', HTML))
    # Skip the skip-link anchor target style matches and non-element ids.
    missing = {i for i in css_ids if i not in ids_in_html}
    assert missing == set(), f"CSS targets ids absent from index.html: {sorted(missing)}"
