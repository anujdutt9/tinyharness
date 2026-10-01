"""Bundled assets: offline charts for HTML pages the agent makes."""

from pathlib import Path

from tinyharness.assets import CHART_JS, INLINE_SCRIPT
from tinyharness.tools.screenshot import _remote_assets_note


def _inline(html: str, base: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("inline_scripts", INLINE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.inline(html, base)


def test_the_bundled_chart_js_is_pinned_and_licensed():
    head = CHART_JS.read_text(encoding="utf-8")[:200]
    assert "Chart.js v4.5.1" in head and "MIT" in head
    assert (CHART_JS.parent / "CHARTJS-LICENSE.md").is_file()


def test_a_cdn_chart_js_link_and_local_scripts_are_inlined(tmp_path):
    (tmp_path / "app.js").write_text("console.log('</script>')")
    html = (
        '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>'
        '<script defer src="app.js"></script><script src="https://example.com/other.js"></script>'
    )
    out, done, left = _inline(html, tmp_path)
    assert done == ["https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js", "app.js"]
    assert left == ["https://example.com/other.js"]
    assert "Chart.js v4.5.1" in out and "<script defer>" in out
    assert "console.log('<\\/script>')" in out  # can't close the inline script early


def test_screenshot_says_when_a_local_page_needs_the_network(tmp_path):
    page = tmp_path / "my report.html"
    page.write_text(
        '<link href="https://fonts.example/x.css" rel=stylesheet><script src="//cdn.x/c.js"></script>'
    )
    note = _remote_assets_note(page.as_uri())
    assert "can't load here" in note and "https://fonts.example/x.css" in note and str(INLINE_SCRIPT) in note
    (tmp_path / "ok.html").write_text('<script src="chart.umd.min.js"></script>')
    assert _remote_assets_note((tmp_path / "ok.html").as_uri()) == ""
