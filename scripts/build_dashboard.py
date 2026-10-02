"""Inject data/derived/dashboard.json into the template -> data/derived/dashboard.html"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
tpl = (ROOT / "scripts" / "dashboard_template.html").read_text()
data = (ROOT / "data" / "derived" / "dashboard.json").read_text().replace("</", "<\\/")
out = ROOT / "data" / "derived" / "dashboard.html"
out.write_text(tpl.replace("/*__DATA__*/", data))
print("wrote", out, round(out.stat().st_size / 1e6, 2), "MB")
