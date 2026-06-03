"""Build a self-contained static demo page (docs/demo/index.html) from a real run.

It spins the API up in-process (TestClient, ephemeral SQLite), ingests
data/events.jsonl, queries every endpoint for each store in pipeline/config/stores.json,
and bakes the real JSON into one static HTML file - no backend/GPU needed to view it.

    python scripts/build_demo.py

Host it in ~1 min: drag docs/demo/ onto https://app.netlify.com/drop, or push to a
PUBLIC GitHub repo and enable Pages. (The page shows only aggregate numbers - no
footage - so it is safe to publish.) If you later host the API, open the page with
?api=https://your-api to make it fetch live instead of the baked snapshot.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)  # so `app` imports when run as a script

# Public repo (used for footer links on the demo page). Override with --repo.
REPO_URL = os.environ.get("DEMO_REPO", "https://github.com/Ikshvaku24/store-intelligence")
os.environ["DB_URL"] = "sqlite:///" + tempfile.mktemp(suffix=".db").replace("\\", "/")
os.environ.setdefault("POS_CSV_PATH", "data/pos_transactions.csv")


def collect() -> dict:
    from fastapi.testclient import TestClient

    from app import db

    db.reset_engine()
    from app.main import app

    stores = json.load(open("pipeline/config/stores.json", encoding="utf-8"))["stores"]
    events = [json.loads(l) for l in open("data/events.jsonl", encoding="utf-8-sig") if l.strip()]
    out: dict = {"stores": [], "n_events": len(events)}
    with TestClient(app) as c:
        for i in range(0, len(events), 500):
            c.post("/events/ingest", json=events[i : i + 500])
        out["health"] = c.get("/health").json()
        for key, s in stores.items():
            sid = s["store_id"]
            out["stores"].append({
                "key": key,
                "store_id": sid,
                "metrics": c.get(f"/stores/{sid}/metrics").json(),
                "funnel": c.get(f"/stores/{sid}/funnel").json(),
                "heatmap": c.get(f"/stores/{sid}/heatmap").json(),
                "anomalies": c.get(f"/stores/{sid}/anomalies").json(),
            })
    return out


TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Store Intelligence - Live Demo</title>
<style>
:root{--bg:#0f1117;--card:#171b24;--line:#262d3a;--ink:#e7edf5;--mut:#94a1b2;--acc:#6d7cff;--ok:#3fb950;--warn:#d29922;--bad:#f85149;--rev:#e3a008}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:24px}
h1{font-size:22px;margin:0 0 2px}.sub{color:var(--mut);margin:0 0 18px;font-size:13px}
.tabs{display:flex;gap:8px;margin:16px 0}.tab{padding:8px 14px;border:1px solid var(--line);border-radius:9px;background:#10141c;color:var(--ink);cursor:pointer;font-weight:600}
.tab.on{background:var(--acc);border-color:var(--acc);color:#0a0e18}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:12px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}
.kpi .v{font-size:26px;font-weight:700}.kpi .l{color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.05em}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:12px 0}
.card h3{margin:0 0 12px;font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut)}
.bar{height:26px;border-radius:6px;background:var(--acc);display:flex;align-items:center;padding:0 8px;color:#0a0e18;font-weight:600;white-space:nowrap;min-width:34px}
.row{display:flex;align-items:center;gap:10px;margin:7px 0}.row .nm{width:150px;color:var(--mut);font-size:13px;flex:none}
.row .track{flex:1;background:#10141c;border-radius:6px;overflow:hidden}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700}
.INFO{background:#10233a;color:#7cc4ff}.WARN{background:#2a2410;color:#ffd479}.CRITICAL{background:#2a1416;color:#ff9b95}
.muted{color:var(--mut);font-size:12px}.foot{color:var(--mut);font-size:12px;margin-top:24px;border-top:1px solid var(--line);padding-top:14px}
a{color:var(--acc)}.pill{font-size:11px;border:1px solid var(--line);border-radius:10px;padding:1px 7px;color:var(--mut)}
</style></head><body><div class="wrap">
<h1>Store Intelligence &mdash; Live Demo</h1>
<p class="sub">Real output from the detection pipeline + intelligence API. <span id="meta"></span></p>
<div class="tabs" id="tabs"></div>
<div id="view"></div>
<div class="foot" id="foot"></div>
</div>
<script>
const BAKED = __DATA__;
const REPO = "__REPO__";
const qApi = new URLSearchParams(location.search).get('api');
let DATA = BAKED, cur = 0;
const $=s=>document.querySelector(s);
const pct=(n)=>Math.max(2,Math.round(n));
function kpi(v,l){return `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`}
function render(){
  const s = DATA.stores[cur]; const m=s.metrics, f=s.funnel, h=s.heatmap, a=s.anomalies;
  $('#meta').textContent = `${DATA.n_events} events ingested across ${DATA.stores.length} stores`+(qApi?` (live: ${qApi})`:` (snapshot)`);
  $('#tabs').innerHTML = DATA.stores.map((x,i)=>`<div class="tab ${i==cur?'on':''}" data-i="${i}">${x.store_id}</div>`).join('');
  document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{cur=+t.dataset.i;render()});
  const conv=(m.conversion_rate*100).toFixed(1)+'%';
  let html = `<div class="grid">
    ${kpi(m.unique_visitors,'Unique visitors')}
    ${kpi(conv,'Conversion rate')}
    ${kpi(m.purchases,'Purchases')}
    ${kpi(m.current_queue_depth,'Queue depth')}
    ${kpi((m.abandonment_rate*100).toFixed(0)+'%','Queue abandon')}
  </div>
  <div class="muted">basis: ${m.visitor_basis} &middot; data confidence: ${m.data_confidence}${m.avg_queue_wait_seconds?` &middot; avg wait ${m.avg_queue_wait_seconds}s`:''}</div>`;
  // funnel
  const fmax = Math.max(1,...f.stages.map(s=>s.count));
  html += `<div class="card"><h3>Conversion funnel</h3>`+
    f.stages.map(s=>`<div class="row"><div class="nm">${s.stage}${s.drop_off_pct?` <span class="pill">-${s.drop_off_pct}%</span>`:''}</div>
      <div class="track"><div class="bar" style="width:${pct(100*s.count/fmax)}%">${s.count}</div></div></div>`).join('')+
    `<div class="muted">${f.note||''}</div></div>`;
  // zones / heatmap
  const zs=(h.zones||[]).slice().sort((x,y)=>(y.avg_dwell_ms||0)-(x.avg_dwell_ms||0));
  if(zs.length){const zmax=Math.max(1,...zs.map(z=>z.avg_dwell_ms||0));
    html += `<div class="card"><h3>Zone dwell &amp; visits</h3>`+
    zs.map(z=>`<div class="row"><div class="nm">${z.zone_id}</div>
      <div class="track"><div class="bar" style="width:${pct(100*(z.avg_dwell_ms||0)/zmax)}%;background:var(--rev)">${((z.avg_dwell_ms||0)/1000).toFixed(1)}s</div></div>
      <div class="muted" style="width:70px">${z.visit_count} visits</div></div>`).join('')+`</div>`;}
  // demographics + groups
  const d=m.demographics||{}, g=m.groups||{};
  if((d.n_classified||0)||(g.group_count||0)){
    const gen=Object.entries(d.by_gender||{}).map(([k,v])=>`${k}: ${v}`).join(' &middot; ')||'-';
    const age=Object.entries(d.by_age_bucket||{}).map(([k,v])=>`${k}: ${v}`).join(' &middot; ')||'-';
    html += `<div class="card"><h3>Demographics &amp; groups</h3>
      <div class="row"><div class="nm">Gender</div><div>${gen}</div></div>
      <div class="row"><div class="nm">Age bucket</div><div>${age}</div></div>
      <div class="row"><div class="nm">Groups</div><div>${g.group_count||0} groups &middot; ${g.visitors_in_groups||0} in groups &middot; avg size ${g.avg_group_size||0}</div></div></div>`;}
  // anomalies
  const al=a.anomalies||a.active||[];
  html += `<div class="card"><h3>Anomalies</h3>`+
    (al.length? al.map(x=>`<div class="row"><span class="badge ${x.severity}">${x.severity}</span>
      <b>${x.type||x.kind}</b><span class="muted">${x.suggested_action||x.detail||''}</span></div>`).join('')
      :`<div class="muted">No active anomalies in this window.</div>`)+`</div>`;
  $('#view').innerHTML = html;
  const hh=DATA.health||{};
  $('#foot').innerHTML = `API health: <b>${hh.status||'?'}</b> &middot; a real snapshot from a detection-pipeline run; `+
    `the full live system runs via <code>docker compose up</code> (API + dashboard) with the GPU detection pipeline. `+
    `Conversion is 0 when the sample POS feed's store_id doesn't match the footage store - the correlation mechanism is built &amp; tested.`+
    `<div style="margin-top:10px">`+
    `<a href="${REPO}">GitHub repo</a> &middot; `+
    `<a href="PRESENTATION.pdf">Slides (PDF)</a> &middot; `+
    `<a href="${REPO}/blob/master/docs/DESIGN.md">DESIGN.md</a> &middot; `+
    `<a href="${REPO}/blob/master/docs/CHOICES.md">CHOICES.md</a> &middot; `+
    `<a href="${REPO}/blob/master/README.md">README</a></div>`;
}
async function boot(){
  if(qApi){ try{
    const stores = BAKED.stores.map(s=>s.store_id); const out={stores:[],n_events:BAKED.n_events};
    out.health = await (await fetch(qApi+'/health')).json();
    for(const sid of stores){ out.stores.push({store_id:sid,
      metrics:await (await fetch(`${qApi}/stores/${sid}/metrics`)).json(),
      funnel:await (await fetch(`${qApi}/stores/${sid}/funnel`)).json(),
      heatmap:await (await fetch(`${qApi}/stores/${sid}/heatmap`)).json(),
      anomalies:await (await fetch(`${qApi}/stores/${sid}/anomalies`)).json()}); }
    DATA=out;
  }catch(e){ console.warn('live fetch failed, using snapshot',e); } }
  render();
}
boot();
</script></body></html>"""


def main() -> None:
    data = collect()
    docs = os.path.join(ROOT, "docs")
    os.makedirs(docs, exist_ok=True)
    html = TEMPLATE.replace("__DATA__", json.dumps(data)).replace("__REPO__", REPO_URL)
    # docs/index.html = the GitHub Pages root (Pages source: master /docs).
    with open(os.path.join(docs, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(html)
    # Disable Jekyll so files serve as-is.
    open(os.path.join(docs, ".nojekyll"), "w").close()
    s0 = data["stores"][0]["metrics"]
    print(f"Wrote docs/index.html + docs/.nojekyll  ({data['n_events']} events, "
          f"{len(data['stores'])} stores; e.g. {data['stores'][0]['store_id']} "
          f"visitors={s0['unique_visitors']})")


if __name__ == "__main__":
    main()
