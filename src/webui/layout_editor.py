"""Interactive layout editor adapted from the original GlyphAnchor demo."""

CANVAS_HTML = """
<style>
/* ── 把 Gradio layout_json textbox 用 CSS 完全隐藏（但 Svelte 仍能监听其 input 事件） ── */
#layout-json-store { display:none !important; }

#glyph-editor-root { position:relative; user-select:none; }
#glyph-canvas-wrap { width:100%; display:flex; justify-content:flex-start; }
#glyph-canvas {
  position:relative; height:520px; width:520px; max-width:100%;
  background:
    repeating-linear-gradient(0deg,transparent,transparent 31px,#ddd 31px,#ddd 32px),
    repeating-linear-gradient(90deg,transparent,transparent 31px,#ddd 31px,#ddd 32px);
  background-color:#f8f8f8;
  border:2px solid #ccc; border-radius:8px; overflow:hidden; flex-shrink:0;
}
.glyph-box {
  position:absolute; border:2px solid; border-radius:4px; box-sizing:border-box;
  cursor:move; display:flex; align-items:flex-start; overflow:visible;
  min-width:20px; min-height:16px;
}
.glyph-box:hover { z-index:20; }
.glyph-box::before { content:""; position:absolute; top:-28px; left:0; right:0; height:28px; }
.glyph-box .box-label {
  overflow:hidden; max-height:100%; box-sizing:border-box;
  font-size:12px; font-weight:600; padding:2px 4px; line-height:1.3;
  white-space:pre-wrap; word-break:break-all; pointer-events:none; width:100%;
}
.glyph-box .box-del {
  position:absolute; top:-27px; right:0px; width:18px; height:18px;
  border-radius:50%; background:#e74c3c; color:#fff; font-size:12px;
  line-height:18px; text-align:center; cursor:pointer; z-index:10; display:none;
}
.glyph-box:hover .box-del { display:block; }
.glyph-box .box-align { position:absolute; top:-29px; left:0; height:24px;
  z-index:11; display:none; background:white; color:#444; border:1px solid #ccc;
  border-radius:4px; white-space:nowrap; }
.glyph-box .box-align button { width:24px; height:24px; padding:2px; border:0;
  background:white; color:#555; cursor:pointer; display:inline-flex; }
.glyph-box .box-align button.active { background:#dbeafe; color:#2563eb; }
.glyph-box .box-align button:nth-child(4) { border-left:1px solid #ccc; }
.glyph-box:hover .box-align, .glyph-box:focus-within .box-align { display:flex; }
.glyph-box .rh {
  position:absolute; width:10px; height:10px; background:#fff;
  border:2px solid; border-radius:2px; z-index:5;
}
.rh.se{bottom:-5px;right:-5px;cursor:se-resize;}
.rh.sw{bottom:-5px;left:-5px;cursor:sw-resize;}
.rh.ne{top:-5px;right:-5px;cursor:ne-resize;}
.rh.nw{top:-5px;left:-5px;cursor:nw-resize;}
.rh.n{top:-5px;left:calc(50% - 5px);cursor:n-resize;}
.rh.s{bottom:-5px;left:calc(50% - 5px);cursor:s-resize;}
.rh.e{right:-5px;top:calc(50% - 5px);cursor:e-resize;}
.rh.w{left:-5px;top:calc(50% - 5px);cursor:w-resize;}
#glyph-text-editor {
  position:absolute; z-index:100; background:#fff; border:2px solid #3498db;
  border-radius:4px; padding:4px 6px; font-size:13px; min-width:80px; min-height:28px;
  outline:none; white-space:pre-wrap; word-break:break-all;
  box-shadow:0 4px 16px rgba(0,0,0,.18); resize:none; overflow:hidden;
}
</style>

<div id="glyph-editor-root">
  <div style="display:flex;gap:8px;align-items:center;margin-bottom:8px;flex-wrap:wrap;">
    <button id="btn-add-box"
      style="padding:5px 14px;border-radius:6px;border:1px solid #aaa;
             background:#f0f4ff;cursor:pointer;font-size:13px;">
      + 添加文字框
    </button>
    <button id="btn-clear-all"
      style="padding:5px 14px;border-radius:6px;border:1px solid #f99;
             background:#fff0f0;cursor:pointer;font-size:13px;">
      清空
    </button>
    <span style="font-size:12px;color:#888;">
      拖动移动 · 拖角/边缘缩放 · 双击编辑文字 · 点 × 删除
    </span>
  </div>
  <div id="glyph-canvas-wrap">
    <div id="glyph-canvas"></div>
  </div>
  <div style="font-size:11px;color:#aaa;margin-top:4px;">
    画布高度固定 520px，宽度随 Width/Height 比例自动调整 · 坐标为归一化 xywh (0~1)
  </div>
</div>
"""

CANVAS_JS = """
() => {
  if (window._glyphEditorInited) return;
  window._glyphEditorInited = true;

  const LOG = (...a) => console.log('[GlyphEditor]', ...a);
  LOG('CANVAS_JS injected via launch(head=)');

  const COLORS = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6',
                  '#1abc9c','#e67e22','#34495e','#e91e63','#00bcd4'];
  let boxes = [], nextId = 1, lastLoaded = '';
  let _canvasW = 1536, _canvasH = 1536;

  const $ = id => document.getElementById(id);

  /* ── find hidden Gradio textarea ── */
  function getInp() {
    const w = $('layout-json-store');
    if (!w) return null;
    return w.querySelector('textarea') || w.querySelector('input');
  }

  /* ── sync boxes → Gradio textbox（visible=True 后 Svelte 能监听 input 事件） ── */
  function sync() {
    const inp = getInp();
    if (!inp) return;
    const data = boxes.map(b => ({
      idx: b.idx, text: b.text, align: b.align || 'left', valign: b.valign || 'center',
      bbox: [+b.x.toFixed(4), +b.y.toFixed(4), +b.w.toFixed(4), +b.h.toFixed(4)]
    }));
    inp.value = JSON.stringify(data);
    lastLoaded = inp.value;
    inp.dispatchEvent(new Event('input', { bubbles: true }));
  }

  /* ── load JSON → canvas ── */
  function load(s) {
    if (!s || !s.trim()) return;
    try {
      const data = JSON.parse(s);
      LOG('load', data.length, 'items');
      boxes = data.map((it, i) => {
        const bb = it.bbox;
        return { id: nextId++, idx: it.idx || (i+1), text: it.text || '', align: it.align || 'left', valign: it.valign || 'center',
                 x: bb[0], y: bb[1], w: bb[2], h: bb[3] };
      });
      renderAll();
    } catch(e) { LOG('load err', e); }
  }

  /* ── setAspect: 固定高度 520px，宽度按比例 ── */
  function setAspect(w, h) {
    _canvasW = w || 1536; _canvasH = h || 1536;
    const canvas = $('glyph-canvas');
    if (!canvas) return;
    const ratio = _canvasW / _canvasH;
    const maxW = (canvas.parentElement
      ? canvas.parentElement.getBoundingClientRect().width : 0) || 800;
    const displayWidth = Math.min(520 * ratio, maxW);
    canvas.style.width = displayWidth + 'px';
    canvas.style.height = (displayWidth / ratio) + 'px';
    renderAll();
  }

  /* ── renderAll ── */
  function renderAll() {
    const canvas = $('glyph-canvas');
    if (!canvas) { LOG('renderAll: no canvas'); return; }
    canvas.innerHTML = '';
    boxes.forEach(b => renderBox(canvas, b));
  }

  function renderBox(canvas, b) {
    const r = canvas.getBoundingClientRect();
    const color = COLORS[(b.idx - 1) % COLORS.length];
    const div = document.createElement('div');
    div.className = 'glyph-box';
    div.dataset.id = b.id;
    div.style.cssText =
      `left:${b.x*r.width}px;top:${b.y*r.height}px;` +
      `width:${b.w*r.width}px;height:${b.h*r.height}px;` +
      `border-color:${color};background:${color}22;`;

    const lbl = document.createElement('div');
    lbl.className = 'box-label';
    lbl.style.color = color;
    lbl.textContent = b.text;
    lbl.style.textAlign = 'left';
    div.style.alignItems = 'flex-start';
    div.appendChild(lbl);

    const align = document.createElement('div');
    align.className = 'box-align';
    const icons = {
      left: '<path d="M3 4h14M3 8h9M3 12h14M3 16h9"/>',
      center: '<path d="M3 4h14M6 8h8M3 12h14M6 16h8"/>',
      right: '<path d="M3 4h14M8 8h9M3 12h14M8 16h9"/>',
      top: '<path d="M3 3h14M7 6v11M13 6v7"/>',
      middle: '<path d="M3 10h14M7 4v12M13 6v8"/>',
      bottom: '<path d="M3 17h14M7 3v11M13 7v7"/>'
    };
    for (const [field, value, label, icon] of [
      ['align','left','左对齐','left'], ['align','center','水平居中','center'], ['align','right','右对齐','right'],
      ['valign','top','顶部对齐','top'], ['valign','center','垂直居中','middle'], ['valign','bottom','底部对齐','bottom']
    ]) {
      const button = document.createElement('button');
      button.type = 'button'; button.title = label; button.setAttribute('aria-label', label);
      const active = (b[field] || (field === 'align' ? 'left' : 'center')) === value;
      button.className = active ? 'active' : '';
      button.setAttribute('aria-pressed', String(active));
      button.innerHTML = '<svg width="18" height="18" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5">'+icons[icon]+'</svg>';
      button.addEventListener('mousedown', e => e.stopPropagation());
      button.addEventListener('click', e => { e.stopPropagation(); b[field] = value; renderAll(); sync(); });
      button.addEventListener('dblclick', e => e.stopPropagation());
      align.appendChild(button);
    }
    const toolbarLeft = Math.max(-b.x * r.width, Math.min(0, r.width - b.x * r.width - 174));
    align.style.left = toolbarLeft + 'px';
    div.appendChild(align);

    const del = document.createElement('div');
    del.className = 'box-del';
    del.style.left = (toolbarLeft + 152) + 'px'; del.style.right = 'auto';
    if (b.y * r.height < 28) { align.style.top = '12px'; del.style.top = '12px'; }
    del.textContent = '\u00d7';
    del.addEventListener('mousedown', e => {
      e.stopPropagation();
      boxes = boxes.filter(x => x.id !== b.id);
      renderAll(); sync();
    });
    div.appendChild(del);

    ['nw','n','ne','w','e','sw','s','se'].forEach(d => {
      const h = document.createElement('div');
      h.className = `rh ${d}`;
      h.style.borderColor = color;
      h.addEventListener('mousedown', e => startResize(e, canvas, b, d));
      div.appendChild(h);
    });

    div.addEventListener('mousedown', e => {
      if (e.target.classList.contains('rh') ||
          e.target.classList.contains('box-del')) return;
      startMove(e, canvas, b);
    });
    div.addEventListener('dblclick', e => { e.stopPropagation(); openEditor(canvas, b); });
    canvas.appendChild(div);
  }

  function startMove(e, canvas, b) {
    e.preventDefault();
    const r = canvas.getBoundingClientRect();
    const mx = e.clientX, my = e.clientY, bx = b.x, by = b.y;
    const mv = ev => {
      b.x = Math.max(0, Math.min(1-b.w, bx+(ev.clientX-mx)/r.width));
      b.y = Math.max(0, Math.min(1-b.h, by+(ev.clientY-my)/r.height));
      const el = canvas.querySelector(`[data-id="${b.id}"]`);
      if (el) { el.style.left = b.x*r.width+'px'; el.style.top = b.y*r.height+'px'; }
    };
    const up = () => { document.removeEventListener('mousemove',mv); document.removeEventListener('mouseup',up); sync(); };
    document.addEventListener('mousemove', mv);
    document.addEventListener('mouseup', up);
  }

  function startResize(e, canvas, b, dir) {
    e.preventDefault(); e.stopPropagation();
    const r = canvas.getBoundingClientRect();
    const mx = e.clientX, my = e.clientY, o = {x:b.x,y:b.y,w:b.w,h:b.h};
    const mv = ev => {
      const dx=(ev.clientX-mx)/r.width, dy=(ev.clientY-my)/r.height;
      let {x,y,w,h} = o;
      if (dir.includes('e')) w = Math.max(.02, w+dx);
      if (dir.includes('s')) h = Math.max(.02, h+dy);
      if (dir.includes('w')) { x=Math.min(o.x+o.w-.02,x+dx); w=o.x+o.w-x; }
      if (dir.includes('n')) { y=Math.min(o.y+o.h-.02,y+dy); h=o.y+o.h-y; }
      x=Math.max(0,x); y=Math.max(0,y);
      w=Math.min(1-x,w); h=Math.min(1-y,h);
      b.x=x; b.y=y; b.w=w; b.h=h;
      const el = canvas.querySelector(`[data-id="${b.id}"]`);
      if (el) {
        el.style.left=x*r.width+'px'; el.style.top=y*r.height+'px';
        el.style.width=w*r.width+'px'; el.style.height=h*r.height+'px';
      }
    };
    const up = () => { document.removeEventListener('mousemove',mv); document.removeEventListener('mouseup',up); sync(); };
    document.addEventListener('mousemove', mv);
    document.addEventListener('mouseup', up);
  }

  function openEditor(canvas, b) {
    const old = $('glyph-text-editor');
    if (old) old.remove();
    const r = canvas.getBoundingClientRect();
    const ta = document.createElement('textarea');
    ta.id = 'glyph-text-editor';
    ta.value = b.text;
    ta.style.cssText =
      `left:${b.x*r.width+2}px;top:${b.y*r.height+2}px;` +
      `width:${Math.max(120,b.w*r.width-4)}px;height:${Math.max(40,b.h*r.height-4)}px;`;
    canvas.appendChild(ta);
    ta.focus(); ta.select();
    const commit = () => { b.text=ta.value; ta.remove(); renderAll(); sync(); };
    ta.addEventListener('blur', commit);
    ta.addEventListener('keydown', e => {
      if (e.key==='Escape') ta.remove();
      if (e.key==='Enter' && (e.ctrlKey||e.shiftKey)) { e.preventDefault(); commit(); }
    });
  }

  /* ── 等待 DOM 元素出现后绑定事件 ── */
  let _tries = 0;
  function init() {
    _tries++;
    const canvas = $('glyph-canvas');
    const btnAdd = $('btn-add-box');
    const btnClr = $('btn-clear-all');
    LOG(`init #${_tries}: canvas=${!!canvas} btnAdd=${!!btnAdd} btnClr=${!!btnClr}`);
    if (!canvas || !btnAdd || !btnClr) {
      if (_tries < 150) setTimeout(init, 200);
      else LOG('ERROR: gave up');
      return;
    }
    LOG('init OK');

    setAspect(1152, 2048);

    btnAdd.addEventListener('click', () => {
      const nb = { id:nextId++, idx:boxes.length+1, text:'新文字', align:'left', valign:'center', x:.05, y:.05, w:.4, h:.12 };
      boxes.push(nb); renderAll(); sync();
      openEditor(canvas, nb);
    });
    btnClr.addEventListener('click', () => {
      if (!confirm('确定清空所有文字框？')) return;
      boxes=[]; renderAll(); sync();
    });

    new ResizeObserver(() => setAspect(_canvasW, _canvasH)).observe(canvas.parentElement);

    /* 轮询 layout_json textbox */
    setInterval(() => {
      const inp = getInp();
      if (!inp) return;
      const v = inp.value;
      if (v && v !== lastLoaded) { LOG('poll: new layout, len=', v.length); lastLoaded=v; load(v); }
    }, 400);

    window._glyphEditor = { load, setAspect, sync };
  }

  init();
}
"""


def editor_layout(items):
    """Materialize the alignment defaults shown by the editor controls."""
    return [{**item, 'align': item.get('align', 'left'),
             'valign': item.get('valign', 'center')} for item in items]
