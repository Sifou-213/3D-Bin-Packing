// static/js/app.js: Colis workspace (un camion, plusieurs camions, une palette)
import { LoadViewer } from './viewer.js';
import { loadingOrder } from './plan.js';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

// Cardboard tones first, then the muted blue and greys of the reference plan
const PALETTE = ['#B9824A', '#4E79A6', '#D2A468', '#8C5B2E', '#7E95AE', '#C79A5B', '#A0703F', '#5D6F82', '#DDBB86', '#966A43', '#8FA3B8', '#B08A62'];
const PALLETS = [
    { key: 'EUR', label: 'Europe 120 × 80', length: 1.2, width: 0.8, empty_height: 0.144, max_load_height: 1.65, max_weight: 1000 },
    { key: 'ISO', label: 'Standard 120 × 100', length: 1.2, width: 1.0, empty_height: 0.15, max_load_height: 1.65, max_weight: 1200 },
    { key: 'US', label: 'US 122 × 102', length: 1.219, width: 1.016, empty_height: 0.14, max_load_height: 1.6, max_weight: 1000 },
];
const QUALITY = {
    fast: { npop: 12, max_generations: 8, nrep: 4, nmerge: 4, max_time_limit: 8 },
    normal: { npop: 20, max_generations: 15, nrep: 6, nmerge: 6, max_time_limit: 15 },
    best: { npop: 30, max_generations: 40, nrep: 8, nmerge: 8, max_time_limit: 40 },
};
const UPRIGHT = [1, 3];          // hga.py rotations that keep the height vertical
const NO_LIMIT = 100000;
const STORE_KEY = 'colis-v1';
const SYNONYMS = { frigo: 'refrigerateur', tele: 'tv', television: 'tv', clim: 'climatiseur', micro: 'micro-ondes', 'lave-linge': 'machine a laver', congel: 'congelateur' };

// ---------- State ----------
const freshState = () => ({
    vehicle: { name: 'Camion 10T', dc: 8, wc: 2.5, hc: 2.6, capT: 10 },
    maxTrucks: 5,
    pallet: { ...PALLETS[0] },
    items: [],
    quality: 'normal',
    seq: 1,
});
function loadState() {
    try { const s = JSON.parse(localStorage.getItem(STORE_KEY)); return s ? { ...freshState(), ...s } : null; } catch (e) { return null; }
}
function save() { try { localStorage.setItem(STORE_KEY, JSON.stringify(state)); } catch (e) { /* stockage indisponible */ } }

let state = loadState() || freshState();
let mode = ['truck', 'fleet', 'pallet'].includes(document.body.dataset.mode) ? document.body.dataset.mode : 'truck';
let trucks = [], catalog = null, demos = null;
let result = null, lastSolve = null, currentView = 0;

// ---------- Helpers ----------
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const num = v => { const n = parseFloat(String(v).replace(',', '.')); return Number.isFinite(n) ? n : NaN; };
const fr = (n, d = 0) => Number.isFinite(n) ? n.toLocaleString('fr-FR', { minimumFractionDigits: d, maximumFractionDigits: d }) : '–';
const cm = m => Number.isFinite(num(m)) && m !== '' ? Math.round(num(m) * 1000) / 10 : '';
const short = n => String(n).replace(/\s*\(.*?\)\s*/g, ' ').trim();
const norm = s => String(s).toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '');
const colorOf = it => PALETTE[(it.colorIdx ?? 0) % PALETTE.length];
const itemByUid = uid => state.items.find(i => i.uid === uid);
const plural = (n, one, many) => `${fr(n)} ${n > 1 ? many : one}`;

function toast(msg) {
    const t = $('#toast');
    t.textContent = msg;
    t.classList.add('show');
    clearTimeout(toast.t);
    toast.t = setTimeout(() => t.classList.remove('show'), 2400);
}

function newItem(d = {}) {
    const n = state.seq++;
    return {
        uid: `colis_${n}`, colorIdx: n - 1, name: d.name || `Colis ${n}`, qty: d.qty || 1,
        d: d.d ?? '', w: d.w ?? '', h: d.h ?? '', weight: d.weight ?? '',
        lbs: d.lbs ?? '', upright: !!d.upright, fragile: !!d.fragile,
    };
}

// ---------- Viewer ----------
const tooltip = $('#tooltip');
const viewer = new LoadViewer($('#stage'), {
    onStep: updateStepUI,
    onHover(box, e) {
        if (!box) { tooltip.classList.add('hidden'); return; }
        const it = itemByUid(box.uid);
        tooltip.innerHTML = `<b>${esc(it ? it.name : 'Colis')}</b><span class="mono">${box.step ? `N° ${box.step} · ` : ''}${fr(box.dx * 100)} × ${fr(box.dy * 100)} × ${fr(box.dz * 100)} cm${it && num(it.weight) ? ` · ${fr(num(it.weight), 1)} kg` : ''}</span>`;
        tooltip.classList.remove('hidden');
        tooltip.style.left = `${Math.min(e.clientX + 14, innerWidth - 270)}px`;
        tooltip.style.top = `${e.clientY + 14}px`;
    },
});

function binView() {
    if (mode === 'pallet') {
        const p = state.pallet;
        const deck = num(p.empty_height) || 0;
        return { L: num(p.length) || 1.2, W: num(p.width) || 0.8, H: (num(p.max_load_height) || 1.6) + deck, z0: deck, boxes: [] };
    }
    const v = state.vehicle;
    return { L: num(v.dc) || 1, W: num(v.wc) || 1, H: num(v.hc) || 1, z0: 0, boxes: [] };
}

function vehicleLabel() {
    if (mode === 'pallet') {
        const p = state.pallet;
        const name = PALLETS.find(x => x.key === p.key)?.label.split(' ')[0] || 'sur mesure';
        return `Palette ${name} · ${fr(num(p.length), 1)} × ${fr(num(p.width), 1)} m`;
    }
    const v = state.vehicle;
    const name = v.name === 'custom' ? 'Camion sur mesure' : short(v.name);
    return `${name} · ${fr(num(v.dc), 1)} × ${fr(num(v.wc), 1)} × ${fr(num(v.hc), 1)} m`;
}

/** Shows the empty truck (or pallet) whenever the setup changes and there is no result. */
function showEmptyBin() {
    result = null; lastSolve = null;
    $('#result').classList.add('hidden');
    $('#plan-tabs-row').classList.add('hidden');
    $('#plan-steps').classList.add('hidden');
    $('#plan-vehicle b').textContent = vehicleLabel();
    $('#stat-count').textContent = `0/${totals().count}`;
    $('#stat-vol').textContent = '0 %';
    $('#plan-note').textContent = state.items.length
        ? 'Tout est prêt. Appuyez sur « Ranger les colis ».'
        : 'Ajoutez des colis, puis appuyez sur « Ranger les colis ».';
    viewer.setView(binView(), { animate: false });
}

// ---------- Step 1: where ----------
function setMode(m) {
    mode = m;
    document.body.dataset.mode = m;
    $$('#mode-tabs button').forEach(b => b.classList.toggle('active', b.dataset.mode === m));
    $('#truck-box').classList.toggle('hidden', m === 'pallet');
    $('#pallet-box').classList.toggle('hidden', m !== 'pallet');
    $$('.fleet-only').forEach(el => el.classList.toggle('hidden', m !== 'fleet'));
    const url = new URL(location.href);
    url.searchParams.set('mode', m);
    url.searchParams.delete('demo');
    history.replaceState(null, '', url);
    renderTotal();
    showEmptyBin();
}

function renderTruckChips() {
    const box = $('#truck-chips');
    box.innerHTML = '';
    trucks.forEach(t => {
        const b = document.createElement('button');
        b.className = 'chip' + (state.vehicle.name === t.Nom ? ' active' : '');
        b.textContent = short(t.Nom);
        b.title = `${t.Nom} · ${t['Longueur (m)']} × ${t['Largeur (m)']} × ${t['Hauteur (m)']} m`;
        b.onclick = () => {
            state.vehicle = { name: t.Nom, dc: t['Longueur (m)'], wc: t['Largeur (m)'], hc: t['Hauteur (m)'], capT: t['Capacité de poids (T)'] };
            save(); syncTruck(); renderTruckChips(); renderTotal(); showEmptyBin();
        };
        box.appendChild(b);
    });
    const c = document.createElement('button');
    c.className = 'chip' + (state.vehicle.name === 'custom' ? ' active' : '');
    c.textContent = 'Autre taille';
    c.onclick = () => { state.vehicle = { ...state.vehicle, name: 'custom', capT: null }; save(); renderTruckChips(); $('#v-dc').focus(); };
    box.appendChild(c);
}
function syncTruck() {
    $('#v-dc').value = state.vehicle.dc; $('#v-wc').value = state.vehicle.wc; $('#v-hc').value = state.vehicle.hc;
    $('#max-trucks').value = state.maxTrucks;
}

function renderPalletChips() {
    const box = $('#pallet-chips');
    box.innerHTML = '';
    [...PALLETS, { key: 'custom', label: 'Autre taille' }].forEach(p => {
        const b = document.createElement('button');
        b.className = 'chip' + (state.pallet.key === p.key ? ' active' : '');
        b.textContent = p.label;
        b.onclick = () => {
            state.pallet = p.key === 'custom' ? { ...state.pallet, key: 'custom' } : { ...p };
            save(); syncPallet(); renderPalletChips(); renderTotal(); showEmptyBin();
        };
        box.appendChild(b);
    });
}
const PALLET_FIELDS = [['#p-length', 'length'], ['#p-width', 'width'], ['#p-maxh', 'max_load_height'], ['#p-maxw', 'max_weight']];
function syncPallet() { PALLET_FIELDS.forEach(([s, k]) => { $(s).value = state.pallet[k]; }); }

// ---------- Step 2: what ----------
function renderItems() {
    const box = $('#items');
    box.innerHTML = state.items.length ? '' : '<div class="empty">Aucun colis pour l\'instant.<br>Choisissez un produit dans la liste, ou ajoutez un autre colis.</div>';
    state.items.forEach(it => box.appendChild(itemEl(it)));
    renderTotal();
}

function itemEl(it) {
    const el = document.createElement('div');
    el.className = 'item';
    el.innerHTML = `
        <div class="item-head">
            <span class="dot" style="background:${colorOf(it)}"></span>
            <input class="item-name" data-k="name" value="${esc(it.name)}" title="${esc(it.name)}" aria-label="Nom du colis">
            <div class="stepper">
                <button type="button" data-q="-1" aria-label="Un de moins">−</button>
                <input type="number" min="1" data-k="qty" value="${it.qty}" aria-label="Nombre">
                <button type="button" data-q="1" aria-label="Un de plus">+</button>
            </div>
            <button class="icon-btn" data-del title="Retirer ce colis" aria-label="Retirer"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 6l12 12M18 6L6 18"/></svg></button>
        </div>
        <div class="item-dims">
            <label class="field"><span>Longueur</span><div class="unit" data-unit="cm"><input class="input" type="number" min="0" step="1" data-k="d" value="${cm(it.d)}"></div></label>
            <label class="field"><span>Largeur</span><div class="unit" data-unit="cm"><input class="input" type="number" min="0" step="1" data-k="w" value="${cm(it.w)}"></div></label>
            <label class="field"><span>Hauteur</span><div class="unit" data-unit="cm"><input class="input" type="number" min="0" step="1" data-k="h" value="${cm(it.h)}"></div></label>
            <label class="field"><span>Poids</span><div class="unit" data-unit="kg"><input class="input" type="number" min="0" step="0.1" data-k="weight" value="${it.weight}"></div></label>
        </div>
        <div class="item-opts">
            <label class="check" title="Le colis ne sera jamais couché"><input type="checkbox" data-k="upright" ${it.upright ? 'checked' : ''}>Toujours debout</label>
            <label class="check" title="Rien ne sera posé dessus"><input type="checkbox" data-k="fragile" ${it.fragile ? 'checked' : ''}>Fragile : rien dessus</label>
        </div>`;
    el.addEventListener('input', e => {
        const k = e.target.dataset.k;
        if (!k) return;
        if (e.target.type === 'checkbox') it[k] = e.target.checked;
        else if (k === 'qty') it.qty = Math.max(1, parseInt(e.target.value) || 1);
        else if (['d', 'w', 'h'].includes(k)) it[k] = e.target.value === '' ? '' : num(e.target.value) / 100;
        else it[k] = e.target.value;
        save(); renderTotal();
        if (result) showEmptyBin();
    });
    el.addEventListener('click', e => {
        const q = e.target.closest('[data-q]');
        if (q) {
            it.qty = Math.max(1, (parseInt(it.qty) || 1) + parseInt(q.dataset.q));
            $('[data-k="qty"]', el).value = it.qty;
            save(); renderTotal();
            if (result) showEmptyBin();
        }
        if (e.target.closest('[data-del]')) {
            state.items = state.items.filter(x => x !== it);
            save(); renderItems(); showEmptyBin();
        }
    });
    return el;
}

function addItem(d, { quiet = false } = {}) {
    const same = d.name && state.items.find(i => i.name === d.name && +i.d === +d.d && +i.w === +d.w && +i.h === +d.h);
    if (same) same.qty += d.qty || 1;
    else state.items.push(newItem(d));
    save(); renderItems(); showEmptyBin();
    if (!quiet) toast(same ? `${short(d.name)} : ${same.qty} colis` : `Ajouté : ${short(d.name)}`);
}

function capacity() {
    if (mode === 'pallet') { const p = state.pallet; return { vol: num(p.length) * num(p.width) * num(p.max_load_height), kg: num(p.max_weight) }; }
    const v = state.vehicle;
    return { vol: num(v.dc) * num(v.wc) * num(v.hc), kg: v.capT ? v.capT * 1000 : NaN };
}
function totals() {
    let count = 0, vol = 0, kg = 0;
    state.items.forEach(i => {
        const q = parseInt(i.qty) || 0;
        count += q;
        const v = num(i.d) * num(i.w) * num(i.h);
        if (Number.isFinite(v)) vol += v * q;
        if (Number.isFinite(num(i.weight))) kg += num(i.weight) * q;
    });
    return { count, vol, kg };
}

function renderTotal() {
    const t = totals(), cap = capacity();
    $('#items-count').textContent = state.items.length ? plural(t.count, 'colis', 'colis') : '';
    const box = $('#total');
    if (!state.items.length) { box.innerHTML = ''; return; }
    let html = `<div class="line label"><span>En tout</span><span><b>${plural(t.count, 'colis', 'colis')}</b> · ${fr(t.vol, 1)} m³ · ${fr(t.kg)} kg</span></div>`;
    if (cap.vol > 0) {
        if (mode === 'fleet') {
            const need = Math.max(1, Math.ceil(t.vol / cap.vol));
            html += `<div class="line label"><span>Il faudra au moins</span><b>${plural(need, 'camion', 'camions')}</b></div>`;
            if (need > state.maxTrucks) html += `<div class="warn">C'est plus que ${plural(state.maxTrucks, 'camion', 'camions')}. Augmentez le nombre de camions.</div>`;
        } else {
            const pct = t.vol / cap.vol * 100;
            const what = mode === 'pallet' ? 'La palette' : 'Le camion';
            html += `<div class="line label"><span>${what} sera rempli${mode === 'pallet' ? 'e' : ''} à environ</span><b>${fr(Math.min(pct, 999))} %</b></div>`;
            html += `<div class="gauge"><span style="width:${Math.min(100, pct)}%"></span></div>`;
            if (pct > 100) html += `<div class="warn">${mode === 'pallet' ? 'Tout ne tiendra pas sur une palette.' : 'Tout ne tiendra pas dans un seul camion. Essayez « Plusieurs camions ».'}</div>`;
        }
    }
    if (mode !== 'fleet' && Number.isFinite(cap.kg) && t.kg > cap.kg) html += `<div class="warn">Attention : ${fr(t.kg)} kg, c'est plus que le poids autorisé (${fr(cap.kg)} kg).</div>`;
    box.innerHTML = html;
}

// ---------- Step 3: pack ----------
function check() {
    const errs = [];
    if (!state.items.length) errs.push('Ajoutez au moins un colis.');
    const bin = mode === 'pallet' ? [num(state.pallet.length), num(state.pallet.width), num(state.pallet.max_load_height)] : [num(state.vehicle.dc), num(state.vehicle.wc), num(state.vehicle.hc)];
    if (!bin.every(x => x > 0)) errs.push(mode === 'pallet' ? 'Indiquez la taille de la palette.' : 'Indiquez la taille du camion.');
    state.items.forEach(i => {
        const dims = [num(i.d), num(i.w), num(i.h)];
        if (!dims.every(x => x > 0)) { errs.push(`« ${short(i.name)} » : indiquez la longueur, la largeur et la hauteur.`); return; }
        if (bin.every(x => x > 0) && !fits(dims, bin, i.upright)) errs.push(`« ${short(i.name)} » est trop grand pour ${mode === 'pallet' ? 'la palette' : 'le camion'}.`);
    });
    return errs;
}
function fits([d, w, h], [L, W, H], upright) {
    const opts = upright ? [[d, w, h], [w, d, h]] : [[d, w, h], [d, h, w], [w, d, h], [w, h, d], [h, d, w], [h, w, d]];
    return opts.some(([a, b, c]) => a <= L + 1e-9 && b <= W + 1e-9 && c <= H + 1e-9);
}

function boxesPayload() {
    const out = [];
    let id = 1;
    state.items.forEach(i => {
        const lbs = i.fragile ? 0 : (num(i.lbs) > 0 ? num(i.lbs) : NO_LIMIT);
        for (let k = 0; k < (parseInt(i.qty) || 0); k++) {
            out.push({
                id: id++, name: i.name, original_type_id: i.uid,
                db: num(i.d), wb: num(i.w), hb: num(i.h), weight: num(i.weight) || 0, lbs,
                is_stackable: !i.fragile, can_support_stacking: true,
                allowed_orientations: i.upright ? UPRIGHT : [1, 2, 3, 4, 5, 6],
            });
        }
    });
    return out;
}

async function pack() {
    const errs = check();
    if (errs.length) { $('#plan-note').textContent = errs.join(' '); toast(errs[0]); return; }

    const ga = { ...QUALITY[state.quality], max_containers: state.maxTrucks };
    const container = mode === 'pallet'
        ? { dc: num(state.pallet.length), wc: num(state.pallet.width), hc: num(state.pallet.max_load_height) }
        : { dc: num(state.vehicle.dc), wc: num(state.vehicle.wc), hc: num(state.vehicle.hc) };
    const payload = { container, boxes: boxesPayload(), ga_settings: ga };
    const endpoint = mode === 'fleet' ? '/solve_shipment_multi' : '/solve_shipment_single';

    const btn = $('#pack-btn');
    btn.disabled = true;
    $('#busy').classList.remove('hidden');
    const t0 = Date.now();
    const timer = setInterval(() => { $('#busy-time').textContent = `${Math.round((Date.now() - t0) / 1000)} s`; }, 500);
    try {
        const res = await fetch(endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
        let data;
        try { data = await res.json(); } catch (e) { throw new Error('Le serveur a répondu de façon inattendue.'); }
        if (!res.ok || data.error) throw new Error(data.error || `Erreur du serveur (${res.status}).`);
        lastSolve = { mode, raw: data, setup: payload };
        result = normalize(data);
        currentView = 0;
        showResult();
    } catch (e) {
        $('#plan-note').textContent = `Le rangement n'a pas marché : ${e.message}`;
        toast("Le rangement n'a pas marché.");
    } finally {
        clearInterval(timer);
        $('#busy').classList.add('hidden');
        btn.disabled = false;
    }
}

function normalize(data) {
    const asView = (viz, title, metrics) => {
        const deck = mode === 'pallet' ? (num(state.pallet.empty_height) || 0) : 0;
        const raw = (viz.layers || []).flatMap(l => (l.boxes || []).map(b => {
            const it = itemByUid(b.original_type_id);
            return {
                x: b.ox, y: b.oy, z: b.oz + deck, dx: b.dx, dy: b.dy, dz: b.dz, uid: b.original_type_id,
                name: it ? it.name : 'Colis', weight: it ? num(it.weight) || 0 : 0, color: it ? colorOf(it) : '#9A9A9A',
            };
        }));
        const boxes = loadingOrder(raw);
        boxes.forEach((b, i) => { b.step = i + 1; });
        const fill = metrics && metrics.container_volume ? metrics.total_packed_volume / metrics.container_volume * 100 : 0;
        return { title, L: viz.container_dims.dc, W: viz.container_dims.wc, H: viz.container_dims.hc + deck, z0: deck, boxes, fill };
    };
    const views = mode === 'fleet'
        ? (Array.isArray(data) ? data : []).map((c, i) => asView(c.visualization_data, `Camion ${i + 1}`, c.metrics))
        : [asView(data.visualization_data, mode === 'pallet' ? 'Palette' : 'Camion', data.metrics)];

    const wanted = {};
    state.items.forEach(i => { wanted[i.uid] = parseInt(i.qty) || 0; });
    const packed = {};
    views.forEach(v => v.boxes.forEach(b => { packed[b.uid] = (packed[b.uid] || 0) + 1; }));
    const left = {};
    Object.entries(wanted).forEach(([uid, q]) => { if (q - (packed[uid] || 0) > 0) left[uid] = q - (packed[uid] || 0); });
    const total = Object.values(wanted).reduce((a, b) => a + b, 0);
    const kgOf = v => v.boxes.reduce((a, b) => a + (num(itemByUid(b.uid)?.weight) || 0), 0);
    const limit = mode === 'pallet' ? num(state.pallet.max_weight) : (mode === 'truck' && state.vehicle.capT ? state.vehicle.capT * 1000 : NaN);
    return { mode, views, packed, left, total, overweight: Number.isFinite(limit) && kgOf(views[0] || { boxes: [] }) > limit, limit };
}

function showResult() {
    const r = result;
    const v = r.views[currentView];
    $('#plan-vehicle b').textContent = vehicleLabel();
    const tabs = $('#plan-tabs');
    $('#plan-tabs-row').classList.toggle('hidden', r.views.length < 2);
    tabs.innerHTML = r.views.map((x, i) => `<button class="chip ${i === currentView ? 'active' : ''}" data-view="${i}">${esc(x.title)} · ${fr(x.fill)} %</button>`).join('');
    $('#plan-steps').classList.remove('hidden');
    viewer.setView(v, { animate: true });

    const leftCount = Object.values(r.left).reduce((a, b) => a + b, 0);
    let verdict;
    if (!leftCount) {
        verdict = `<div><div class="big">Tout rentre !</div><div class="sub">${plural(r.total, 'colis rangé', 'colis rangés')}${r.views.length > 1 ? ` dans ${plural(r.views.length, 'camion', 'camions')}` : ''}.</div></div>`;
    } else {
        const tip = r.mode === 'truck' ? 'Essayez avec plusieurs camions.' : r.mode === 'fleet' ? 'Autorisez plus de camions.' : 'Essayez une palette plus grande.';
        verdict = `<div><div class="big">${plural(leftCount, 'colis ne rentre pas', 'colis ne rentrent pas')}</div><div class="sub">${tip}</div></div>`;
    }
    if (r.overweight) verdict += `<div class="sub" style="flex-basis:100%">Attention : le chargement dépasse le poids autorisé (${fr(r.limit)} kg).</div>`;
    const action = leftCount && r.mode === 'truck' ? '<button class="btn btn-primary btn-sm" data-go="fleet">Utiliser plusieurs camions →</button>'
        : leftCount && r.mode === 'fleet' ? '<button class="btn btn-primary btn-sm" data-go="more">+ 2 camions</button>' : '';
    $('#verdict').innerHTML = `${verdict}<div class="actions">${action}<button class="btn btn-sm" id="export-btn">↓ Rapport Excel</button></div>`;

    const rows = state.items.map(it => {
        const p = r.packed[it.uid] || 0, l = r.left[it.uid] || 0;
        let html = '';
        if (p) html += `<li><span class="dot" style="background:${colorOf(it)}"></span><span class="nm" title="${esc(it.name)}">${esc(it.name)}</span><span class="ct">${p}</span></li>`;
        if (l) html += `<li class="out"><span class="dot" style="background:transparent;border:1.5px solid ${colorOf(it)}"></span><span class="nm" title="${esc(it.name)}">${esc(it.name)}</span><span class="ct">${l} hors</span></li>`;
        return html;
    }).join('');
    $('#legend').innerHTML = `<span class="label">Les colis</span><ul>${rows}</ul>`;
    $('#result').classList.remove('hidden');
}
const totalPacked = () => Object.values(result.packed).reduce((a, b) => a + b, 0);

/** Keeps the step bar, the counters and the sentence under the plan in sync with the viewer. */
function updateStepUI(k, n, box, playing, highlight) {
    if (!result) return;
    const v = result.views[currentView];
    const range = $('#step-range');
    range.max = n;
    range.value = k;
    $('#step-n').textContent = k;
    $('#step-total').textContent = n;
    $('#step-play').textContent = playing ? '❚❚ Pause' : '▶ Lecture';
    $('#step-first').disabled = $('#step-prev').disabled = k === 0;
    $('#step-next').disabled = $('#step-last').disabled = k >= n;

    const loaded = v.boxes.slice(0, k).reduce((a, b) => a + b.dx * b.dy * b.dz, 0);
    const room = v.L * v.W * (v.H - v.z0);
    $('#stat-count').textContent = result.views.length > 1 ? `${k}/${n} · total ${totalPacked()}/${result.total}` : `${k}/${result.total}`;
    $('#stat-vol').textContent = `${fr(room ? loaded / room * 100 : 0)} %`;

    const note = $('#plan-note');
    const pallet = result.mode === 'pallet';
    if (playing) {
        note.textContent = pallet ? 'Les colis sont posés couche par couche, en partant du coin de la palette.'
            : 'Le chargement se construit couche par couche, du fond du camion vers la porte.';
    } else if (k === 0) {
        note.innerHTML = `${pallet ? 'Palette vide' : 'Camion vide'}. Appuyez sur <b>▶</b> (ou <span class="kbd">→</span>) pour poser le premier colis.`;
    } else if (!highlight || !box) {
        note.innerHTML = `Tout est chargé. Utilisez <b>◀</b> <b>▶</b> (ou <span class="kbd">←</span> <span class="kbd">→</span>) pour revoir le chargement colis par colis.`;
    } else {
        const z = box.z - v.z0;
        const where = `à ${fr(box.x, 2)} m du ${pallet ? 'bord' : 'fond'}, ${z > 0.001 ? `posé à ${fr(z, 2)} m de hauteur` : 'posé au sol'}`;
        note.innerHTML = `Colis <b>${k}</b> sur ${n} : <b>${esc(box.name)}</b> · ${fr(box.dx * 100)} × ${fr(box.dy * 100)} × ${fr(box.dz * 100)} cm, ${where}.`;
    }
}

// ---------- Excel ----------
async function exportExcel(btn) {
    if (!lastSolve) return;
    const label = btn.textContent;
    btn.disabled = true;
    btn.textContent = 'Préparation…';
    try {
        // One picture per truck, taken from the same view the user sees, then back to where they were.
        const step = viewer.step;
        const images = [];
        for (const v of result.views) { viewer.setView(v, { animate: false }); images.push(viewer.snapshot()); }
        viewer.setView(result.views[currentView], { step });

        const r = result;
        const packedVol = r.views.reduce((a, v) => a + v.boxes.reduce((s, b) => s + b.dx * b.dy * b.dz, 0), 0);
        const roomVol = r.views.reduce((a, v) => a + v.L * v.W * (v.H - v.z0), 0);
        const report = {
            mode: r.mode,
            vehicle: vehicleLabel(),
            quality: state.quality,
            packed: totalPacked(),
            total: r.total,
            fill: roomVol ? packedVol / roomVol * 100 : 0,
            views: r.views.map((v, i) => ({
                title: v.title, fill: v.fill, image: images[i], L: v.L, W: v.W, H: v.H - v.z0,
                boxes: v.boxes.map(b => ({ step: b.step, name: b.name, color: b.color, weight: b.weight, dx: b.dx, dy: b.dy, dz: b.dz, x: b.x, y: b.y, z: b.z - v.z0 })),
            })),
            items: state.items.map(it => ({
                name: it.name, color: colorOf(it), qty: parseInt(it.qty) || 0, packed: r.packed[it.uid] || 0, left: r.left[it.uid] || 0,
                d: num(it.d), w: num(it.w), h: num(it.h), weight: num(it.weight) || 0, upright: it.upright, fragile: it.fragile,
            })),
        };
        const res = await fetch('/export_report', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(report),
        });
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'erreur');
        toast('Le fichier se télécharge…');
        if (data.download_url) location.href = data.download_url;
    } catch (e) {
        toast(`Le téléchargement n'a pas marché : ${e.message}`);
    } finally {
        btn.disabled = false;
        btn.textContent = label;
    }
}

// ---------- Catalog ----------
let catType = '';
async function openCatalog() {
    $('#catalog').showModal();
    $('#catalog-search').value = '';
    if (!catalog) {
        $('#catalog-list').innerHTML = '<p class="muted" style="padding:12px">Chargement…</p>';
        try {
            catalog = await (await fetch('/get_items_data')).json();
            if (!Array.isArray(catalog)) throw new Error();
        } catch (e) {
            catalog = null;
            $('#catalog-list').innerHTML = '<p class="muted" style="padding:12px">Impossible de charger la liste.</p>';
            return;
        }
        const types = [...new Set(catalog.map(c => c.Type))].sort((a, b) => a.localeCompare(b, 'fr'));
        $('#catalog-types').innerHTML = ['', ...types].map(t => `<button class="chip ${t ? '' : 'active'}" data-type="${esc(t)}">${t ? esc(t) : 'Tout'}</button>`).join('');
    }
    renderCatalog();
    setTimeout(() => $('#catalog-search').focus(), 50);
}
function renderCatalog() {
    if (!catalog) return;
    let q = norm($('#catalog-search').value.trim());
    Object.entries(SYNONYMS).forEach(([k, v]) => { if (q === k || q.startsWith(k + ' ')) q = q.replace(k, v); });
    const rows = catalog.filter(c => (!catType || c.Type === catType) && (!q || norm(c.Product).includes(q)));
    $('#catalog-list').innerHTML = rows.length ? rows.slice(0, 250).map(c => `
        <div class="cat" data-i="${catalog.indexOf(c)}">
            <div class="txt"><div class="nm" title="${esc(c.Product)}">${esc(c.Product)}</div>
            <div class="meta">${fr(c['Depth (m)'] * 100)} × ${fr(c['Width (m)'] * 100)} × ${fr(c['Height (m)'] * 100)} cm · ${fr(c['Weight (kg)'], 1)} kg</div></div>
            <button class="btn btn-sm">+ Ajouter</button>
        </div>`).join('') : '<p class="muted" style="padding:12px">Aucun produit trouvé.</p>';
}
function fromCatalog(c) {
    const allowed = Array.isArray(c['Allowed Orientations']) ? c['Allowed Orientations'] : [1, 1, 1, 1, 1, 1];
    const lbs = num(c['LBS (kg)']);
    return {
        name: c.Product, qty: 1, d: c['Depth (m)'], w: c['Width (m)'], h: c['Height (m)'], weight: c['Weight (kg)'],
        lbs: lbs > 0 ? lbs : '', upright: allowed.filter(Boolean).length < 6, fragile: !(lbs > 0),
    };
}

// ---------- Demo / reset ----------
async function loadDemo() {
    try { if (!demos) demos = await (await fetch('/static/Data/demo_shipments.json')).json(); } catch (e) { toast("Impossible de charger l'exemple"); return; }
    const d = demos[mode];
    state.items = []; state.seq = 1;
    d.items.forEach(i => state.items.push(newItem({ ...i, fragile: !(i.lbs > 0) })));
    if (d.vehicle) {
        const t = trucks.find(x => x.Nom === d.vehicle);
        if (t) state.vehicle = { name: t.Nom, dc: t['Longueur (m)'], wc: t['Largeur (m)'], hc: t['Hauteur (m)'], capT: t['Capacité de poids (T)'] };
    }
    if (d.maxTrucks) state.maxTrucks = d.maxTrucks;
    if (d.pallet) state.pallet = { ...PALLETS.find(p => p.key === d.pallet) };
    save(); syncAll(); showEmptyBin();
    toast('Exemple chargé. Appuyez sur « Ranger les colis ».');
}
function syncAll() {
    syncTruck(); renderTruckChips(); syncPallet(); renderPalletChips(); renderItems();
    $$('#quality-chips .chip').forEach(c => c.classList.toggle('active', c.dataset.q === state.quality));
}

// ---------- Wiring ----------
$$('#mode-tabs button').forEach(b => b.addEventListener('click', () => { if (b.dataset.mode !== mode) setMode(b.dataset.mode); }));
[['#v-dc', 'dc'], ['#v-wc', 'wc'], ['#v-hc', 'hc']].forEach(([s, k]) => $(s).addEventListener('input', e => {
    state.vehicle[k] = e.target.value;
    if (state.vehicle.name !== 'custom') { state.vehicle.name = 'custom'; state.vehicle.capT = null; renderTruckChips(); }
    save(); renderTotal(); showEmptyBin();
}));
PALLET_FIELDS.forEach(([s, k]) => $(s).addEventListener('input', e => {
    state.pallet[k] = e.target.value;
    if (state.pallet.key !== 'custom') { state.pallet.key = 'custom'; renderPalletChips(); }
    save(); renderTotal(); showEmptyBin();
}));
$('#max-trucks').addEventListener('input', e => { state.maxTrucks = Math.min(20, Math.max(1, parseInt(e.target.value) || 1)); save(); renderTotal(); });
$$('[data-step]').forEach(b => b.addEventListener('click', () => {
    const i = $('#max-trucks');
    i.value = Math.min(20, Math.max(1, (parseInt(i.value) || 1) + parseInt(b.dataset.step)));
    i.dispatchEvent(new Event('input'));
}));
$('#catalog-btn').addEventListener('click', openCatalog);
$('#add-item-btn').addEventListener('click', () => {
    addItem({}, { quiet: true });
    const last = $('#items').lastElementChild;
    if (last) { last.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); $('[data-k="d"]', last).focus(); }
});
$('#catalog-close').addEventListener('click', () => $('#catalog').close());
$('#catalog').addEventListener('click', e => { if (e.target === $('#catalog')) $('#catalog').close(); });
$('#catalog-search').addEventListener('input', renderCatalog);
$('#catalog-types').addEventListener('click', e => {
    const c = e.target.closest('[data-type]');
    if (!c) return;
    catType = c.dataset.type;
    $$('#catalog-types .chip').forEach(x => x.classList.toggle('active', x === c));
    renderCatalog();
});
$('#catalog-list').addEventListener('click', e => {
    const row = e.target.closest('.cat');
    if (!row || !e.target.closest('button')) return;
    addItem(fromCatalog(catalog[+row.dataset.i]));
    const b = row.querySelector('button');
    b.textContent = 'Ajouté ✓';
    b.classList.add('btn-primary');
    setTimeout(() => { b.textContent = '+ Ajouter'; b.classList.remove('btn-primary'); }, 1200);
});
$$('#quality-chips .chip').forEach(c => c.addEventListener('click', () => {
    state.quality = c.dataset.q; save();
    $$('#quality-chips .chip').forEach(x => x.classList.toggle('active', x === c));
}));
$('#pack-btn').addEventListener('click', pack);
$('#demo-btn').addEventListener('click', loadDemo);
$('#reset-btn').addEventListener('click', () => {
    if (state.items.length && !confirm('Tout effacer et recommencer ?')) return;
    state = freshState(); save(); syncAll(); showEmptyBin();
});
$('#replay-btn').addEventListener('click', () => viewer.replay());
$('#step-first').addEventListener('click', () => viewer.first());
$('#step-prev').addEventListener('click', () => viewer.prev());
$('#step-next').addEventListener('click', () => viewer.next());
$('#step-last').addEventListener('click', () => viewer.last());
$('#step-play').addEventListener('click', () => viewer.toggle());
$('#step-range').addEventListener('input', e => viewer.goTo(+e.target.value));
document.addEventListener('keydown', e => {
    if (!result || e.altKey || e.ctrlKey || e.metaKey) return;
    const target = e.target instanceof Element ? e.target : document.body;
    if (target.closest('input, textarea, select, dialog') || $('#catalog').open) return;
    const actions = { ArrowRight: () => viewer.next(), ArrowLeft: () => viewer.prev(), Home: () => viewer.first(), End: () => viewer.last(), ' ': () => viewer.toggle() };
    if (!actions[e.key] || (e.key === ' ' && target.closest('button'))) return;
    e.preventDefault();
    actions[e.key]();
});
$('#recenter-btn').addEventListener('click', () => viewer.recenter());
$('#plan-tabs').addEventListener('click', e => {
    const b = e.target.closest('[data-view]');
    if (!b || !result) return;
    currentView = +b.dataset.view;
    showResult();
});
$('#verdict').addEventListener('click', e => {
    if (e.target.id === 'export-btn') { exportExcel(e.target); return; }
    const go = e.target.closest('[data-go]');
    if (!go) return;
    if (go.dataset.go === 'fleet') { setMode('fleet'); pack(); }
    if (go.dataset.go === 'more') { state.maxTrucks = Math.min(20, state.maxTrucks + 2); save(); syncTruck(); renderTotal(); pack(); }
});

// ---------- Start ----------
const wantsDemo = new URLSearchParams(location.search).has('demo');
syncAll();
setMode(mode);
try { trucks = await (await fetch('/get_truck_data')).json(); if (!Array.isArray(trucks)) trucks = []; } catch (e) { trucks = []; }
renderTruckChips();
if (wantsDemo) loadDemo();
