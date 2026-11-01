// static/js/landing.js: plays the ready-made demo load plan on the home page, on a loop
import { LoadViewer } from './viewer.js';
import { loadingOrder } from './plan.js';

const $ = sel => document.querySelector(sel);
const fr = n => n.toLocaleString('fr-FR', { maximumFractionDigits: 0 });
const LOOP_PAUSE_MS = 4500;

let view = null;
let loopTimer = null;

const viewer = new LoadViewer($('#demo-stage'), {
    onStep(k, n, box, playing) {
        if (!view) return;
        const loaded = view.boxes.slice(0, k).reduce((a, b) => a + b.dx * b.dy * b.dz, 0);
        $('#demo-count').textContent = `${k}/${n}`;
        $('#demo-vol').textContent = `${fr(loaded / (view.L * view.W * view.H) * 100)} %`;
        clearTimeout(loopTimer);
        if (k === n && !playing) loopTimer = setTimeout(() => viewer.replay(), LOOP_PAUSE_MS);
    },
});

$('#demo-replay').addEventListener('click', () => viewer.replay());

try {
    const plan = await (await fetch('/static/Data/demo_plan.json')).json();
    $('#demo-vehicle').textContent = plan.vehicle;
    const boxes = loadingOrder(plan.boxes.map(b => ({ ...b, name: plan.kinds[b.k].name, color: plan.kinds[b.k].color })));
    boxes.forEach((b, i) => { b.step = i + 1; });
    view = { L: plan.L, W: plan.W, H: plan.H, z0: 0, boxes };
    viewer.setView(view, { animate: true });
} catch (e) {
    $('#demo-vehicle').textContent = '';
}
