// static/js/plan.js: helpers shared by the planner and the home page

/**
 * Orders boxes the way they would be loaded: a box comes only after every box it rests on,
 * and among the boxes that can go next, the one closest to the back of the truck goes first
 * (then lowest, then closest to the side wall).
 */
export function loadingOrder(boxes) {
    const n = boxes.length;
    const overlap = (a0, a1, b0, b1) => a0 < b1 - 1e-6 && b0 < a1 - 1e-6;
    const waitingOn = new Array(n).fill(0);
    const holds = boxes.map(() => []);
    for (let i = 0; i < n; i++) {
        const b = boxes[i];
        for (let j = 0; j < n; j++) {
            const o = boxes[j];
            if (i !== j && Math.abs(o.z + o.dz - b.z) < 1e-4 && overlap(o.x, o.x + o.dx, b.x, b.x + b.dx) && overlap(o.y, o.y + o.dy, b.y, b.y + b.dy)) {
                waitingOn[i]++;
                holds[j].push(i);
            }
        }
    }
    const key = b => [Math.round(b.x * 100), Math.round(b.z * 100), Math.round(b.y * 100)];
    const before = (a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2];
    const placed = new Array(n).fill(false);
    const order = [];
    for (let s = 0; s < n; s++) {
        let best = -1, bestKey = null;
        for (let i = 0; i < n; i++) {
            if (placed[i] || waitingOn[i] > 0) continue;
            const k = key(boxes[i]);
            if (best < 0 || before(k, bestKey) < 0) { best = i; bestKey = k; }
        }
        if (best < 0) best = placed.indexOf(false); // safety net, should not happen
        placed[best] = true;
        order.push(boxes[best]);
        holds[best].forEach(i => { waitingOn[i]--; });
    }
    return order;
}
