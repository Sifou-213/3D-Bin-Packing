// static/js/viewer.js: interactive 3D load plan (cardboard boxes in a wireframe truck)
// Boxes are shown in the order given by view.boxes (the loading order). The plan can be
// played like a film or stepped through one box at a time.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const BG = 0x0d0f12;
const LINE = 0x6b7480;
const GRID = 0x2c323a;
const DROP_MS = 320;

const prefersReducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const easeOut = t => 1 - Math.pow(1 - t, 3);

function lighten(hex, amt) {
    const c = new THREE.Color(hex);
    c.lerp(new THREE.Color(0xffffff), amt);
    return c;
}

export class LoadViewer {
    constructor(host, { onHover, onStep } = {}) {
        this.host = host;
        this.onHover = onHover;
        this.onStep = onStep;
        this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        this.renderer.setClearColor(BG);
        host.appendChild(this.renderer.domElement);

        this.scene = new THREE.Scene();
        this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, -1000, 1000);
        this.scene.add(new THREE.AmbientLight(0xffffff, 0.62));
        const sun = new THREE.DirectionalLight(0xffffff, 0.9);
        sun.position.set(0.25, 1, 0.55);
        this.scene.add(sun);

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.enablePan = false;
        this.controls.minZoom = 0.6;
        this.controls.maxZoom = 5;
        this.controls.maxPolarAngle = Math.PI / 2 - 0.02;

        this.root = new THREE.Group();
        this.scene.add(this.root);
        this.items = [];
        this.materials = new Map();
        this.step = 0;
        this.playing = false;
        this.hovered = null;
        this.raycaster = new THREE.Raycaster();
        this.pointer = new THREE.Vector2();

        new ResizeObserver(() => this.resize()).observe(host);
        this.renderer.domElement.addEventListener('pointermove', e => this.handlePointer(e));
        this.renderer.domElement.addEventListener('pointerleave', () => this.setHovered(null, null));
        this.resize();
        this.loop();
    }

    // ---------- Scene building ----------
    clear() {
        this.root.traverse(o => { if (o.geometry) o.geometry.dispose(); });
        this.root.clear();
        this.materials.forEach(m => m.dispose());
        this.materials.clear();
        this.items = [];
        this.hovered = null;
    }

    material(key, make) {
        if (!this.materials.has(key)) this.materials.set(key, make());
        return this.materials.get(key);
    }

    setView(view, { animate = true, step = null } = {}) {
        this.pause();
        this.clear();
        this.view = view;
        this.addContainer(view);
        view.boxes.forEach(b => {
            const group = this.makeBox(b);
            group.userData.target = group.position.y;
            this.root.add(group);
            this.items.push(group);
        });
        this.frame(view.L, view.W, view.H);
        if (step !== null) this.showStep(step, { highlight: step < this.items.length });
        else if (animate && !prefersReducedMotion() && this.items.length) { this.showStep(0); this.play(); }
        else this.showStep(this.items.length);
    }

    addContainer({ L, W, H, z0 = 0 }) {
        const solid = this.material('line', () => new THREE.LineBasicMaterial({ color: LINE, transparent: true, opacity: 0.9 }));
        const dashed = this.material('dash', () => new THREE.LineDashedMaterial({ color: LINE, dashSize: 0.12, gapSize: 0.09, transparent: true, opacity: 0.8 }));
        const grid = this.material('grid', () => new THREE.LineDashedMaterial({ color: GRID, dashSize: 0.08, gapSize: 0.08 }));
        const seg = (pts, mat) => {
            const g = new THREE.BufferGeometry().setFromPoints(pts.map(p => new THREE.Vector3(...p)));
            const l = new THREE.LineSegments(g, mat);
            l.computeLineDistances();
            this.root.add(l);
        };
        // Three.js axes: x = length (cab at 0, door at L), y = height, z = width.
        seg([[0, 0, 0], [L, 0, 0], [0, 0, 0], [0, 0, W], [0, 0, 0], [0, H, 0], [0, H, 0], [L, H, 0], [0, H, 0], [0, H, W],
             [0, 0, W], [0, H, W], [0, H, W], [L, H, W], [L, 0, 0], [L, H, 0], [L, H, 0], [L, H, W]], solid);
        seg([[0, 0, W], [L, 0, W], [L, 0, 0], [L, 0, W], [L, 0, W], [L, H, W]], dashed);

        const step = L > 3 ? 1 : 0.2;
        const pts = [];
        for (let x = step; x < L - 1e-6; x += step) pts.push([x, 0.001, 0], [x, 0.001, W]);
        for (let z = step; z < W - 1e-6; z += step) pts.push([0, 0.001, z], [L, 0.001, z]);
        if (pts.length) seg(pts, grid);

        if (z0 > 0) {
            // Pallet deck: three runners under a top board
            const wood = this.material('wood', () => new THREE.MeshLambertMaterial({ color: 0xb48c5a }));
            const top = new THREE.Mesh(new THREE.BoxGeometry(L, z0 * 0.35, W), wood);
            top.position.set(L / 2, z0 - z0 * 0.175, W / 2);
            this.root.add(top);
            [0.08, 0.5, 0.92].forEach(f => {
                const r = new THREE.Mesh(new THREE.BoxGeometry(L, z0 * 0.65, W * 0.12), wood);
                r.position.set(L / 2, z0 * 0.325, W * f);
                this.root.add(r);
            });
        }
    }

    makeBox(b) {
        const group = new THREE.Group();
        const inset = 0.006;
        const sx = Math.max(b.dx - inset, 0.001), sy = Math.max(b.dz - inset, 0.001), sz = Math.max(b.dy - inset, 0.001);
        const body = this.material(`box:${b.color}`, () => new THREE.MeshLambertMaterial({ color: b.color }));
        const mesh = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz), body);
        mesh.userData.group = group;
        group.add(mesh);

        const edgeMat = this.material('edge', () => new THREE.LineBasicMaterial({ color: 0x1b1206, transparent: true, opacity: 0.75 }));
        group.add(new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry), edgeMat));

        // Packing tape along the longer side of the lid, folding down over both ends
        const tapeMat = this.material(`tape:${b.color}`, () => new THREE.MeshLambertMaterial({
            color: lighten(b.color, 0.35), polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
        }));
        const alongX = sx >= sz;
        const long = alongX ? sx : sz, short = alongX ? sz : sx;
        const tw = Math.min(0.09, short * 0.2);
        const flap = Math.min(sy * 0.28, 0.18);
        const top = new THREE.Mesh(new THREE.PlaneGeometry(alongX ? long : tw, alongX ? tw : long), tapeMat);
        top.rotation.x = -Math.PI / 2;
        top.position.y = sy / 2 + 0.001;
        group.add(top);
        [-1, 1].forEach(sign => {
            const end = new THREE.Mesh(new THREE.PlaneGeometry(tw, flap), tapeMat);
            end.position.y = sy / 2 - flap / 2;
            if (alongX) { end.rotation.y = sign * Math.PI / 2; end.position.x = sign * (sx / 2 + 0.001); }
            else { end.rotation.y = sign > 0 ? 0 : Math.PI; end.position.z = sign * (sz / 2 + 0.001); }
            group.add(end);
        });

        // White outline marking the box of the current step
        const outlineMat = this.material('outline', () => new THREE.LineBasicMaterial({ color: 0xffffff }));
        const outline = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(sx + 0.02, sy + 0.02, sz + 0.02)), outlineMat);
        outline.visible = false;
        group.add(outline);
        group.userData.outline = outline;

        group.position.set(b.x + b.dx / 2, b.z + b.dz / 2, b.y + b.dy / 2);
        group.userData.box = b;
        return group;
    }

    frame(L, W, H) {
        const center = new THREE.Vector3(L / 2, H / 2, W / 2);
        const dir = new THREE.Vector3(0.95, 0.95, 1.35).normalize();
        const r = 0.5 * Math.sqrt(L * L + W * W + H * H);
        this.camera.position.copy(center).addScaledVector(dir, r * 4);
        this.camera.zoom = 1;
        this.controls.target.copy(center);
        this.camera.lookAt(center);
        this.controls.update();
        this.resize();
    }

    recenter() { if (this.view) this.frame(this.view.L, this.view.W, this.view.H); }

    // ---------- Steps ----------
    get total() { return this.items.length; }

    /** Shows the first k boxes. `drop` animates the newest one; `highlight` outlines it. */
    showStep(k, { drop = false, highlight = true } = {}) {
        const n = this.items.length;
        k = Math.max(0, Math.min(n, Math.round(k)));
        const now = performance.now();
        this.items.forEach((g, i) => {
            const visible = i < k;
            if (visible && i === k - 1 && drop && !prefersReducedMotion()) g.userData.dropStart = now;
            else if (!visible || !drop) { g.userData.dropStart = null; g.position.y = g.userData.target; }
            g.visible = visible;
            g.userData.outline.visible = highlight && i === k - 1;
        });
        this.step = k;
        if (this.onStep) this.onStep(k, n, k ? this.items[k - 1].userData.box : null, this.playing, highlight);
    }

    next() { this.pause(); this.showStep(this.step + 1, { drop: true }); }
    prev() { this.pause(); this.showStep(this.step - 1); }
    first() { this.pause(); this.showStep(0); }
    last() { this.pause(); this.showStep(this.items.length, { highlight: false }); }
    goTo(k) { this.pause(); this.showStep(k); }

    play() {
        if (!this.items.length) return;
        if (this.step >= this.items.length) this.showStep(0);
        this.playing = true;
        this.interval = Math.max(45, Math.min(260, 2800 / this.items.length));
        this.nextAt = performance.now();
        if (this.onStep) this.onStep(this.step, this.items.length, null, true, false);
    }
    pause() {
        if (!this.playing) return;
        this.playing = false;
        this.showStep(this.step);
    }
    toggle() { this.playing ? this.pause() : this.play(); }
    replay() { this.showStep(0); this.play(); }

    tick(now) {
        if (this.playing && now >= this.nextAt) {
            if (this.step < this.items.length) {
                this.showStep(this.step + 1, { drop: true });
                this.nextAt = now + this.interval;
            }
            if (this.step >= this.items.length) {
                this.playing = false;
                this.showStep(this.items.length, { drop: true, highlight: false });
            }
        }
        const drop = (this.view?.H || 2) * 0.35;
        this.items.forEach(g => {
            const s = g.userData.dropStart;
            if (s == null) return;
            const t = Math.min(1, (now - s) / DROP_MS);
            g.position.y = g.userData.target + drop * (1 - easeOut(t));
            if (t >= 1) g.userData.dropStart = null;
        });
    }

    // ---------- Interaction ----------
    handlePointer(e) {
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.pointer.set(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
        this.raycaster.setFromCamera(this.pointer, this.camera);
        const meshes = this.items.filter(g => g.visible).map(g => g.children[0]);
        const hit = this.raycaster.intersectObjects(meshes, false)[0];
        this.setHovered(hit ? hit.object.userData.group : null, e);
    }

    setHovered(group, e) {
        if (this.hovered && this.hovered !== group) this.hovered.children[0].material = this.hovered.userData.baseMat;
        if (group && this.hovered !== group) {
            const mesh = group.children[0];
            group.userData.baseMat = mesh.material;
            mesh.material = this.material(`hi:${group.userData.box.color}`, () => new THREE.MeshLambertMaterial({ color: lighten(group.userData.box.color, 0.3) }));
        }
        this.hovered = group;
        this.renderer.domElement.style.cursor = group ? 'pointer' : 'grab';
        if (this.onHover) this.onHover(group ? group.userData.box : null, e);
    }

    // ---------- Rendering ----------
    resize() {
        const w = this.host.clientWidth || 1, h = this.host.clientHeight || 1;
        this.renderer.setSize(w, h, false);
        const aspect = w / h;
        // Fit the container's corners, as seen from the current camera direction, into the frame.
        let halfX = 1, halfY = 1;
        if (this.view) {
            const { L, W, H } = this.view;
            this.camera.updateMatrixWorld();
            const inv = this.camera.matrixWorldInverse;
            const c = this.controls.target.clone().applyMatrix4(inv);
            halfX = 0; halfY = 0;
            for (const x of [0, L]) for (const y of [0, H]) for (const z of [0, W]) {
                const p = new THREE.Vector3(x, y, z).applyMatrix4(inv);
                halfX = Math.max(halfX, Math.abs(p.x - c.x));
                halfY = Math.max(halfY, Math.abs(p.y - c.y));
            }
        }
        const halfH = Math.max(halfY, halfX / aspect) * 1.1;
        this.camera.left = -halfH * aspect;
        this.camera.right = halfH * aspect;
        this.camera.top = halfH;
        this.camera.bottom = -halfH;
        this.camera.updateProjectionMatrix();
    }

    loop() {
        const frame = now => {
            this.tick(now);
            this.controls.update();
            this.renderer.render(this.scene, this.camera);
            requestAnimationFrame(frame);
        };
        requestAnimationFrame(frame);
    }

    /** Renders the complete plan to a PNG data URL, then restores the current step. */
    snapshot() {
        const k = this.step, outlined = this.items.map(g => g.userData.outline.visible);
        this.items.forEach(g => { g.visible = true; g.position.y = g.userData.target; g.userData.outline.visible = false; });
        this.renderer.render(this.scene, this.camera);
        const url = this.renderer.domElement.toDataURL('image/png');
        this.items.forEach((g, i) => { g.visible = i < k; g.userData.outline.visible = outlined[i]; });
        return url;
    }
}
