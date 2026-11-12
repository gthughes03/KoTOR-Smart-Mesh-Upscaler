#!/usr/bin/env python3
"""
KotOR MDL/MDX Mesh Upscaler  (KotOR 1 & 2, PC binary format)

* Own native MDL/MDX reader + writer (no PyKotor needed; PyKotor cannot round-trip vertex data).
* Real 3D viewer window (software rasteriser, orbit / zoom / pan, wireframe, original-vs-upscaled).
* Feature-aware upscaling: crease / hard-edge detection, smooth-group normals, crack-free
  curved PN-triangle tessellation, UV / normal / skin-weight interpolation.
* Writes by appending new data and repointing the mesh headers, so animations, controllers,
  bones, dangly/saber/walkmesh nodes and every other pointer in the file stay valid.

* v3.1: the viewer draws both sides of every face (rooms no longer lose walls seen from outside) and shows
  meshes it can't edit yet in grey.
* v3.2: a show/hide toggle per mesh in the mesh list (click its circle), plus Show all / Hide all.

Requires: python 3.8+, numpy, Pillow.   (tkinter ships with python)
"""
import os, sys, struct, math, shutil, re
import numpy as np

# ----------------------------------------------------------------------------- binary model
NODE_MESH = 0x20
NODE_SKIN = 0x40
MDX_VERT, MDX_UV1, MDX_NORM = 0x1, 0x2, 0x20

# offsets inside the trimesh header (header starts at node+80)
H_FACES, H_FCOUNT, H_FCOUNT2 = 8, 12, 16
H_BBOX = 20
H_TEX0 = 88
H_IDXCNT_ARR = 176
H_IDXOFF_ARR = 188
H_STRIDE, H_FLAGS = 252, 256
H_OFF_VERT, H_OFF_NORM, H_OFF_UV1 = 260, 264, 272
H_VCOUNT = 304
H_MDXOFF = 332
H_MDLVERT = 336
SKIN_W_OFF, SKIN_I_OFF = 340 + 12, 340 + 16        # MDX offsets of weights / bone indices


class Mesh:
    pass


class Model:
    def __init__(self, mdl_path):
        self.mdl_path = mdl_path
        self.mdx_path = os.path.splitext(mdl_path)[0] + ".mdx"
        raw = open(mdl_path, "rb").read()
        if len(raw) < 100 or struct.unpack_from("<I", raw, 0)[0] != 0:
            raise ValueError("Not a binary KotOR MDL (ASCII models are not supported)")
        self.D = bytearray(raw[12:])
        self.X = bytearray(open(self.mdx_path, "rb").read()) if os.path.exists(self.mdx_path) else None
        if self.X is None:
            raise ValueError("Matching .mdx file not found next to the .mdl")
        D = self.D
        self.name = bytes(D[8:40]).split(b"\0")[0].decode("latin1")
        root = self.u32(40)
        nnames = self.u32(192)
        noff = 196
        self.names = []
        for i in range(nnames):
            o = self.u32(noff + 4 * i)
            self.names.append(bytes(D[o:D.index(b"\0", o)]).decode("latin1"))
        self.meshes = []
        self._walk(root, None, np.eye(4))

    # -- helpers
    def u32(self, o): return struct.unpack_from("<I", self.D, o)[0]
    def u16(self, o): return struct.unpack_from("<H", self.D, o)[0]
    def f32(self, o): return struct.unpack_from("<f", self.D, o)[0]

    def _walk(self, o, parent, pm):
        D = self.D
        t = self.u16(o)
        nm = self.u16(o + 4)
        pos = struct.unpack_from("<3f", D, o + 16)
        w, x, y, z = struct.unpack_from("<4f", D, o + 28)
        n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
        w, x, y, z = w / n, x / n, y / n, z / n
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        M = np.eye(4); M[:3, :3] = R; M[:3, 3] = pos
        W = pm @ M
        if t & NODE_MESH:
            m = self._read_mesh(o, t, self.names[nm] if nm < len(self.names) else "node%d" % nm, W)
            if m is not None:
                self.meshes.append(m)
        co, cn = self.u32(o + 44), self.u32(o + 48)
        for i in range(cn):
            self._walk(self.u32(co + 4 * i), o, W)

    def _read_mesh(self, o, t, name, W):
        D, X = self.D, self.X
        h = o + 80
        m = Mesh()
        m.node, m.type, m.name, m.h, m.world = o, t, name, h, W
        m.skin = bool(t & NODE_SKIN)
        m.render = bool(D[h + 313])
        m.vc = self.u16(h + H_VCOUNT)
        m.fc = self.u32(h + H_FCOUNT)
        m.stride = self.u32(h + H_STRIDE)
        m.flags = self.u32(h + H_FLAGS)
        m.mdxoff = self.u32(h + H_MDXOFF)
        m.tex = bytes(D[h + H_TEX0:h + H_TEX0 + 32]).split(b"\0")[0].decode("latin1")
        m.ok = True
        m.why = ""
        if m.vc == 0 or m.fc == 0 or m.stride == 0 or not (m.flags & MDX_VERT):
            m.ok, m.why = False, "no renderable geometry"
            return m
        if m.mdxoff + m.vc * m.stride > len(X):
            m.ok, m.why = False, "mdx range"
            return m
        raw = np.frombuffer(bytes(X[m.mdxoff:m.mdxoff + m.vc * m.stride]), dtype=np.uint8).reshape(m.vc, m.stride)
        fo = self.u32(h + H_FACES)
        fb = np.frombuffer(bytes(D[fo:fo + 32 * m.fc]), dtype=np.uint8).reshape(m.fc, 32)
        m.faces = fb[:, 26:32].copy().view("<u2").reshape(m.fc, 3).astype(np.int64)
        m.surf = fb[:, 16:20].copy().view("<u4").reshape(m.fc)
        def fl(off, n): return raw[:, off:off + 4 * n].copy().view("<f4").reshape(m.vc, n).astype(np.float64)
        m.pos = fl(self.u32(h + H_OFF_VERT), 3)
        m.norm = fl(self.u32(h + H_OFF_NORM), 3) if (m.flags & MDX_NORM) and self.u32(h + H_OFF_NORM) != 0xFFFFFFFF else None
        m.uv = fl(self.u32(h + H_OFF_UV1), 2) if (m.flags & MDX_UV1) and self.u32(h + H_OFF_UV1) != 0xFFFFFFFF else None
        if m.skin:
            wo, io = self.u32(h + SKIN_W_OFF), self.u32(h + SKIN_I_OFF)
            m.sw = fl(wo, 4)
            m.si = fl(io, 4)
            m.skin_offs = (wo, io)
        else:
            m.sw = m.si = None
        if m.faces.max() >= m.vc:
            m.ok, m.why = False, "bad indices"
        known = MDX_VERT | MDX_UV1 | MDX_NORM
        if m.flags & ~known:
            m.ok, m.why = False, "extra vertex channels (tangents/colours/uv2) not supported yet"
        return m

    # world-space geometry for the viewer
    def world_pos(self, m, pos=None):
        p = m.pos if pos is None else pos
        return p @ m.world[:3, :3].T + m.world[:3, 3]

    # ------------------------------------------------------------------ writing
    def apply(self, m, new):
        """Append the upscaled mesh `new` (dict) and repoint header of mesh m."""
        D, X = self.D, self.X
        h = m.h
        vc, nf = len(new["pos"]), len(new["faces"])
        if vc > 65535:
            raise ValueError("%s: %d vertices exceeds 16-bit index limit" % (m.name, vc))
        stride = m.stride
        rows = np.zeros((vc, stride), dtype=np.uint8)
        def put(off, arr):
            a = np.ascontiguousarray(arr.astype("<f4"))
            rows[:, off:off + 4 * a.shape[1]] = a.view(np.uint8).reshape(vc, -1)
        put(self.u32(h + H_OFF_VERT), new["pos"])
        if m.norm is not None: put(self.u32(h + H_OFF_NORM), new["norm"])
        if m.uv is not None: put(self.u32(h + H_OFF_UV1), new["uv"])
        if m.skin:
            put(m.skin_offs[0], new["sw"]); put(m.skin_offs[1], new["si"])
        term = np.zeros((1, stride), dtype=np.uint8)
        tv = np.zeros(stride // 4, dtype="<f4"); tv[:3] = 1e7
        if stride >= 24: tv[3:6] = 0
        term[0, :] = tv.view(np.uint8)
        # mdx
        mdxoff = len(X)
        X += rows.tobytes() + term.tobytes()
        # mdl: faces
        while len(D) % 4: D += b"\0"
        faces_off = len(D)
        F = new["faces"]
        P = new["pos"].astype(np.float32)
        a, b, c = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
        fn = np.cross(b - a, c - a)
        ln = np.linalg.norm(fn, axis=1, keepdims=True); ln[ln == 0] = 1
        fn = fn / ln
        dist = -(fn * a).sum(1)
        adj = adjacency(F)
        fb = np.zeros((nf, 32), dtype=np.uint8)
        fb[:, 0:12] = np.ascontiguousarray(fn.astype("<f4")).view(np.uint8).reshape(nf, 12)
        fb[:, 12:16] = np.ascontiguousarray(dist.astype("<f4")).view(np.uint8).reshape(nf, 4)
        fb[:, 16:20] = np.ascontiguousarray(new["surf"].astype("<u4")).view(np.uint8).reshape(nf, 4)
        fb[:, 20:26] = np.ascontiguousarray(adj.astype("<u2")).view(np.uint8).reshape(nf, 6)
        fb[:, 26:32] = np.ascontiguousarray(F.astype("<u2")).view(np.uint8).reshape(nf, 6)
        D += fb.tobytes()
        # index count + index buffer
        cnt_off = len(D); D += struct.pack("<I", nf * 3)
        ptr_off = len(D); D += struct.pack("<I", 0)
        idx_off = len(D); D += np.ascontiguousarray(F.astype("<u2")).tobytes()
        struct.pack_into("<I", D, ptr_off, idx_off)
        while len(D) % 4: D += b"\0"
        vert_off = len(D)
        D += np.ascontiguousarray(P).astype("<f4").tobytes()
        # header repoint
        struct.pack_into("<III", D, h + H_FACES, faces_off, nf, nf)
        struct.pack_into("<I", D, h + H_IDXCNT_ARR, cnt_off)
        struct.pack_into("<I", D, h + H_IDXOFF_ARR, ptr_off)
        struct.pack_into("<H", D, h + H_VCOUNT, vc)
        struct.pack_into("<I", D, h + H_MDXOFF, mdxoff)
        struct.pack_into("<I", D, h + H_MDLVERT, vert_off)
        lo, hi = P.min(0), P.max(0)
        old = struct.unpack_from("<6f", D, h + H_BBOX)
        lo, hi = np.minimum(lo, old[:3]), np.maximum(hi, old[3:])
        struct.pack_into("<6f", D, h + H_BBOX, *lo, *hi)

    def save(self, path):
        struct.pack_into("<I", self.D, 176, len(self.X))      # model header: MDX data size (engine sizes its buffer from it)
        out = struct.pack("<III", 0, len(self.D), len(self.X)) + bytes(self.D)
        mdx = os.path.splitext(path)[0] + ".mdx"
        for p, data in ((path, out), (mdx, bytes(self.X))):
            if os.path.exists(p) and not os.path.exists(p + ".bak"):
                shutil.copyfile(p, p + ".bak")
            with open(p, "wb") as f:
                f.write(data)


def adjacency(F):
    nf = len(F)
    adj = np.full((nf, 3), 0xFFFF, dtype=np.int64)
    emap = {}
    for f in range(nf):
        for e in range(3):
            a, b = int(F[f, e]), int(F[f, (e + 1) % 3])
            emap.setdefault((min(a, b), max(a, b)), []).append((f, e))
    for lst in emap.values():
        if len(lst) == 2:
            (f0, e0), (f1, e1) = lst
            adj[f0, e0] = f1; adj[f1, e1] = f0
    return adj

# ----------------------------------------------------------------------------- upscaling

def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n[n == 0] = 1
    return v / n


def analyse(m, crease_deg=40.0, face_crease=None):
    """Feature detection. Returns welded ids, per-face-corner smooth-group normals and the edge crease map."""
    P, F = m.pos, m.faces
    key = np.round(P / 1e-5).astype(np.int64)
    _, wid = np.unique(key, axis=0, return_inverse=True)
    wid = wid.reshape(-1)
    W = wid[F]                                              # welded id per corner
    fnv = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])   # area weighted
    fn = _unit(fnv)
    cosc = math.cos(math.radians(crease_deg))
    fcos = None if face_crease is None else np.cos(np.radians(np.asarray(face_crease, dtype=float)))
    edges = {}
    for f in range(len(F)):
        for e in range(3):
            a, b = int(W[f, e]), int(W[f, (e + 1) % 3])
            edges.setdefault((min(a, b), max(a, b)), []).append((f, e))
    parent = {}
    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    crease = {}
    nrm = m.norm
    for k, lst in edges.items():
        smooth = False
        if len(lst) == 2:
            (f0, e0), (f1, e1) = lst
            # a region can ask for a higher crease angle; an edge uses it only if BOTH faces are in the region
            ce = cosc if fcos is None else float(max(fcos[f0], fcos[f1]))
            smooth = float(fn[f0] @ fn[f1]) >= ce
            if smooth and nrm is not None:
                # artist-authored hard edge: the shading normals differ across the edge
                c0 = [F[f0, e0], F[f0, (e0 + 1) % 3]]
                c1 = {int(W[f1, e1]): F[f1, e1], int(W[f1, (e1 + 1) % 3]): F[f1, (e1 + 1) % 3]}
                for c in c0:
                    w = int(wid[c])
                    if w in c1 and float(nrm[c] @ nrm[c1[w]]) < ce:
                        smooth = False
            if smooth:
                a, b = k
                union((a, f0), (a, f1)); union((b, f0), (b, f1))
        crease[k] = not smooth
    for f in range(len(F)):
        for e in range(3):
            find((int(W[f, e]), f))
    acc = {}
    for f in range(len(F)):
        for e in range(3):
            r = find((int(W[f, e]), f))
            # angle weighting
            p0, p1, p2 = P[F[f, e]], P[F[f, (e + 1) % 3]], P[F[f, (e + 2) % 3]]
            u, v = _unit(p1 - p0), _unit(p2 - p0)
            ang = math.acos(max(-1.0, min(1.0, float(u @ v))))
            acc[r] = acc.get(r, 0) + fn[f] * ang
    cn = np.zeros((len(F), 3, 3))
    for f in range(len(F)):
        for e in range(3):
            cn[f, e] = acc[find((int(W[f, e]), f))]
    cn = _unit(cn)
    return dict(W=W, cn=cn, crease=crease, fn=fn)


# ---- texture loading (TGA via Pillow, TPC decoded here) ------------------------------------------
def load_texture(path):
    """Return float32 array (H,W,C) in ENGINE row order (row index = v * H, v=0 bottom)."""
    from PIL import Image
    ext = os.path.splitext(path)[1].lower()
    if ext == ".tpc":
        d = open(path, "rb").read()
        size, _alpha, w, h, enc, mips = struct.unpack_from("<IfHHBB", d, 0)
        w &= 0x7FFF
        body = d[128:]
        if size:                                       # compressed
            fourcc = b"DXT1" if enc == 2 else b"DXT5"
            hdr = struct.pack("<4sI", b"DDS ", 124) + struct.pack("<IIIII", 0x81007, h, w, size, 0) \
                + struct.pack("<I", 1) + b"\0" * 44 + struct.pack("<II4s", 32, 4, fourcc) + b"\0" * 20 \
                + struct.pack("<I", 0x1000) + b"\0" * 16
            img = Image.open(__import__("io").BytesIO(hdr + body[:size])).convert("RGBA")
            return np.asarray(img, dtype=np.float32) / 255.0      # stored order == engine order
        ch = 4 if enc == 4 else 3 if enc == 2 else 1
        a = np.frombuffer(body[:w * h * ch], dtype=np.uint8).reshape(h, w, ch).astype(np.float32) / 255.0
        return a
    img = Image.open(path).convert("RGBA")
    return np.asarray(img, dtype=np.float32)[::-1] / 255.0        # Pillow gives top-down; engine is bottom-up


def find_normal_map(model_path, tex):
    base = os.path.dirname(os.path.abspath(model_path))
    names = [tex + "_n", tex + "_N", tex + "_normal"]
    dirs = [base, os.path.join(base, "pbr"), os.path.join(base, "..", "Override"), os.path.join(base, "..", "Override", "pbr"),
            os.path.join(base, "..")]
    for d in dirs:
        for n in names:
            for e in (".tga", ".tpc", ".TGA", ".TPC"):
                p = os.path.join(d, n + e)
                if os.path.exists(p):
                    return os.path.normpath(p)
    return None


def height_from_normals(nm, invert_y=False, hp_sigma=24.0):
    """Integrate a tangent-space normal map into a height field (units: texels), high-passed."""
    nx = nm[..., 0] * 2 - 1
    ny = nm[..., 1] * 2 - 1
    if invert_y: ny = -ny
    nz = np.clip(nm[..., 2] * 2 - 1, 0.25, 1.0) if nm.shape[2] > 2 else np.ones_like(nx)
    gx, gy = -nx / nz, -ny / nz                      # dh/du, dh/dv in height-per-texel
    # mirror-pad for a non periodic Poisson solve
    H, W = gx.shape
    def mir(a, sx, sy):
        a = np.concatenate([a, sx * a[:, ::-1]], axis=1)
        return np.concatenate([a, sy * a[::-1]], axis=0)
    gxm, gym = mir(gx, -1, 1), mir(gy, 1, -1)
    fy = np.fft.fftfreq(gxm.shape[0])[:, None] * 2 * np.pi
    fx = np.fft.fftfreq(gxm.shape[1])[None, :] * 2 * np.pi
    Gx, Gy = np.fft.fft2(gxm), np.fft.fft2(gym)
    den = fx ** 2 + fy ** 2
    den[0, 0] = 1
    Hf = (-1j * fx * Gx - 1j * fy * Gy) / den
    Hf[0, 0] = 0
    h = np.real(np.fft.ifft2(Hf))[:H, :W]
    # high-pass: remove what the base mesh already models
    Fh = np.fft.fft2(h)
    ky = np.fft.fftfreq(H)[:, None]; kx = np.fft.fftfreq(W)[None, :]
    gauss = np.exp(-2 * (np.pi * hp_sigma) ** 2 * (kx ** 2 + ky ** 2))
    h = np.real(np.fft.ifft2(Fh * (1 - gauss)))
    return h.astype(np.float32)


def sample_bilinear(img, uv):
    H, W = img.shape[:2]
    x = uv[:, 0] * W - 0.5
    y = uv[:, 1] * H - 0.5
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x - x0, y - y0
    if img.ndim == 3: fx, fy = fx[:, None], fy[:, None]
    x0m, x1m = x0 % W, (x0 + 1) % W
    y0m, y1m = y0 % H, (y0 + 1) % H
    return (img[y0m, x0m] * ((1 - fx) * (1 - fy)) + img[y0m, x1m] * (fx * (1 - fy))
            + img[y1m, x0m] * ((1 - fx) * fy) + img[y1m, x1m] * (fx * fy))


# ---- the upscaler ---------------------------------------------------------------------------------
def _patch_controls(m, an, crease_pts):
    """Per original face: 10 cubic Bezier control points of the crease-aware PN patch."""
    P, F, W, cn, crease = m.pos, m.faces, an["W"], an["cn"], an["crease"]
    nf = len(F)
    p = [P[F[:, i]] for i in range(3)]
    def edge_pt(i, j):
        a, b = p[i], p[j]
        N = cn[:, i]
        w = ((b - a) * N).sum(1, keepdims=True)
        curved = (2 * a + b - w * N) / 3
        straight = (2 * a + b) / 3
        flag = np.array([crease[(min(int(W[f, i]), int(W[f, j])), max(int(W[f, i]), int(W[f, j])))] for f in range(nf)])[:, None]
        return np.where(flag, straight, curved)
    b01, b10, b12, b21, b20, b02 = (edge_pt(0, 1), edge_pt(1, 0), edge_pt(1, 2), edge_pt(2, 1), edge_pt(2, 0), edge_pt(0, 2))
    E = (b01 + b10 + b12 + b21 + b20 + b02) / 6.0
    V = (p[0] + p[1] + p[2]) / 3.0
    b111 = E + (E - V) / 2.0
    return np.stack([p[0], p[1], p[2], b01, b10, b12, b21, b20, b02, b111], axis=1)   # (nf,10,3)


def _eval_patch(C, fid, B, strength):
    u, v, w = B[:, 0:1], B[:, 1:2], B[:, 2:3]
    c = C[fid]
    flat = u * c[:, 0] + v * c[:, 1] + w * c[:, 2]
    cub = (u ** 3 * c[:, 0] + v ** 3 * c[:, 1] + w ** 3 * c[:, 2]
           + 3 * u * u * v * c[:, 3] + 3 * u * v * v * c[:, 4] + 3 * v * v * w * c[:, 5]
           + 3 * v * w * w * c[:, 6] + 3 * w * w * u * c[:, 7] + 3 * w * u * u * c[:, 8]
           + 6 * u * v * w * c[:, 9])
    return flat + (cub - flat) * strength


def upscale(m, quality=3.0, crease_deg=40.0, strength=0.85, disp=None, max_verts=40000, progress=None, region=None, target_faces=None):
    """Adaptive, feature-aware refinement.
    quality  : 1..6  (geometric error tolerance halves every step)
    disp     : None or dict(height=float32 HxW array (texels), scale=float 0..1 strength, flipv=bool)"""
    region = region or {}
    an = analyse(m, crease_deg, region.get("crease"))
    F, cn = m.faces, an["cn"]
    nf = len(F)
    keepf = region.get("keep")          # faces left alone: never subdivided, never displaced
    nodisp = region.get("nodisp")       # faces that get no displacement
    dmask = np.ones(nf, bool)
    if nodisp is not None: dmask &= ~np.asarray(nodisp, bool)
    if keepf is not None: dmask &= ~np.asarray(keepf, bool)
    C = _patch_controls(m, an, None)
    P = m.pos
    diag = float(np.linalg.norm(P.max(0) - P.min(0))) or 1.0
    tol_rel = 0.004 * 0.5 ** (float(quality) - 1.0)
    dl_mul = 1.0
    tgt = int(target_faces) if target_faces else 0
    if tgt:
        # face-budget mode: even edge length everywhere (bisection on the longest allowed edge), no curvature/relief steering
        a3_ = 0.5 * np.linalg.norm(np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]]), axis=1).sum()
        l0 = float(np.sqrt(a3_ / (0.433 * max(tgt, 8)))) * 1.35
        lo_l, hi_l = l0 * 0.3, l0 * 4.0          # lo_l: surely too many faces; hi_l: surely too few
        cur_l = l0; best = None; tries = 0
        relmap = None
        if disp is not None and m.uv is not None:
            # where the map has relief, edges may be shorter; flat areas (cloth, legs) get coarser ones
            ah = np.abs(disp["height"]).astype(np.float32)
            Hr, Wr = ah.shape
            ky_ = np.fft.fftfreq(Hr)[:, None]; kx_ = np.fft.fftfreq(Wr)[None, :]
            rl = np.real(np.fft.ifft2(np.fft.fft2(ah) * np.exp(-2 * (np.pi * 12.0) ** 2 * (kx_ ** 2 + ky_ ** 2))))
            relmap = np.clip(rl / max(float(np.percentile(rl, 99)), 1e-9), 0, 1).astype(np.float32)
    while True:
        tol = diag * tol_rel
        lmin = diag * 0.004 * 0.5 ** (float(quality) - 1.0) * 6
        lmax = np.inf
        if tgt: tol = 1e30; lmin = 0.0; lmax = cur_l
        fid = np.arange(nf)
        B = np.tile(np.eye(3)[None], (nf, 1, 1))              # (T,3corners,3bary)
        ok = True
        for it in range(10):
            T = len(fid)
            pos = _eval_patch(C, np.repeat(fid, 3), B.reshape(-1, 3), strength).reshape(T, 3, 3)
            nrm = (B[:, :, 0:1] * cn[fid][:, 0][:, None, :] + B[:, :, 1:2] * cn[fid][:, 1][:, None, :] + B[:, :, 2:3] * cn[fid][:, 2][:, None, :])
            nrm = _unit(nrm)
            key = np.round(pos.reshape(-1, 3) / 1e-5).astype(np.int64)
            _, vid = np.unique(key, axis=0, return_inverse=True)
            vid = vid.reshape(T, 3)
            M = int(vid.max()) + 2
            ea = np.stack([vid[:, 0], vid[:, 1], vid[:, 2]], 1)       # edge e connects corner e and (e+1)%3
            eb = np.roll(ea, -1, axis=1)
            ekey = np.minimum(ea, eb).astype(np.int64) * M + np.maximum(ea, eb)
            pa, pb = pos, np.roll(pos, -1, axis=1)
            na, nb = nrm, np.roll(nrm, -1, axis=1)
            L = np.linalg.norm(pb - pa, axis=2)
            theta = np.arccos(np.clip((na * nb).sum(2), -1, 1))
            if tgt and relmap is not None:
                UVr = m.uv[F[fid]]
                cen = np.einsum("tcb,tbk->tck", B, UVr).mean(1)
                if disp.get("flipv"): cen = np.stack([cen[:, 0], 1 - cen[:, 1]], axis=-1)
                rr = sample_bilinear(relmap[..., None], cen)[:, 0]
                sc = 1.55 - 1.05 * np.clip(rr / 0.35, 0, 1)
                want = (L > lmax * sc[:, None])
            else:
                want = ((L * theta / 8.0 > tol) | (L > lmax)) & (L > lmin)
            if keepf is not None:
                want &= ~np.asarray(keepf, bool)[fid][:, None]
            if disp is not None and not tgt:
                # relief-aware: split edges whose displacement is not well described by a straight blend
                UVf = m.uv[F[fid]]
                uvc = np.einsum("tcb,tbk->tck", B, UVf)
                if disp.get("flipv"): uvc = np.stack([uvc[..., 0], 1 - uvc[..., 1]], axis=-1)
                ua, ub = uvc, np.roll(uvc, -1, axis=1)
                ts = (0.0, 0.25, 0.5, 0.75, 1.0)
                hs = []
                for t_ in ts:
                    uvt = (ua * (1 - t_) + ub * t_).reshape(-1, 2)
                    hs.append(sample_bilinear(disp["height"][..., None], uvt)[:, 0].reshape(T, 3))
                hs = np.stack(hs, axis=-1)
                lin = hs[..., :1] * (1 - np.array(ts)) + hs[..., -1:] * np.array(ts)
                err = np.abs(hs - lin)[..., 1:4].max(-1) * disp["wpf"][fid][:, None] * disp["scale"]
                want |= (err > disp["tol"] * dl_mul) & (L > disp["lmin"]) & dmask[fid][:, None]
            if not want.any():
                break
            marked = np.unique(ekey[want])
            longest = np.argmax(L - 1e-9 * np.arange(3)[None, :], axis=1)
            for _ in range(60):                                       # Rivara propagation
                mk = np.isin(ekey, marked)
                need = mk.any(1) & ~mk[np.arange(T), longest]
                if not need.any(): break
                marked = np.union1d(marked, ekey[np.arange(T)[need], longest[need]])
            mk = np.isin(ekey, marked)
            cnt = mk.sum(1)
            nfid, nB = [fid[cnt == 0]], [B[cnt == 0]]
            def mid(Bt, i, j): return (Bt[:, i] + Bt[:, j]) / 2
            def add(sel, tri_list):
                for tri in tri_list:
                    nfid.append(fid[sel]); nB.append(np.stack(tri, axis=1))
            # 1 marked edge
            for e in range(3):
                sel = (cnt == 1) & mk[:, e]
                if sel.any():
                    Bt = B[sel]; i, j, o = e, (e + 1) % 3, (e + 2) % 3
                    mm = mid(Bt, i, j)
                    add(sel, [(Bt[:, i], mm, Bt[:, o]), (mm, Bt[:, j], Bt[:, o])])
            # 2 marked edges: unmarked edge e0 -> split the quad on the shorter diagonal
            for e in range(3):
                sel = (cnt == 2) & ~mk[:, e]
                if sel.any():
                    Bt = B[sel]
                    # corners: c0=e, c1=e+1, c2=e+2 ; marked edges are (e+1)->(e+2) and (e+2)->e
                    c0, c1, c2 = e, (e + 1) % 3, (e + 2) % 3
                    m12 = mid(Bt, c1, c2); m20 = mid(Bt, c2, c0)
                    add(sel, [(Bt[:, c2], m20, m12)])
                    # remaining quad c0,c1,m12,m20 : choose diagonal by 3D length of the patch positions
                    Pq = [_eval_patch(C, fid[sel], x, strength) for x in (Bt[:, c0], Bt[:, c1], m12, m20)]
                    d1 = np.linalg.norm(Pq[0] - Pq[2], axis=1); d2 = np.linalg.norm(Pq[1] - Pq[3], axis=1)
                    A = d1 <= d2
                    q = [Bt[:, c0], Bt[:, c1], m12, m20]
                    t1a = np.where(A[:, None], q[0], q[1]); t1b = np.where(A[:, None], q[1], q[2]); t1c = np.where(A[:, None], q[2], q[3])
                    # diag c0-m12 :(c0,c1,m12),(c0,m12,m20) ; diag c1-m20: (c1,m12,m20),(c1,m20,c0)
                    ta = [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
                    tb = [(q[1], q[2], q[3]), (q[1], q[3], q[0])]
                    for k in range(2):
                        tri = [np.where(A[:, None], ta[k][z], tb[k][z]) for z in range(3)]
                        add(sel, [tuple(tri)])
            sel = cnt == 3
            if sel.any():
                Bt = B[sel]
                m01, m12, m20 = mid(Bt, 0, 1), mid(Bt, 1, 2), mid(Bt, 2, 0)
                add(sel, [(Bt[:, 0], m01, m20), (m01, Bt[:, 1], m12), (m20, m12, Bt[:, 2]), (m01, m12, m20)])
            fid = np.concatenate(nfid); B = np.concatenate(nB)
            if tgt:
                if len(fid) > tgt * 3:
                    ok = False
                    break
            elif len(fid) * 0.55 > max_verts:
                ok = False
                break
        if tgt:
            tries += 1
            nfc = len(fid)
            if ok and nfc <= tgt * 1.03 and (best is None or nfc > best[0]):
                best = (nfc, fid.copy(), B.copy(), cur_l)
            if (ok and tgt * 0.93 <= nfc <= tgt * 1.03) or tries >= 16:
                if best is not None:
                    _, fid, B, cur_l = best
                break
            if (not ok) or nfc > tgt * 1.03: lo_l = cur_l
            else: hi_l = cur_l
            cur_l = float(np.sqrt(lo_l * hi_l))
            continue
        # vertex budget check
        pos = _eval_patch(C, np.repeat(fid, 3), B.reshape(-1, 3), strength)
        nv = len(np.unique(np.round(pos / 1e-5).astype(np.int64), axis=0))
        if ok and nv < max_verts:
            break
        # over budget: give up crack detail first (displacement tolerance), and only then coarsen the base smoothing,
        # so a densely cracked area can't starve the rest of the mesh (belts, legs, hands) of smoothing
        if disp is not None and dl_mul < 60.0:
            dl_mul *= 1.5
        else:
            tol_rel *= 1.6; dl_mul *= 1.5
        if tol_rel > 1: break
    T = len(fid)
    Bf = B.reshape(-1, 3)
    ff = np.repeat(fid, 3)
    def bary(attr):
        if attr is None: return None
        return sum(Bf[:, i:i + 1] * attr[F[ff, i]] for i in range(3))
    pos = _eval_patch(C, ff, Bf, strength)
    pos0 = pos.copy()
    nrm = bary(m.norm)
    if nrm is not None: nrm = _unit(nrm)
    uv = bary(m.uv)
    smooth_n = _unit(sum(Bf[:, i:i + 1] * cn[ff, i] for i in range(3)))
    # --- displacement from the normal map (applied on the smooth surface, shading normals untouched)
    if disp is not None and uv is not None:
        uvs = uv.copy()
        if disp.get("flipv"): uvs[:, 1] = 1 - uvs[:, 1]
        hmap = disp["height"]
        if tgt:
            # detail finer than the mesh can carry would just alias into noise: band-limit the height to the edge length
            e_tex = 0.75 * cur_l / max(float(np.median(disp["wpf"])), 1e-9)
            sg = max(0.25 * e_tex, 0.5)
            Hh_, Ww_ = hmap.shape
            ky_ = np.fft.fftfreq(Hh_)[:, None]; kx_ = np.fft.fftfreq(Ww_)[None, :]
            hb = np.real(np.fft.ifft2(np.fft.fft2(hmap) * np.exp(-2 * (np.pi * sg) ** 2 * (kx_ ** 2 + ky_ ** 2)))).astype(np.float32)
            gain = min(float(hmap.std()) / max(float(hb.std()), 1e-9), 4.0)
            hmap = hb * gain
        h = sample_bilinear(hmap[..., None], uvs)[:, 0]          # texels
        if tgt:
            # face-budget mode: depth is relative to the edge length, so cracks stay visible however coarse the mesh is.
            # strength 1.0 = the deepest 10% of the relief reaches about 45% of an edge length
            d = h * disp["wpf"][ff] * dmask[ff]
            ad = np.abs(d[dmask[ff]]) if dmask.any() else np.abs(d)
            p90 = float(np.percentile(ad, 90)) if len(ad) else 0.0
            e_w = 0.75 * cur_l
            d = d * (0.38 * e_w / max(p90, 1e-12)) * disp["scale"]
            d = np.where(d > 0, d * 0.25, d)          # cracks cut inward; the high-pass halo around them must not raise spiky rims
            d = np.clip(d, -0.9 * e_w, 0.9 * e_w)
        else:
            d = h * disp["wpf"][ff] * disp["scale"] * dmask[ff]
            d = np.clip(d, -disp["clamp"] * diag, disp["clamp"] * diag)
        # weld across UV/normal seams: coincident vertices must move together or the surface tears open
        wk = np.round(pos0 / 1e-5).astype(np.int64)
        _, winv = np.unique(wk, axis=0, return_inverse=True)
        winv = winv.reshape(-1)
        wc = np.bincount(winv).astype(np.float64)
        d = np.bincount(winv, weights=d)[winv] / wc[winv]
        wn = np.stack([np.bincount(winv, weights=smooth_n[:, k])[winv] for k in range(3)], 1)
        smooth_n = _unit(wn)
        pos = pos + smooth_n * d[:, None]
        dstat = float(np.sqrt((d ** 2).mean()))
    # --- skin weights
    sw = si = None
    if m.skin:
        sw = np.zeros((len(Bf), 4)); si = np.full((len(Bf), 4), -1.0)
        cache = {}
        for q in range(len(Bf)):
            f = ff[q]
            acc = {}
            for i in range(3):
                vi = F[f, i]
                for k in range(4):
                    if m.si[vi, k] >= 0 and m.sw[vi, k] > 0:
                        acc[int(m.si[vi, k])] = acc.get(int(m.si[vi, k]), 0.0) + m.sw[vi, k] * Bf[q, i]
            top = sorted(acc.items(), key=lambda t: -t[1])[:4]
            s = sum(t[1] for t in top) or 1.0
            for k, (bi, bw) in enumerate(top):
                si[q, k] = bi; sw[q, k] = bw / s
    # --- dedupe
    cols = [np.round(pos / 1e-5)]
    if uv is not None: cols.append(np.round(uv / 1e-5))
    if nrm is not None: cols.append(np.round(nrm / 1e-3))
    key = np.concatenate(cols, axis=1).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    pick = lambda arr: None if arr is None else arr[first]
    faces = inv.reshape(T, 3)
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    out = dict(pos=pick(pos), norm=pick(nrm), uv=pick(uv), sw=pick(sw), si=pick(si),
               faces=faces[keep], surf=m.surf[fid][keep], level=quality)
    if disp is not None and uv is not None:
        out["dstat"] = dstat
    if disp is not None:
        out["pos"] = _weld_displacement(out["pos"], pick(pos0), pick(smooth_n))
    return out


def make_disp(m, height, scale=1.0, flipv=False, sens=5.0, clamp=0.01):
    """Displacement settings for upscale(). sens 1..10: higher = refine for smaller relief (relief-aware)."""
    P, F = m.pos, m.faces
    a3 = 0.5 * np.linalg.norm(np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]]), axis=1)
    uvf = m.uv[F]
    d1, d2 = uvf[:, 1] - uvf[:, 0], uvf[:, 2] - uvf[:, 0]
    aU = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    g = aU > 1e-10
    Hh, Ww = height.shape
    wpf = np.sqrt(a3 / np.maximum(aU, 1e-12)) / np.sqrt(Hh * Ww)           # world units per texel, per face
    med = float(np.median(wpf[g])) if g.any() else 0.001
    wpf = np.where(g, wpf, med)
    diag = float(np.linalg.norm(P.max(0) - P.min(0))) or 1.0
    tol = diag * 0.002 / (1.35 ** (float(sens) - 1.0))
    return dict(height=height, scale=scale, flipv=flipv, wpf=wpf, tol=tol, lmin=med * 2.0, clamp=clamp)


def _weld_displacement(posd, pos0, nrm):
    """Average the displacement of vertices that share an undisplaced position (UV seams) so no cracks open."""
    key = np.round(pos0 / 1e-5).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    off = posd - pos0
    n = inv.max() + 1
    acc = np.zeros((n, 3)); cnt = np.zeros(n)
    np.add.at(acc, inv, off); np.add.at(cnt, inv, 1)
    return pos0 + (acc / cnt[:, None])[inv]


# ----------------------------------------------------------------------------- FBX import / export
# Binary FBX 7.4.  Export writes world-space geometry (Z up, metres) with normals + UVs, one
# object per KotOR mesh node named after the node.  Import reads any binary FBX (Blender, Maya, ...),
# applies its transforms / axes / units and returns objects ready to be written back into the MDL.
import zlib

_FBX_MAGIC = b"Kaydara FBX Binary  \x00\x1a\x00"


class _Node:
    def __init__(self, name, props=(), children=()):
        self.name, self.props, self.children = name, list(props), list(children)


def _fbx_prop(buf, p):
    t, v = p
    if t == "I": buf += b"I" + struct.pack("<i", v)
    elif t == "L": buf += b"L" + struct.pack("<q", v)
    elif t == "D": buf += b"D" + struct.pack("<d", v)
    elif t == "C": buf += b"C" + struct.pack("<B", 1 if v else 0)
    elif t == "S":
        b = v if isinstance(v, bytes) else v.encode("utf8")
        buf += b"S" + struct.pack("<I", len(b)) + b
    elif t in ("d", "i", "l", "f"):
        dt = {"d": "<f8", "i": "<i4", "l": "<i8", "f": "<f4"}[t]
        a = np.ascontiguousarray(np.asarray(v, dtype=dt)).reshape(-1)
        buf += t.encode() + struct.pack("<III", len(a), 0, a.nbytes) + a.tobytes()
    else:
        raise ValueError(t)


def _fbx_write_node(buf, n):
    start = len(buf)
    buf += b"\0" * 12                                  # end offset, nprops, proplen  (patched below)
    buf += struct.pack("<B", len(n.name)) + n.name.encode()
    pstart = len(buf)
    for p in n.props: _fbx_prop(buf, p)
    plen = len(buf) - pstart
    if n.children:
        for c in n.children: _fbx_write_node(buf, c)
        buf += b"\0" * 13
    struct.pack_into("<III", buf, start, len(buf), len(n.props), plen)


def _P(name, typ, label, flags, *vals):
    ps = [("S", name), ("S", typ), ("S", label), ("S", flags)]
    for v in vals:
        ps.append(("D", v) if isinstance(v, float) else ("I", v) if isinstance(v, int) else ("S", v))
    return _Node("P", ps)


def write_fbx(path, objs):
    """objs: list of dict(name, pos (N,3), faces (M,3), norm (N,3)|None, uv (N,2)|None)."""
    nodes = []
    nodes.append(_Node("FBXHeaderExtension", children=[
        _Node("FBXHeaderVersion", [("I", 1003)]), _Node("FBXVersion", [("I", 7400)]),
        _Node("Creator", [("S", "KotOR Mesh Upscaler")])]))
    nodes.append(_Node("GlobalSettings", children=[_Node("Version", [("I", 1000)]), _Node("Properties70", children=[
        _P("UpAxis", "int", "Integer", "", 2), _P("UpAxisSign", "int", "Integer", "", 1),
        _P("FrontAxis", "int", "Integer", "", 1), _P("FrontAxisSign", "int", "Integer", "", -1),
        _P("CoordAxis", "int", "Integer", "", 0), _P("CoordAxisSign", "int", "Integer", "", 1),
        _P("OriginalUpAxis", "int", "Integer", "", 2), _P("OriginalUpAxisSign", "int", "Integer", "", 1),
        _P("UnitScaleFactor", "double", "Number", "", 100.0), _P("OriginalUnitScaleFactor", "double", "Number", "", 100.0)])]))
    nodes.append(_Node("Definitions", children=[_Node("Version", [("I", 100)]), _Node("Count", [("I", 1 + 2 * len(objs))]),
        _Node("ObjectType", [("S", "GlobalSettings")], [_Node("Count", [("I", 1)])]),
        _Node("ObjectType", [("S", "Model")], [_Node("Count", [("I", len(objs))])]),
        _Node("ObjectType", [("S", "Geometry")], [_Node("Count", [("I", len(objs))])])]))
    objn, conn = [], []
    for k, o in enumerate(objs):
        gid, mid = 1000 + 2 * k, 1001 + 2 * k
        P, F = np.asarray(o["pos"], float), np.asarray(o["faces"], np.int64)
        pvi = F.copy(); pvi[:, 2] = -(pvi[:, 2] + 1)
        gch = [_Node("Properties70"), _Node("GeometryVersion", [("I", 124)]),
               _Node("Vertices", [("d", P.reshape(-1))]), _Node("PolygonVertexIndex", [("i", pvi.reshape(-1))])]
        layer = [_Node("Version", [("I", 100)])]
        if o.get("norm") is not None:
            N = np.asarray(o["norm"], float)[F.reshape(-1)]
            gch.append(_Node("LayerElementNormal", [("I", 0)], [_Node("Version", [("I", 101)]), _Node("Name", [("S", "")]),
                _Node("MappingInformationType", [("S", "ByPolygonVertex")]), _Node("ReferenceInformationType", [("S", "Direct")]),
                _Node("Normals", [("d", N.reshape(-1))])]))
            layer.append(_Node("LayerElement", children=[_Node("Type", [("S", "LayerElementNormal")]), _Node("TypedIndex", [("I", 0)])]))
        if o.get("uv") is not None:
            U = np.asarray(o["uv"], float)
            gch.append(_Node("LayerElementUV", [("I", 0)], [_Node("Version", [("I", 101)]), _Node("Name", [("S", "UVMap")]),
                _Node("MappingInformationType", [("S", "ByPolygonVertex")]), _Node("ReferenceInformationType", [("S", "IndexToDirect")]),
                _Node("UV", [("d", U.reshape(-1))]), _Node("UVIndex", [("i", F.reshape(-1))])]))
            layer.append(_Node("LayerElement", children=[_Node("Type", [("S", "LayerElementUV")]), _Node("TypedIndex", [("I", 0)])]))
        gch.append(_Node("Layer", [("I", 0)], layer))
        nm = o["name"]
        objn.append(_Node("Geometry", [("L", gid), ("S", nm + "\x00\x01Geometry"), ("S", "Mesh")], gch))
        objn.append(_Node("Model", [("L", mid), ("S", nm + "\x00\x01Model"), ("S", "Mesh")], [
            _Node("Version", [("I", 232)]), _Node("Properties70"), _Node("Shading", [("C", True)]),
            _Node("Culling", [("S", "CullingOff")])]))
        conn.append(_Node("C", [("S", "OO"), ("L", mid), ("L", 0)]))
        conn.append(_Node("C", [("S", "OO"), ("L", gid), ("L", mid)]))
    nodes.append(_Node("Objects", children=objn))
    nodes.append(_Node("Connections", children=conn))
    buf = bytearray(_FBX_MAGIC + struct.pack("<I", 7400))
    for n in nodes: _fbx_write_node(buf, n)
    buf += b"\0" * 13
    with open(path, "wb") as f: f.write(bytes(buf))


def _fbx_read_node(d, o, v64):
    if v64: end, npr, plen = struct.unpack_from("<QQQ", d, o); o += 24
    else: end, npr, plen = struct.unpack_from("<III", d, o); o += 12
    nl = d[o]; o += 1
    if end == 0: return None, o
    name = d[o:o + nl].decode("latin1"); o += nl
    props = []
    for _ in range(npr):
        t = chr(d[o]); o += 1
        if t == "I": props.append(struct.unpack_from("<i", d, o)[0]); o += 4
        elif t == "L": props.append(struct.unpack_from("<q", d, o)[0]); o += 8
        elif t == "D": props.append(struct.unpack_from("<d", d, o)[0]); o += 8
        elif t == "F": props.append(struct.unpack_from("<f", d, o)[0]); o += 4
        elif t == "Y": props.append(struct.unpack_from("<h", d, o)[0]); o += 2
        elif t == "C": props.append(bool(d[o])); o += 1
        elif t in "SR":
            ln = struct.unpack_from("<I", d, o)[0]; o += 4
            props.append(bytes(d[o:o + ln])); o += ln
        elif t in "dilfb":
            ln, enc, cl = struct.unpack_from("<III", d, o); o += 12
            raw = d[o:o + cl]; o += cl
            if enc == 1: raw = zlib.decompress(raw)
            dt = {"d": "<f8", "i": "<i4", "l": "<i8", "f": "<f4", "b": "u1"}[t]
            props.append(np.frombuffer(raw, dtype=dt, count=ln))
        else:
            raise ValueError("bad FBX property type %r" % t)
    children = []
    while o < end:
        c, o = _fbx_read_node(d, o, v64)
        if c is None: break
        children.append(c)
    o = end
    n = _Node(name, props, children)
    return n, o


def _fbx_parse(path):
    d = open(path, "rb").read()
    if not d.startswith(b"Kaydara FBX Binary"):
        raise ValueError("Only binary FBX files are supported (export 'FBX Binary', not ASCII)")
    ver = struct.unpack_from("<I", d, 23)[0]
    o, top = 27, []
    while o < len(d) - 13:
        n, o = _fbx_read_node(d, o, ver >= 7500)
        if n is None: break
        top.append(n)
    return top


def _eul(rx, ry, rz, order=0):
    rx, ry, rz = math.radians(rx), math.radians(ry), math.radians(rz)
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    X = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]); Y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return {0: Z @ Y @ X, 1: Y @ Z @ X, 2: Y @ X @ Z, 3: X @ Z @ Y, 4: X @ Y @ Z, 5: Z @ X @ Y}.get(order, Z @ Y @ X)


def read_fbx(path):
    """Return list of dict(name, pos, faces, norm, uv) in KotOR world space (Z up, metres)."""
    top = _fbx_parse(path)
    byname = {n.name: n for n in top}
    def prop70(node):
        out = {}
        for p in [c for c in node.children if c.name == "Properties70"]:
            for q in p.children:
                if q.name == "P" and len(q.props) >= 5:
                    out[q.props[0].decode() if isinstance(q.props[0], bytes) else q.props[0]] = q.props[4:]
        return out
    gs = prop70(byname["GlobalSettings"]) if "GlobalSettings" in byname else {}
    g = lambda k, dflt: gs[k][0] if k in gs else dflt
    up, ups = g("UpAxis", 1), g("UpAxisSign", 1)
    fr, frs = g("FrontAxis", 2), g("FrontAxisSign", 1)
    co, cos_ = g("CoordAxis", 0), g("CoordAxisSign", 1)
    unit = g("UnitScaleFactor", 1.0) / 100.0
    A = np.zeros((3, 3))                                 # file coords -> KotOR (x right, y forward, z up)
    A[0, co] = cos_; A[1, fr] = -frs; A[2, up] = ups
    geoms, models, parent = {}, {}, {}
    for c in byname.get("Objects", _Node("")).children:
        if c.name == "Geometry" and len(c.props) >= 3 and c.props[2] == b"Mesh":
            geoms[c.props[0]] = c
        elif c.name == "Model":
            models[c.props[0]] = c
    g2m = {}
    for c in byname.get("Connections", _Node("")).children:
        if c.name == "C" and c.props[0] == b"OO":
            child, par = c.props[1], c.props[2]
            if child in geoms and par in models: g2m[child] = par
            elif child in models: parent[child] = par
    def local(mid):
        p = prop70(models[mid])
        v = lambda k, d_: np.array(p[k][:3], float) if k in p else np.array(d_, float)
        T, R, S = v("Lcl Translation", (0, 0, 0)), v("Lcl Rotation", (0, 0, 0)), v("Lcl Scaling", (1, 1, 1))
        pre, post = v("PreRotation", (0, 0, 0)), v("PostRotation", (0, 0, 0))
        order = int(p["RotationOrder"][0]) if "RotationOrder" in p else 0
        Rm = _eul(*pre) @ _eul(*R, order) @ np.linalg.inv(_eul(*post))
        M = np.eye(4); M[:3, :3] = Rm @ np.diag(S); M[:3, 3] = T
        return M
    def world(mid):
        M = local(mid)
        while mid in parent and parent[mid] in models:
            mid = parent[mid]; M = local(mid) @ M
        return M
    out = []
    for gid, gn in geoms.items():
        mid = g2m.get(gid)
        nm = (models[mid].props[1] if mid else gn.props[1]).split(b"\x00")[0].decode("latin1")
        for pre in ("Model::", "Geometry::"):
            if nm.startswith(pre): nm = nm[len(pre):]
        ch = {c.name: c for c in gn.children}
        V = np.asarray(ch["Vertices"].props[0], float).reshape(-1, 3)
        pvi = np.asarray(ch["PolygonVertexIndex"].props[0], np.int64)
        ends = np.nonzero(pvi < 0)[0]
        idx = np.where(pvi < 0, -pvi - 1, pvi)
        poly_of = np.repeat(np.arange(len(ends)), np.diff(np.concatenate([[-1], ends])))
        # triangulate (fan)
        tris, tcorner = [], []
        s = 0
        for e in ends:
            n = e - s + 1
            for k in range(1, n - 1):
                tris.append((idx[s], idx[s + k], idx[s + k + 1])); tcorner.append((s, s + k, s + k + 1))
            s = e + 1
        tris, tcorner = np.array(tris, np.int64), np.array(tcorner, np.int64)
        def layer(name, dname, iname):
            L = ch.get(name)
            if L is None: return None
            lc = {c.name: c for c in L.children}
            data = np.asarray(lc[dname].props[0], float)
            comp = 3 if dname == "Normals" else 2
            data = data.reshape(-1, comp)
            mapping = lc["MappingInformationType"].props[0].decode()
            ref = lc["ReferenceInformationType"].props[0].decode()
            if ref == "IndexToDirect" and iname in lc: data = data[np.asarray(lc[iname].props[0], np.int64)]
            if mapping == "ByPolygonVertex": return data                               # per corner
            if mapping in ("ByVertice", "ByVertex"): return data[idx]
            if mapping == "ByPolygon": return data[poly_of]
            if mapping == "AllSame": return np.repeat(data[:1], len(idx), 0)
            return None
        Nc, Uc = layer("LayerElementNormal", "Normals", "NormalsIndex"), layer("LayerElementUV", "UV", "UVIndex")
        M = world(mid) if mid else np.eye(4)
        lin = A @ M[:3, :3]
        pos = (V @ M[:3, :3].T + M[:3, 3]) @ A.T * unit
        if Nc is not None:
            Rn = A @ np.linalg.inv(M[:3, :3]).T
            Nc = Nc @ Rn.T
            Nc /= np.maximum(np.linalg.norm(Nc, axis=1, keepdims=True), 1e-12)
        # unique (position, normal, uv) corners -> vertices
        cols = [tris.reshape(-1, 1).astype(float)]
        ci = tcorner.reshape(-1)
        if Nc is not None: cols.append(np.round(Nc[ci] / 1e-4))
        if Uc is not None: cols.append(np.round(Uc[ci] / 1e-5))
        key = np.concatenate([cols[0]] + cols[1:], axis=1)
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        inv = inv.reshape(-1)
        v_pos = pos[tris.reshape(-1)[first]]
        out.append(dict(name=nm, pos=v_pos, faces=inv.reshape(-1, 3),
                        norm=None if Nc is None else Nc[ci[first]], uv=None if Uc is None else Uc[ci[first]],
                        mflip=bool(np.linalg.det(lin) < 0)))
    for o in out:                                      # mirrored transforms flip winding
        if o["mflip"]: o["faces"] = o["faces"][:, ::-1].copy()
    return out


def fbx_to_mesh(m, obj):
    """Turn an imported FBX object into an upscale()-style dict for KotOR mesh m (node-local space)."""
    inv = np.linalg.inv(m.world)
    pos = obj["pos"] @ inv[:3, :3].T + inv[:3, 3]
    nrm = obj["norm"]
    if nrm is not None:
        nrm = _unit(nrm @ inv[:3, :3].T)
    uv = obj["uv"]
    F = obj["faces"]
    if len(pos) > 65535:
        raise ValueError("%s: %d vertices exceeds the 16-bit index limit" % (m.name, len(pos)))
    if nrm is None:
        fn = np.cross(pos[F[:, 1]] - pos[F[:, 0]], pos[F[:, 2]] - pos[F[:, 0]])
        acc = np.zeros_like(pos)
        for i in range(3): np.add.at(acc, F[:, i], fn)
        nrm = _unit(acc)
    if m.uv is not None and uv is None: uv = np.zeros((len(pos), 2))
    new = dict(pos=pos, norm=nrm if m.norm is not None else None, uv=uv if m.uv is not None else None, faces=F)
    # transfer surface ids and skin weights from the nearest original geometry
    cen = pos[F].mean(1)
    ocen = m.pos[m.faces].mean(1)
    nearest_face = np.empty(len(cen), np.int64)
    for s in range(0, len(cen), 2048):
        d = ((cen[s:s + 2048, None, :] - ocen[None]) ** 2).sum(2); nearest_face[s:s + 2048] = d.argmin(1)
    new["surf"] = m.surf[nearest_face]
    if m.skin:
        sw = np.zeros((len(pos), 4)); si = np.full((len(pos), 4), -1.0)
        for s in range(0, len(pos), 1024):
            d = ((pos[s:s + 1024, None, :] - m.pos[None]) ** 2).sum(2)
            nn = np.argsort(d, axis=1)[:, :3]
            for r in range(nn.shape[0]):
                acc = {}
                for j in nn[r]:
                    wgt = 1.0 / (math.sqrt(d[r, j]) + 1e-6)
                    for k in range(4):
                        if m.si[j, k] >= 0 and m.sw[j, k] > 0:
                            acc[int(m.si[j, k])] = acc.get(int(m.si[j, k]), 0.0) + m.sw[j, k] * wgt
                top = sorted(acc.items(), key=lambda t: -t[1])[:4]
                tot = sum(t[1] for t in top) or 1.0
                for k, (bi, bw) in enumerate(top):
                    si[s + r, k] = bi; sw[s + r, k] = bw / tot
        new["sw"], new["si"] = sw, si
    return new


# ----------------------------------------------------------------------------- rendering

class Camera:
    def __init__(self):
        self.yaw, self.pitch, self.dist = 0.6, 0.25, 1.0
        self.pan = np.zeros(2)
        self.center = np.zeros(3)
        self.size = 1.0

    def fit(self, pts):
        lo, hi = pts.min(0), pts.max(0)
        self.center = (lo + hi) / 2
        self.size = float(np.linalg.norm(hi - lo)) or 1.0
        self.dist = 1.0
        self.pan[:] = 0

    def matrix(self):
        cy, sy, cp, sp = math.cos(self.yaw), math.sin(self.yaw), math.cos(self.pitch), math.sin(self.pitch)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        return Rx @ Rz            # z-up model -> view (x right, y depth, z up)


def render(items, cam, W, H, wire=False, bg=(24, 26, 30)):
    """items: list of (pos(N,3) world, normals or None, faces(M,3), colour(3,))  -> PIL image."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (W, H), bg)
    dr = ImageDraw.Draw(img)
    R = cam.matrix()
    scale = min(W, H) * 0.95 / cam.size * cam.dist
    polys = []
    for pos, nrm, faces, col in items:
        q = (pos - cam.center) @ R.T
        sx = W / 2 + (q[:, 0] + cam.pan[0] * cam.size) * scale
        sy = H / 2 - (q[:, 2] + cam.pan[1] * cam.size) * scale
        depth = q[:, 1]
        fa = faces
        a, b, c = pos[fa[:, 0]], pos[fa[:, 1]], pos[fa[:, 2]]
        fn = _unit(np.cross(b - a, c - a))
        if nrm is not None:
            vn = _unit(nrm[fa[:, 0]] + nrm[fa[:, 1]] + nrm[fa[:, 2]])
            fn = _unit(fn + vn)
        vnv = fn @ R.T
        facing = vnv[:, 1] < 0.0                                  # towards the viewer (view dir = +y into screen)
        # both sides are drawn: rooms and buildings are modelled to be seen from inside, so from outside their walls
        # face away and would vanish until the view turned. A back face is lit as if turned round, a little darker.
        vl = np.where(facing[:, None], vnv, -vnv)
        light = np.clip(-vl[:, 1] * 0.75 + vl[:, 2] * 0.35 + vl[:, 0] * 0.15 + 0.2, 0.05, 1.0) * np.where(facing, 1.0, 0.7)
        zz = depth[fa].mean(1)
        for i in range(len(fa)):
            polys.append((zz[i], (sx[fa[i, 0]], sy[fa[i, 0]], sx[fa[i, 1]], sy[fa[i, 1]], sx[fa[i, 2]], sy[fa[i, 2]]), light[i], col))
    polys.sort(key=lambda t: -t[0])
    for z, xy, l, col in polys:
        fill = tuple(int(min(255, c * l)) for c in col)
        if wire:
            dr.polygon(xy, fill=fill, outline=(235, 235, 235))
        else:
            dr.polygon(xy, fill=fill, outline=fill)
    return img

# ----------------------------------------------------------------------------- v2 additions
# (1) texture/mesh agreement check + UV repair   (2) validation before/after save
# (3) diffuse-guided carving that is COMBINED with the existing normal map's relief (normal map is only read)
def _weld_ids(pos, tol=1e-5):
    _, inv = np.unique(np.round(pos / tol).astype(np.int64), axis=0, return_inverse=True)
    return inv.reshape(-1)


def uv_report(m, tex_px=2048):
    """Per-mesh UV health: flipped (orientation disagrees with neighbours), degenerate, stretched, density outliers."""
    P, UV, F = m.pos, m.uv, m.faces
    a, b, c = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    area3 = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
    d1, d2 = UV[F[:, 1]] - UV[F[:, 0]], UV[F[:, 2]] - UV[F[:, 0]]
    det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    ok = np.abs(det) > 1e-14
    e1, e2 = b - a, c - a
    sd = np.where(ok, det, 1.0)
    Ju = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / sd[:, None]
    Jv = (-e1 * d2[:, 0:1] + e2 * d1[:, 0:1]) / sd[:, None]
    g11, g22, g12 = (Ju * Ju).sum(1), (Jv * Jv).sum(1), (Ju * Jv).sum(1)
    tr, dt = g11 + g22, g11 * g22 - g12 ** 2
    disc = np.sqrt(np.maximum(tr * tr / 4 - dt, 0))
    s1, s2 = np.sqrt(np.maximum(tr / 2 + disc, 0)), np.sqrt(np.maximum(tr / 2 - disc, 0))
    mmpx = np.sqrt(s1 * s2) / tex_px * 1000
    aniso = np.where(s2 > 1e-12, s1 / np.maximum(s2, 1e-12), np.inf)
    w = _weld_ids(P)
    ed = {}
    for i, f in enumerate(F):
        for k in range(3):
            ed.setdefault((min(w[f[k]], w[f[(k + 1) % 3]]), max(w[f[k]], w[f[(k + 1) % 3]])), []).append(i)
    nb = [[] for _ in range(len(F))]
    for fs in ed.values():
        if len(fs) == 2: nb[fs[0]].append(fs[1]); nb[fs[1]].append(fs[0])
    sg = np.sign(det)
    flipped = np.zeros(len(F), bool)
    for i, ns in enumerate(nb):
        if sg[i] != 0 and ns and (sg[ns] == -sg[i]).sum() > len(ns) / 2: flipped[i] = True
    degenerate = ~ok
    good = ok & (area3 > 1e-9)
    med = float(np.median(mmpx[good])) if good.any() else 0.0
    rep = dict(faces=len(F), median_mm_per_px=med, flipped=int(flipped.sum()), degenerate=int(degenerate.sum()),
               stretched=int((good & (aniso > 3)).sum()),
               density_outliers=int((good & ((mmpx > 4 * med) | (mmpx < med / 4))).sum()))
    return rep, dict(flipped=flipped, degenerate=degenerate)


def uv_repair(m, flags, iters=60):
    """Relax the UVs of vertices used only by flipped/degenerate faces towards their neighbours' average."""
    F, UV = m.faces, m.uv.copy()
    bad = flags["flipped"] | flags["degenerate"]
    free = np.zeros(m.vc, bool); free[F[bad].ravel()] = True
    good = np.zeros(m.vc, bool); good[F[~bad].ravel()] = True
    free &= ~good
    if not free.any(): return UV, 0
    adj = [set() for _ in range(m.vc)]
    for f in F:
        for k in range(3): adj[f[k]].update((f[(k + 1) % 3], f[(k + 2) % 3]))
    idx = np.nonzero(free)[0]
    for _ in range(iters):
        new = UV.copy()
        for v in idx:
            nn = list(adj[v])
            if nn: new[v] = UV[nn].mean(0)
        UV = new
    return UV, int(free.sum())


def validate_model(mdl_path):
    """Crash / mis-load checks. Returns (errors, warnings)."""
    err, warn = [], []
    raw = open(mdl_path, "rb").read()
    mdxp = os.path.splitext(mdl_path)[0] + ".mdx"
    mdx = open(mdxp, "rb").read() if os.path.exists(mdxp) else None
    if len(raw) < 100: return ["file too small"], warn
    h0, bsz, xsz = struct.unpack_from("<3I", raw, 0)
    if bsz != len(raw) - 12: err.append("MDL header says %d body bytes, file has %d" % (bsz, len(raw) - 12))
    if mdx is None: return err + ["no .mdx next to the .mdl"], warn
    if xsz != len(mdx): err.append("MDL header says MDX is %d bytes, .mdx file is %d" % (xsz, len(mdx)))
    tot = struct.unpack_from("<I", raw, 12 + 176)[0]
    if tot != len(mdx): err.append("MDX total size at body offset 176 is %d but .mdx is %d bytes (the game crashes on load)" % (tot, len(mdx)))
    M = Model(mdl_path)
    for m in M.meshes:
        if not m.ok:
            if m.why != "no renderable geometry": warn.append("mesh '%s' skipped: %s" % (m.name, m.why))
            continue
        tag = "mesh '%s'" % m.name
        if m.faces.max() >= m.vc: err.append("%s: face index out of range" % tag)
        if M.u32(M.u32(m.h + H_IDXCNT_ARR)) != m.fc * 3: err.append("%s: index count != 3 x faces" % tag)
        end = m.mdxoff + (m.vc + 1) * m.stride
        if end > len(M.X): err.append("%s: MDX rows (with sentinel) run past end of .mdx" % tag)
        if not np.isfinite(m.pos).all(): err.append("%s: NaN/inf positions" % tag)
        if m.norm is not None:
            nl = np.linalg.norm(m.norm, axis=1)
            if (np.abs(nl - 1) > 0.05).mean() > 0.01: warn.append("%s: %.1f%% non-unit normals" % (tag, 100 * (np.abs(nl - 1) > 0.05).mean()))
        if m.vc > 40000: warn.append("%s: %d vertices - heavy for one mesh" % (tag, m.vc))
    return err, warn


def carve_height(diffuse, normal_height, med_wpf, depth_mm=2.0, sigma=18.0):
    """Combine the normal map's integrated height (kept) with cavities found in the DIFFUSE (dark vs local average).
    Returns a height field in texels (negative = deeper). The normal map itself is never modified."""
    H, W = normal_height.shape
    if diffuse.shape[0] != H or diffuse.shape[1] != W:
        from PIL import Image
        im = Image.fromarray((np.clip(diffuse[..., :3], 0, 1) * 255).astype(np.uint8)).resize((W, H))
        diffuse = np.asarray(im, np.float32) / 255.0
    L = diffuse[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32)
    ky = np.fft.fftfreq(H)[:, None]; kx = np.fft.fftfreq(W)[None, :]
    bg = np.real(np.fft.ifft2(np.fft.fft2(L) * np.exp(-2 * (np.pi * sigma) ** 2 * (kx ** 2 + ky ** 2))))
    cav = np.clip(np.clip((bg - L) / np.maximum(bg, 0.05), 0, 1) * 2.0, 0, 1)
    depth_tex = (depth_mm / 1000.0) / max(med_wpf, 1e-9)
    return (normal_height - cav * depth_tex).astype(np.float32)


# (4) anatomy presets (measured from the Vitruvian Project mesh by sweeping its Race/Age/Gender/BodyType shape keys)
#     value = (head width scale, head depth scale, face protrusion scale) relative to the neutral Vitruvian head
HEAD_NEUTRAL = dict(w_over_h=0.758, d_over_h=0.931, head_h=0.225,
                    landmarks={"chin": 0.0, "teeth_line": 0.231, "nose_base": 0.289, "eye_line": 0.507, "head_top": 1.0})
PRESETS = {"neutral": (1.0, 1.0, 1.0)}
PRESETS.update({"gender_female": [0.984,0.985,0.977],"gender_male": [1.016,1.015,1.029],"race_bengali": [1.038,1.006,0.959],"race_tamil": [1.026,1.0,1.0],"race_punjabi": [1.0,1.0,1.0],"race_marathi": [1.0,1.002,1.012],"race_telegu": [1.001,1.0,1.001],"race_sinhalese": [1.0,1.0,1.002],"race_kannada": [1.0,1.002,1.009],"race_hispanic": [1.037,1.0,1.0],"race_white": [1.0,0.993,0.954],"race_african": [1.0,1.001,1.008],"race_eastasian": [1.0,0.997,0.984],"race_middleeastern": [1.0,1.003,1.015],"bodytype_fat": [1.001,1.014,1.002],"bodytype_emaciated": [0.992,0.997,0.986],"bodytype_muscular": [1.017,1.012,1.03],"bodytype_lean": [1.003,1.003,1.005],"bodytype_longproportions": [1.0,1.0,1.0],"age_old": [1.025,1.009,1.033]})


def proportion_report(m, preset="neutral", head_h=None):
    """Rough head-proportion check against a Vitruvian preset. Takes the head end of the mesh's long axis
    (the narrower end), measures width/height and depth/height of a head-sized slab and compares them."""
    P = m.pos
    c = P - P.mean(0)
    u, s, vt = np.linalg.svd(c, full_matrices=False)
    axes = vt                                              # principal axes, long first
    hh = float(head_h or HEAD_NEUTRAL["head_h"])
    ext = np.ptp(P, axis=0)
    if float(ext.max()) < hh * 1.6:                         # a head-only mesh: use all of it
        slab = P; long_ax = int(np.argmax(ext)); lat_axes = [a for a in range(3) if a != long_ax]
        hgt = float(ext[long_ax]); w, d = sorted((float(ext[lat_axes[0]]), float(ext[lat_axes[1]])), reverse=True)
        note = "head-only mesh"
    else:
        t = c @ axes[0]
        best = None
        for sgn in (1, -1):
            end = t * sgn
            sel = end > end.max() - hh
            lat = c[sel] @ axes[1:].T
            width = float(np.ptp(lat, axis=0).max())
            if best is None or width < best[0]: best = (width, sel)
        sel = best[1]
        lat = c[sel] @ axes[1:].T
        w, d = sorted(map(float, np.ptp(lat, axis=0)), reverse=True)
        hgt = hh
        note = "head end of the long axis (%.2f m slab; rough on full bodies - shoulders can widen it, most reliable on head-only meshes)" % hh
    ws, ds, ps = PRESETS.get(preset, PRESETS["neutral"])
    want_w, want_d = HEAD_NEUTRAL["w_over_h"] * ws, HEAD_NEUTRAL["d_over_h"] * ds
    got_w, got_d = w / hgt, d / hgt
    rep = dict(preset=preset, note=note, width_over_height=got_w, depth_over_height=got_d,
               expected_width=want_w, expected_depth=want_d,
               width_off_pct=100 * (got_w / want_w - 1), depth_off_pct=100 * (got_d / want_d - 1))
    rep["verdict"] = "OK (within 8%)" if max(abs(rep["width_off_pct"]), abs(rep["depth_off_pct"])) <= 8 else "differs from the %s preset by more than 8 percent" % preset
    return rep


# (5) safe install: validate, keep a first-time .bak of anything replaced, copy, verify by md5
def safe_install(mdl_path, dest_dir):
    import hashlib
    errs, warns = validate_model(mdl_path)
    if errs: raise RuntimeError("not installed - validation failed:\n" + "\n".join(errs))
    base = os.path.splitext(mdl_path)[0]
    out = []
    for src in (base + ".mdl", base + ".mdx"):
        dst = os.path.join(dest_dir, os.path.basename(src))
        data = open(src, "rb").read()
        if os.path.exists(dst) and not os.path.exists(dst + ".bak"):
            shutil.copyfile(dst, dst + ".bak")
        with open(dst, "wb") as f: f.write(data)
        got = hashlib.md5(open(dst, "rb").read()).hexdigest()
        if got != hashlib.md5(data).hexdigest(): raise RuntimeError("md5 mismatch after copying " + dst)
        out.append((dst, got))
    return out


# ----------------------------------------------------------------------------- GUI

REGION_MODES = ("Smoother (crease 75\u00b0)", "Leave alone", "No displacement")


def find_diffuse(model_path, tex):
    base = os.path.dirname(os.path.abspath(model_path))
    for d in (base, os.path.join(base, ".."), os.path.join(base, "..", "Override")):
        for n in (tex, tex.lower(), tex.upper()):
            for e in (".tga", ".tpc", ".TGA", ".TPC"):
                p = os.path.join(d, n + e)
                if os.path.exists(p): return os.path.normpath(p)
    return None


def region_arrays(m, regions, crease_deg):
    """Per-face arrays from UV rectangles [(u0, v0, u1, v1, mode_index)] (face centroid decides membership)."""
    if not regions or m.uv is None: return None
    c = m.uv[m.faces].mean(1)
    crease = np.full(len(m.faces), float(crease_deg)); keep = np.zeros(len(m.faces), bool); nodisp = np.zeros(len(m.faces), bool)
    for (u0, v0, u1, v1, mode) in regions:
        inside = (c[:, 0] >= min(u0, u1)) & (c[:, 0] <= max(u0, u1)) & (c[:, 1] >= min(v0, v1)) & (c[:, 1] <= max(v0, v1))
        if mode == 0: crease[inside] = np.maximum(crease[inside], 75.0)
        elif mode == 1: keep |= inside
        else: nodisp |= inside
    return dict(crease=crease, keep=keep, nodisp=nodisp)


def run_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from PIL import ImageTk

    root = tk.Tk()
    root.title("KotOR Mesh Upscaler 3.2")
    root.geometry("1240x1060")
    hcache = {}
    S = dict(model=None, up={}, cam=Camera(), img=None, drag=None)

    left = ttk.Frame(root, padding=8); left.pack(side="left", fill="y")
    cv = tk.Canvas(root, bg="#181a1e", highlightthickness=0); cv.pack(side="right", fill="both", expand=True)

    pathv = tk.StringVar()
    ttk.Label(left, text="Model (.mdl, .mdx next to it)").pack(anchor="w")
    ttk.Entry(left, textvariable=pathv, width=44).pack(anchor="w")
    ttk.Button(left, text="Open...", command=lambda: open_file()).pack(anchor="w", pady=4)

    ttk.Label(left, text="Meshes (ctrl/shift-click to choose what to upscale; click \u25c9 to show / hide)").pack(anchor="w", pady=(8, 0))
    tv = ttk.Treeview(left, columns=("eye", "v", "f", "d", "tex"), show="headings", height=8, selectmode="extended")
    for c, w_ in (("eye", 30), ("v", 44), ("f", 44), ("d", 40), ("tex", 116)):
        tv.heading(c, text={"eye": "show", "v": "verts", "f": "faces", "d": "disp", "tex": "name / texture"}[c]); tv.column(c, width=w_, anchor="center" if c == "eye" else "w")
    EYE_ON, EYE_OFF = "\u25c9", "\u25cb"            # filled / empty circle (plain BMP characters: Tk 8.6 cannot draw emoji)
    def eye_click(e):
        # a click in the 'show' column toggles that mesh in the viewer and leaves the selection alone
        if tv.identify_region(e.x, e.y) != "cell" or tv.identify_column(e.x) != "#1": return None
        row = tv.identify_row(e.y)
        if not row: return "break"
        i = int(row)
        hid = S.setdefault("hidden", set())
        if i in hid: hid.discard(i)
        else: hid.add(i)
        vals = list(tv.item(row, "values")); vals[0] = EYE_OFF if i in hid else EYE_ON; tv.item(row, values=vals)
        draw()
        return "break"
    tv.bind("<Button-1>", eye_click, add="+")
    def set_all_shown(show):
        S["hidden"] = set() if show else set(int(r) for r in tv.get_children())
        for r in tv.get_children():
            vals = list(tv.item(r, "values")); vals[0] = EYE_ON if show else EYE_OFF; tv.item(r, values=vals)
        draw()
    tv.pack(anchor="w", fill="x")
    S["dmesh"] = {}; S["regions"] = {}
    def set_disp(on):
        for i in selected():
            S["dmesh"][i] = on
            vals = list(tv.item(str(i), "values")); vals[3] = "yes" if on else "no"; tv.item(str(i), values=vals)
    brow = ttk.Frame(left); brow.pack(fill="x", pady=2)
    ttk.Button(brow, text="Displace: on", command=lambda: set_disp(True)).pack(side="left", fill="x", expand=True, padx=(0, 2))
    ttk.Button(brow, text="Displace: off", command=lambda: set_disp(False)).pack(side="left", fill="x", expand=True, padx=2)
    ttk.Button(brow, text="UV regions...", command=lambda: region_editor()).pack(side="left", fill="x", expand=True, padx=(2, 0))
    srow = ttk.Frame(left); srow.pack(fill="x", pady=(0, 2))
    ttk.Button(srow, text="Show all", command=lambda: set_all_shown(True)).pack(side="left", fill="x", expand=True, padx=(0, 2))
    ttk.Button(srow, text="Hide all", command=lambda: set_all_shown(False)).pack(side="left", fill="x", expand=True, padx=(2, 0))

    def slider(label, lo, hi, init, res, fmt):
        var = tk.DoubleVar(value=init)
        f = ttk.Frame(left); f.pack(fill="x", pady=2)
        lab = ttk.Label(f, text=label, width=22); lab.pack(side="left")
        out = ttk.Label(f, text=fmt(init), width=6); out.pack(side="right")
        sc = ttk.Scale(f, from_=lo, to=hi, variable=var, command=lambda _v: out.config(text=fmt(var.get())))
        sc.pack(side="left", fill="x", expand=True)
        return var
    lvl = slider("Smoothness quality", 1, 6, 3, 1, lambda v: "%.1f" % v)
    mvv = slider("Max verts / mesh (k)", 2, 60, 40, 1, lambda v: "%dk" % v)
    fbv = slider("Face budget, total (k)", 0, 60, 0, 0.5, lambda v: "off" if v < 0.25 else "%.1fk" % (round(v * 2) / 2))
    crs = slider("Crease angle (deg)", 10, 90, 40, 1, lambda v: "%d" % v)
    stg = slider("Curvature strength", 0, 1, 0.85, 0.01, lambda v: "%.2f" % v)
    ttk.Separator(left).pack(fill="x", pady=4)
    dispv = tk.BooleanVar(value=False)
    ttk.Checkbutton(left, text="Add real geometry detail from normal map (displacement)", variable=dispv).pack(anchor="w")
    nmv = tk.StringVar(value="")
    nmrow = ttk.Frame(left); nmrow.pack(fill="x")
    ttk.Entry(nmrow, textvariable=nmv, width=34).pack(side="left")
    ttk.Button(nmrow, text="...", width=3, command=lambda: nmv.set(filedialog.askopenfilename(
        filetypes=[("Texture", "*.tga *.tpc")]) or nmv.get())).pack(side="left", padx=2)
    ttk.Label(left, text="(blank = auto-find <texture>_n.tga/.tpc, also in pbr/)", foreground="#777").pack(anchor="w")
    dst = slider("Displacement strength", 0, 3, 1.0, 0.05, lambda v: "%.2f" % v)
    dtx = slider("Relief sensitivity", 1, 10, 5, 1, lambda v: "%d" % v)
    invy = tk.BooleanVar(value=False); flv = tk.BooleanVar(value=False)
    ff_ = ttk.Frame(left); ff_.pack(anchor="w")
    ttk.Checkbutton(ff_, text="Invert normal Y", variable=invy).pack(side="left")
    ttk.Checkbutton(ff_, text="Flip texture V", variable=flv).pack(side="left", padx=8)

    carv = tk.BooleanVar(value=False)
    ttk.Checkbutton(left, text="Carve cavities from diffuse (diffuse + normal map together)", variable=carv).pack(anchor="w")
    cdp = slider("Cavity depth (mm)", 0, 6, 2.0, 0.1, lambda v: "%.1f" % v)
    ttk.Button(left, text="Check textures / UVs...", command=lambda: uv_check()).pack(fill="x", pady=2)
    prow = ttk.Frame(left); prow.pack(fill="x", pady=2)
    ttk.Label(prow, text="Anatomy preset").pack(side="left")
    presv = tk.StringVar(value="neutral")
    ttk.Combobox(prow, textvariable=presv, values=sorted(PRESETS), width=20, state="readonly").pack(side="left", padx=4)
    ttk.Button(prow, text="Check proportions", command=lambda: prop_check()).pack(side="left")
    vrow = ttk.Frame(left); vrow.pack(fill="x", pady=2)
    ttk.Button(vrow, text="Validate model", command=lambda: val_check()).pack(side="left", fill="x", expand=True, padx=(0, 2))
    ttk.Button(vrow, text="Safe install to Override...", command=lambda: do_install()).pack(side="left", fill="x", expand=True, padx=(2, 0))

    viewv = tk.StringVar(value="upscaled")
    vf = ttk.Frame(left); vf.pack(anchor="w", pady=4)
    for t in ("original", "upscaled", "split"):
        ttk.Radiobutton(vf, text=t, value=t, variable=viewv, command=lambda: draw()).pack(side="left", padx=3)
    wirev = tk.BooleanVar(value=False)
    ttk.Checkbutton(left, text="Wireframe", variable=wirev, command=lambda: draw()).pack(anchor="w")
    info = tk.StringVar(value="Open a model.")
    ttk.Label(left, textvariable=info, wraplength=330, justify="left").pack(anchor="w", pady=6)
    ttk.Button(left, text="Preview upscale", command=lambda: do_up()).pack(fill="x", pady=2)
    ttk.Button(left, text="Save upscaled model...", command=lambda: save()).pack(fill="x", pady=2)
    fb = ttk.Frame(left); fb.pack(fill="x", pady=2)
    ttk.Button(fb, text="Export FBX...", command=lambda: export_fbx()).pack(side="left", fill="x", expand=True, padx=(0, 2))
    ttk.Button(fb, text="Import FBX...", command=lambda: import_fbx()).pack(side="left", fill="x", expand=True, padx=(2, 0))
    ttk.Label(left, text="Drag: orbit   Right-drag: pan   Wheel: zoom\nSave makes a .bak copy of any file it overwrites.",
              foreground="#777").pack(anchor="w", pady=6)

    def selected():
        sel = [int(i) for i in tv.selection()]
        return sel

    def open_file(p=None):
        p = p or filedialog.askopenfilename(filetypes=[("KotOR model", "*.mdl")])
        if not p: return
        try:
            mdl = Model(p)
        except Exception as e:
            messagebox.showerror("Open failed", str(e)); return
        S["model"], S["up"] = mdl, {}
        S["dmesh"], S["regions"], S["hidden"] = {}, {}, set()
        pathv.set(p)
        tv.delete(*tv.get_children())
        pre = []
        for i, m in enumerate(mdl.meshes):
            S["dmesh"][i] = bool(m.fc >= 100)          # small meshes (eyes, teeth, lids) default to no displacement
            tv.insert("", "end", iid=str(i), values=(EYE_ON, m.vc, m.fc, "yes" if S["dmesh"][i] else "no", "%s  [%s]%s" % (m.name, m.tex, "" if m.render else "  (hidden)")))
            if m.ok and m.render and m.tex.upper() not in ("NULL", ""): pre.append(str(i))
        tv.selection_set(pre)
        allp = np.concatenate([mdl.world_pos(m) for m in mdl.meshes if m.render and hasattr(m, "pos") and len(m.pos)])
        S["cam"].fit(allp)
        nro = sum(1 for m in mdl.meshes if not m.ok and "vertex channels" in m.why)
        info.set("%d meshes; %d selected (textured, renderable).%s" % (len(mdl.meshes), len(pre),
                 ("  %d shown grey can't be edited yet (extra vertex data, e.g. lightmapped room parts)." % nro) if nro else ""))
        draw()

    def region_editor():
        mdl = S["model"]
        sel = selected()
        if not mdl or not sel:
            messagebox.showinfo("UV regions", "Open a model and select a mesh first."); return
        i = sel[0]; m = mdl.meshes[i]
        if m.uv is None:
            messagebox.showinfo("UV regions", "This mesh has no UVs."); return
        win = tk.Toplevel(root); win.title("UV regions - %s" % m.name)
        Z = 640
        cvu = tk.Canvas(win, width=Z, height=Z, bg="#101010", highlightthickness=0); cvu.pack(side="left")
        side = ttk.Frame(win, padding=8); side.pack(side="left", fill="y")
        regs = S["regions"].setdefault(i, [])
        bg = {"img": None}
        tp = find_diffuse(mdl.mdl_path, m.tex)
        from PIL import Image
        if tp:
            try:
                a = load_texture(tp)[::-1, :, :3]
                bg["img"] = ImageTk.PhotoImage(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((Z, Z)))
            except Exception:
                bg["img"] = None
        cen = m.uv[m.faces].mean(1)
        drag = {}
        def U(x): return x / Z
        def V(y): return 1.0 - y / Z
        def redraw():
            cvu.delete("all")
            if bg["img"]: cvu.create_image(0, 0, anchor="nw", image=bg["img"])
            tri = m.uv[m.faces]
            inreg = np.zeros(len(m.faces), bool)
            for (u0, v0, u1, v1, mode) in regs:
                inreg |= (cen[:, 0] >= min(u0, u1)) & (cen[:, 0] <= max(u0, u1)) & (cen[:, 1] >= min(v0, v1)) & (cen[:, 1] <= max(v0, v1))
            for f in range(len(tri)):
                pts = [(tri[f, k, 0] * Z, (1 - tri[f, k, 1]) * Z) for k in range(3)]
                cvu.create_polygon(pts, outline="#ffcc44" if inreg[f] else "#66aaff", fill="#ffcc4440" if False else "", width=1)
            for (u0, v0, u1, v1, mode) in regs:
                cvu.create_rectangle(u0 * Z, (1 - v0) * Z, u1 * Z, (1 - v1) * Z, outline=("#44ff88", "#ff5555", "#ffaa00")[mode], width=2)
            if "r" in drag:
                x0, y0, x1, y1 = drag["r"]; cvu.create_rectangle(x0, y0, x1, y1, outline="#ffffff", dash=(4, 3))
            lst.delete(0, "end")
            for r in regs: lst.insert("end", "%s  u %.2f-%.2f  v %.2f-%.2f" % (REGION_MODES[r[4]].split(" (")[0], min(r[0], r[2]), max(r[0], r[2]), min(r[1], r[3]), max(r[1], r[3])))
            n.config(text="%d of %d faces in a region" % (int(inreg.sum()), len(inreg)))
        def down(e): drag["p"] = (e.x, e.y); drag["r"] = (e.x, e.y, e.x, e.y)
        def mv(e):
            if "p" in drag: drag["r"] = (drag["p"][0], drag["p"][1], e.x, e.y); redraw()
        def up(e):
            if "r" in drag and abs(drag["r"][2] - drag["r"][0]) > 3 and abs(drag["r"][3] - drag["r"][1]) > 3:
                drag["done"] = drag["r"]
            drag.pop("p", None); redraw()
        cvu.bind("<Button-1>", down); cvu.bind("<B1-Motion>", mv); cvu.bind("<ButtonRelease-1>", up)
        ttk.Label(side, text="Drag a rectangle on the UV map,\nchoose what to do, then 'Add region'.", justify="left").pack(anchor="w")
        modev = tk.IntVar(value=0)
        for k, nm in enumerate(REGION_MODES):
            ttk.Radiobutton(side, text=nm, value=k, variable=modev).pack(anchor="w")
        def add():
            if "done" not in drag:
                messagebox.showinfo("UV regions", "Drag a rectangle first."); return
            x0, y0, x1, y1 = drag.pop("done"); drag.pop("r", None)
            regs.append((U(x0), V(y0), U(x1), V(y1), modev.get())); redraw()
        ttk.Button(side, text="Add region", command=add).pack(fill="x", pady=4)
        lst = tk.Listbox(side, height=8, width=34); lst.pack(fill="x")
        def rm():
            for k in reversed(lst.curselection()): regs.pop(k)
            redraw()
        ttk.Button(side, text="Remove selected", command=rm).pack(fill="x", pady=2)
        ttk.Button(side, text="Clear all", command=lambda: (regs.clear(), redraw())).pack(fill="x")
        n = ttk.Label(side, text=""); n.pack(anchor="w", pady=6)
        ttk.Label(side, text="Smoother: hair/neck-like areas get a\nhigher crease angle so rough edges round off.\nLeave alone: never subdivided or displaced.\nBlue = UV triangles, yellow = in a region.", foreground="#777", justify="left").pack(anchor="w")
        ttk.Button(side, text="Done", command=win.destroy).pack(fill="x", pady=8)
        redraw()

    def uv_check():
        mdl = S["model"]
        if not mdl: return
        lines = []
        for i in (selected() or range(len(mdl.meshes))):
            m = mdl.meshes[i]
            if not m.ok or m.uv is None: continue
            tp = find_diffuse(mdl.mdl_path, m.tex)
            px = 2048
            if tp:
                try: px = int(load_texture(tp).shape[1])
                except Exception: pass
            r, fl = uv_report(m, px)
            lines.append("%s: %d flipped, %d degenerate, %d stretched, %d density outliers, %.2f mm/px" % (
                m.name, r["flipped"], r["degenerate"], r["stretched"], r["density_outliers"], r["median_mm_per_px"]))
            if r["flipped"] + r["degenerate"] and messagebox.askyesno("Repair UVs", "%s has %d flipped/degenerate UV faces.\nRe-solve them for the upscale?" % (m.name, r["flipped"] + r["degenerate"])):
                m.uv, n = uv_repair(m, fl); lines[-1] += "  (re-solved %d vertices)" % n
        info.set("\n".join(lines) or "No UV meshes selected.")

    def prop_check():
        mdl = S["model"]
        if not mdl: return
        lines = []
        for i in selected():
            m = mdl.meshes[i]
            if not m.ok: continue
            r = proportion_report(m, presv.get())
            lines.append("%s [%s]: width/height %.3f (want %.3f, %+.0f%%), depth/height %.3f (want %.3f, %+.0f%%) - %s. Based on the %s." % (
                m.name, r["preset"], r["width_over_height"], r["expected_width"], r["width_off_pct"],
                r["depth_over_height"], r["expected_depth"], r["depth_off_pct"], r["verdict"], r["note"]))
        info.set("\n".join(lines) or "Select a mesh first.")

    def val_check():
        mdl = S["model"]
        if not mdl: return
        try:
            errs, warns = validate_model(mdl.mdl_path)
        except Exception as e:
            info.set("Validate failed: %s" % e); return
        info.set(("FAIL: " + "; ".join(errs) if errs else "OK - no crash or load problems found.") + ("  Warnings: " + "; ".join(warns) if warns else ""))

    def do_install():
        mdl = S["model"]
        if not mdl: return
        dest = filedialog.askdirectory(title="Game Override folder to install into")
        if not dest: return
        try:
            res = safe_install(mdl.mdl_path, dest)
        except Exception as e:
            messagebox.showerror("Install failed", str(e)); return
        info.set("Installed (originals kept as .bak, md5 verified):\n" + "\n".join("%s  %s" % (os.path.basename(p), h[:10]) for p, h in res) + "\nFully restart the game.")

    def do_up():
        mdl = S["model"]
        if not mdl: return
        S["up"] = {}
        tot0 = tot1 = 0
        budget = int(round(fbv.get() * 2) / 2 * 1000)
        areas = {}
        for i in selected():
            mm = mdl.meshes[i]
            if mm.ok:
                Pp = mm.pos[mm.faces]
                areas[i] = float(0.5 * np.linalg.norm(np.cross(Pp[:, 1] - Pp[:, 0], Pp[:, 2] - Pp[:, 0]), axis=1).sum())
        atot = sum(areas.values()) or 1.0
        for i in selected():
            m = mdl.meshes[i]
            if not m.ok:
                continue
            try:
                dsp = None
                if dispv.get() and m.uv is not None and S["dmesh"].get(i, True):
                    key = (nmv.get() or m.tex, invy.get())
                    if key not in hcache:
                        p = nmv.get() or find_normal_map(S["model"].mdl_path, m.tex)
                        if not p or not os.path.exists(p):
                            raise RuntimeError("no normal map found for texture '%s' - pick one with the '...' button" % m.tex)
                        info.set("Integrating normal map %s ..." % os.path.basename(p)); root.update()
                        nmimg = load_texture(p)
                        mr, mg, mb = (float(nmimg[..., k].mean()) for k in range(3))
                        if abs(mr - 0.5) > 0.12 or abs(mg - 0.5) > 0.12 or mb < 0.7:
                            raise RuntimeError("%s doesn't look like a normal map (average colour %.2f/%.2f/%.2f, a normal map is about 0.5/0.5/1.0). "
                                               "Clear the box to auto-find <texture>_n, or pick the _n.tga file." % (os.path.basename(p), mr, mg, mb))
                        hcache[key] = height_from_normals(nmimg, invert_y=invy.get())
                    hmap_ = hcache[key]
                    if carv.get():
                        dp_ = find_diffuse(S["model"].mdl_path, m.tex)
                        if not dp_: raise RuntimeError("no diffuse texture '%s' found for carving" % m.tex)
                        ck_ = ("carve", key, dp_, round(cdp.get(), 2))
                        if ck_ not in hcache:
                            tmp_ = make_disp(m, hmap_, 1.0, flv.get(), 5)
                            hcache[ck_] = carve_height(load_texture(dp_), hmap_, float(np.median(tmp_["wpf"])), cdp.get())
                        hmap_ = hcache[ck_]
                    dsp = make_disp(m, hmap_, dst.get(), flv.get(), dtx.get())
                info.set("Upscaling %s ..." % m.name); root.update()
                tf = max(int(budget * areas.get(i, 0) / atot), m.fc) if budget > 0 else None
                S["up"][i] = upscale(m, lvl.get(), crs.get(), stg.get(), dsp, int(mvv.get() * 1000),
                                     region=region_arrays(m, S["regions"].get(i), crs.get()), target_faces=tf)
            except Exception as e:
                messagebox.showerror("Upscale failed", "%s: %s" % (m.name, e)); return
            tot0 += m.fc; tot1 += len(S["up"][i]["faces"])
            root.update()
        info.set("Upscaled %d meshes: %d -> %d faces (%.1fx)." % (len(S["up"]), tot0, tot1, tot1 / max(tot0, 1)))
        viewv.set("split" if viewv.get() == "original" else viewv.get())
        draw()

    def items(which):
        mdl = S["model"]
        out = []
        for i, m in enumerate(mdl.meshes):
            if i in S.get("hidden", ()): continue          # hidden with its eye in the mesh list
            if not m.ok:
                # meshes this version can't edit (lightmapped room parts with extra vertex data) are still shown,
                # plain grey, so a room or building is seen whole
                if m.render and hasattr(m, "pos") and hasattr(m, "faces") and len(m.faces) and m.faces.max() < len(m.pos):
                    out.append((mdl.world_pos(m), None if m.norm is None else _xf_n(m, m.norm), m.faces, (105, 110, 120)))
                continue
            if (not m.render and i not in S["up"]) or (m.tex.upper() in ("NULL", "") and i not in S["up"]): continue
            col = (200, 190, 175) if i in S["up"] else (150, 160, 175)
            if which == "up" and i in S["up"]:
                u = S["up"][i]
                out.append((mdl.world_pos(m, u["pos"]), None if u["norm"] is None else _xf_n(m, u["norm"]), u["faces"], col))
            else:
                out.append((mdl.world_pos(m), None if m.norm is None else _xf_n(m, m.norm), m.faces, col))
        return out

    def _xf_n(m, n):
        return n @ m.world[:3, :3].T

    def draw():
        if not S["model"]: return
        W, H = max(cv.winfo_width(), 200), max(cv.winfo_height(), 200)
        mode = viewv.get()
        if mode == "split":
            a = render(items("orig"), S["cam"], W // 2, H, wirev.get())
            b = render(items("up"), S["cam"], W - W // 2, H, wirev.get())
            from PIL import Image
            img = Image.new("RGB", (W, H)); img.paste(a, (0, 0)); img.paste(b, (W // 2, 0))
        else:
            img = render(items("up" if mode == "upscaled" else "orig"), S["cam"], W, H, wirev.get())
        S["img"] = ImageTk.PhotoImage(img)
        cv.delete("all"); cv.create_image(0, 0, anchor="nw", image=S["img"])

    def press(e, mode): S["drag"] = (e.x, e.y, mode)
    def move(e):
        if not S["drag"]: return
        x0, y0, mode = S["drag"]; dx, dy = e.x - x0, e.y - y0
        c = S["cam"]
        if mode == "orbit":
            c.yaw += dx * 0.01; c.pitch = max(-1.5, min(1.5, c.pitch + dy * 0.01))
        else:
            sc = min(cv.winfo_width(), cv.winfo_height()) * 0.95 / c.size * c.dist
            c.pan += np.array([dx, -dy]) / sc / c.size
        S["drag"] = (e.x, e.y, mode); draw()
    def wheel(e):
        d = e.delta if e.delta else (120 if e.num == 4 else -120)
        S["cam"].dist = max(0.1, min(30, S["cam"].dist * (1.1 if d > 0 else 1 / 1.1))); draw()
    cv.bind("<ButtonPress-1>", lambda e: press(e, "orbit"))
    cv.bind("<ButtonPress-3>", lambda e: press(e, "pan"))
    cv.bind("<B1-Motion>", move); cv.bind("<B3-Motion>", move)
    cv.bind("<MouseWheel>", wheel); cv.bind("<Button-4>", wheel); cv.bind("<Button-5>", wheel)
    cv.bind("<Configure>", lambda e: draw())

    def export_fbx():
        mdl = S["model"]
        if not mdl: return
        sel = selected() or [i for i, m in enumerate(mdl.meshes) if m.ok and m.render]
        objs = []
        for i in sel:
            m = mdl.meshes[i]
            if not m.ok: continue
            u = S["up"].get(i)
            pos = m.pos if u is None else u["pos"]
            nrm = m.norm if u is None else u["norm"]
            uv = m.uv if u is None else u["uv"]
            F = m.faces if u is None else u["faces"]
            objs.append(dict(name=m.name, pos=mdl.world_pos(m, pos), faces=F,
                             norm=None if nrm is None else _xf_n(m, nrm), uv=uv))
        if not objs: return
        out = filedialog.asksaveasfilename(defaultextension=".fbx", initialfile=mdl.name + ".fbx",
                                           filetypes=[("FBX binary", "*.fbx")])
        if not out: return
        try:
            write_fbx(out, objs)
        except Exception as e:
            messagebox.showerror("Export failed", str(e)); return
        info.set("Exported %d mesh(es) to %s  (Z up, metres; edit in Blender and re-import)." % (len(objs), os.path.basename(out)))

    def import_fbx():
        mdl = S["model"]
        if not mdl: return
        p = filedialog.askopenfilename(filetypes=[("FBX binary", "*.fbx")])
        if not p: return
        try:
            objs = read_fbx(p)
        except Exception as e:
            messagebox.showerror("Import failed", str(e)); return
        norm = lambda n: re.sub(r"\.\d+$", "", n).lower()
        byname = {}
        for i, m in enumerate(mdl.meshes):
            if m.ok: byname.setdefault(m.name.lower(), i)
        done, skipped = [], []
        for o in objs:
            i = byname.get(norm(o["name"]))
            if i is None: skipped.append(o["name"]); continue
            info.set("Importing %s (transferring skin weights) ..." % o["name"]); root.update()
            try:
                S["up"][i] = fbx_to_mesh(mdl.meshes[i], o)
            except Exception as e:
                messagebox.showerror("Import failed", "%s: %s" % (o["name"], e)); return
            done.append(o["name"])
        viewv.set("split")
        info.set("Imported %s.%s" % (", ".join(done) or "nothing", ("  Not matched: " + ", ".join(skipped)) if skipped else ""))
        draw()

    def save():
        mdl = S["model"]
        if not mdl or not S["up"]:
            messagebox.showinfo("Nothing to save", "Run 'Preview upscale' first."); return
        out = filedialog.asksaveasfilename(defaultextension=".mdl", initialfile=os.path.basename(mdl.mdl_path),
                                           initialdir=os.path.dirname(mdl.mdl_path), filetypes=[("KotOR model", "*.mdl")])
        if not out: return
        try:
            fresh = Model(mdl.mdl_path)          # always patch a pristine copy
            for i, u in S["up"].items():
                fresh.apply(fresh.meshes[i], u)
            fresh.save(out)
            errs, warns = validate_model(out)
            if errs: raise RuntimeError("validation failed:\n" + "\n".join(errs))
            verify = Model(out)
            bad = [m.name for m in verify.meshes if m.ok and (m.faces.max() >= m.vc)]
            if bad: raise RuntimeError("verification failed: " + ", ".join(bad))
        except Exception as e:
            messagebox.showerror("Save failed", str(e)); return
        messagebox.showinfo("Saved", "Wrote %s and its .mdx (originals backed up as .bak)." % out)

    if len(sys.argv) > 1: root.after(100, lambda: open_file(sys.argv[1]))
    root.mainloop()


if __name__ == "__main__":
    run_gui()
