# PhyRC 2027 - policy: 2 robot cung gap 2 ben GAU AO, xoay ao, dua toi 2 tay nguoi nom va keo ao vao
# Chi dung obs duoc phep: rgb 5 camera, joint_position, base_pose_world cua chinh robot, simulation_time_s.
# Khong doc vi tri ao / nguoi tu mo phong, khong chay lai quy dao demo.
import os, time, zlib, struct
import numpy as np

# ============================== ANH DEBUG ==============================
SNAP = False                     # dat False truoc khi nop bai
SNAP_DIR = "/output/anh"
VID_FPS = 4.0                   # quay "video" 5 camera (anh PNG 4 khung/giay) khi SNAP=True; 0 = tat
SNAP_EVERY = 10.0

# ============================== GAP AO ==============================
# Do tu anh camera dau trong 3 demo luc gap (tam vung xanh cua ao):
#   robot_0 thay ao o cot ~115, hang ~177 ; robot_1 thay ao o cot ~178, hang ~182
HEAD_T = ((125, 177), (178, 182))   # r0 125 (truoc 115): anh 22h28 kep robot_0 (ben phai) lech ra ngoai ~4 cm
LIFT_T = 0.54
# cong thuc gap rieng tung robot, lay tu dung lan gap cua robot do trong demo:
# (lift, R, wrist_yaw, wrist_pitch, wrist_roll) ; tay vuon = R + y_robot
LOW = (0.39, 0.94, 0.01, -0.03, -0.04)   # demo1 robot_0 luc gap: tay ha sat ban, co tay thang
RECIPES = (
    [LOW,
     (0.537, 0.94, -0.327,  0.287,  2.761),   # demo2 robot_0
     (0.536, 0.88,  0.001,  0.010, -0.835),   # demo3 robot_0
     (0.536, 0.93, -0.275,  0.283,  0.300)],
    [LOW,
     (0.536, 0.99, -0.275,  0.283,  0.368),   # demo2 robot_1
     (0.460, 0.89,  0.001, -0.266,  0.220),   # demo1 robot_1
     (0.600, 0.98,  0.000,  0.020,  1.940)],  # demo3 robot_1
)
# camera co tay: 2 dau ngon kep nam quanh hang 155, cot 55 va 200 -> o cua so nay phai thay GAU (vang)
TIP_WIN = (138, 178, 45, 212)   # hang tu..den, cot tu..den: dai ngay duoi 2 dau ngon (anh sm19: dau ngon o hang ~140)
KEP_LOST = 0.106                # dang giu ma ngon khep het muc nay = tuot ao
LIFT_UP, HOLD_S = 0.12, 3.5
GRAB_HOLD = 4.0                # WAIT: phai dung yen >= 4 s moi nang (env chi dinh ao vao kep sau GRAB_SETTLE_FRAMES=60 khung kep dung yen)
DRAG_BACK = 0.0                 # gap xong thu tay ve 3 cm luc nang
HOVER_LIFT = 0.55               # do cao an toan khi vuon tay tren ban (sm19-21 vuon o 0.537 khong va ban)
FIND_UP = 0.05                  # DIRECT: do cao tim gau = do cao gap + 5 cm (khong cham vai)
CORNER_U = [148.0, 150.0]       # cot tren camera dau ma kep cua robot nam (anh 14h29) -> dua goc gau ao ve day
FIND_MAX = 0.32                 # rut tay toi da 18 cm khi tim gau
SLIDE_IN, SLIDE_V, SLIDE_MAX = 0.03, 0.001, 0.15
KEO_N = [0.0, 0.0]             # quang keo ngang RIENG tung robot (c7: R1 keo 5.2 cm thi vuot qua goc gau ra mep suon ao)
KEO_NGANG, KEO_V = 0.0, 0.10    # >0: (cach cua thay) ngon da khieu vao khe gau -> moi tay keo RA NGOAI KEO_NGANG m (toi 2 goc gau) roi moi kep
SEQ_PUSH, SEQ_MAX = False, 20.0  # True: R1 dung cho (ngon duoi sat ban) toi khi R0 luon khe + keo ngang xong moi day; cho toi da SEQ_MAX s
KEP_EP = 0.0                   # 07/11 e5: >0: do mo ngon jp[11] < KEP_EP khi ha/day = MAT BAN EP NGON lam kep dong (R0 0.50->0.16) -> coi la cham, nang tay
FIND_XA = 9.0                  # 07/11: >0 nho: FIND thay mep gau con XA hon SLIDE_GAP + FIND_XA thi VUON TOI TU TU (dung voi arm_ofs am: REACH dung truoc gau, khong vuon qua ao)
REACH_V, SLIDE_T, NAV_TOL = 0.5, 20.0, 8.0   # 07/11: toc do vuon REACH (m/s), thoi gian day toi da (s), sai so canh goc gau NAV (px)
GIU_THAN = 0.0                 # >0: khi day vao khe, banh xe day than TOI bu phan bi lui (lenh = GIU_THAN * than_lui) -> ngon co luc chen vao
DRESS_STEPS = 0                # >0: chi lam N buoc dau cua KF (vd 2 = RA_CHO_TRONG + NANG_LEN) roi DUNG YEN GIU AO (khong tha)
SLIDE_LUI = 0.03                # SLIDE: than bi day lui qua muc nay (m) thi dung day
SLIDE_H = 0.004                 # SLIDE: dau ngon cach mat ban 4 mm khi truot (khong ep xuong ban)
SLIDE_GAP, D_TIP = 0.02, 0.20       # SLIDE: mep gau phai nam truoc dau ngon >= 2 cm; D_TIP = do sau dau ngon neu khong do duoc
FIND_FWD = 0.25                 # khi KHONG thay gau: vuon them toi da 25 cm (ban dau chi rut ve)
ARM_OFS = 0.0                   # vuon them so voi demo (m)
TABLE_Z = 0.57                  # mat ban (BOX_TOP_Z = 0.6*0.95, code moi truong cua BTC)
FT_CLEAR = 0.006                # dung ha khi dau ngon cach mat ban 6 mm (fingertip_position_base)
DLIFT = 0.21                    # ha them so voi do cao gap cua demo 0.39 (m); am = thap hon
TILT_HARD = 4.0                 # che do DIRECT: de nghieng them qua 4 do thi dung ha (an toan)
DOWN_V = 0.02                   # ha kep 2 cm/s cho toi khi cham vai/mat ban
CONTACT_UP = -0.02              # cham ban roi nang len bao nhieu truoc khi dong kep
GRASP_ONLY = False              # True khi toi uu tham so: chi gap + giu, khong mac ao
KQ_TIME = 75.0                  # che do toi uu: qua thoi diem nay thi in ket qua du chua gap xong
SCAN_V, SCAN_L, SCAN_R, FWD0, SEE_N = 0.8, 5.0, 5.0, 0.4, 1000
R1_OFFSET_X = 0.47              # demo: robot_1 dung ben +x robot_0 ~0.47 m luc gap
CACH_R1 = 0.38                  # R0 khong lai gan R1 hon (m, theo x) khi R1 dang di theo R0 (chong dam nhau)
READY_WAIT = 15.0
PAIR_WAIT = 120.0
RATE = dict(lift=1.54, arm=1.21)

# ============================== MAC AO ==============================
# Nguoi nom ngoi o khoang x=PX, quay mat ve -y, 2 tay gio ve phia truoc (-y).
# Cac diem dung duoi day do doi tu thiet ke (doi xung quanh nguoi), tham khao cach lai trong demo.
PX = 0.03                       # nguoi nom ngoi o x ~ PX, y ~ 0.45, quay mat ve -y, 2 tay gio ve -y
C_GRIP = 0.45                   # kep nam truoc tam de robot (arm + C_GRIP) m theo huong mat robot
                                # (kiem tra: voi so nay, goc xoay than tinh ra -104/-76 do, demo1 do duoc -96..-109 / -72..-81)
STAGE = (PX, -0.38)             # cho trong de xoay ao
# Moi buoc:
#  ("pair", cx, cy, goc_xoay_do, lift0, lift1, arm0, arm1, gio): 2 de robot cach nhau dung bang luc gap (S0),
#        than quay thang -y; goc 0 = robot_0 ben -x, 180 = 2 robot da doi cho (ao da xoay 180 do)
#  ("side", y_de, x_de, lift0, lift1, arm0, arm1, gio): 2 de robot dung 2 ben nguoi (PX +- x_de),
#        than xoay vao trong sao cho 2 kep van cach nhau S0 (khong keo cang ao)
KF = [
    ("RA_CHO_TRONG", "pair", STAGE[0], STAGE[1],   0, 0.46, 0.46, 0.02, 0.02, 40),
    ("NANG_LEN",     "pair", STAGE[0], STAGE[1],   0, 0.70, 0.70, 0.02, 0.02, 10),   # ao lo lung, khong cham san
    ("XOAY_1",      "pair", STAGE[0], STAGE[1],  22, 0.70, 0.70, 0.02, 0.02, 10),
    ("XOAY_2",      "pair", STAGE[0], STAGE[1],  45, 0.70, 0.70, 0.02, 0.02, 10),
    ("XOAY_3",      "pair", STAGE[0], STAGE[1],  67, 0.70, 0.70, 0.02, 0.02, 10),
    ("XOAY_4",      "pair", STAGE[0], STAGE[1],  90, 0.70, 0.70, 0.02, 0.02, 10),
    ("XOAY_5",      "pair", STAGE[0], STAGE[1], 112, 0.70, 0.70, 0.02, 0.02, 10),
    ("XOAY_6",      "pair", STAGE[0], STAGE[1], 135, 0.70, 0.70, 0.02, 0.02, 10),
    ("XOAY_7",      "pair", STAGE[0], STAGE[1], 157, 0.70, 0.70, 0.02, 0.02, 10),
    ("XOAY_XONG",   "pair", STAGE[0], STAGE[1], 180, 0.70, 0.70, 0.02, 0.02, 10),
    ("RA_TRUOC",     "side", -0.20, 0.45, 0.61, 0.69, 0.52, 0.52, 20),   # dung 2 ben, mieng ao truoc 2 tay
    ("CANH_NGUOI",   "side",  0.42, 0.45, 0.61, 0.69, 0.52, 0.52, 20),
    ("NANG_CAO",     "side",  0.45, 0.45, 0.84, 0.84, 0.52, 0.52, 12),
    ("XO_TAY_1",     "side",  0.46, 0.43, 0.70, 0.84, 0.26, 0.30, 12),   # thu tay: mieng ao truot vao 2 tay
    ("XO_TAY_2",     "side",  0.46, 0.43, 0.77, 0.84, 0.06, 0.22, 12),
    ("QUA_DAU",      "side",  0.72, 0.42, 0.77, 0.84, 0.02, 0.03, 15),   # di ra sau: ao qua dau
    ("SAU_LUNG",     "side",  1.05, 0.31, 0.80, 0.82, 0.15, 0.20, 15),
    ("KEO_XUONG_1",  "side",  1.05, 0.31, 0.85, 0.50, 0.30, 0.25, 10),   # keo ao xuong lung
    ("KEO_XUONG_2",  "side",  1.05, 0.31, 0.95, 0.40, 0.40, 0.23, 8),
]
# Goc co tay (yaw, pitch, roll) tung buoc. Diem mau chot ma ban truoc thieu: luc thu tay xo ao, demo1 cup co tay
# robot_0 xuong (pitch ~ -0.83) va xoay co tay robot_1 (yaw ~ +0.83) de quay mieng ao ve phia 2 ban tay nguoi.
W_GIU = ((0.00, 0.01, 0.56), (0.00, -0.27, 0.00))      # luc mang ao di (demo1 t=44..77)
WRIST = {
    "XO_TAY_1":    ((0.00, -0.83, 0.55), (0.83, -0.27, 0.00)),
    "XO_TAY_2":    ((0.00, -0.82, 0.55), (0.83, -0.27, 0.00)),
    "QUA_DAU":     ((0.27, -0.53, 0.55), (0.30, -0.24, 0.00)),
    "SAU_LUNG":    ((0.40, -0.25, 0.55), (-0.04, -0.13, 0.02)),
    "KEO_XUONG_1": ((-0.07, -0.07, 0.56), (-0.10, -1.13, 0.04)),
    "KEO_XUONG_2": ((-0.07, -0.05, 0.56), (-0.10, -1.13, 0.04)),
}
POS_TOL, LIFT_TOL, ARM_TOL, WR_TOL = 0.07, 0.04, 0.05, 0.08


def cmd(err, rate, gain=6.0):
    return float(np.clip(gain * err / rate, -1, 1))


def masks(img):
    r, g, b = [img[..., i].astype(int) for i in range(3)]
    return (b > r + 60) & (b > g + 60), (r > 170) & (g > 170) & (b < 110)


def yaw_of(q):
    return float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))


def drive(a, r, vx, vy, yaw, vmax=0.6):
    # van toc mong muon (toa do the gioi) -> lenh fwd/left cua robot r (dau +1, da kiem tra tu demo)
    f = vx * np.cos(yaw) + vy * np.sin(yaw)
    l = -vx * np.sin(yaw) + vy * np.cos(yaw)
    a[r, 0] = float(np.clip(f, -vmax, vmax))
    a[r, 1] = float(np.clip(l, -vmax, vmax))


def wrap(v):
    return float((v + np.pi) % (2 * np.pi) - np.pi)


def blue_center(hb):
    n = int(hb.sum())
    if n <= 400:
        return n, -1.0, -1.0
    ys, xs = np.nonzero(hb)
    return n, float(xs.mean()), float(ys.mean())


def hem_corner(img, cy, r):
    """Camera dau: goc GAU AO phia robot r (2 dau dai vang o phia duoi tam ao).
    robot_0 dung ben phai -> goc ben phai anh (cot lon nhat); robot_1 -> goc ben trai."""
    yel = masks(img)[1]
    ys, xs = np.nonzero(yel[int(max(cy, 0)):])
    if len(xs) < 30:
        return None
    ys = ys + int(max(cy, 0))
    band = ys >= np.percentile(ys, 40)              # chi lay dai vang duoi cung (gau), bo co ao
    xs, ys = xs[band], ys[band]
    u = float(np.percentile(xs, 98)) if r == 0 else float(np.percentile(xs, 2))
    v = float(np.median(ys[np.abs(xs - u) < 6])) if np.any(np.abs(xs - u) < 6) else float(np.median(ys))
    return u, v


VIDEO_HTML = r'''<!doctype html><html><head><meta charset="utf-8"><title>Video lan chay</title>
<style>body{background:#111;color:#eee;font-family:monospace;margin:10px}img{width:100%;max-width:1600px;image-rendering:pixelated;display:block}
.cap{display:flex;font-size:13px;margin:4px 0;max-width:1600px}.cap div{flex:1;text-align:center}#txt{font-size:16px;margin:6px 0;white-space:pre}
button{font-size:16px;margin-right:6px}input[type=range]{width:100%;max-width:1600px}</style></head><body>
<div id="txt"></div><div class="cap"><div>overview</div><div>r0 wrist</div><div>r1 wrist</div><div>r0 head</div><div>r1 head</div></div>
<img id="im"><input type="range" id="sl" min="1" value="1"><div>
<button onclick="play=!play">Chay / Dung</button><button onclick="go(i-1)">&lt; lui</button><button onclick="go(i+1)">toi &gt;</button>
toc do <select id="sp"><option value="1">x1</option><option value="2" selected>x2</option><option value="4">x4</option><option value="8">x8</option></select></div>
<script src="info.js"></script><script>
var N=INFO.length,i=1,play=true,im=document.getElementById('im'),sl=document.getElementById('sl'),tx=document.getElementById('txt');sl.max=N;
function p(n){return String(n).padStart(5,'0')}
function f(a){return a.s+'  xanh_dam='+a.xanh_dam+'  dz='+a.dz}
function go(k){i=Math.max(1,Math.min(N,k));im.src='f_'+p(i)+'.png';sl.value=i;var d=INFO[i-1];
tx.textContent='t='+d.t+' s   ('+i+'/'+N+')   '+d.mode+'\nR0: '+f(d.r0)+'\nR1: '+f(d.r1);}
sl.oninput=function(){go(+sl.value)};go(1);
(function loop(){setTimeout(function(){if(play&&i<N)go(i+1);loop()},250/(+document.getElementById('sp').value))})();
</script></body></html>
'''


def save_png(path, img, level=6):
    img = np.ascontiguousarray(img[..., :3], dtype=np.uint8)
    h, w = img.shape[:2]
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))
    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, level)) + chunk(b"IEND", b""))


def mark(img, x, y, col, size=6):
    h, w = img.shape[:2]
    x, y = int(round(x)), int(round(y))
    for d in (-1, 0, 1):
        if 0 <= y + d < h:
            img[y + d, max(0, x - size):min(w, x + size + 1)] = col
        if 0 <= x + d < w:
            img[max(0, y - size):min(h, y + size + 1), x + d] = col


def box(img, r0, r1, c0, c1, col):
    img[r0, c0:c1] = col
    img[r1 - 1, c0:c1] = col
    img[r0:r1, c0] = col
    img[r0:r1, c1 - 1] = col


class Arm:
    """May trang thai gap gau ao cho 1 robot."""

    def __init__(self, r):
        self.r, self.k = r, 0
        self.s, self.t0, self.ok, self.lost, self.opened = "IDLE", 0.0, None, None, False
        self.cpre, self.arm_t, self.hold_t, self.kep, self.fails = 0, 0.3, None, 0.0, 0
        self.tries, self.first_ok = 0, None
        self.events = []
        self.fresh = False
        self.dk = (0, -1.0, -1.0)
        self.dn, self.dt = None, None   # SLIDE: do sau vai gan nhat / dau ngon (m)
        self.ug = 0                     # SLIDE: so diem VAI ngay giua/duoi 2 dau ngon (hang 138-175) tren camera co tay       # xanh dam tren camera co tay: (so diem, cot trung vi, hang trung vi)
        self.dz = None                  # DEPTH camera co tay: (n diem vai giua 2 ngon, depth trung vi m, depth gan nhat m)
        self.ftx = None                 # vi tri dau ngon (he base): x = huong tay vuon, y = ngang
        self.fty = None
        self.ftz = None                 # do cao dau ngon thap nhat (m, he base) tu fingertip_position_base
        self.tilt, self.tilt0 = 0.0, 0.0  # do nghieng de robot (do), cap nhat moi buoc
        self.farm0 = 0.3
        self.y_ref, self.lift_t, self.lchk, self.contact = 0.0, 0.5, (0.0, 0.5), 0.45

    def recipe(self):
        # r000 7h30: thu lan 2 dung cong thuc so 1 (demo cu: lift 0.537, roll 2.76) -> 2 tay vuon len than ao, nhin len troi.
        # Moi lan thu deu dung cong thuc 0 (tu the da chon); moi lan thu lai chi ep sau them 1 cm (xem LOWER).
        return RECIPES[self.r][0]

    def go(self, s, t):
        self.s, self.t0, self.ok, self.lost, self.opened = s, t, None, None, False
        self.fresh = True
        if s == "PUSH":
            self.seq_ok = False
        print(f"[R{self.r}] >>> {s} t={t:.1f} cong_thuc={self.k % len(RECIPES[self.r])}", flush=True)

    def wrist(self, jp, tgt, a):
        for j in range(3):
            e = tgt[j] - float(jp[8 + j])
            a[self.r, 5 + j] = 0.0 if abs(e) < 0.03 else float(np.clip(1.5 * e, -0.5, 0.5))
        return np.abs(tgt - jp[8:11])

    def nang_kep(self, t, jp):
        """e5: mat ban ep ngon duoi -> kep tu dong (jp[11] giam tu 0.50). Nang dan dau ngon len toi khi het bi ep."""
        nang = getattr(self, "nang", 0.0)
        if KEP_EP > 0 and float(jp[11]) < KEP_EP and nang < 0.03:
            nang += 0.0005
            if int(t * 2) != int((t - 0.05) * 2):
                print(f"[R{self.r}] NGON BI EP (kep {float(jp[11]):.3f} < {KEP_EP}) -> nang ngon them, tong {nang * 1000:.1f} mm", flush=True)
        self.nang = nang
        return nang

    def step(self, t, jp, bp, n, cx, cy, cloth, hem, other_ready, other_holding, a):
        r = self.r
        lift, arm = float(jp[3]), float(jp[4:8].sum())
        lt, Rt, wy, wp, wr = self.recipe()
        tw = np.array([wy, wp, wr])
        tgt = self.contact + CONTACT_UP + LIFT_UP
        hx, hy = HEAD_T[r]
        ex, ey = cx - hx, hy - cy
        s = self.s
        fresh, self.fresh = self.fresh, False   # True o buoc dau tien cua moi trang thai

        def hold_arm():
            a[r, 4] = float(np.clip(cmd(self.arm_t - arm, RATE["arm"], 3.0), -0.5, 0.5))

        if s == "NAV":                      # dua ao ve dung cho tren camera dau (giong demo)
            a[r, 3] = cmd(LIFT_T - lift, RATE["lift"])
            if n > 400:
                self.lost = None
                a[r, 1] = -float(np.clip(ex / 80, -1, 1)) * 0.5
                a[r, 0] = float(np.clip(ey / 60, -1, 1)) * 0.5
                if abs(ex) < NAV_TOL and abs(ey) < 8 and abs(LIFT_T - lift) < 0.02:
                    self.ok = self.ok or t
                    if t - self.ok > 0.5:
                        self.go("OPEN", t)
                else:
                    self.ok = None
            else:
                self.lost = self.lost or t
                if t - self.lost > 2:
                    self.go("LOST", t)
            if t - self.t0 > 20:
                print(f"[R{self.r}] di toi goc gau het gio (lech {ex:.0f},{ey:.0f} px)", flush=True)
                self.go("OPEN", t)

        elif s == "OPEN":
            a[r, 3] = cmd(LIFT_T - lift, RATE["lift"])
            if not self.opened:
                a[r, 8] = -1.0
                self.opened = True
            if t - self.t0 > 1.0:
                self.go("PREP", t)

        elif s == "PREP":
            el, ea = HOVER_LIFT - lift, 0.02 - arm     # thu tay, kep o tren cao (tranh dap vao mep ban)
            a[r, 3] = cmd(el, RATE["lift"])
            a[r, 4] = cmd(ea, RATE["arm"])
            we = self.wrist(jp, tw, a)
            if abs(el) < 0.02 and abs(ea) < 0.03 and np.all(we < 0.06):
                self.ok = self.ok or t
                if t - self.ok > 0.3:
                    self.go("REACH", t)
            else:
                self.ok = None
            if t - self.t0 > 14:
                self.go("REACH", t)

        elif s == "REACH":                  # vuon toi truoc gau ao, kep o do cao an toan tren mat ban
            if fresh:
                self.y_ref = float(bp[1])       # chot vi tri de luc vuon (de bi day lui thi khong vuon duoi theo)
            self.arm_t = float(np.clip(Rt + self.y_ref + ARM_OFS, 0.02, 0.50))
            a[r, 3] = cmd(HOVER_LIFT - lift, RATE["lift"])
            self.wrist(jp, tw, a)
            ea = self.arm_t - arm
            a[r, 4] = float(np.clip(cmd(ea, RATE["arm"], 3.0), -REACH_V, REACH_V))   # 07/11: vuon cham (REACH_V)
            if (abs(ea) < 0.015 and abs(HOVER_LIFT - lift) < 0.02) or t - self.t0 > 10:
                print(f"[R{r}] vuon {arm:.3f} -> ha xuong {LOW[0] + DLIFT + FIND_UP:.3f}, rut tay cho toi khi dau ngon o mep gau", flush=True)
                self.go("FIND", t)

        elif s == "FIND":                   # DIRECT: kep cach mat ban 5 cm, rut tay ve cho toi khi dau ngon nam dung mep gau
            # anh sm r000: vuon theo demo thi dau ngon nam QUA gau 5-10 cm (tren than ao) -> phai rut ve.
            # camera co tay: gau (vang) hien o phia duoi anh (gan co tay) khi ngon con qua xa; rut tay -> vang di len.
            if fresh:
                self.farm0, self.ok = arm, None
                self.arm_t = arm
            # do cao tim gau theo DAU NGON (input fingertip_position_base), khong theo so lift co dinh:
            # r000 20h: co tay thang -> lift 0.435 la cham ban (ngon_z 0.575) -> ngon quet tren mat ban. Muon ngon cao hon ban FIND_UP.
            hgt = (lift + (TABLE_Z + FIND_UP - self.ftz)) if self.ftz is not None else (LOW[0] + DLIFT + FIND_UP)
            a[r, 3] = cmd(hgt - lift, RATE["lift"], 3.0)
            self.wrist(jp, tw, a)
            hold_arm()
            # SLIDE: rut tay cho toi khi GIUA 2 NGON KHONG CON VAI (ngon dung truoc mep gau, tren mat ban)
            ncl = self.ug
            dtip = self.dt if self.dt is not None else D_TIP
            gap = (self.dn - dtip) if self.dn is not None else 9.0     # mep vai gan nhat nam TRUOC dau ngon bao xa (m)
            if abs(hgt - lift) > 0.015:
                pass
            elif gap > SLIDE_GAP + FIND_XA and arm < self.farm0 + 0.30:
                # e3 anh 17.3 s: REACH vuon qua gau len than ao -> ngon duoi (co tay nghieng) quet LAT GAU AO.
                # -> REACH dung truoc gau (arm_ofs am), o day vuon toi CHAM 1 mm/buoc cho toi khi mep gau con cach SLIDE_GAP..SLIDE_GAP+FIND_XA
                self.ok = None
                if abs(self.arm_t - arm) < 0.015:
                    self.arm_t = min(0.50, self.arm_t + 0.001)
            elif gap > SLIDE_GAP:
                self.ok = self.ok or t
                if t - self.ok > 0.3:
                    print(f"[R{r}] NGON DUNG TRUOC GAU AO (mep vai cach dau ngon {gap:.3f} m: vai {self.dn} ngon {dtip:.3f}) arm={arm:.3f} -> ha sat ban roi day tu tu vao", flush=True)
                    self.events.append(f"R{r}_truoc_gau")
                    self.go("LOWER", t)
            else:
                self.ok = None
                if abs(self.arm_t - arm) < 0.015:
                    self.arm_t = max(0.05, self.farm0 - 0.20, self.arm_t - 0.002)
            if int(t * 2) != int((t - 0.05) * 2):
                print(f"[R{r}] lui tay ra truoc gau: mep_vai_truoc_ngon={gap:.3f} (vai {self.dn} ngon {self.dt}) mau_vai={ncl} arm={arm:.3f} ngon_z={self.ftz if self.ftz is not None else -1:.3f}", flush=True)
            _end = (self.arm_t <= max(self.farm0 - FIND_MAX, 0.02) + 1e-3
                    or self.arm_t >= min(self.farm0 + FIND_FWD, 0.52) - 1e-3 or t - self.t0 > 25)
            if _end:
                print(f"[R{r}] het tam tim (arm={arm:.3f}, gioi han them {FIND_FWD}) ngon_x={self.ftx if self.ftx is not None else -1:.3f} -> ha xuong kep o day", flush=True)
                self.events.append(f"R{r}_khong_thay_mep_gau")
                self.go("LOWER", t)

        elif s == "LOWER":                  # ha tu tu cho toi khi kep cham vai / mat ban (tu do do cao ban)
            if fresh:
                self.lift_t, self.lchk = lift, (t, lift)
                self.nang = 0.0
                self.tilt0 = self.tilt
            goal = LOW[0] + DLIFT - 0.01 * min(self.k, 3)    # DIRECT: do cao gap demo; moi lan thu lai ep sau them 1 cm
            if abs(self.arm_t - arm) < 0.01:           # tay da ve dung cho thi moi ha
                if self.ftz is not None:
                    # do cao theo DAU NGON (dung cho MOI goc co tay): duoc ha toi khi dau ngon thap hon mat ban 1 cm.
                    # (r008 22h: co tay thang -> san cu goal-0.04=0.56 lai KEO TAY LEN lift 0.60, ngon_z 0.78 -> kep vao khong khi)
                    floor_ = min(self.lift_t, lift - (self.ftz - (TABLE_Z - 0.01)))
                else:
                    floor_ = goal
                gan = self.ftz is not None and self.ftz < TABLE_Z + 0.03      # con < 3 cm toi mat ban: ha CHAM (khong dap kep xuong ban)
                self.lift_t = max(floor_, self.lift_t - DOWN_V * 2.5 * (0.2 if gan else 1.0) * 0.05)
            a[r, 3] = cmd(self.lift_t - lift, RATE["lift"], 3.0)
            self.wrist(jp, tw, a)
            hold_arm()
            ep = KEP_EP > 0 and float(jp[11]) < KEP_EP and not fresh
            if ep:
                print(f"[R{r}] NGON BI EP khi ha (kep {float(jp[11]):.3f}) -> dung ha, coi nhu cham ban", flush=True)
            if self.ftz is not None and (self.ftz <= TABLE_Z + FT_CLEAR or ep) and not fresh:
                self.contact = lift - CONTACT_UP
                print(f"[R{r}] DAU NGON CHAM MAT BAN: ngon_z={self.ftz:.3f} (ban {TABLE_Z}) lift={lift:.3f} -> kep", flush=True)
                self.events.append(f"R{r}_ha_xong")
                self.go("PUSH", t)
            else:
                hard = self.tilt - self.tilt0 > TILT_HARD or self.tilt > 6.0
                if (self.ftz is None and (abs(lift - goal) < 0.008 or lift < goal - 0.035)) or hard or (t - self.t0 > 12):
                    self.contact = (lift if hard else goal) - CONTACT_UP   # de buoc sau giu dung do cao nay
                    print(f"[R{r}] ha toi lift={lift:.3f} (muon {goal:.3f}, nghieng them {self.tilt - self.tilt0:.1f} do"
                          f"{' -> DUNG VI NGHIENG' if hard else ''}) -> kep", flush=True)
                    self.events.append(f"R{r}_ha_xong")
                    self.go("PUSH", t)

        elif s == "PUSH":                   # nang nhe 1 cm, day tay them 3 cm cho ngon trong luon vao lo gau
            if SEQ_PUSH and r == 1 and not self.seq_ok:
                if other_ready or other_holding or t - self.t0 > SEQ_MAX:
                    print(f"[R1] {'R0 da luon khe xong' if (other_ready or other_holding) else 'cho R0 qua lau'} -> R1 bat dau day", flush=True)
                    self.seq_ok, self.t0, fresh = True, t, True      # tinh gio + moc than_lui tu luc bat dau day
                else:                                                # R1 dung cho: ngon duoi sat ban, chua day
                    if self.ftz is not None:
                        a[r, 3] = cmd(TABLE_Z + SLIDE_H + self.nang_kep(t, jp) - self.ftz, RATE["lift"], 3.0)
                    self.wrist(jp, tw, a)
                    hold_arm()
                    return
            if fresh:
                self.events.append(f"R{r}_chinh_tay_xong")
                # anh mau cua thay (9h18): kep sat mat ban, GOC AO nam giua 2 ngon, ngon TRONG ngay mieng lo gau
                # -> day thang toi: ngon trong luon vao lo gau, ngon ngoai di ngoai suon ao -> khep = kep suon ao
                # anh r000 23h10: day CO DINH 5 cm -> robot_0 trung gau, robot_1 con thieu 3-4 cm (gau o hang ~118, ngon o ~138).
                # -> day THEO PHAN HOI: tang tay cho toi khi hang duoi cua dai vang (hemy) cham hang dau ngon PUSH_ROW,
                # do o do cao da ha (cung hinh hoc luc kep). Toi da PUSH_MAX.
                self.arm_t, self.parm0, self.pok, self.py0 = arm, arm, None, float(bp[1])
            a[r, 3] = cmd(self.contact + CONTACT_UP - lift, RATE["lift"], 3.0)
            self.wrist(jp, tw, a)
            hold_arm()
            hard = self.tilt - self.tilt0 >= TILT_HARD
            # day TU TU sat mat ban: khi vai lot vao giua 2 ngon thi day them SLIDE_IN roi kep
            if fresh:
                self.sl_in = None
            if self.ftz is not None:        # giu DAU NGON cach mat ban SLIDE_H (r016/017: ngon ep xuong 0.536-0.555 < ban 0.57 -> ma sat day LUI ca robot, dung sau 3 cm)
                a[r, 3] = cmd(TABLE_Z + SLIDE_H + self.nang_kep(t, jp) - self.ftz, RATE["lift"], 3.0)
            ncl = self.ug
            dtip = self.dt if self.dt is not None else D_TIP
            gap = (self.dn - dtip) if self.dn is not None else 9.0
            if self.sl_in is None and gap <= 0.005:
                self.sl_in = arm
                print(f"[R{r}] MEP GAU DA TOI DAU NGON (gap {gap:.3f} m, mau_vai={ncl}, xanh_dam={self.dk[0]}) arm={arm:.3f} -> day them {SLIDE_IN} m", flush=True)
            # 06/11: luon gioi han quang day <= SLIDE_MAX (c5: day 14 cm -> ui gau ao don len, anh 31.8 s)
            tgt = min(0.52, self.parm0 + SLIDE_MAX, (self.sl_in + SLIDE_IN) if self.sl_in is not None else 9.0)
            cao_du = self.ftz is None or self.ftz > TABLE_Z + SLIDE_H - 0.004   # NANG NGON LEN DU CAO roi moi day (test offline: day khi ngon con cham ban -> ket)
            if not hard and cao_du and abs(self.arm_t - arm) < 0.012:
                self.arm_t = min(tgt, self.arm_t + SLIDE_V)
            # e2 (07/11): ngon chui vao khe -> het can, tay tu vuon qua tgt 8 cm; lenh rut ve keo ca than robot toi 6 cm
            # -> KHONG rut tay ve: giu tay o cho hien tai va ket thuc day
            qua_sau = arm > tgt + 0.01
            if qua_sau:
                self.arm_t = arm
            if int(t * 2) != int((t - 0.05) * 2):
                print(f"[R{r}] truot vao: than_lui={float(bp[1]) - self.py0:.3f} lift={lift:.3f} mep_vai_truoc_ngon={gap:.3f} mau_vai={ncl} xanh_dam={self.dk[0]} da day {arm - self.parm0:.3f} m ngon_z={self.ftz if self.ftz is not None else -1:.3f}", flush=True)
            lui = float(bp[1]) - self.py0
            if GIU_THAN > 0 and lui > 0.003:         # than bi day lui -> banh xe day toi bu lai
                a[r, 0] = float(np.clip(GIU_THAN * lui, 0.0, 0.15))
            if ((self.arm_t >= tgt - 1e-3 and abs(self.arm_t - arm) < 0.008) or hard or t - self.t0 > SLIDE_T
                    or lui > SLIDE_LUI or qua_sau):
                ly = ("TAY VUON QUA SAU" if qua_sau else "du tam" if self.arm_t >= tgt - 1e-3 else "NGHIENG" if hard else "het gio" if t - self.t0 > SLIDE_T else "THAN BI DAY LUI")
                # e2: READY/CLOSE ha tay ve do cao luc LOWER (lift 0.395) trong khi luc day lift da 0.44 -> ep ngon xuong 4.5 cm, robot NGHIENG 8 do
                # -> giu DO CAO HIEN TAI cho cac buoc sau
                self.contact = lift - CONTACT_UP
                print(f"[R{r}] day ngon vao gau xong (truot, ly do: {ly}, than lui {lui:.3f} m): arm={arm:.3f} vai_giua_ngon={ncl} xanh_dam={self.dk[0]} "
                      f"{'CO VAI' if self.sl_in is not None else 'KHONG CO VAI giua ngon'}", flush=True)
                self.events.append(f"R{r}_day_vao_gau")
                self.go("KEO" if KEO_N[r] > 0 else "READY", t)

        elif s == "KEO":                    # cach cua thay: ngon dang trong khe gau -> keo ngang ra ngoai toi 2 goc gau, giu nguyen do cao
            if fresh:
                self.kx0 = float(bp[0])
            if self.ftz is not None:
                a[r, 3] = cmd(TABLE_Z + SLIDE_H + self.nang_kep(t, jp) - self.ftz, RATE["lift"], 3.0)
            self.wrist(jp, tw, a)
            hold_arm()
            sgn = -1.0 if r == 0 else 1.0       # robot_1 dung ben +x cua robot_0 (R1_OFFSET_X) -> ra ngoai: r0 ve -x, r1 ve +x
            moved = (float(bp[0]) - self.kx0) * sgn
            kn = KEO_N[r]
            if moved < kn:
                drive(a, r, sgn * KEO_V, 0.0, yaw_of(bp[3:7]), vmax=KEO_V)
            if int(t * 2) != int((t - 0.05) * 2):
                print(f"[R{r}] keo ra goc gau: da keo {moved:.3f}/{kn:.3f} m xanh_dam={self.dk[0]} ngon_z={self.ftz if self.ftz is not None else -1:.3f}", flush=True)
            if moved >= kn or t - self.t0 > 8.0:
                print(f"[R{r}] keo ra goc gau xong: {moved:.3f} m ({'du' if moved >= kn else 'HET GIO'}) xanh_dam={self.dk[0]}", flush=True)
                self.events.append(f"R{r}_keo_ra_goc")
                self.contact = lift - CONTACT_UP      # e2: READY giu do cao hien tai, khong ep ngon xuong
                self.go("READY", t)

        elif s == "READY":                  # cho robot kia, roi 2 robot dong kep cung luc
            a[r, 3] = cmd(self.contact + CONTACT_UP - lift, RATE["lift"])
            self.wrist(jp, tw, a)
            hold_arm()
            if t - self.t0 > 0.5 and (other_ready or other_holding or t - self.t0 > READY_WAIT):
                self.go("CLOSE", t)

        elif s == "CLOSE":
            a[r, 8] = 1.0
            we = np.abs(tw - jp[8:11])
            print(f"[R{r}] CLOSE lift={lift:.3f}(cham o {self.contact:.3f}) arm={arm:.3f} y={float(bp[1]):.2f} "
                  f"ngon_z={self.ftz if self.ftz is not None else -1:.3f} ngon_x={self.ftx if self.ftx is not None else -1:.3f} ngon_y={self.fty if self.fty is not None else -1:.3f} dz={self.dz} vai={cloth} gau={hem} sai_co_tay=({we[0]:.2f},{we[1]:.2f},{we[2]:.2f})", flush=True)
            self.events.append(f"R{r}_dong_kep")
            self.go("WAIT", t)

        elif s == "WAIT":
            if fresh:
                self.wup, self.wok = 0.0, None
            fk = float(jp[11])
            # 06/11: KHONG nhich tay len, KHONG xet do mo ngon. Moi truong dinh ao vao kep bang code khi kep da dong
            # va dung yen GRAB_SETTLE_FRAMES (log r033: R1 "grabbed 256 particles" du ngon mo 0.341) -> dung yen GRAB_HOLD s roi nang.
            a[r, 3] = cmd(self.contact + CONTACT_UP - lift, RATE["lift"], 3.0)
            self.wrist(jp, tw, a)
            hold_arm()
            if t - self.t0 > GRAB_HOLD:
                print(f"[R{r}] kep da khep: ngon={fk:.3f} (nhich len {self.wup * 100:.1f} cm) lift={lift:.3f}", flush=True)
                self.cpre, self.kep = cloth, float(jp[11])
                # demo1: gap xong robot_0 vua nang nhe vua THU TAY ve (arm 0.24 -> 0) -> keo lop tren, mo mieng gau ao
                self.arm_t = max(0.02, self.arm_t - DRAG_BACK)
                self.go("LIFT", t)

        elif s == "LIFT":
            err = tgt - lift
            a[r, 3] = cmd(err, RATE["lift"], 4.0)
            self.wrist(jp, tw, a)
            hold_arm()
            if abs(err) < 0.02 or t - self.t0 > 5:
                self.go("CHECK", t)

        elif s == "CHECK":                  # nang len: camera co tay van dinh vai = cam duoc
            a[r, 3] = cmd(tgt - lift, RATE["lift"], 4.0)
            self.wrist(jp, tw, a)
            hold_arm()
            if t - self.t0 > 1.0:
                # 06/11: KHONG xet do mo ngon (r034: R0 "grabbed 256 particles" ma ngon chi 0.099 < KEP_HELD -> bao TRUOT
                # roi tu mo kep lam rot ao). Chi xet camera co tay van thay vai sau khi nang.
                good = cloth >= 1500
                print(f"[R{r}] CHECK vai={cloth} truoc={self.cpre} ngon_kep={self.kep:.3f}->{float(jp[11]):.3f} -> "
                      f"{'CAM DUOC' if good else 'TRUOT'}", flush=True)
                self.events.append(f"R{r}_{'CAM_DUOC' if good else 'TRUOT'}")
                self.tries += 1
                if self.first_ok is None:
                    self.first_ok = bool(good)
                if good:
                    self.hold_t = t
                    self.go("HOLD", t)
                else:
                    self.fails += 1
                    self.go("RESET", t)

        elif s == "HOLD":
            a[r, 3] = cmd(tgt - lift, RATE["lift"], 4.0)
            self.wrist(jp, tw, a)
            hold_arm()
            # 06/11: bo kiem tra KEP_LOST. Moi truong chi tha ao khi policy ra lenh MO kep, nen dang giu thi
            # KHONG BAO GIO tu mo kep (truoc day ngon 0.099 < KEP_LOST 0.106 -> RESET -> mo kep -> rot ao).

        elif s == "RESET":
            nxt = RECIPES[r][(self.k + 1) % len(RECIPES[r])][0]
            if fresh:
                self.rl0 = lift
            if t - self.t0 < 1.5:               # TRUOT: MO KEP + NANG TAY LEN 6 cm truoc, KHONG rut tay (khoi keo ao ve phia robot)
                a[r, 8] = -1.0
                a[r, 3] = cmd(self.rl0 + 0.06 - lift, RATE["lift"])
                self.wrist(jp, tw, a)
                hold_arm()
                return
            if not self.opened:
                a[r, 8] = -1.0
                self.opened = True
            a[r, 3] = cmd(nxt - lift, RATE["lift"])
            a[r, 4] = cmd(0.02 - arm, RATE["arm"])
            self.wrist(jp, tw, a)
            done = abs(arm - 0.02) < 0.03 and abs(nxt - lift) < 0.03 and t - self.t0 > 2
            if done or t - self.t0 > 10:
                self.k += 1
                self.go("NAV", t)       # can lai ao truoc khi thu lan nua (ao co the da bi xe dich)

        elif s == "DRESS":              # bo phoi hop dieu khien de, lift, arm ; o day chi giu co tay
            self.wrist(jp, tw, a)


def apply_params(path):
    """Doc bo tham so gap tu file JSON (checkpoint.json). Khong co file -> giu gia tri mac dinh."""
    global HEAD_T, CONTACT_UP, DRAG_BACK, RECIPES, GRASP_ONLY, SNAP, KQ_TIME
    global ARM_OFS, DLIFT, FIND_UP, GRAB_HOLD, FT_CLEAR, FIND_FWD
    global SLIDE_H, SLIDE_LUI, SLIDE_GAP, D_TIP, SLIDE_IN, SLIDE_V, SLIDE_MAX
    global DRESS_STEPS, KEO_NGANG, KEO_V, SEQ_PUSH, SEQ_MAX, GIU_THAN
    global REACH_V, SLIDE_T, NAV_TOL, FIND_XA, KEP_EP
    import json
    if not path or not str(path).endswith(".json") or not os.path.exists(str(path)):
        return None
    p = json.load(open(str(path)))
    HEAD_T = ((p.get("head_cx0", HEAD_T[0][0]), HEAD_T[0][1]), (p.get("head_cx1", HEAD_T[1][0]), HEAD_T[1][1]))
    CONTACT_UP = p.get("contact_up", CONTACT_UP)
    DRAG_BACK = p.get("drag_back", DRAG_BACK)
    lows = []
    for r in (0, 1):
        c0 = RECIPES[r][0]
        lows.append((LOW[0], LOW[1], p.get(f"wy{r}", c0[2]), p.get(f"wp{r}", c0[3]), p.get(f"wr{r}", c0[4])))
    RECIPES = ([lows[0]] + list(RECIPES[0][1:]), [lows[1]] + list(RECIPES[1][1:]))
    ARM_OFS = p.get("arm_ofs", ARM_OFS)
    DLIFT = p.get("dlift", DLIFT)
    FIND_UP = p.get("find_up", FIND_UP)
    FIND_FWD = float(p.get("find_fwd", FIND_FWD))
    GRAB_HOLD = float(p.get("grab_hold", GRAB_HOLD))
    FT_CLEAR = p.get("ft_clear", FT_CLEAR)
    CORNER_U[0] = p.get("corner_u0", CORNER_U[0])
    CORNER_U[1] = p.get("corner_u1", CORNER_U[1])
    SLIDE_GAP = float(p.get("slide_gap", SLIDE_GAP))
    SLIDE_H = float(p.get("slide_h", SLIDE_H))
    SLIDE_LUI = float(p.get("slide_lui", SLIDE_LUI))
    D_TIP = float(p.get("d_tip", D_TIP))
    SLIDE_IN = float(p.get("slide_in", SLIDE_IN))
    SLIDE_V = float(p.get("slide_v", SLIDE_V))
    SLIDE_MAX = float(p.get("slide_max", SLIDE_MAX))
    SLIDE_T = float(p.get("slide_t", SLIDE_T))
    SEQ_PUSH = bool(p.get("seq_push", SEQ_PUSH))
    SEQ_MAX = float(p.get("seq_max", SEQ_MAX))
    GIU_THAN = float(p.get("giu_than", GIU_THAN))
    KEO_NGANG = float(p.get("keo_ngang", KEO_NGANG))
    KEO_N[0] = float(p.get("keo_ngang0", KEO_NGANG))
    KEO_N[1] = float(p.get("keo_ngang1", KEO_NGANG))
    KEO_V = float(p.get("keo_v", KEO_V))
    KEP_EP = float(p.get("kep_ep", KEP_EP))
    FIND_XA = float(p.get("find_xa", FIND_XA))
    REACH_V = float(p.get("reach_v", REACH_V))
    NAV_TOL = float(p.get("nav_tol", NAV_TOL))
    DRESS_STEPS = int(p.get("dress_steps", DRESS_STEPS))
    GRASP_ONLY = bool(p.get("grasp_only", GRASP_ONLY))
    SNAP = bool(p.get("snap", SNAP))
    KQ_TIME = float(p.get("kq_time", KQ_TIME))
    print(f"[THAM_SO] nap {path}: {json.dumps(p)}", flush=True)
    return p


def set_wrist(r, wy, wp, wr):
    global RECIPES
    L = list(RECIPES[r])
    L[0] = (LOW[0], LOW[1], wy, wp, wr)
    RECIPES = tuple(tuple(L) if i == r else RECIPES[i] for i in range(2))


# lan 7 & 9 (2/11): robot_1 giu duoc ao khi co tay xoay wr ~ +0.8 rad (nghieng ~45 do, giong anh cua thay)
set_wrist(1, 0.0, 0.83, 0.0)
set_wrist(0, 0.0, 0.83, 0.0)       # robot_0 o phia doi dien -> xoay nguoc chieu (dang kiem tra)


def load_policy(checkpoint, observation_space, action_space):
    apply_params(checkpoint)
    class Policy:
        def __init__(self):
            self.reset()

        def reset(self):
            self.arms = [Arm(0), Arm(1)]
            self.mode, self.t_mode = "SCAN", 0.0
            self.first, self.p0, self.d = True, None, 1
            self.ki, self.t_kf = 0, 0.0
            self.stuck, self.blocked = [None, None], [False, False]
            self.fallen = False
            self.nga_r = set()
            self.S0 = 0.40
            self.wcur = [None, None]
            self.events, self.snap_t, self.dir = [], -1e9, None
            self.vid_t, self.vn, self.vinfo = -1e9, 0, []
            self.el = 0.0
            self.kq_done = False

        def set_mode(self, m, t):
            self.mode, self.t_mode = m, t
            print(f"[PAIR] >>> {m} t={t:.1f}", flush=True)
            self.events.append(f"PAIR_{m}")

        def snap(self, t, rgb, tags, info):
            if not SNAP:
                return
            try:
                if self.dir is None:
                    self.dir = os.path.join(SNAP_DIR, time.strftime("lan_%m%d_%H%M%S"))
                    os.makedirs(self.dir, exist_ok=True)
                    print(f"[ANH] luu anh vao {self.dir}", flush=True)
                ims = [np.array(rgb[i][..., :3], dtype=np.uint8) for i in range(5)]
                for r in (0, 1):
                    hi = ims[3 + r]
                    mark(hi, HEAD_T[r][0], HEAD_T[r][1], (0, 255, 0), 8)     # xanh la: cho ao can nam
                    if info[r][0] > 400:
                        mark(hi, info[r][1], info[r][2], (255, 0, 0), 6)    # do: tam ao dang thay
                    c = getattr(self, "corner", [None, None])[r]
                    if c is not None:
                        mark(hi, c[0], c[1], (255, 0, 255), 6)              # tim: goc gau ao dang thay
                        mark(hi, CORNER_U[r], c[1], (0, 255, 255), 6)       # xanh lo: cho kep cua minh se cham
                    box(ims[1 + r], *TIP_WIN, (0, 255, 0))                  # khung tim gau giua 2 ngon
                h = min(im.shape[0] for im in ims)
                sep = np.full((h, 4, 3), 255, np.uint8)
                row = []
                for i in range(5):
                    row += [ims[i][:h], sep]
                name = f"t{t:06.1f}_{'__'.join(tags)}.png"
                save_png(os.path.join(self.dir, name), np.concatenate(row[:-1], 1))
                print(f"[ANH] {name}", flush=True)
            except Exception as e:
                print(f"[ANH] loi luu anh: {e!r}", flush=True)

        def video(self, t, rgb, A0, A1):
            # luu 1 khung video: 5 camera canh nhau (overview, r0 wrist, r1 wrist, r0 head, r1 head) + thong tin trang thai
            try:
                if self.dir is None:
                    self.dir = os.path.join(SNAP_DIR, time.strftime("lan_%m%d_%H%M%S"))
                    os.makedirs(self.dir, exist_ok=True)
                    print(f"[ANH] luu anh vao {self.dir}", flush=True)
                vd = os.path.join(self.dir, "video")
                if self.vn == 0:
                    os.makedirs(vd, exist_ok=True)
                    with open(os.path.join(vd, "xem_video.html"), "w") as f:
                        f.write(VIDEO_HTML)
                ims = [np.array(rgb[i][..., :3], dtype=np.uint8) for i in range(5)]
                for r in (0, 1):
                    box(ims[1 + r], *TIP_WIN, (0, 255, 0))
                h = min(im.shape[0] for im in ims)
                sep = np.full((h, 4, 3), 255, np.uint8)
                row = []
                for i in range(5):
                    row += [ims[i][:h], sep]
                self.vn += 1
                save_png(os.path.join(vd, f"f_{self.vn:05d}.png"), np.concatenate(row[:-1], 1), 1)
                def inf(A):
                    return dict(s=A.s, xanh_dam=int(A.dk[0]), dz=(None if A.dz is None else round(float(A.dz[1]), 3)))
                self.vinfo.append(dict(t=round(t, 2), mode=self.mode, r0=inf(A0), r1=inf(A1)))
                import json as _j
                with open(os.path.join(vd, "info.js"), "w") as f:
                    f.write("var INFO=" + _j.dumps(self.vinfo) + ";\n")
            except Exception as e:
                print(f"[ANH] loi luu video: {e!r}", flush=True)

        def __call__(self, obs):
            _t0 = time.perf_counter()
            try:
                a = self.step(obs)
            except Exception as e:
                print("[DBG] loi:", repr(e), flush=True)
                a = np.zeros((2, 9))
            self.el += time.perf_counter() - _t0 + 0.05    # thoi gian tap tinh nhu evaluator (mo phong + thoi gian policy)
            return np.clip(np.nan_to_num(a), -1, 1).astype(np.float32).reshape(2, 9)

        def step(self, obs):
            t = float(np.asarray(obs["simulation_time_s"]).ravel()[0])
            JP = np.asarray(obs["joint_position"], dtype=float).reshape(2, 13)
            BP = np.asarray(obs["base_pose_world"], dtype=float).reshape(2, 7)
            rgb = np.asarray(obs["rgb"])
            yaw = [yaw_of(BP[0, 3:7]), yaw_of(BP[1, 3:7])]
            if "fingertip_position_base" in obs:        # input BTC cho phep: vi tri 2 dau ngon trong he base
                FT = np.asarray(obs["fingertip_position_base"], dtype=float).reshape(2, 2, 3)
                for r in (0, 1):
                    self.arms[r].ftz = float(FT[r, :, 2].min())
                    self.arms[r].ftx = float(FT[r, :, 0].mean())
                    self.arms[r].fty = float(FT[r, :, 1].mean())
            a = np.zeros((2, 9))
            lift = [float(JP[0, 3]), float(JP[1, 3])]
            arm = [float(JP[0, 4:8].sum()), float(JP[1, 4:8].sum())]

            # camera: 0 tong quan, 1 co tay r0, 2 co tay r1, 3 dau r0, 4 dau r1
            hd = [blue_center(masks(rgb[3])[0]), blue_center(masks(rgb[4])[0])]
            # di toi sao cho GOC GAU AO (cho co khe gau) nam ngay truoc kep cua minh tren camera dau,
            # thay vi dua TAM AO ve diem demo (anh 14h29: 2 kep gap o giua gau, khong o 2 goc co khe)
            hg = list(hd)
            self.corner = [None, None]
            for r in (0, 1):
                if hd[r][0] > 400:
                    c = hem_corner(rgb[3 + r], hd[r][2], r)
                    if c is not None:
                        self.corner[r] = c
                        hg[r] = (hd[r][0], c[0] - CORNER_U[r] + HEAD_T[r][0], hd[r][2])
            cloth, hem = [], []
            r0w, r1w, c0w, c1w = TIP_WIN
            for k, i in enumerate((1, 2)):
                wb, wy = masks(rgb[i])
                cloth.append(int(wb.sum() + wy.sum()))
                hem.append(int(wy[r0w:r1w, c0w:c1w].sum()))
                im_ = rgb[i][..., :3].astype(int)
                dm = (im_[..., 2] > im_[..., 0] + 40) & (im_[..., 2] > im_[..., 1] + 40) & (im_[..., 2] < 180) & (im_[..., 0] < 60)
                if dm.sum() > 50:
                    yd_, xd_ = np.nonzero(dm)
                    self.arms[k].dk = (int(dm.sum()), float(np.median(xd_)), float(np.median(yd_)))
                else:
                    self.arms[k].dk = (int(dm.sum()), -1.0, -1.0)
                self.arms[k].ug = int((wb | wy)[138:175, 60:196].sum())
                self.arms[k].dz = None
                try:                                    # depth (input BTC cho phep): do khoang cach THAT toi vai giua 2 dau ngon
                    if "depth" in obs:
                        dd = np.asarray(obs["depth"])[i][..., 0].astype(float)
                        ok = (np.asarray(obs["depth_valid"])[i][..., 0] if "depth_valid" in obs else dd > 0) & (dd > 0)
                        cm = (wb | wy) & ok
                        win = cm[120:180, 95:160]
                        dw = dd[120:180, 95:160][win]
                        if dw.size >= 30:
                            self.arms[k].dz = (int(dw.size), float(np.median(dw)), float(np.percentile(dd[cm], 5)))
                        # SLIDE: dn = do sau VAI GAN NHAT (mep gau), dt = do sau DAU NGON (diem gan nhat trong 2 o quanh 2 dau ngon, khong phai vai)
                        cmx = cm.copy(); cmx[:, :40] = False; cmx[:, 216:] = False; cmx[185:, :] = False
                        self.arms[k].dn = float(np.percentile(dd[cmx], 5)) if cmx.sum() >= 50 else None
                        pads = []
                        for (y0_, y1_, x0_, x1_) in ((115, 155, 30, 80), (115, 155, 176, 226)):
                            okk = ok[y0_:y1_, x0_:x1_] & ~cm[y0_:y1_, x0_:x1_]
                            if okk.sum() >= 20:
                                pads.append(float(np.percentile(dd[y0_:y1_, x0_:x1_][okk], 10)))
                        self.arms[k].dt = min(pads) if pads else None
                except Exception:
                    self.arms[k].dz = None

            if self.p0 is None:
                self.p0 = (float(BP[0, 0]), float(BP[0, 1]), float(np.cos(yaw[0])), float(np.sin(yaw[0])))
            x0, yy0, fx, fy = self.p0
            dx, dy = float(BP[0, 0]) - x0, float(BP[0, 1]) - yy0
            dl, df = -dx * fy + dy * fx, dx * fx + dy * fy
            if self.first:
                self.first = False
                ov = int(masks(rgb[0])[0].sum())
                self.d = 1 if ov >= 30 else -1
                print(f"[DBG] yaw spawn r0={yaw[0]:.2f} r1={yaw[1]:.2f} | quet {'trai' if self.d > 0 else 'phai'} truoc", flush=True)

            A0, A1 = self.arms
            for r in (0, 1):
                w, x, y, z = BP[r, 3:7]
                tilt = float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))))
                self.arms[r].tilt = tilt
                if tilt > 8 and r not in self.nga_r:
                    # r000 11h10: robot_1 nghieng -> dung CA 2 robot -> robot_0 dang CHECK (da kep duoc) bi cat ngang.
                    # Gio: chi dung robot bi nghieng; dung het khi ca 2 cung nghieng.
                    self.nga_r.add(r)
                    self.fallen = len(self.nga_r) == 2
                    print(f"[R{r}] NGHIENG {tilt:.0f} do -> dung robot nay (robot kia van chay)", flush=True)
                    self.events.append(f"R{r}_NGHIENG")
            if GRASP_ONLY and not self.kq_done and max(t, self.el) > KQ_TIME:
                # het gio ma chua in ket qua (ket o SCAN, bi nga, ...) -> van in de may toi uu cham diem
                mieng = 0
                for i in (3, 4):
                    im = rgb[i][..., :3].astype(int)
                    rr, gg, bb = im[..., 0], im[..., 1], im[..., 2]
                    mieng += int(((bb > rr + 40) & (bb > gg + 40) & (bb < 150)).sum())
                print(f"[KQ] giu0={int(A0.s == 'HOLD')} giu1={int(A1.s == 'HOLD')} "
                      f"dau0={int(bool(A0.first_ok))} dau1={int(bool(A1.first_ok))} "
                      f"lan0={A0.tries} lan1={A1.tries} t={t:.1f} mieng={mieng}", flush=True)
                print(f"[KQ_LYDO] mode={self.mode} nga={sorted(self.nga_r)} r0={A0.s} r1={A1.s} "
                      f"thay_ao_dau r0={hd[0][0]} r1={hd[1][0]}", flush=True)
                self.kq_done = True
                self.snap(t, rgb, ["KET_QUA"], hd)
            if self.fallen:
                return np.zeros((2, 9))

            def follow_r0():
                ex1 = float(BP[0, 0]) + R1_OFFSET_X - float(BP[1, 0])
                ey1 = float(BP[0, 1]) - float(BP[1, 1])
                # e10 ban 1: R0 bi CACH_R1 ham, R1 chi nhich 1.5*sai so -> 2 robot bo 0.06 m/s. Them van toc R0 (ve +x) vao R1 de R1 chu dong tranh.
                ff = max(0.0, float(a[0, 0] * np.cos(yaw[0]) - a[0, 1] * np.sin(yaw[0])))
                drive(a, 1, 1.5 * ex1 + ff, 1.5 * ey1, yaw[1])
                a[1, 3] = cmd(LIFT_T - lift[1], RATE["lift"])
                return np.hypot(ex1, ey1)

            if self.mode == "SCAN":
                a[0, 3] = cmd(LIFT_T - lift[0], RATE["lift"])
                if hd[0][0] > SEE_N:
                    print(f"[R0] thay ao: n={hd[0][0]} cx={hd[0][1]:.0f}", flush=True)
                    A0.go("NAV", t)
                    self.set_mode("GRASP", t)
                else:
                    if df < FWD0:
                        a[0, 0] = 0.3
                    if self.d > 0 and dl > SCAN_L:
                        self.d = -1
                    elif self.d < 0 and dl < -SCAN_R:
                        self.d = 1
                    a[0, 1] = self.d * SCAN_V
                follow_r0()

            elif self.mode == "GRASP":
                if A0.s == "LOST":
                    A0.s = "IDLE"
                    self.set_mode("SCAN", t)
                else:
                    A0.step(t, JP[0], BP[0], *hg[0], cloth[0], hem[0],
                            A1.s in ("READY", "CLOSE", "WAIT"), A1.s == "HOLD", a)
                if A1.s in ("IDLE", "LOST"):
                    if follow_r0() < 0.15 and hd[1][0] > SEE_N:
                        print(f"[R1] thay ao: n={hd[1][0]} cx={hd[1][1]:.0f}", flush=True)
                        A1.go("NAV", t)
                else:
                    A1.step(t, JP[1], BP[1], *hg[1], cloth[1], hem[1],
                            A0.s in ("READY", "CLOSE", "WAIT"), A0.s == "HOLD", a)
                both = A0.s == "HOLD" and A1.s == "HOLD" and t - A0.hold_t > HOLD_S and t - A1.hold_t > HOLD_S
                if GRASP_ONLY and not self.kq_done and (both or max(t, self.el) > KQ_TIME):
                    # do do mo mieng gau: so diem xanh DAM (mat trong ao) tren 2 camera dau
                    mieng = 0
                    for i in (3, 4):
                        im = rgb[i][..., :3].astype(int)
                        rr, gg, bb = im[..., 0], im[..., 1], im[..., 2]
                        mieng += int(((bb > rr + 40) & (bb > gg + 40) & (bb < 150)).sum())
                    print(f"[KQ] giu0={int(A0.s == 'HOLD')} giu1={int(A1.s == 'HOLD')} "
                          f"dau0={int(bool(A0.first_ok))} dau1={int(bool(A1.first_ok))} "
                          f"lan0={A0.tries} lan1={A1.tries} t={t:.1f} mieng={mieng}", flush=True)
                    self.kq_done = True
                    self.snap(t, rgb, ["KET_QUA"], hd)
                if GRASP_ONLY:
                    pass
                elif both:
                    print("[PAIR] CA 2 ROBOT DA CAM GAU AO -> bat dau mac ao", flush=True)
                    A0.go("DRESS", t)
                    A1.go("DRESS", t)
                    self.ki, self.t_kf = 0, t
                    g = [BP[r, :2] + (arm[r] + C_GRIP) * np.array([np.cos(yaw[r]), np.sin(yaw[r])]) for r in (0, 1)]
                    self.S0 = float(np.clip(np.hypot(*(g[0] - g[1])), 0.25, 0.55))
                    print(f"[PAIR] 2 kep cach nhau {self.S0:.2f} m luc gap -> giu dung khoang nay khi mac ao", flush=True)
                    self.set_mode("DRESS", t)
                    print(f"[PAIR] buoc 1/{len(KF)}: {KF[0][0]}", flush=True)
                elif A0.s == "HOLD" and t - A0.hold_t > PAIR_WAIT:
                    print("[PAIR] robot_1 khong gap duoc -> robot_0 giu ao (chi diem gap)", flush=True)
                    self.set_mode("SOLO", t)

            elif self.mode == "SOLO":
                A0.step(t, JP[0], BP[0], *hd[0], cloth[0], hem[0], False, False, a)

            elif self.mode == "DRESS":
                kf = KF[self.ki]
                name, kind, tmax = kf[0], kf[1], kf[-1]
                S = self.S0
                if kind == "pair":
                    _, _, cx, cy, ang, l0, l1, m0, m1, _ = kf
                    th = np.radians(ang)
                    # robot_0 di vong qua phia +y, robot_1 qua phia -y; 2 de luon cach nhau S
                    tgt = [np.array([cx + S / 2 * np.cos(np.pi - th), cy + S / 2 * np.sin(np.pi - th)]),
                           np.array([cx + S / 2 * np.cos(-th), cy + S / 2 * np.sin(-th)])]
                    gx = [None, None]
                else:
                    _, _, by, bx, l0, l1, m0, m1, _ = kf
                    tgt = [np.array([PX + bx, by]), np.array([PX - bx, by])]
                    gx = [PX + S / 2, PX - S / 2]          # kep phai nam o day (sau khi da xoay ao)
                done = True
                for r, l, m in ((0, l0, m0), (1, l1, m1)):
                    e = tgt[r] - BP[r, :2]
                    dist = float(np.hypot(*e))
                    if self.stuck[r] is None or t - self.stuck[r][0] > 2.0:
                        if self.stuck[r] is not None and dist > 0.1 and \
                                np.hypot(*(BP[r, :2] - self.stuck[r][1])) < 0.02 and not self.blocked[r]:
                            print(f"[R{r}] bi chan o ({BP[r,0]:.2f},{BP[r,1]:.2f}), con cach {dist:.2f} m -> dung lai", flush=True)
                            self.blocked[r] = True
                        self.stuck[r] = (t, BP[r, :2].copy())
                    traveling = dist > 0.12 and not self.blocked[r]
                    if traveling:
                        if name.startswith("XOAY"):
                            vmax = 0.10
                        elif dist > 0.6 and kind == "pair":
                            vmax = 0.35
                        else:
                            vmax = 0.18
                        drive(a, r, 1.2 * e[0], 1.2 * e[1], yaw[r], vmax)
                        lt = min(l, 0.55) if kind == "side" else l
                        mt = min(m, 0.08)
                    else:
                        if not self.blocked[r]:
                            drive(a, r, 1.2 * e[0], 1.2 * e[1], yaw[r], 0.15)
                        lt, mt = l, m
                    # goc than: "pair" quay thang -y ; "side" xoay vao trong de kep o dung gx (theo do dai tay HIEN TAI)
                    if gx[r] is None:
                        yt = -np.pi / 2
                    else:
                        L = arm[r] + C_GRIP
                        c = float(np.clip((gx[r] - BP[r, 0]) / L, -0.6, 0.6))
                        yt = -float(np.arccos(c))
                    a[r, 2] = float(np.clip(2.0 * wrap(yt - yaw[r]), -0.5, 0.5))
                    a[r, 3] = cmd(lt - lift[r], RATE["lift"], 3.0)
                    a[r, 4] = float(np.clip(cmd(mt - arm[r], RATE["arm"], 3.0), -0.4, 0.4))
                    if traveling or abs(l - lift[r]) > LIFT_TOL or abs(m - arm[r]) > ARM_TOL or abs(wrap(yt - yaw[r])) > 0.08:
                        done = False
                    if float(JP[r, 11]) < KEP_LOST and int(t * 20) % 40 == 0:
                        print(f"[R{r}] ngon kep khep het ({float(JP[r, 11]):.3f}) - da tuot ao", flush=True)
                if int(t * 20) % 40 == 0:
                    g = [BP[r, :2] + (arm[r] + C_GRIP) * np.array([np.cos(yaw[r]), np.sin(yaw[r])]) for r in (0, 1)]
                    print(f"[PAIR] {name}: 2 kep cach nhau {np.hypot(*(g[0] - g[1])):.2f} m (luc gap {S:.2f}) "
                          f"yaw=({np.degrees(yaw[0]):.0f},{np.degrees(yaw[1]):.0f})", flush=True)
                wt = WRIST.get(name, W_GIU)
                for r in (0, 1):
                    # co tay doi goc tu tu (toi da 0.25 rad/s) de khong giat tuot ao
                    if self.wcur[r] is None:
                        self.wcur[r] = np.array(JP[r, 8:11], dtype=float)
                    step_w = np.clip(np.array(wt[r]) - self.wcur[r], -0.0125, 0.0125)
                    self.wcur[r] = self.wcur[r] + step_w
                    for j in range(3):
                        ew = float(self.wcur[r][j]) - float(JP[r, 8 + j])
                        if abs(wt[r][j] - self.wcur[r][j]) > WR_TOL:
                            done = False
                        a[r, 5 + j] = 0.0 if abs(ew) < 0.03 else float(np.clip(1.5 * ew, -0.5, 0.5))
                        if abs(ew) > WR_TOL:
                            done = False
                late = t - self.t_kf > tmax
                if done or late:
                    if late:
                        print(f"[PAIR] buoc {name} HET GIO (van di tiep)", flush=True)
                    self.events.append(f"xong_{self.ki + 1:02d}_{name}")
                    self.ki += 1
                    self.t_kf = t
                    self.stuck, self.blocked = [None, None], [False, False]
                    if DRESS_STEPS and self.ki >= DRESS_STEPS:
                        print(f"[PAIR] xong {DRESS_STEPS} buoc dau -> DUNG YEN GIU AO (dress_steps)", flush=True)
                        self.set_mode("GIU", t)
                    elif self.ki >= len(KF):
                        self.set_mode("RELEASE", t)
                    else:
                        print(f"[PAIR] buoc {self.ki + 1}/{len(KF)}: {KF[self.ki][0]}", flush=True)

            elif self.mode == "GIU":        # dress_steps: dung yen, KHONG mo kep, de xem ao co tuot khong
                A0.wrist(JP[0], W_GIU[0], a)
                A1.wrist(JP[1], W_GIU[1], a)

            elif self.mode == "RELEASE":    # tha robot_1 truoc, roi robot_0 ha thap va tha
                k = t - self.t_mode
                if k < 0.2:
                    a[1, 8] = -1.0
                a[1, 3] = cmd(0.30 - lift[1], RATE["lift"], 3.0)
                a[0, 3] = cmd((0.95 if k < 3 else 0.15) - lift[0], RATE["lift"], 3.0)
                if 6.0 < k < 6.2:
                    a[0, 8] = -1.0
                if k > 8:
                    self.set_mode("DONE", t)

            tags = self.events + A0.events + A1.events
            self.events, A0.events, A1.events = [], [], []
            if not tags and t - self.snap_t >= SNAP_EVERY:
                tags = [f"dinh_ky_{self.mode}_r0-{A0.s}_r1-{A1.s}"]
            if tags:
                self.snap_t = t
                self.snap(t, rgb, tags, hd)

            if SNAP and VID_FPS > 0 and t - self.vid_t >= 1.0 / VID_FPS - 1e-6:
                self.vid_t = t
                self.video(t, rgb, A0, A1)

            if int(t * 20) % 20 == 0:
                print(f"[DBG] t={t:.1f} {self.mode}{'/' + KF[self.ki][0] if self.mode == 'DRESS' else ''} | "
                      f"r0={A0.s} xy=({BP[0,0]:.2f},{BP[0,1]:.2f}) lift={lift[0]:.2f} arm={arm[0]:.2f} kep={float(JP[0,11]):.3f} ngon_z={A0.ftz if A0.ftz is not None else -1:.3f} vai={cloth[0]} gau={hem[0]} | "
                      f"r1={A1.s} xy=({BP[1,0]:.2f},{BP[1,1]:.2f}) lift={lift[1]:.2f} arm={arm[1]:.2f} kep={float(JP[1,11]):.3f} ngon_z={A1.ftz if A1.ftz is not None else -1:.3f} vai={cloth[1]} gau={hem[1]}", flush=True)
            # 08/11 seed 43/44: R0 (SCAN / NAV) chay ve +x DAM VAO R1 (R1 theo sau R0 cham hon, 2 robot trung cho, camera dau R1 den).
            # Khi R1 chua tu di (SCAN hoac R1 IDLE/LOST): R0 chay ve phia R1 thi giam toc theo khoang cach, gan qua thi dung cho R1 tranh.
            if self.mode == "SCAN" or (self.mode == "GRASP" and A1.s in ("IDLE", "LOST")):
                gap = float(BP[1, 0] - BP[0, 0])
                cy, sy = float(np.cos(yaw[0])), float(np.sin(yaw[0]))
                vx0 = a[0, 0] * cy - a[0, 1] * sy          # van toc the gioi theo x cua lenh R0
                if vx0 > 0 and gap < CACH_R1 + 0.15:
                    k = float(np.clip((gap - CACH_R1) / 0.15, 0.0, 1.0))
                    a[0, 0] *= k
                    a[0, 1] *= k
                    if t - getattr(self, "cach_t", -9) > 1.0:
                        self.cach_t = t
                        print(f"[R0] GAN R1 (cach {gap:.2f} m) -> giam toc x{k:.2f} cho R1 tranh", flush=True)
            for r_ in self.nga_r:
                a[r_, :] = 0.0              # robot da nghieng: dung han, khong dong/mo kep
            return a
    return Policy()
