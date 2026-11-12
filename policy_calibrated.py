# PhyRC 2027 - policy (Embedded 12-Demo Relative Reach Calibration + Fixed Grasp Check + Fast Lower/Fine)
# Chi dung obs duoc phep: rgb 5 camera, joint_position, base_pose_world cua chinh robot, simulation_time_s.
# Khong doc vi tri ao / nguoi tu mo phong, khong chay lai quy dao demo.
import os, time, zlib, struct, json
import numpy as np

# ============================== ANH DEBUG ==============================
SNAP = True                     # Dat False truoc khi nop bai de tang toc toi da
SNAP_DIR = "/output/anh"
SNAP_EVERY = 10.0

# ============================== THAM SO GAP AO (DA NHUNG TU CALIBRATION_3.JSON - 12 DEMOS) ==============================
# Tam ngam pixel tren camera dau (robot_0, robot_1) tu 12 mau doc lap
HEAD_T = ((103.5, 174.7), (172.7, 174.3))

# Tam vuon tuong doi D_reach (met) tinh tu vi tri dung NAV va he so hoi quy k_D_per_cy
D_REACH_BASE = (0.3627, 0.2987)
K_D_PER_CY = (-0.0060, -0.0060)
R_BASE = (0.9665, 0.9338)
LIFT_CONTACT_MEDIAN = (0.460, 0.460)

LIFT_T = 0.54
# Giu co tay phang (LOW) cho MOI lan thu, chi thay doi do bu tam vuon (reach_offset)
# de camera co tay luon nhin thang xuong ban, khong bi lech khung TIP_WIN khi retry!
LOW_WRIST = (0.01, -0.03, -0.04)
RETRY_OFFSETS = (0.00, -0.035, 0.030, -0.060, 0.050)

TIP_WIN = (138, 178, 45, 212)   # Cua so dau ngon tay
PINCH_WIN = (100, 182, 55, 200) # Cua so khe giua 2 ngon tay khi da khep kep
SEEK_FROM, SEEK_TO, SEEK_V = -0.08, 0.08, 0.035
HEM_N = 220

# Khi kep 1-2 lop vai mong trong Isaac Sim, jp[11] = 0.103..0.104.
# Ha nguong KEP_HELD xuong 0.098 va KEP_LOST xuong 0.092, ket hop kiem tra anh co tay!
KEP_HELD = 0.098
KEP_LOST = 0.092

WAIT_S, LIFT_UP, HOLD_S = 1.2, 0.12, 3.2
DRAG_BACK = 0.025
HOVER_LIFT = 0.55
FAST_LIFT_TO = 0.455            # Ha nhanh xuong 0.455m roi moi do cham ban (tiet kiem 5s)
TILT_STOP = 1.4
FINE_ROW = 161.0                # Trung binh fine_row_median (162.0 va 160.0) tu calibration_3.json
DOWN_V = 0.025
TIP_MID = 128
TIP_END = (120, 136)
PUSH = 0.025
CONTACT_UP = 0.0
GRASP_ONLY = False
KQ_TIME = 75.0
SCAN_V, SCAN_L, SCAN_R, FWD0, SEE_N = 0.8, 5.0, 5.0, 0.4, 1000
R1_OFFSET_X = 0.47
READY_WAIT = 6.0                # Giam tu 15s xuong 6s de khong bi het gio 60s
PAIR_WAIT = 120.0
RATE = dict(lift=1.54, arm=1.21)

# ============================== MAC AO ==============================
PX = 0.03
C_GRIP = 0.45
STAGE = (PX, -0.38)
KF = [
    ("RA_CHO_TRONG", "pair", STAGE[0], STAGE[1],   0, 0.46, 0.46, 0.02, 0.02, 40),
    ("NANG_LEN",     "pair", STAGE[0], STAGE[1],   0, 0.70, 0.70, 0.02, 0.02, 10),
    ("XOAY_1",      "pair", STAGE[0], STAGE[1],  22, 0.70, 0.70, 0.02, 0.02, 10),
    ("XOAY_2",      "pair", STAGE[0], STAGE[1],  45, 0.70, 0.70, 0.02, 0.02, 10),
    ("XOAY_3",      "pair", STAGE[0], STAGE[1],  67, 0.70, 0.70, 0.02, 0.02, 10),
    ("XOAY_4",      "pair", STAGE[0], STAGE[1],  90, 0.70, 0.70, 0.02, 0.02, 10),
    ("XOAY_5",      "pair", STAGE[0], STAGE[1], 112, 0.70, 0.70, 0.02, 0.02, 10),
    ("XOAY_6",      "pair", STAGE[0], STAGE[1], 135, 0.70, 0.70, 0.02, 0.02, 10),
    ("XOAY_7",      "pair", STAGE[0], STAGE[1], 157, 0.70, 0.70, 0.02, 0.02, 10),
    ("XOAY_XONG",   "pair", STAGE[0], STAGE[1], 180, 0.70, 0.70, 0.02, 0.02, 10),
    ("RA_TRUOC",     "side", -0.20, 0.45, 0.61, 0.69, 0.52, 0.52, 20),
    ("CANH_NGUOI",   "side",  0.42, 0.45, 0.61, 0.69, 0.52, 0.52, 20),
    ("NANG_CAO",     "side",  0.45, 0.45, 0.84, 0.84, 0.52, 0.52, 12),
    ("XO_TAY_1",     "side",  0.46, 0.43, 0.70, 0.84, 0.26, 0.30, 12),
    ("XO_TAY_2",     "side",  0.46, 0.43, 0.77, 0.84, 0.06, 0.22, 12),
    ("QUA_DAU",      "side",  0.72, 0.42, 0.77, 0.84, 0.02, 0.03, 15),
    ("SAU_LUNG",     "side",  1.05, 0.31, 0.80, 0.82, 0.15, 0.20, 15),
    ("KEO_XUONG_1",  "side",  1.05, 0.31, 0.85, 0.50, 0.30, 0.25, 10),
    ("KEO_XUONG_2",  "side",  1.05, 0.31, 0.95, 0.40, 0.40, 0.23, 8),
]
W_GIU = ((0.00, 0.01, 0.56), (0.00, -0.27, 0.00))
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


def save_png(path, img):
    img = np.ascontiguousarray(img[..., :3], dtype=np.uint8)
    h, w = img.shape[:2]
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))
    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


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
        self.cpre, self.arm_t, self.hold_t, self.kep, self.fails = 0, 0.25, None, 0.0, 0
        self.tries, self.first_ok = 0, None
        self.events = []
        self.seen = 0
        self.best_hem, self.best_arm = 0, None
        self.fresh = False
        self.hemx = (0, 0.0, 0.0)
        self.tilt, self.tilt0 = 0.0, 0.0
        self.hemy = (0, 0.0, 0.0)
        self.pinch_cloth, self.pinch_yellow = 0, 0
        self.fsg, self.ferr, self.ft, self.farm0 = 1.0, None, 0.0, 0.25
        self.y_ref, self.base_reach = 0.0, D_REACH_BASE[r]
        self.lift_t, self.lchk, self.contact = 0.5, (0.0, 0.5), 0.42
        self.asg, self.aerr, self.at = 1.0, None, 0.0

    def go(self, s, t):
        self.s, self.t0, self.ok, self.lost, self.opened = s, t, None, None, False
        self.fresh = True
        print(f"[R{self.r}] >>> {s} t={t:.1f} lan_thu={self.k}", flush=True)

    def wrist(self, jp, tgt, a):
        for j in range(3):
            e = tgt[j] - float(jp[8 + j])
            a[self.r, 5 + j] = 0.0 if abs(e) < 0.03 else float(np.clip(1.5 * e, -0.5, 0.5))
        return np.abs(tgt - jp[8:11])

    def step(self, t, jp, bp, other_bp_x, n, cx, cy, cloth, hem, other_ready, other_holding, a):
        r = self.r
        lift, arm = float(jp[3]), float(jp[4:8].sum())
        tw = np.array(LOW_WRIST, dtype=float)
        retry_off = RETRY_OFFSETS[self.k % len(RETRY_OFFSETS)]
        tgt = self.contact + CONTACT_UP + LIFT_UP
        hx, hy = HEAD_T[r]
        ex, ey = cx - hx, hy - cy
        s = self.s
        fresh, self.fresh = self.fresh, False

        def hold_arm():
            a[r, 4] = float(np.clip(cmd(self.arm_t - arm, RATE["arm"], 3.5), -0.5, 0.5))

        if s == "NAV":
            a[r, 3] = cmd(LIFT_T - lift, RATE["lift"])
            if n > 400:
                self.lost = None
                lat_cmd = -float(np.clip(ex / 75.0, -1.0, 1.0)) * 0.5
                # Khong cho R1 di ngang qua sat R0 (giu khoang cach X >= 0.38m)
                if r == 1 and (float(bp[0]) - other_bp_x) < 0.38 and lat_cmd < 0:
                    lat_cmd = 0.0
                a[r, 1] = lat_cmd
                a[r, 0] = float(np.clip(ey / 55.0, -1.0, 1.0)) * 0.5
                if abs(ex) < 8 and abs(ey) < 8 and abs(LIFT_T - lift) < 0.02:
                    self.ok = self.ok or t
                    if t - self.ok > 0.4:
                        self.go("OPEN", t)
                else:
                    self.ok = None
            else:
                self.lost = self.lost or t
                if t - self.lost > 2:
                    self.go("LOST", t)
            if t - self.t0 > 25:
                self.go("OPEN", t)

        elif s == "OPEN":
            a[r, 3] = cmd(LIFT_T - lift, RATE["lift"])
            if not self.opened:
                a[r, 8] = -1.0
                self.opened = True
            if t - self.t0 > 0.6:
                self.go("PREP", t)

        elif s == "PREP":
            el, ea = HOVER_LIFT - lift, 0.02 - arm
            a[r, 3] = cmd(el, RATE["lift"])
            a[r, 4] = cmd(ea, RATE["arm"])
            we = self.wrist(jp, tw, a)
            if abs(el) < 0.02 and abs(ea) < 0.03 and np.all(we < 0.06):
                self.ok = self.ok or t
                if t - self.ok > 0.2:
                    self.go("REACH", t)
            else:
                self.ok = None
            if t - self.t0 > 6:
                self.go("REACH", t)

        elif s == "REACH":
            if fresh:
                self.y_ref = float(bp[1])
                corr = float(np.clip(K_D_PER_CY[r] * (cy - hy), -0.08, 0.08)) if n > 400 else 0.0
                self.base_reach = float(np.clip(D_REACH_BASE[r] + corr + retry_off, 0.16, 0.44))
            # Bu tru neu de xe bi dich chuyen so voi luc bat dau REACH
            drift_y = float(bp[1]) - self.y_ref
            self.arm_t = float(np.clip(self.base_reach + drift_y + SEEK_FROM, 0.05, 0.46))
            a[r, 3] = cmd(HOVER_LIFT - lift, RATE["lift"])
            self.wrist(jp, tw, a)
            ea = self.arm_t - arm
            a[r, 4] = float(np.clip(cmd(ea, RATE["arm"], 3.5), -0.5, 0.5))
            if (abs(ea) < 0.015 and abs(HOVER_LIFT - lift) < 0.02) or t - self.t0 > 5:
                self.asg, self.aerr, self.at = 1.0, None, t
                self.go("ALIGN", t)

        elif s == "ALIGN":
            a[r, 3] = cmd(HOVER_LIFT - lift, RATE["lift"])
            self.wrist(jp, tw, a)
            hold_arm()
            hn, cmin, cmax = self.hemx
            if hn < 300:
                if t - self.t0 > 1.2:
                    print(f"[R{r}] camera co tay khong thay mep ao de can -> bo qua buoc can", flush=True)
                    self.go("SEEK", t)
            else:
                end = cmax if r == 0 else cmin
                err = end - TIP_END[r]
                if t - self.at > 0.8:
                    if self.aerr is not None and abs(err) > abs(self.aerr) + 6:
                        self.asg *= -1.0
                    self.aerr, self.at = err, t
                if abs(err) < 10:
                    self.ok = self.ok or t
                    if t - self.ok > 0.35:
                        print(f"[R{r}] mep gau o cot {end:.0f} (muon {TIP_END[r]})", flush=True)
                        self.go("SEEK", t)
                else:
                    self.ok = None
                    lat_cmd = self.asg * float(np.clip(-err / 128 * 0.5, -0.18, 0.18))
                    if r == 1 and (float(bp[0]) - other_bp_x) < 0.38 and lat_cmd < 0:
                        lat_cmd = 0.0
                    a[r, 1] = lat_cmd
            if t - self.t0 > 4.5:
                self.go("SEEK", t)

        elif s == "SEEK":
            if fresh:
                self.seen = 0
                self.best_hem, self.best_arm = 0, None
            a[r, 3] = cmd(HOVER_LIFT - lift, RATE["lift"])
            self.wrist(jp, tw, a)
            drift_y = float(bp[1]) - self.y_ref
            end = float(np.clip(self.base_reach + drift_y + SEEK_TO, 0.10, 0.48))
            if hem > self.best_hem:
                self.best_hem, self.best_arm = hem, arm
            self.seen = self.seen + 1 if hem >= HEM_N else 0
            if self.seen >= 2:
                print(f"[R{r}] thay GAU AO o dau ngon: vang={hem} arm={arm:.3f} -> ha xuong", flush=True)
                self.arm_t = arm
                self.go("LOWER", t)
            elif self.arm_t >= end - 1e-3:
                if self.best_hem >= 60 and self.best_arm is not None:
                    self.arm_t = float(np.clip(self.best_arm, 0.08, 0.44))
                else:
                    self.arm_t = float(np.clip(self.base_reach + drift_y, 0.08, 0.42))
                print(f"[R{r}] vuon het (max_vang={self.best_hem}) -> dat tay ve {self.arm_t:.2f} roi ha", flush=True)
                self.go("LOWER", t)
            elif float(bp[1]) - self.y_ref > 0.03:
                self.arm_t = arm
                self.go("LOWER", t)
            else:
                if abs(self.arm_t - arm) < 0.02:
                    self.arm_t = min(end, self.arm_t + SEEK_V * 0.05)
            hold_arm()

        elif s == "LOWER":
            if fresh:
                # Ha nhanh xuong FAST_LIFT_TO (0.455m) truoc khi ha cham do cham ban
                self.lift_t = min(lift, FAST_LIFT_TO)
                self.lchk = (t, lift)
                self.tilt0 = self.tilt
            if abs(self.arm_t - arm) < 0.03:
                if lift > FAST_LIFT_TO + 0.01:
                    self.lift_t = FAST_LIFT_TO
                else:
                    self.lift_t = max(0.30, self.lift_t - DOWN_V * 0.05)
            a[r, 3] = cmd(self.lift_t - lift, RATE["lift"], 3.5)
            self.wrist(jp, tw, a)
            hold_arm()
            if self.tilt - self.tilt0 > TILT_STOP or (lift <= FAST_LIFT_TO and lift - self.lift_t > 0.02):
                self.contact = lift
                print(f"[R{r}] cham o lift={lift:.3f} -> sang FINE chinh gau", flush=True)
                self.go("FINE", t)
            elif t - self.lchk[0] > 0.45:
                stalled = (self.lchk[1] - lift) < 0.003 and (self.lift_t < lift - 0.025)
                self.lchk = (t, lift)
                if stalled or lift < 0.31:
                    self.contact = lift
                    print(f"[R{r}] kep cham ban o lift={lift:.3f} -> sang FINE", flush=True)
                    self.go("FINE", t)

        elif s == "FINE":
            if fresh:
                self.fsg, self.ferr, self.ft, self.farm0 = 1.0, None, t, arm
                self.arm_t = arm
            a[r, 3] = cmd(self.contact + 0.022 - lift, RATE["lift"], 3.0)
            self.wrist(jp, tw, a)
            hold_arm()
            n_, row, col = self.hemy
            if n_ < 80:
                if t - self.t0 > 1.5:
                    self.go("PUSH", t)
            else:
                err = row - FINE_ROW
                if abs(err) < 12:
                    self.ok = self.ok or t
                    if t - self.ok > 0.3:
                        print(f"[R{r}] GAU AO chuan o dau ngon (hang {row:.0f}, vang={n_}) arm={arm:.3f} -> PUSH", flush=True)
                        self.go("PUSH", t)
                else:
                    self.ok = None
                    if abs(self.arm_t - arm) < 0.025:
                        step_arm = float(np.clip(-self.fsg * err * 0.0007, -0.007, 0.007))
                        self.arm_t = float(np.clip(self.arm_t + step_arm, 0.05, 0.48))
            if t - self.t0 > 5.0 or float(bp[1]) - self.y_ref > 0.04:
                print(f"[R{r}] FINE het gio (hang {self.hemy[1]:.0f}, arm={arm:.3f}) -> sang PUSH", flush=True)
                self.go("PUSH", t)

        elif s == "PUSH":
            if fresh:
                n_, row, _ = self.hemy
                # Neu gau ao van nam phia sau dau ngon (row > FINE_ROW + 14),
                # KHONG day tay ve phia truoc ma thu nhe tay ve de dau ngon trung gau ao!
                if n_ >= 80 and row > FINE_ROW + 14:
                    pull_back = float(np.clip((row - FINE_ROW) * 0.0012, 0.01, 0.05))
                    self.arm_t = max(0.05, arm - pull_back)
                else:
                    self.arm_t = min(0.50, arm + (PUSH if self.tilt < TILT_STOP else 0.0))
            a[r, 3] = cmd(self.contact + CONTACT_UP - lift, RATE["lift"], 3.0)
            self.wrist(jp, tw, a)
            hold_arm()
            if abs(self.arm_t - arm) < 0.012 or t - self.t0 > 2.0 or float(bp[1]) - self.y_ref > 0.04:
                self.go("READY", t)

        elif s == "READY":
            a[r, 3] = cmd(self.contact + CONTACT_UP - lift, RATE["lift"])
            self.wrist(jp, tw, a)
            hold_arm()
            if t - self.t0 > 0.3 and (other_ready or other_holding or t - self.t0 > READY_WAIT):
                self.go("CLOSE", t)

        elif s == "CLOSE":
            a[r, 8] = 1.0
            we = np.abs(tw - jp[8:11])
            print(f"[R{r}] CLOSE lift={lift:.3f}(cham o {self.contact:.3f}) arm={arm:.3f} y={float(bp[1]):.2f} "
                  f"vai={cloth} gau={hem} sai_co_tay=({we[0]:.2f},{we[1]:.2f},{we[2]:.2f})", flush=True)
            self.events.append(f"R{r}_dong_kep")
            self.go("WAIT", t)

        elif s == "WAIT":
            a[r, 3] = cmd(self.contact + CONTACT_UP - lift, RATE["lift"])
            self.wrist(jp, tw, a)
            hold_arm()
            if t - self.t0 > WAIT_S:
                self.cpre, self.kep = cloth, float(jp[11])
                self.arm_t = max(0.02, self.arm_t - DRAG_BACK)
                self.go("LIFT", t)

        elif s == "LIFT":
            err = tgt - lift
            a[r, 3] = cmd(err, RATE["lift"], 4.0)
            self.wrist(jp, tw, a)
            hold_arm()
            if abs(err) < 0.02 or t - self.t0 > 3.5:
                self.go("CHECK", t)

        elif s == "CHECK":
            a[r, 3] = cmd(tgt - lift, RATE["lift"], 4.0)
            self.wrist(jp, tw, a)
            hold_arm()
            if t - self.t0 > 0.8:
                # Phan biet kep dinh vs kep truot tren camera co tay khi da nhac len cao:
                # Khi kep dinh: van con soc vang o khe ngon (hem >= 6 hoac pinch_yellow >= 8)
                # hoac vai nam sat giua 2 ngon (pinch_cloth >= 450 va tong vai < 6500, khong phai nhin toan canh ban).
                finger_ok = float(jp[11]) >= KEP_HELD
                visual_ok = (hem >= 6) or (self.pinch_yellow >= 8) or (self.pinch_cloth >= 450 and cloth < 6500)
                good = finger_ok and cloth >= 800 and visual_ok
                print(f"[R{r}] CHECK vai={cloth} pinch_vai={self.pinch_cloth} gau={hem} pinch_vang={self.pinch_yellow} "
                      f"ngon_kep={self.kep:.3f}->{float(jp[11]):.3f} -> {'CAM DUOC' if good else 'TRUOT'}", flush=True)
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
            # Chi bao roi ao neu ca ngon kep khep nho hon KEP_LOST (0.092) lan mat sach vai tren camera co tay
            if float(jp[11]) < KEP_LOST or (cloth < 400 and self.pinch_cloth < 100):
                print(f"[R{r}] roi ao luc dang giu (ngon kep={float(jp[11]):.3f}, vai={cloth})", flush=True)
                self.events.append(f"R{r}_ROI_AO")
                self.fails += 1
                self.go("RESET", t)

        elif s == "RESET":
            if not self.opened:
                a[r, 8] = -1.0
                self.opened = True
            a[r, 3] = cmd(HOVER_LIFT - lift, RATE["lift"])
            a[r, 4] = cmd(0.02 - arm, RATE["arm"])
            self.wrist(jp, tw, a)
            done = abs(arm - 0.02) < 0.03 and abs(HOVER_LIFT - lift) < 0.03 and t - self.t0 > 1.2
            if done or t - self.t0 > 5.0:
                self.k += 1
                self.go("NAV", t)

        elif s == "DRESS":
            self.wrist(jp, tw, a)


def apply_params(path):
    """Doc bo tham so tu calibration.json hoac checkpoint.json 1 lan luc khoi tao."""
    global TIP_END, HEAD_T, D_REACH_BASE, K_D_PER_CY, FINE_ROW, PUSH, CONTACT_UP, DRAG_BACK, GRASP_ONLY, SNAP, KQ_TIME
    cand_paths = []
    if path and str(path).endswith(".json"):
        cand_paths.append(str(path))
    cand_paths.append(os.path.join(os.path.dirname(__file__), "calibration.json"))

    loaded_path = next((cp for cp in cand_paths if os.path.exists(cp)), None)
    if not loaded_path:
        return None

    with open(loaded_path, "r", encoding="utf-8") as f:
        p = json.load(f)

    if "robot_0" in p and "robot_1" in p:
        r0, r1 = p["robot_0"], p["robot_1"]
        HEAD_T = (tuple(r0.get("head_target_px", HEAD_T[0])), tuple(r1.get("head_target_px", HEAD_T[1])))
        D_REACH_BASE = (
            float(r0.get("D_reach_base", D_REACH_BASE[0])),
            float(r1.get("D_reach_base", D_REACH_BASE[1]))
        )
        K_D_PER_CY = (
            float(r0.get("k_D_per_cy", r0.get("k_R_per_cy", K_D_PER_CY[0]))),
            float(r1.get("k_D_per_cy", r1.get("k_R_per_cy", K_D_PER_CY[1])))
        )
        if "fine_row_median" in r0:
            FINE_ROW = float(0.5 * (r0["fine_row_median"] + r1.get("fine_row_median", r0["fine_row_median"])))
        print(f"[CALIB] Da nap {loaded_path}: HEAD_T={HEAD_T}, D_REACH_BASE={D_REACH_BASE}, K_D={K_D_PER_CY}", flush=True)
    else:
        TIP_END = (p.get("tip_end0", TIP_END[0]), p.get("tip_end1", TIP_END[1]))
        HEAD_T = ((p.get("head_cx0", HEAD_T[0][0]), HEAD_T[0][1]), (p.get("head_cx1", HEAD_T[1][0]), HEAD_T[1][1]))
        PUSH = p.get("push", PUSH)
        CONTACT_UP = p.get("contact_up", CONTACT_UP)
        DRAG_BACK = p.get("drag_back", DRAG_BACK)
        GRASP_ONLY = bool(p.get("grasp_only", GRASP_ONLY))
        SNAP = bool(p.get("snap", SNAP))
        KQ_TIME = float(p.get("kq_time", KQ_TIME))
        print(f"[THAM_SO] Da nap {loaded_path}: {json.dumps(p)}", flush=True)
    return p


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
            self.S0 = 0.40
            self.wcur = [None, None]
            self.events, self.snap_t, self.dir = [], -1e9, None
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
                    mark(hi, HEAD_T[r][0], HEAD_T[r][1], (0, 255, 0), 8)
                    if info[r][0] > 400:
                        mark(hi, info[r][1], info[r][2], (255, 0, 0), 6)
                    box(ims[1 + r], *TIP_WIN, (0, 255, 0))
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

        def __call__(self, obs):
            try:
                a = self.step(obs)
            except Exception as e:
                print("[DBG] loi:", repr(e), flush=True)
                a = np.zeros((2, 9))
            return np.clip(np.nan_to_num(a), -1, 1).astype(np.float32).reshape(2, 9)

        def step(self, obs):
            t = float(np.asarray(obs["simulation_time_s"]).ravel()[0])
            JP = np.asarray(obs["joint_position"], dtype=float).reshape(2, 13)
            BP = np.asarray(obs["base_pose_world"], dtype=float).reshape(2, 7)
            rgb = np.asarray(obs["rgb"])
            yaw = [yaw_of(BP[0, 3:7]), yaw_of(BP[1, 3:7])]
            a = np.zeros((2, 9))
            lift = [float(JP[0, 3]), float(JP[1, 3])]
            arm = [float(JP[0, 4:8].sum()), float(JP[1, 4:8].sum())]

            hd = [blue_center(masks(rgb[3])[0]), blue_center(masks(rgb[4])[0])]
            cloth, hem = [], []
            r0w, r1w, c0w, c1w = TIP_WIN
            pr0, pr1, pc0, pc1 = PINCH_WIN
            for k, i in enumerate((1, 2)):
                wb, wy = masks(rgb[i])
                cloth.append(int(wb.sum() + wy.sum()))
                hem.append(int(wy[r0w:r1w, c0w:c1w].sum()))
                self.arms[k].pinch_cloth = int((wb | wy)[pr0:pr1, pc0:pc1].sum())
                self.arms[k].pinch_yellow = int(wy[pr0:pr1, pc0:pc1].sum())
                ys_, xs_ = np.nonzero((wb | wy)[120:190])
                yy_, xx_ = np.nonzero(wy)
                self.arms[k].hemy = ((len(yy_), float(np.median(yy_)), float(np.median(xx_))) if len(yy_) >= 80
                                     else (len(yy_), 0.0, 0.0))
                if len(xs_) >= 300:
                    self.arms[k].hemx = (len(xs_), float(np.percentile(xs_, 3)), float(np.percentile(xs_, 97)))
                else:
                    self.arms[k].hemx = (len(xs_), 0.0, 0.0)

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
                if tilt > 8 and not self.fallen:
                    self.fallen = True
                    print(f"[R{r}] NGHIENG {tilt:.0f} do -> dung moi chuyen dong", flush=True)
                    self.events.append(f"R{r}_NGHIENG")
            if GRASP_ONLY and not self.kq_done and t > KQ_TIME:
                mieng = 0
                for i in (3, 4):
                    im = rgb[i][..., :3].astype(int)
                    rr, gg, bb = im[..., 0], im[..., 1], im[..., 2]
                    mieng += int(((bb > rr + 40) & (bb > gg + 40) & (bb < 150)).sum())
                print(f"[KQ] giu0={int(A0.s == 'HOLD')} giu1={int(A1.s == 'HOLD')} "
                      f"dau0={int(bool(A0.first_ok))} dau1={int(bool(A1.first_ok))} "
                      f"lan0={A0.tries} lan1={A1.tries} t={t:.1f} mieng={mieng}", flush=True)
                print(f"[KQ_LYDO] mode={self.mode} nga={self.fallen} r0={A0.s} r1={A1.s} "
                      f"thay_ao_dau r0={hd[0][0]} r1={hd[1][0]}", flush=True)
                self.kq_done = True
                self.snap(t, rgb, ["KET_QUA"], hd)
            if self.fallen:
                return np.zeros((2, 9))

            def follow_r0():
                ex1 = float(BP[0, 0]) + R1_OFFSET_X - float(BP[1, 0])
                ey1 = float(BP[0, 1]) - float(BP[1, 1])
                drive(a, 1, 1.5 * ex1, 1.5 * ey1, yaw[1])
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
                    A0.step(t, JP[0], BP[0], float(BP[1, 0]), *hd[0], cloth[0], hem[0],
                            A1.s in ("READY", "CLOSE", "WAIT"), A1.s == "HOLD", a)
                if A1.s in ("IDLE", "LOST"):
                    if follow_r0() < 0.15 and hd[1][0] > SEE_N:
                        print(f"[R1] thay ao: n={hd[1][0]} cx={hd[1][1]:.0f}", flush=True)
                        A1.go("NAV", t)
                else:
                    A1.step(t, JP[1], BP[1], float(BP[0, 0]), *hd[1], cloth[1], hem[1],
                            A0.s in ("READY", "CLOSE", "WAIT"), A0.s == "HOLD", a)
                both = A0.s == "HOLD" and A1.s == "HOLD" and t - A0.hold_t > HOLD_S and t - A1.hold_t > HOLD_S
                if GRASP_ONLY and not self.kq_done and (both or t > KQ_TIME):
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
                A0.step(t, JP[0], BP[0], float(BP[1, 0]), *hd[0], cloth[0], hem[0], False, False, a)

            elif self.mode == "DRESS":
                kf = KF[self.ki]
                name, kind, tmax = kf[0], kf[1], kf[-1]
                S = self.S0
                if kind == "pair":
                    _, _, cx, cy, ang, l0, l1, m0, m1, _ = kf
                    th = np.radians(ang)
                    tgt = [np.array([cx + S / 2 * np.cos(np.pi - th), cy + S / 2 * np.sin(np.pi - th)]),
                           np.array([cx + S / 2 * np.cos(-th), cy + S / 2 * np.sin(-th)])]
                    gx = [None, None]
                else:
                    _, _, by, bx, l0, l1, m0, m1, _ = kf
                    tgt = [np.array([PX + bx, by]), np.array([PX - bx, by])]
                    gx = [PX + S / 2, PX - S / 2]
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
                    if self.ki >= len(KF):
                        self.set_mode("RELEASE", t)
                    else:
                        print(f"[PAIR] buoc {self.ki + 1}/{len(KF)}: {KF[self.ki][0]}", flush=True)

            elif self.mode == "RELEASE":
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

            if int(t * 20) % 20 == 0:
                print(f"[DBG] t={t:.1f} {self.mode}{'/' + KF[self.ki][0] if self.mode == 'DRESS' else ''} | "
                      f"r0={A0.s} xy=({BP[0,0]:.2f},{BP[0,1]:.2f}) lift={lift[0]:.2f} arm={arm[0]:.2f} vai={cloth[0]} gau={hem[0]} | "
                      f"r1={A1.s} xy=({BP[1,0]:.2f},{BP[1,1]:.2f}) lift={lift[1]:.2f} arm={arm[1]:.2f} vai={cloth[1]} gau={hem[1]}", flush=True)
            return a
    return Policy()
