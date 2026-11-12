# PhyRC 2027 - hem/check, measured PUSH, coordinated DRESS (experimental route)
# Chi dung obs duoc phep: rgb 5 camera, joint_position, base_pose_world cua chinh robot, simulation_time_s.
# Khong doc vi tri ao / nguoi tu mo phong, khong chay lai quy dao demo.
import os, time, zlib, struct, json
import numpy as np

# ============================== ANH DEBUG ==============================
SNAP = True                     # Dat False truoc khi nop bai de tang toc toi da
SNAP_DIR = "/output/anh"
SNAP_EVERY = 10.0

# Embedded candidate for an explicitly requested experimental rollout.
# policy_ready remains false; this is not a deployment validation claim.
EMBEDDED_CALIBRATION = {'schema': 'phyrc-reach-candidate-v2',
 'policy_ready': False,
 'robots': {'robot_0': {'model': 'nav_plus',
                        'D_reach_base': 0.323504834191373,
                        'cy_ref': 181.18127659756934,
                        'y_nav_ref': -0.6420206427574158,
                        'k_D_per_cy': 0.001115297472468595,
                        'base_y_coefficient': 1.0,
                        'raw_k_D_per_cy': 0.001115297472468595,
                        'slope_was_clipped': False,
                        'ridge_alpha': 50.0,
                        'num_samples': 12,
                        'head_target_px': [98.658797488152, 181.18127659756934],
                        'lift_contact_median': 0.45955832302570343,
                        'fine_row_median': 166.0,
                        'policy_ready': False},
            'robot_1': {'model': 'nav_plus',
                        'D_reach_base': 0.3003295946007839,
                        'cy_ref': 178.89236363690614,
                        'y_nav_ref': -0.6352040469646454,
                        'k_D_per_cy': 0.0013167425386852547,
                        'base_y_coefficient': 1.0,
                        'raw_k_D_per_cy': 0.0013167425386852547,
                        'slope_was_clipped': False,
                        'ridge_alpha': 50.0,
                        'num_samples': 12,
                        'head_target_px': [173.4872305002975, 178.89236363690614],
                        'lift_contact_median': 0.4600083976984024,
                        'fine_row_median': 161.0,
                        'policy_ready': False}}}
CALIBRATION = tuple(EMBEDDED_CALIBRATION["robots"][f"robot_{r}"] for r in (0, 1))
HEAD_T = tuple(tuple(c["head_target_px"]) for c in CALIBRATION)
D_REACH_BASE = tuple(c["D_reach_base"] for c in CALIBRATION)
K_D_PER_CY = tuple(c["k_D_per_cy"] for c in CALIBRATION)
CY_REF = tuple(c["cy_ref"] for c in CALIBRATION)
Y_NAV_REF = tuple(c["y_nav_ref"] for c in CALIBRATION)
BASE_Y_COEFF = tuple(c["base_y_coefficient"] for c in CALIBRATION)
FINE_ROWS = tuple(c["fine_row_median"] for c in CALIBRATION)
LIFT_CONTACT_MEDIAN = tuple(c["lift_contact_median"] for c in CALIBRATION)

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
CONTACT_TILT_DELTA = 1.4
TILT_STOP = CONTACT_TILT_DELTA  # legacy alias; never gates PUSH
HEM_COLS = (60, 195)
CHECK_YELLOW_MIN = 24
CHECK_STABLE_S = 0.5
CHECK_TIMEOUT_S = 2.5
PUSH_STEP_M = 0.005
PUSH_SPEED_M_S = 0.015
PUSH_TOL_M = 0.003
PUSH_TIMEOUT_S = 4.0
PUSH_ABS_TILT_DEG = 3.5
PUSH_TILT_RISE_DEG = 1.0
PUSH_TILT_RATE_DEG_S = 4.0
DRESS_SOFT_TILT_DEG = 4.0
DRESS_TILT_RATE_DEG_S = 6.0
DRESS_SYNC_ERROR_M = 0.12
DRESS_GRIP_SPAN_MAX_M = 0.65
DRESS_BASE_MIN_M = 0.38
APPROACH_GRIP_SPAN_MAX_M = 0.75
# Experimental corridor for the reported scene, not a mapped obstacle boundary.
# Route inside the reported collision at (-0.44, 0.04); validate in simulator.
TRANSIT_BX = 0.22
COLLISION_POINT = np.array([-0.44, 0.04])
COLLISION_CLEARANCE_M = 0.23
POLICY_VERSION = "hem-check-push-dress-candidate-v2"
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
    ("THU_DOI_HINH", "side", -0.20, TRANSIT_BX, 0.61, 0.69, 0.52, 0.52, 15),
    ("QUA_BAN_PHU",   "side",  0.42, TRANSIT_BX, 0.61, 0.69, 0.52, 0.52, 20),
    ("CANH_NGUOI",   "side",  0.42, 0.45, 0.61, 0.69, 0.52, 0.52, 15),
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


def wrist_hem(wy):
    """Fine row and local closed-gripper evidence; never use outside columns."""
    c0, c1 = HEM_COLS
    yy, xx = np.nonzero(wy[:, c0:c1])
    hemy = ((len(yy), float(np.median(yy)), float(np.median(xx) + c0))
            if len(yy) >= 80 else (len(yy), 0.0, 0.0))
    local = wy[PINCH_WIN[0]:PINCH_WIN[1], c0:c1]
    # Reject isolated yellow speckles: require horizontal and vertical support.
    count = int(local.sum())
    supported = (count >= CHECK_YELLOW_MIN
                 and np.count_nonzero(local.any(axis=0)) >= 4
                 and np.count_nonzero(local.any(axis=1)) >= 2)
    return hemy, count, bool(supported)


def inferred_grips(bp, jp):
    # Same geometric approximation as the original policy. Not force sensing.
    result = []
    for r in (0, 1):
        yaw = yaw_of(bp[r, 3:7])
        result.append(bp[r, :2] + (float(jp[r, 4:8].sum()) + C_GRIP)
                      * np.array([np.cos(yaw), np.sin(yaw)]))
    return np.array(result)


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
        self.local_yellow, self.local_hem_ok = 0, False
        self.tilt_rate, self.tilt_sample = 0.0, None
        self.push_start, self.push_goal, self.push_tilt0 = 0.0, 0.0, 0.0
        self.push_progress = (0.0, 0.0)
        self.seek_phase = 0
        self.last_step_t = None
        self.nav_y, self.nav_cy = None, None

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
        if self.last_step_t is not None and (t <= self.last_step_t or t - self.last_step_t > 0.15):
            self.ok = None
        self.last_step_t = t
        if self.tilt_sample is not None and t > self.tilt_sample[0]:
            raw_rate = (self.tilt - self.tilt_sample[1]) / (t - self.tilt_sample[0])
            self.tilt_rate = 0.5 * self.tilt_rate + 0.5 * raw_rate
        self.tilt_sample = (t, self.tilt)

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
                # Use one NAV snapshot for both camera and base terms.
                if self.nav_y is None or self.nav_cy is None:
                    if n <= 400:
                        self.go("RESET", t)
                        return
                    self.nav_y, self.nav_cy = float(bp[1]), float(cy)
                    print(f"[R{r}] CALIB NAV snapshot unavailable; use current REACH observation", flush=True)
                self.y_ref = self.nav_y
                raw = calibrated_reach(r, self.nav_cy, self.nav_y) + retry_off
                self.base_reach = float(np.clip(raw, 0.16, 0.44))
                print(f"[R{r}] CALIB cy={self.nav_cy:.2f} y_nav={self.nav_y:.4f} "
                      f"cy_ref={CY_REF[r]:.2f} y_ref={Y_NAV_REF[r]:.4f} "
                      f"raw={raw:.4f} clipped={self.base_reach:.4f}", flush=True)
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
                self.seek_phase = 0
            a[r, 3] = cmd(HOVER_LIFT - lift, RATE["lift"])
            self.wrist(jp, tw, a)
            drift_y = float(bp[1]) - self.y_ref
            offsets = (SEEK_TO, -0.16, 0.04)
            end = float(np.clip(self.base_reach + drift_y + offsets[self.seek_phase], 0.05, 0.48))
            if hem > self.best_hem:
                self.best_hem, self.best_arm = hem, arm
            self.seen = self.seen + 1 if hem >= HEM_N else 0
            if self.seen >= 2:
                print(f"[R{r}] thay GAU AO o dau ngon: vang={hem} arm={arm:.3f} -> ha xuong", flush=True)
                self.arm_t = arm
                self.go("LOWER", t)
            elif abs(arm - end) < 0.008:
                if self.seek_phase < 2:
                    self.seek_phase += 1
                    print(f"[R{r}] SEEK reverse/pass={self.seek_phase} max_yellow={self.best_hem}", flush=True)
                else:
                    self.fails += 1
                    self.go("RESET", t)
            else:
                if abs(self.arm_t - arm) < 0.012:
                    self.arm_t += float(np.clip(end - self.arm_t, -SEEK_V * 0.05, SEEK_V * 0.05))
            hold_arm()
            if t - self.t0 > 15.0 or abs(float(bp[1]) - self.y_ref) > 0.04:
                a[r, 4] = 0.0
                self.fails += 1
                self.go("RESET", t)

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
            if self.tilt - self.tilt0 > CONTACT_TILT_DELTA or (lift <= FAST_LIFT_TO and lift - self.lift_t > 0.02):
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
                self.ok = None
                if t - self.t0 > 1.5:
                    a[r, 4] = 0.0
                    self.fails += 1
                    self.go("RESET", t)
            else:
                err = row - FINE_ROWS[r]
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
            if t - self.t0 > 5.0 or abs(float(bp[1]) - self.y_ref) > 0.04:
                print(f"[R{r}] FINE unresolved -> RESET", flush=True)
                a[r, 4] = 0.0
                self.fails += 1
                self.go("RESET", t)

        elif s == "PUSH":
            if fresh:
                self.push_start = arm
                self.push_goal = arm + PUSH
                self.push_tilt0 = self.tilt
                self.push_progress = (t, arm)
                self.arm_t = arm
            a[r, 3] = cmd(self.contact + CONTACT_UP - lift, RATE["lift"], 3.0)
            self.wrist(jp, tw, a)
            unsafe = (self.tilt >= PUSH_ABS_TILT_DEG
                      or self.tilt - self.push_tilt0 >= PUSH_TILT_RISE_DEG
                      or (self.tilt_rate > PUSH_TILT_RATE_DEG_S
                          and self.tilt - self.push_tilt0 > 0.3)
                      or not 0 < PUSH <= 0.04
                      or arm > 0.505
                      or self.push_goal > 0.50
                      or abs(float(bp[1]) - self.y_ref) > 0.04)
            stalled = False
            if t - self.push_progress[0] >= 0.6:
                stalled = (self.arm_t - arm > PUSH_TOL_M
                           and arm - self.push_progress[1] < 0.001)
                self.push_progress = (t, arm)
            if unsafe or stalled or t - self.t0 > PUSH_TIMEOUT_S:
                print(f"[R{r}] PUSH abort: moved={arm-self.push_start:.4f} "
                      f"tilt={self.tilt:.2f} rate={self.tilt_rate:.2f} jam={stalled}", flush=True)
                a[r, 3:5] = 0.0
                self.arm_t = arm
                self.fails += 1
                self.go("RESET", t)
            elif self.push_goal - arm <= PUSH_TOL_M:
                self.arm_t = self.push_goal
                hold_arm()
                self.ok = t if self.ok is None else self.ok
                if t - self.ok >= 0.15:
                    print(f"[R{r}] PUSH complete: measured={arm-self.push_start:.4f}m", flush=True)
                    self.go("READY", t)
            else:
                self.ok = None
                if self.arm_t - arm <= 0.002:
                    self.arm_t = min(self.push_goal, self.arm_t + PUSH_STEP_M)
                a[r, 4] = float(np.clip(cmd(self.arm_t - arm, RATE["arm"], 3.5),
                                       -PUSH_SPEED_M_S / RATE["arm"],
                                       PUSH_SPEED_M_S / RATE["arm"]))

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
            if abs(err) < 0.02 and lift - self.contact >= 0.08:
                self.go("CHECK", t)
            elif t - self.t0 > 3.5:
                a[r, 3:5] = 0.0
                self.fails += 1
                self.go("RESET", t)

        elif s == "CHECK":
            a[r, 3] = cmd(tgt - lift, RATE["lift"], 4.0)
            self.wrist(jp, tw, a)
            hold_arm()
            finger_ok = float(jp[11]) >= KEP_HELD
            raised = abs(tgt - lift) < 0.02 and lift - self.contact >= 0.08
            wrist_ok = np.all(np.abs(tw - jp[8:11]) < 0.08)
            evidence = finger_ok and raised and wrist_ok and self.local_hem_ok
            # Wait for settling, then require continuous independent observations.
            if t - self.t0 < 0.8 or not evidence:
                self.ok = None
            elif self.ok is None:
                self.ok = t
            stable = self.ok is not None and t - self.ok >= CHECK_STABLE_S
            expired = t - self.t0 >= CHECK_TIMEOUT_S
            if stable or expired:
                good = bool(stable)
                print(f"[R{r}] CHECK cloth={cloth} local_yellow={self.local_yellow} "
                      f"finger={float(jp[11]):.3f} raised={raised} stable={stable} "
                      f"-> {'CAM DUOC' if good else 'TRUOT'}", flush=True)
                self.events.append(f"R{r}_{'CAM_DUOC' if good else 'TRUOT'}")
                self.tries += 1
                if self.first_ok is None:
                    self.first_ok = good
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
            evidence = float(jp[11]) >= KEP_LOST and self.local_hem_ok
            if evidence:
                self.lost = None
            else:
                self.lost = t if self.lost is None else self.lost
                grace = 0.3 if float(jp[11]) < KEP_LOST else 0.8
                if t - self.lost >= grace:
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
        if s == "NAV" and self.s == "OPEN":
            if n > 400:
                self.nav_y, self.nav_cy = float(bp[1]), float(cy)
            else:
                self.nav_y, self.nav_cy = None, None
                self.go("RESET", t)



def calibrated_reach(r, cy, y_nav):
    return (D_REACH_BASE[r] + K_D_PER_CY[r] * (cy - CY_REF[r])
            + BASE_Y_COEFF[r] * (y_nav - Y_NAV_REF[r]))


def apply_params(path):
    """Embedded candidate by default; explicit JSON may replace it as a whole."""
    global CALIBRATION, HEAD_T, D_REACH_BASE, K_D_PER_CY, CY_REF, Y_NAV_REF, BASE_Y_COEFF, FINE_ROWS, LIFT_CONTACT_MEDIAN
    if path is not None:
        if not str(path).endswith(".json"):
            raise ValueError("This policy accepts only an explicit candidate JSON checkpoint")
        with open(path, "r", encoding="utf-8-sig") as f:
            candidate = json.load(f)
        if candidate.get("schema") != "phyrc-reach-candidate-v2":
            raise ValueError("Expected phyrc-reach-candidate-v2; cannot mix legacy coefficients with NAV compensation")
        rows = tuple(candidate["robots"][f"robot_{r}"] for r in (0, 1))
        for c in rows:
            values = [c[k] for k in ("D_reach_base", "k_D_per_cy", "cy_ref", "y_nav_ref", "base_y_coefficient", "fine_row_median", "lift_contact_median")]
            if not np.isfinite(values).all() or len(c["head_target_px"]) != 2 or not np.isfinite(c["head_target_px"]).all():
                raise ValueError("Nonfinite or invalid candidate parameters")
        CALIBRATION = rows
        HEAD_T = tuple(tuple(c["head_target_px"]) for c in rows)
        D_REACH_BASE = tuple(c["D_reach_base"] for c in rows)
        K_D_PER_CY = tuple(c["k_D_per_cy"] for c in rows)
        CY_REF = tuple(c["cy_ref"] for c in rows)
        Y_NAV_REF = tuple(c["y_nav_ref"] for c in rows)
        BASE_Y_COEFF = tuple(c["base_y_coefficient"] for c in rows)
        FINE_ROWS = tuple(c["fine_row_median"] for c in rows)
        LIFT_CONTACT_MEDIAN = tuple(c["lift_contact_median"] for c in rows)
    print(f"[CALIB] EXPERIMENTAL candidate; source={path or 'embedded'} "
          f"D={D_REACH_BASE} K={K_D_PER_CY} CY_REF={CY_REF} "
          f"Y_NAV_REF={Y_NAV_REF} BASE_Y_COEFF={BASE_Y_COEFF} FINE_ROWS={FINE_ROWS}", flush=True)
    return CALIBRATION


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
            self.pause_jp, self.pause_reason = None, None
            self.dress_entry_bp, self.stage_entry_bp = None, None
            self.dress_lost = [None, None]
            self.tilt_prev, self.tilt_rates = None, np.zeros(2)
            print(f"[POLICY] {POLICY_VERSION} D={D_REACH_BASE} K={K_D_PER_CY}", flush=True)

        def set_mode(self, m, t):
            self.mode, self.t_mode = m, t
            print(f"[PAIR] >>> {m} t={t:.1f}", flush=True)
            self.events.append(f"PAIR_{m}")

        def pause(self, t, jp, reason):
            if self.mode != "PAUSED":
                self.pause_jp = jp.copy()
                self.pause_reason = reason
                print(f"[PAIR] PAUSED: {reason}", flush=True)
                self.set_mode("PAUSED", t)

        def paused_action(self, jp):
            # Preserve gripper intent; no blind navigation/release near mannequin.
            action = np.zeros((2, 9))
            if self.pause_jp is not None:
                for r in (0, 1):
                    action[r, 3] = cmd(self.pause_jp[r, 3] - jp[r, 3], RATE["lift"], 3.0)
                    action[r, 4] = float(np.clip(cmd(self.pause_jp[r, 4:8].sum() - jp[r, 4:8].sum(), RATE["arm"], 3.0), -.1, .1))
                    self.arms[r].wrist(jp[r], self.pause_jp[r, 8:11], action)
            return action

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
                    box(ims[1 + r], 0, 256, *HEM_COLS, (255, 0, 255))
                    box(ims[1 + r], PINCH_WIN[0], PINCH_WIN[1], *HEM_COLS, (255, 255, 0))
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
                hem.append(int(wy[r0w:r1w, HEM_COLS[0]:HEM_COLS[1]].sum()))
                self.arms[k].pinch_cloth = int((wb | wy)[pr0:pr1, pc0:pc1].sum())
                self.arms[k].pinch_yellow = int(wy[pr0:pr1, pc0:pc1].sum())
                ys_, xs_ = np.nonzero((wb | wy)[120:190])
                hemy, local_yellow, local_ok = wrist_hem(wy)
                self.arms[k].hemy = hemy
                self.arms[k].local_yellow = local_yellow
                self.arms[k].local_hem_ok = local_ok
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
            tilts = np.array([robot.tilt for robot in self.arms])
            if self.tilt_prev is not None and t > self.tilt_prev[0]:
                rates = (tilts - self.tilt_prev[1]) / (t - self.tilt_prev[0])
                self.tilt_rates = .5 * self.tilt_rates + .5 * rates
            self.tilt_prev = (t, tilts.copy())
            if self.mode == "DRESS":
                for r in (0, 1):
                    if tilts[r] >= DRESS_SOFT_TILT_DEG or (tilts[r] > 2.5 and self.tilt_rates[r] > DRESS_TILT_RATE_DEG_S):
                        self.pause(t, JP, f"R{r} transit tilt={tilts[r]:.2f} rate={self.tilt_rates[r]:.2f}")
                    # Wrist orientation changes during dressing, so do not reuse
                    # the LOW_WRIST yellow ROI as a mandatory dressing detector.
                    if JP[r, 11] < KEP_LOST:
                        self.dress_lost[r] = t if self.dress_lost[r] is None else self.dress_lost[r]
                        if t - self.dress_lost[r] >= .3:
                            self.pause(t, JP, f"R{r} gripper closed below loss threshold")
                    else:
                        self.dress_lost[r] = None
            if self.mode == "PAUSED":
                return self.paused_action(JP)

            def follow_r0():
                ex1 = float(BP[0, 0]) + R1_OFFSET_X - float(BP[1, 0])
                ey1 = float(BP[0, 1]) - float(BP[1, 1])
                drive(a, 1, 1.5 * ex1, 1.5 * ey1, yaw[1])
                a[1, 3] = cmd(LIFT_T - lift[1], RATE["lift"])
                return np.hypot(ex1, ey1)

            dress_commands = False
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
                    measured_span = float(np.linalg.norm(inferred_grips(BP, JP)[0] - inferred_grips(BP, JP)[1]))
                    self.S0 = measured_span  # do not disguise 0.62m as 0.55m
                    self.dress_entry_jp = JP.copy()
                    self.dress_entry_bp = BP[:, :2].copy()
                    self.stage_entry_bp = BP[:, :2].copy()
                    print(f"[PAIR] 2 kep cach nhau {self.S0:.2f} m luc gap -> giu dung khoang nay khi mac ao", flush=True)
                    self.set_mode("DRESS", t)
                    if not 0.25 <= self.S0 <= DRESS_GRIP_SPAN_MAX_M:
                        self.pause(t, JP, f"invalid initial grip separation {self.S0:.3f}")
                        return self.paused_action(JP)
                    print(f"[PAIR] buoc 1/{len(KF)}: {KF[0][0]}", flush=True)
                elif A0.s == "HOLD" and t - A0.hold_t > PAIR_WAIT:
                    print("[PAIR] robot_1 khong gap duoc -> robot_0 giu ao (chi diem gap)", flush=True)
                    self.set_mode("SOLO", t)

            elif self.mode == "SOLO":
                A0.step(t, JP[0], BP[0], float(BP[1, 0]), *hd[0], cloth[0], hem[0], False, False, a)

            elif self.mode == "DRESS":
                dress_commands = True
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
                if name == "RA_CHO_TRONG":
                    l0, l1 = self.dress_entry_jp[:, 3]
                    m0, m1 = self.dress_entry_jp[:, 4:8].sum(axis=1)
                done = True
                for r, l, m in ((0, l0, m0), (1, l1, m1)):
                    e = tgt[r] - BP[r, :2]
                    dist = float(np.hypot(*e))
                    if self.stuck[r] is None or t - self.stuck[r][0] > 2.0:
                        if self.stuck[r] is not None and dist > 0.1 and \
                                np.hypot(*(BP[r, :2] - self.stuck[r][1])) < 0.02 and not self.blocked[r]:
                            print(f"[R{r}] bi chan o ({BP[r,0]:.2f},{BP[r,1]:.2f}), con cach {dist:.2f} m -> dung lai", flush=True)
                            self.blocked[r] = True
                            self.pause(t, JP, f"R{r} blocked during {name}")
                            return self.paused_action(JP)
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
                        mt = m if name == "RA_CHO_TRONG" else min(m, 0.08)
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
                    if dist > POS_TOL or traveling or abs(l - lift[r]) > LIFT_TOL or abs(m - arm[r]) > ARM_TOL or abs(wrap(yt - yaw[r])) > 0.08:
                        done = False
                    if float(JP[r, 11]) < KEP_LOST and int(t * 20) % 40 == 0:
                        print(f"[R{r}] ngon kep khep het ({float(JP[r, 11]):.3f}) - da tuot ao", flush=True)
                if int(t * 20) % 40 == 0:
                    g = [BP[r, :2] + (arm[r] + C_GRIP) * np.array([np.cos(yaw[r]), np.sin(yaw[r])]) for r in (0, 1)]
                    print(f"[PAIR] {name}: 2 kep cach nhau {np.hypot(*(g[0] - g[1])):.2f} m (luc gap {S:.2f}) "
                          f"yaw=({np.degrees(yaw[0]):.0f},{np.degrees(yaw[1]):.0f})", flush=True)
                wt = (self.dress_entry_jp[:, 8:11] if name == "RA_CHO_TRONG" else WRIST.get(name, W_GIU))
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
                if late and not done:
                    self.pause(t, JP, f"waypoint {name} timeout")
                    return self.paused_action(JP)
                if done:
                    self.events.append(f"xong_{self.ki + 1:02d}_{name}")
                    self.ki += 1
                    self.t_kf = t
                    self.stuck, self.blocked = [None, None], [False, False]
                    self.stage_entry_bp = BP[:, :2].copy()
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

            # Apply coordination constraints after both robots have generated actions.
            predicted_bp = BP.copy()
            predicted_jp = JP.copy()
            horizon = 0.25
            for r in (0, 1):
                fwd, left, turn = a[r, :3]
                predicted_bp[r, 0] += horizon * (fwd*np.cos(yaw[r]) - left*np.sin(yaw[r]))
                predicted_bp[r, 1] += horizon * (fwd*np.sin(yaw[r]) + left*np.cos(yaw[r]))
                next_yaw = yaw[r] + horizon*turn
                predicted_bp[r, 3:7] = (np.cos(next_yaw/2), 0, 0, np.sin(next_yaw/2))
                predicted_jp[r, 4:8] += horizon * a[r, 4] * RATE["arm"] / 4
            base_sep = float(np.linalg.norm(BP[0, :2] - BP[1, :2]))
            predicted_base_sep = float(np.linalg.norm(predicted_bp[0, :2] - predicted_bp[1, :2]))
            if predicted_base_sep < DRESS_BASE_MIN_M and predicted_base_sep < base_sep:
                a[:, :3] = 0.0
            if self.mode == "GRASP" and all(robot.s not in ("IDLE", "LOST") for robot in self.arms):
                now_span = np.linalg.norm(inferred_grips(BP, JP)[0] - inferred_grips(BP, JP)[1])
                pred_span = np.linalg.norm(inferred_grips(predicted_bp, predicted_jp)[0] - inferred_grips(predicted_bp, predicted_jp)[1])
                if pred_span > APPROACH_GRIP_SPAN_MAX_M and pred_span > now_span:
                    a[:, :3] = 0.0
            both_held = all(robot.s in ("HOLD", "DRESS") for robot in self.arms)
            if both_held:
                current_span = float(np.linalg.norm(inferred_grips(BP, JP)[0] - inferred_grips(BP, JP)[1]))
                next_span = float(np.linalg.norm(inferred_grips(predicted_bp, predicted_jp)[0] - inferred_grips(predicted_bp, predicted_jp)[1]))
                upper = min(DRESS_GRIP_SPAN_MAX_M, self.S0 + .06) if self.mode == "DRESS" else DRESS_GRIP_SPAN_MAX_M
                if next_span > upper and next_span > current_span:
                    a[:, :3] = 0.0
                    a[:, 4] = 0.0
                if self.mode == "DRESS" and current_span > upper + .03:
                    self.pause(t, JP, f"grip separation {current_span:.3f}")
            if self.mode == "DRESS" and dress_commands:
                # Slow/stop the leading base if its partner falls behind.
                progress = BP[:, :2] - self.stage_entry_bp
                direction = np.asarray(tgt) - self.stage_entry_bp
                lengths = np.linalg.norm(direction, axis=1)
                fraction = np.clip(np.sum(progress * direction, axis=1)
                                   / np.maximum(lengths ** 2, 1e-6), 0.0, 1.0)
                # Compare progress relative to each path length. Comparing raw
                # metres deadlocks valid asymmetric waypoint paths.
                if min(lengths) > POS_TOL and abs(fraction[0] - fraction[1]) * max(lengths) > DRESS_SYNC_ERROR_M:
                    leader = int(np.argmax(fraction))
                    a[leader, :3] = 0.0
                # Local exclusion around the user-reported collision point.
                # This is NOT a complete obstacle map or a clearance guarantee.
                if name in ("THU_DOI_HINH", "QUA_BAN_PHU", "CANH_NGUOI"):
                    for r in (0, 1):
                        now_dist = np.linalg.norm(BP[r, :2] - COLLISION_POINT)
                        pred_dist = np.linalg.norm(predicted_bp[r, :2] - COLLISION_POINT)
                        if pred_dist < COLLISION_CLEARANCE_M and pred_dist < now_dist:
                            self.pause(t, JP, f"R{r} approaching reported side-table collision region")
            if self.mode == "PAUSED":
                return self.paused_action(JP)

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
