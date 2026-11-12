import os, zlib, struct
import numpy as np

# ===== SỐ ĐO TỪ 3 DEMO CỦA BẠN =====
HEAD_CX = 115                 # vị trí ngang của áo trong cam đầu lúc gắp
HEAD_CY = 177                 # vị trí dọc của áo trong cam đầu lúc gắp
LIFT_T  = 0.54                # độ cao lift khi căn áo
# Mỗi "công thức gắp" = tư thế thật lúc BẠN đóng tay trong demo:
# (lift, tầm với R, cổ tay yaw, pitch, roll).  R = (-y của robot) + arm
RECIPES = [
    (0.537, 0.94, -0.327,  0.287,  2.761),   # demo2 robot_0 - DA GAP THANH CONG
    (0.536, 0.99, -0.275,  0.283,  0.368),   # demo2 robot_1
    (0.536, 0.93, -0.275,  0.283,  0.300),   # nghiêng cổ tay như demo2, xoay nhẹ
    (0.460, 0.89,  0.001, -0.266,  0.220),   # demo1 robot_1
    (0.536, 0.88,  0.001,  0.010, -0.835),   # demo3 robot_0
    (0.600, 0.98,  0.000,  0.020,  1.940),   # demo3 robot_1
]
WAIT_S  = 1.7                 # đứng yên sau khi đóng tay v
LIFT_UP = 0.15                # nhấc thêm sau khi gắp
HOLD_S  = 3.5
CHK_RATIO = 0.85              # sau khi nâng, vải phải còn >= 85% trong cam cổ tay
SCAN_V  = 0.8
SCAN_L  = 5.0
SCAN_R  = 5.0
FWD0    = 0.4
SEE_N   = 1000
SIGN_FWD = 1.0
SIGN_STR = 1.0
SIGN_ARM = 1.0
DEBUG_IMG = True              # CHỈ để debug: đặt False trước khi nộp bài
IMG_DIR = "/output/gap_dbg"
# =====================================
RATE = dict(lift=1.54, arm=1.21)

def cmd(err, rate, gain=6.0):
    return float(np.clip(gain * err / rate, -1, 1))

def masks(img):
    r, g, b = [img[..., i].astype(int) for i in range(3)]
    return (b > r + 60) & (b > g + 60), (r > 170) & (g > 170) & (b < 110)

def yaw_of(q):
    return float(np.arctan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2)))

def save_png(path, arr):
    arr = np.ascontiguousarray(arr.astype(np.uint8))
    h, w, _ = arr.shape
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 3)) + chunk(b"IEND", b""))
    with open(path, "wb") as fh:
        fh.write(png)

def load_policy(checkpoint, observation_space, action_space):
    class Policy:
        def __init__(self):
            self.reset()

        def reset(self):
            self.s, self.t0, self.ok, self.lost = "SCAN", 0.0, None, None
            self.first, self.p0, self.d = True, None, 1
            self.k, self.cmax, self.opened, self.cpre = 0, 0, False, 0
            self.sheet = None
            self.wsgn, self.wu, self.we, self.wt = np.ones(3), np.zeros(3), None, None
            self.arm_t = 0.3

        def go(self, s, t):
            self.s, self.t0, self.ok, self.lost, self.opened = s, t, None, None, False
            print(f"[DBG] >>> {s} t={t:.1f} lan_gap={self.k}", flush=True)

        def wrist(self, jp, tgt, t, a):
            # điều khiển 3 khớp cổ tay (yaw, pitch, roll); tự đổi chiều nếu thấy sai số tăng
            q = np.array([jp[8], jp[9], jp[10]], dtype=float)
            err = tgt - q
            u = np.clip(1.5 * err, -0.5, 0.5)
            u = np.where(np.abs(err) < 0.03, 0.0, u)
            if self.wt is None or t - self.wt >= 0.6:
                if self.we is not None:
                    for j in range(3):
                        if abs(self.wu[j]) > 0.15 and abs(err[j]) > abs(self.we[j]) + 0.04:
                            self.wsgn[j] *= -1.0
                            print(f"[DBG] doi chieu co_tay khop {j}", flush=True)
                self.we, self.wt, self.wu = err.copy(), t, u.copy()
            a[0, 5:8] = self.wsgn * u
            return err

        def dump(self, rgb, col):
            if not DEBUG_IMG:
                return
            try:
                if self.sheet is None:
                    self.sheet = np.full((len(RECIPES) * 512, 3 * 256, 3), 255, np.uint8)
                r0, c0 = (self.k % len(RECIPES)) * 512, col * 256
                self.sheet[r0:r0 + 256, c0:c0 + 256] = rgb[1]
                self.sheet[r0 + 256:r0 + 512, c0:c0 + 256] = rgb[3]
                os.makedirs(IMG_DIR, exist_ok=True)
                save_png(IMG_DIR + "/sheet.png", self.sheet)
            except Exception as e:
                print("[DBG] khong luu duoc anh:", repr(e), flush=True)

        def __call__(self, obs):
            try:
                a = self.step(obs)
            except Exception as e:
                print("[DBG] loi:", repr(e), flush=True)
                a = np.zeros((2, 9))
            return np.clip(np.nan_to_num(a), -1, 1).astype(np.float32).reshape(2, 9)

        def step(self, obs):
            t = float(np.asarray(obs["simulation_time_s"]).ravel()[0])
            jp = np.asarray(obs["joint_position"]).ravel()[:13]
            lift, arm = float(jp[3]), float(jp[4:8].sum())
            bp = np.asarray(obs["base_pose_world"]).ravel()[:7]
            if self.p0 is None:
                y0 = yaw_of(bp[3:7])
                self.p0 = (float(bp[0]), float(bp[1]), float(np.cos(y0)), float(np.sin(y0)))
            x0, yy0, fx, fy = self.p0
            dx, dy = float(bp[0]) - x0, float(bp[1]) - yy0
            dl = -dx * fy + dy * fx
            df = dx * fx + dy * fy
            rgb = np.asarray(obs["rgb"])
            a = np.zeros((2, 9))
            if self.first:
                self.first = False
                ovm = masks(rgb[0])[0]
                ov = int(ovm.sum())
                ovx = float(np.nonzero(ovm)[1].mean()) if ov > 0 else -1.0
                self.d = 1 if ov >= 30 else -1       # camera tổng quan thấy áo => áo ở bên trái, ngược lại bên phải
                print(f"[DBG] rgb shape {rgb.shape} | xanh tong quan={ov} cx_tq={ovx:.0f} -> quet {'trai' if self.d > 0 else 'phai'} truoc "
                      f"| co_tay+kep luc dau={np.round(jp[8:13], 3).tolist()}", flush=True)
            hb, _ = masks(rgb[3])
            wb, wy = masks(rgb[1])
            cloth = int(wb.sum() + wy.sum())
            n = int(hb.sum())
            if n > 400:
                ys, xs = np.nonzero(hb)
                cx, cy = float(xs.mean()), float(ys.mean())
            else:
                cx = cy = -1.0
            ex, ey = cx - HEAD_CX, HEAD_CY - cy
            lt, Rt, wyaw, wpitch, wroll = RECIPES[self.k % len(RECIPES)]
            tgt_w = np.array([wyaw, wpitch, wroll])
            tgt = lt + LIFT_UP
            s = self.s

            def hold_arm():
                a[0, 4] = SIGN_ARM * float(np.clip(cmd(self.arm_t - arm, RATE["arm"], 3.0), -0.5, 0.5))

            if s == "SCAN":                       # 1. tìm áo
                a[0, 3] = cmd(LIFT_T - lift, RATE["lift"])
                if n > SEE_N:
                    print(f"[DBG] thay ao: trai={dl:.2f} n={n} cx={cx:.0f}", flush=True)
                    self.go("NAV", t)
                else:
                    if df < FWD0:
                        a[0, 0] = 0.3 * SIGN_FWD
                    if self.d > 0 and dl > SCAN_L:
                        self.d = -1
                    elif self.d < 0 and dl < -SCAN_R:
                        self.d = 1
                    a[0, 1] = SIGN_STR * self.d * SCAN_V

            elif s == "NAV":                      # 2. căn áo vào đúng vị trí trong cam đầu
                a[0, 3] = cmd(LIFT_T - lift, RATE["lift"])
                if n > 400:
                    self.lost = None
                    a[0, 1] = SIGN_STR * (-np.clip(ex / 80, -1, 1)) * 0.5
                    a[0, 0] = SIGN_FWD * np.clip(ey / 60, -1, 1) * 0.5
                    if abs(ex) < 10 and abs(ey) < 10 and abs(LIFT_T - lift) < 0.02:
                        self.ok = self.ok or t
                        if t - self.ok > 0.5:
                            self.go("OPEN", t)
                    else:
                        self.ok = None
                else:
                    self.lost = self.lost or t
                    if t - self.lost > 2:
                        self.go("SCAN", t)
                if t - self.t0 > 40:
                    self.go("OPEN", t)

            elif s == "OPEN":                     # 3. mở tay gắp cho chắc
                a[0, 3] = cmd(LIFT_T - lift, RATE["lift"])
                if not self.opened:
                    a[0, 8] = -1.0
                    self.opened = True
                if t - self.t0 > 1.0:
                    self.go("PREP", t)

            elif s == "PREP":                     # 4. xoay cổ tay + lift về đúng tư thế của công thức
                el, ea = lt - lift, 0.02 - arm
                a[0, 3] = cmd(el, RATE["lift"])
                a[0, 4] = SIGN_ARM * cmd(ea, RATE["arm"])
                werr = self.wrist(jp, tgt_w, t, a)
                if abs(el) < 0.02 and abs(ea) < 0.03 and np.all(np.abs(werr) < 0.06):
                    self.ok = self.ok or t
                    if t - self.ok > 0.3:
                        self.go("REACH", t)
                else:
                    self.ok = None
                if t - self.t0 > 14:
                    self.go("REACH", t)

            elif s == "REACH":                    # 5. duỗi tay tới đúng tầm với R
                self.arm_t = float(np.clip(Rt + float(bp[1]), 0.05, 0.50))
                a[0, 3] = cmd(lt - lift, RATE["lift"])
                self.wrist(jp, tgt_w, t, a)
                ea = self.arm_t - arm
                a[0, 4] = SIGN_ARM * float(np.clip(cmd(ea, RATE["arm"], 3.0), -0.5, 0.5))
                if abs(ea) < 0.012:
                    self.ok = self.ok or t
                    if t - self.ok > 0.5:
                        self.go("SETTLE", t)
                else:
                    self.ok = None
                if t - self.t0 > 12:
                    self.go("SETTLE", t)

            elif s == "SETTLE":                   # 6. đứng yên cho tay ngừng trôi
                a[0, 3] = cmd(lt - lift, RATE["lift"])
                self.wrist(jp, tgt_w, t, a)
                hold_arm()
                if t - self.t0 > 1.0:
                    self.go("CLOSE", t)

            elif s == "CLOSE":                    # 7. đóng tay đúng 1 lần
                a[0, 8] = 1.0
                self.cmax = cloth
                print(f"[DBG] CLOSE k={self.k} lift={lift:.3f} arm={arm:.3f} y={float(bp[1]):.2f} "
                      f"R={-float(bp[1]) + arm:.3f} co_tay_kep={np.round(jp[8:13], 3).tolist()} cloth={cloth}", flush=True)
                self.dump(rgb, 0)
                self.go("WAIT", t)

            elif s == "WAIT":                     # 8. đứng yên chờ gắp
                self.cmax = max(self.cmax, cloth)
                a[0, 3] = cmd(lt - lift, RATE["lift"])
                self.wrist(jp, tgt_w, t, a)
                hold_arm()
                if t - self.t0 > WAIT_S:
                    self.cpre = cloth
                    self.dump(rgb, 1)
                    self.go("LIFT", t)

            elif s == "LIFT":                     # 9. nâng lên mượt
                err = tgt - lift
                a[0, 3] = cmd(err, RATE["lift"], 4.0)
                self.wrist(jp, tgt_w, t, a)
                hold_arm()
                if abs(err) < 0.02 or t - self.t0 > 5:
                    self.go("CHECK", t)

            elif s == "CHECK":                    # 10. vải còn theo tay không
                a[0, 3] = cmd(tgt - lift, RATE["lift"], 4.0)
                self.wrist(jp, tgt_w, t, a)
                hold_arm()
                if t - self.t0 > 1.0:
                    self.dump(rgb, 2)
                    if cloth >= CHK_RATIO * self.cpre and cloth >= 2500:
                        self.go("HOLD", t)
                    else:
                        self.go("RESET", t)

            elif s == "HOLD":                     # 11. giữ
                a[0, 3] = cmd(tgt - lift, RATE["lift"], 4.0)
                self.wrist(jp, tgt_w, t, a)
                hold_arm()
                if cloth < 0.7 * self.cpre:
                    self.go("RESET", t)
                elif t - self.t0 > HOLD_S:
                    self.go("DONE", t)

            elif s == "RESET":                    # gắp trượt: mở tay, rút tay về, thử công thức kế tiếp
                nxt = RECIPES[(self.k + 1) % len(RECIPES)][0]
                if not self.opened:
                    a[0, 8] = -1.0
                    self.opened = True
                a[0, 3] = cmd(nxt - lift, RATE["lift"])
                a[0, 4] = SIGN_ARM * cmd(0.02 - arm, RATE["arm"])
                self.wrist(jp, tgt_w, t, a)
                done = abs(arm - 0.02) < 0.03 and abs(nxt - lift) < 0.03 and t - self.t0 > 2
                if done or t - self.t0 > 10:
                    self.k += 1
                    self.go("PREP", t)

            if int(t * 20) % 20 == 0:
                print(f"[DBG] t={t:.1f} {self.s} k={self.k} lift={lift:.2f} arm={arm:.2f} y={float(bp[1]):.2f} trai={dl:.2f} "
                      f"co_tay={np.round(jp[8:11], 2).tolist()} kep={np.round(jp[11:13], 2).tolist()} "
                      f"cam_dau n={n} cx={cx:.0f} cy={cy:.0f} cam_tay={cloth} cpre={self.cpre}", flush=True)
            return a
    return Policy()
