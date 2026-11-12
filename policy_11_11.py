# PhyRC 2027 - policy: 2 robot cung gap 2 ben GAU AO, xoay ao, dua toi 2 tay nguoi nom va keo ao vao
# Chi dung obs duoc phep: rgb 5 camera, joint_position, base_pose_world cua chinh robot, simulation_time_s.
# Khong doc vi tri ao / nguoi tu mo phong, khong chay lai quy dao demo.
import os, time, zlib, struct
import numpy as np

# ============================== ANH DEBUG ==============================
SNAP = True                     # dat False truoc khi nop bai
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
GRIP_CLOTH_MIN = 30             # so pixel vai toi thieu trong vung GIUA hai ngon, khong phai toan camera
GRIP_CONFIRM_S = 0.3            # phai thay vai lien tuc moi cho dong kep / xac nhan sau nang
GRIP_LOST_S = 0.5               # HOLD: mat vai giua ngon qua thoi gian nay -> thu lai
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

# ============================== GIAI DOAN (09/11) ==============================
# Chi la LOP DANH DAU tren may trang thai co san (khong doi hanh vi), de sau nay sua tung giai doan rieng:
#   GAP      : mode SCAN / GRASP / SOLO   = tim ao + toan bo cac state gap (NAV ... HOLD)  -> tag gd1_gap_ok
#   MANG     : KF RA_CHO_TRONG .. XOAY_XONG = nhac ao khoi ban, mang ra cho trong, xoay ao 180 do
#   CHUAN_BI : KF RA_TRUOC .. NANG_CAO      = dua ao toi truoc / canh nguoi, nang cao - DUNG TRUOC KHI CHAM NGUOI
#   MAC      : KF XO_TAY_1 .. KEO_XUONG_2   = xo tay, qua dau, sau lung, keo xuong (phan mac ao that)
#   XONG     : mode RELEASE / DONE          = tha ao
# (mode GIU cua dress_steps: giu nguyen giai doan dang lam do)
GIAI_DOAN = ("GAP", "MANG", "CHUAN_BI", "MAC", "XONG")
KF_TEN = [k[0] for k in KF]
KF_I_CHUAN_BI = KF_TEN.index("RA_TRUOC")    # buoc KF dau tien cua CHUAN_BI
KF_I_MAC = KF_TEN.index("XO_TAY_1")         # buoc KF dau tien cua MAC
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

# ============================== MANG MOI (10/11, theo demo7) ==============================
# Ban cu (KF XOAY_1..XOAY_XONG): 2 de di vong doi cho nhung THAN KHOA o -90 do -> duong noi 2 kep quay 180 do
# con moi kep giu nguyen huong -> moi dau ao bi van nua vong, mat mieng ao (log gd0_mac_cu 74-83 s).
# Ban moi = QUAY CUNG ca khoi (2 robot + ao) quanh tam de R1: R1 chi xoay than tai cho, R0 di vong quanh R1,
# goc than R0 quay CUNG goc voi R1 -> khoang cach + huong 2 kep so voi ao khong doi. Co tay giu nguyen goc luc gap.
#   NHAC  : nang lift MANG_LIFT, rut tay tu tu ve MANG_ARM (de dung yen)
#   LUI   : ca khoi lui MANG_LUI m nguoc huong tay (ra khoi ban)
#   XOAY  : quay ca khoi 180 do quanh de R1, chieu quay sao cho R0 di vong phia KHONG co ao/ban
#   VE_LAN: ca khoi di toi y = MANG_LAN_Y (lan giua ban va nguoi), giu nguyen goc
#   DI_TOI: ca khoi di toi tam (PX, MANG_Y) truoc mat nguoi, giu nguyen goc
MANG_MOI = False            # json "mang_moi": true -> dung MANG moi thay cho KF RA_CHO_TRONG..XOAY_XONG
DUNG_SAU_MANG = False       # json "dung_sau_mang": true -> xong MANG thi dung yen giu ao (chua CHUAN_BI / MAC)
MANG_LIFT, MANG_ARM, MANG_ARM_V = 0.70, 0.06, 0.05
MANG_LUI, MANG_W, MANG_V = 0.35, 0.20, 0.20
MANG_LAN_Y, MANG_Y = -0.60, -0.65
MANG_BUOC = (("NHAC", 15), ("LUI", 12), ("XOAY", 45), ("VE_LAN", 12), ("DI_TOI", 45))
# LAT_KEP (10/11): sau XOAY, robot dang dung PHIA NGOAI ao (ao giua robot va nguoi). De giong demo (2 robot dung 2 ben nguoi,
# tay chia ve -y, ao o truoc nguoi) moi robot phai TU DI VONG quanh CHINH KEP CUA NO: kep dung yen, de di vong phia ngoai
# (R0 phia +x, R1 phia -x), than quay ~ +163 / -162 do. Lan luot: R0 truoc, R1 dung giu. Xong thi ca cap TINH TIEN (khong quay):
#   VE_LAN (y de = LAT_LAN_Y, truoc ban) -> DI_NGANG (x = PX) -> DI_TOI (y de = LAT_DICH_Y, 2 ben nguoi, ao chua cham nguoi)
LAT_KEP = False             # json "lat_kep": true
LAT_YAW = [-106.0, -71.0]   # goc than cuoi (do) R0 / R1 - nhu demo7 luc ao cham nguoi
LAT_LIFT, LAT_W = 0.85, 0.20
LAT_LAN_Y, LAT_DICH_Y = -0.35, -0.10
LAT_CANH_BAO = 0.03         # kep lech qua muc nay (m) -> in CANH BAO
MANG_BUOC_LAT = (("NHAC", 15), ("LUI", 12), ("XOAY", 45), ("LAT0", 40), ("LAT1", 40),
                 ("VE_LAN", 20), ("DI_NGANG", 45), ("DI_TOI", 20))
# MO_KHE (10/11): ngay sau LAT1, 2 de DUNG YEN; nang lift ca 2 len KHE_LIFT, roi xoay co tay 2 robot CUNG LUC, cung luong,
# moi nac KHE_NAC do toi KHE_MAX do, thu 3 cach: chi pitch, chi roll, ca hai. Moi nac in [KHE] (xanh_dam 2 co tay = thay long ao).
# Xong chon goc co min(xanh_dam R0, R1) lon nhat, giu goc do (w_mang) cho cac buoc sau.
MO_KHE = False              # json "mo_khe": true -> chen buoc MO_KHE sau LAT1
KHE_DICH = None             # json "khe_dich": [[yaw,pitch,roll] R0, [..] R1] goc TUYET DOI cho cach "dich" (vd demo7: R0 (0,0.30,1.36), R1 (0,-0.25,-1.13))
KHE_SAU = "LAT1"            # json "khe_sau": buoc dung truoc MO_KHE ("LAT1" = canh ban; "DI_TOI" = da toi 2 ben nguoi)
# DIEM LUU (chi khi phat trien, xem chay_tu_moc.sh): json "luu_moc": true -> policy dat self.moc_can = ten moc,
# chay_moc.py thay thi luu trang thai mo phong + bien policy. evaluate_policy.py (bai nop) bo qua thuoc tinh nay.
LUU_MOC = False
# MO KHE RONG (10/11 toi): sau khi van kep xong (GIU), 2 de nhich lai gan nhau DEU 2 BEN cho 2 kep cach nhau KHE_KEP m
# (None = giu nguyen) -> gau chung xuong, 2 lop tach ra. Do khe camera dau KHE_DO_T s (trung binh 2 s cuoi) roi luu moc_mo_khe_xong.
KHE_KEP = None              # json "khe_kep": vd 0.42
KHE_DO_T = 8.0              # json "khe_do_t": s tu luc bat dau GIU toi luc chot so do
# XO TAY (11/11, giai doan MAC doan dau): sau GIU do khe -> mode XO_TAY: NANG (2 kep cao hon tay XT_CAO) -> TIEN (ca cap tinh tien,
# KHONG quay, giua 2 kep toi tren tay) -> HA (lift ha XT_V m/s, XT_HA m) -> giu XT_GIU s -> luu moc_xo_tay_xong.
# TAM THOI: vi tri tay nguoi XT_TAY lay tu metadata danh gia (arms_world_m, seed 42) - CHI de thu; sau nay thay bang camera.
XO_TAY = False              # json "xo_tay"
XT_TAY = (0.11, -0.03, 1.14)  # json "xt_tay": [x, y, z] giua 2 dau ban tay nguoi (the gioi)
XT_LECH = (0.0, 0.0)        # json "xt_lech": [dx, dy] lech dich giua 2 kep so voi XT_TAY
XT_CAO = 0.10               # json "xt_cao": giua 2 kep cao hon tay bao nhieu luc NANG
XT_V = 0.02                 # json "xt_v": toc do ha lift (m/s)
XT_HA = 0.25                # json "xt_ha": ha tong cong bao nhieu (m)
XT_V_DE = 0.05              # json "xt_v_de": toc do tinh tien de (m/s)
XT_GIU = 4.0                # json "xt_giu": s dung yen sau khi ha xong
XT_LIFT_MAX = 1.15          # json "xt_lift_max" (demo7 R0 len 1.13)
XT_XOAY = 28.0              # json "xt_xoay": do, quay CUNG ca khoi quanh giua 2 kep TRUOC khi tien (+ = nguoc chieu kim dong ho, nhin tu tren)
XT_W = 0.10                 # rad/s toc do quay ca khoi
# DEN NGUOI (11/11 theo video demo7): 2 de dung NGANG 2 BEN nguoi, ao ngay TRUOC 2 ban tay (camera dau thay tay nguoi).
# json "xt_duong": true -> sau NANG di theo: LUI (giua 2 kep ra xa tay XT_XA theo huong mat nguoi) -> XOAY XT_XOAY
#   -> NGANG (giua 2 kep thang truoc tay) -> TIEN (giua 2 kep cach dau ngon XT_TRUOC) -> (HA neu xt_ha > 0) -> giu -> luu XT_MOC.
XT_DUONG = False            # json "xt_duong"
XT_YAW_NGUOI = 0.0          # json "xt_yaw_nguoi": do, nguoi quay bao nhieu (0 = mat nhin -y). TAM THOI ghi cung (seed 42: 21.5)
XT_XA = 0.65                # json "xt_xa": giua 2 kep cach dau ngon (m) luc di ngang
XT_TRUOC = 0.12             # json "xt_truoc": giua 2 kep cach dau ngon (m) luc dung cuoi
XT_KEP = None               # json "xt_kep": dat lai khoang cach 2 kep (m) luc NANG (None = giu)
XT_MOC = "moc_xo_tay_xong"  # json "xt_moc": ten moc luu khi xong
# MAC BUOC 2 (11/11, json "b2": true, chay tu moc_den_nguoi): DE DUNG YEN (khong tinh tien), chi lift + arm (+ xoay than tai cho de giu
# khoang cach 2 kep): B2_NANG (lift +b2_nang) -> B2_RUT (2 kep cung lui ve phia nguoi b2_rut m theo huong mat nguoi, b2_v m/s;
# arm huong RA XA nguoi nen phai RUT arm) -> B2_HA (lift -b2_ha) -> giu b2_giu s -> anh B2_XONG.
B2 = False
B2_NANG, B2_RUT, B2_V, B2_HA, B2_GIU, B2_ARM_MIN = 0.0, 0.22, 0.02, 0.0, 4.0, 0.005
B2_DE = 0.0                 # json "b2_de": sau RUT, 2 de lui them ve phia nguoi (m)
B2_DMIN = 0.10              # json "b2_dmin": dung de neu tam de cach diem nguoi (B2_NGUOI) < muc nay
B2_NANG2 = 0.0              # json "b2_nang2": sau RUT/DE nang lift tu tu (XT_V) them (m)
B2_MOC = "moc_tay_vao_khe"  # json "b2_moc" ("" = khong luu)
B2_MOC_DE = ""              # json "b2_moc_de": luu moc ngay sau khi de lui xong (vd "moc_lui_de"), "" = khong
# MAC BUOC 3 (11/11, json "b3": true, chay tu moc_tay_vao_khe; theo demo7 pha 4): 2 de CUNG tien ve phia nguoi (-huong mat nguoi)
# b3_tien m, DONG THOI nang lift b3_nang m (ti le theo quang di, b3_v m/s), arm + goc than + co tay giu nguyen; dung neu de < B2_DMIN toi nguoi.
B3 = False
B3_TIEN, B3_NANG, B3_V, B3_GIU = 0.10, 0.10, 0.02, 4.0
B3_MOC = "moc_xo_tay_xong"  # json "b3_moc" ("" = khong luu)
# MAC BUOC 4 (11/11, json "b4": true, chay tu moc_tay_vao_khe): dua ao DOC CANG TAY ve phia vai: 2 de cung tien (-huong mat nguoi)
# b4_tien m, lift doi b4_dz m moi 1 m tien (cang tay seed 42 doc XUONG ~ -0.51), b4_v m/s; arm/goc than/co tay giu nguyen;
# moi b4_buoc m luu 1 moc "moc_b4_<cm>" de kiem bang cong_cu/kiem_tay_khe.py (CHI KHI PHAT TRIEN). Dung neu de < B2_DMIN toi nguoi.
B4 = False
B4_TIEN, B4_DZ, B4_V, B4_BUOC, B4_GIU = 0.30, -0.51, 0.02, 0.05, 3.0
B2_HA_TRUOC = False         # json "b2_ha_truoc": true -> sau DE: HA truoc roi NANG2 (nhu demo7 147-148 s)
# TAM THOI (seed 42, tu metadata danh gia + uoc luong): diem nguoi tren mat san (x, y)
B2_NGUOI = {"vai T": (0.035, 0.525), "khuyu T": (0.174, 0.304), "co tay T": (0.248, 0.128), "ngon T": (0.291, 0.04),
            "vai P": (-0.201, 0.43), "khuyu P": (-0.153, 0.184), "co tay P": (-0.1, -0.001), "ngon P": (-0.072, -0.095),
            "co": (-0.081, 0.481), "than/ghe": (-0.082, 0.483), "goi T": (0.176, 0.10), "goi P": (-0.010, 0.027)}
ANH_CHI = None              # json "anh_chi": ["KHE_DO_XONG"] -> chi chup anh co tag bat dau bang cac chuoi nay (tat anh tung khung)
KHE_DUNG = True             # json "khe_dung": true -> xong MO_KHE la xong MANG (bo VE_LAN/DI_NGANG/DI_TOI)
KHE_LIFT = 1.00             # ngang tay nguoi (demo7 R0 1.00 / R1 0.91 luc cham nguoi)
KHE_NAC, KHE_MAX = 15.0, 90.0
KHE_CHO = 1.5               # s cho vai lang sau moi nac truoc khi doc xanh_dam
KHE_CACH = ["pitch", "roll", "ca2"]
KHE_PITCH_DAU = -1.0        # 2 robot CUNG dau pitch (cung mat phang tay); am = nhu demo (XO_TAY pitch -0.83)
KHE_CHON = True             # json "khe_chon": false -> thu xong ve goc luc gap (khong chon theo xanh_dam)
KHE_ROLL_DAU = 1.0          # roll DOI XUNG: R0 += dau*goc, R1 -= dau*goc (luc gap R0 -0.6, R1 +0.6 -> dau +1 = quay ve 0 truoc)


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


def do_khe(img):
    """Camera dau: KHE giua 2 dai vang (2 lop gau). Moi cot (va moi hang - lay chieu co dien tich lon hon): khoang trong giua
    2 doan vang lien tiep, rong 3..80 px, >= 60 % la toi (bong) hoac xanh dam (long ao) -> 1 khe; lay khe rong nhat moi cot.
    Tra (rong_max px, rong_trung_vi px, so_cot_co_khe, dien_tich px)."""
    im = np.asarray(img)[..., :3].astype(int)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    yel = (r > 170) & (g > 170) & (b < 110)
    toi = ((b > r + 40) & (b > g + 40) & (b < 180) & (r < 60)) | (im.max(-1) < 70)
    best = (0, 0.0, 0, 0)
    for yv, tv in ((yel, toi), (yel.T, toi.T)):
        w = []
        for x in range(yv.shape[1]):
            ys = np.flatnonzero(yv[:, x])
            if len(ys) < 2:
                continue
            m = 0
            for j in np.flatnonzero(np.diff(ys) > 3):
                a0, a1 = ys[j] + 1, ys[j + 1]
                if a1 - a0 <= 80 and tv[a0:a1, x].mean() >= 0.6:
                    m = max(m, a1 - a0)
            if m:
                w.append(m)
        if w and sum(w) > best[3]:
            best = (int(max(w)), float(np.median(w)), len(w), int(sum(w)))
    return best


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
        self.cloth_between, self.cloth_since = False, None
        self.sl_in = None
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
        if s in ("OPEN", "PUSH", "RESET"):
            self.cloth_between, self.cloth_since = False, None
            self.sl_in = None
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

        # Chi xac nhan vai o GIUA hai ngon; neu co depth thi mep vai phai toi dau ngon.
        # RESET/PUSH xoa co, tranh dung lai ket qua cua lan gap truoc.
        if s in ("PUSH", "KEO", "READY", "CLOSE"):
            near = (self.dn is None or self.dt is None or self.dn - self.dt <= 0.005)
            if self.ug >= GRIP_CLOTH_MIN and near:
                if self.cloth_since is None:
                    self.cloth_since = t
                self.cloth_between = t - self.cloth_since >= GRIP_CONFIRM_S
            else:
                self.cloth_between, self.cloth_since = False, None

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
            if self.sl_in is None and self.cloth_between:
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
                      f"{'CO VAI' if self.cloth_between else 'KHONG CO VAI giua ngon'}", flush=True)
                if self.cloth_between and not hard:
                    self.events.append(f"R{r}_day_vao_gau")
                    self.go("KEO" if KEO_N[r] > 0 else "READY", t)
                else:
                    print(f"[R{r}] CHUA XAC NHAN VAI / GAP CAN ({ly}) -> RESET, khong dong kep", flush=True)
                    self.fails += 1
                    a[r, :] = 0.0
                    self.go("RESET", t)

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
            if not self.cloth_between:
                print(f"[R{r}] READY khong con vai giua ngon -> RESET", flush=True)
                self.fails += 1
                a[r, :] = 0.0
                self.go("RESET", t)
            elif t - self.t0 > 0.5 and (other_ready or other_holding or t - self.t0 > READY_WAIT):
                self.go("CLOSE", t)

        elif s == "CLOSE":
            if not self.cloth_between:
                print(f"[R{r}] CLOSE khong co vai giua ngon -> RESET, khong dong kep", flush=True)
                self.fails += 1
                a[r, :] = 0.0
                self.go("RESET", t)
                return
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

        elif s == "CHECK":                  # nang len: vai phai van o vung GIUA hai ngon
            a[r, 3] = cmd(tgt - lift, RATE["lift"], 4.0)
            self.wrist(jp, tw, a)
            hold_arm()
            if self.cloth_between and self.ug >= GRIP_CLOTH_MIN:
                if self.ok is None:
                    self.ok = t
            else:
                self.ok = None
            if t - self.t0 > 1.0:
                # Nhieu pixel vai tren ban KHONG du de ket luan dang cam ao.
                good = self.ok is not None and t - self.ok >= GRIP_CONFIRM_S and cloth >= 1500
                print(f"[R{r}] CHECK vai={cloth} giua_ngon={self.ug} co_vai={int(self.cloth_between)} truoc={self.cpre} ngon_kep={self.kep:.3f}->{float(jp[11]):.3f} -> "
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
            # Khong dung rieng do mo kep de ket luan tuot; kiem tra vai giua ngon qua nhieu khung.
            if self.ug >= GRIP_CLOTH_MIN:
                self.lost = None
            else:
                if self.lost is None:
                    self.lost = t
                if t - self.lost >= GRIP_LOST_S:
                    print(f"[R{r}] HOLD mat vai giua ngon -> RESET", flush=True)
                    self.fails += 1
                    a[r, :] = 0.0
                    self.go("RESET", t)

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
    global MANG_MOI, DUNG_SAU_MANG, MANG_LIFT, MANG_ARM, MANG_ARM_V, MANG_LUI, MANG_W, MANG_V, MANG_LAN_Y, MANG_Y
    MANG_MOI = bool(p.get("mang_moi", MANG_MOI))
    DUNG_SAU_MANG = bool(p.get("dung_sau_mang", DUNG_SAU_MANG))
    MANG_LIFT = float(p.get("mang_lift", MANG_LIFT))
    MANG_ARM = float(p.get("mang_arm", MANG_ARM))
    MANG_ARM_V = float(p.get("mang_arm_v", MANG_ARM_V))
    MANG_LUI = float(p.get("mang_lui", MANG_LUI))
    MANG_W = float(p.get("mang_w", MANG_W))
    MANG_V = float(p.get("mang_v", MANG_V))
    MANG_LAN_Y = float(p.get("mang_lan_y", MANG_LAN_Y))
    MANG_Y = float(p.get("mang_y", MANG_Y))
    global LAT_KEP, LAT_LIFT, LAT_W, LAT_LAN_Y, LAT_DICH_Y
    LAT_KEP = bool(p.get("lat_kep", LAT_KEP))
    LAT_YAW[0] = float(p.get("lat_yaw0", LAT_YAW[0]))
    LAT_YAW[1] = float(p.get("lat_yaw1", LAT_YAW[1]))
    LAT_LIFT = float(p.get("lat_lift", LAT_LIFT))
    LAT_W = float(p.get("lat_w", LAT_W))
    LAT_LAN_Y = float(p.get("lat_lan_y", LAT_LAN_Y))
    LAT_DICH_Y = float(p.get("lat_dich_y", LAT_DICH_Y))
    global MO_KHE, KHE_DUNG, KHE_LIFT, KHE_NAC, KHE_MAX, KHE_CHO, KHE_CACH, KHE_PITCH_DAU, KHE_ROLL_DAU
    MO_KHE = bool(p.get("mo_khe", MO_KHE))
    KHE_DUNG = bool(p.get("khe_dung", KHE_DUNG))
    KHE_LIFT = float(p.get("khe_lift", KHE_LIFT))
    KHE_NAC = float(p.get("khe_nac", KHE_NAC))
    KHE_MAX = float(p.get("khe_max", KHE_MAX))
    KHE_CHO = float(p.get("khe_cho", KHE_CHO))
    KHE_CACH = list(p.get("khe_cach", KHE_CACH))
    KHE_PITCH_DAU = float(p.get("khe_pitch_dau", KHE_PITCH_DAU))
    KHE_ROLL_DAU = float(p.get("khe_roll_dau", KHE_ROLL_DAU))
    global KHE_CHON, KHE_SAU, LUU_MOC
    KHE_SAU = str(p.get("khe_sau", KHE_SAU))
    global KHE_DICH
    KHE_DICH = p.get("khe_dich", KHE_DICH)
    LUU_MOC = bool(p.get("luu_moc", LUU_MOC))
    global KHE_KEP, KHE_DO_T, ANH_CHI, VID_FPS
    KHE_KEP = p.get("khe_kep", KHE_KEP)
    KHE_DO_T = float(p.get("khe_do_t", KHE_DO_T))
    ANH_CHI = p.get("anh_chi", ANH_CHI)
    VID_FPS = float(p.get("vid_fps", VID_FPS))
    global XO_TAY, XT_TAY, XT_LECH, XT_CAO, XT_V, XT_HA, XT_V_DE, XT_GIU, XT_LIFT_MAX, XT_XOAY
    XT_LIFT_MAX = float(p.get("xt_lift_max", XT_LIFT_MAX))
    XT_XOAY = float(p.get("xt_xoay", XT_XOAY))
    XO_TAY = bool(p.get("xo_tay", XO_TAY))
    XT_TAY = tuple(p.get("xt_tay", XT_TAY))
    XT_LECH = tuple(p.get("xt_lech", XT_LECH))
    XT_CAO = float(p.get("xt_cao", XT_CAO))
    XT_V = float(p.get("xt_v", XT_V))
    XT_HA = float(p.get("xt_ha", XT_HA))
    XT_V_DE = float(p.get("xt_v_de", XT_V_DE))
    XT_GIU = float(p.get("xt_giu", XT_GIU))
    global XT_DUONG, XT_YAW_NGUOI, XT_XA, XT_TRUOC, XT_KEP, XT_MOC
    XT_DUONG = bool(p.get("xt_duong", XT_DUONG))
    XT_YAW_NGUOI = float(p.get("xt_yaw_nguoi", XT_YAW_NGUOI))
    XT_XA = float(p.get("xt_xa", XT_XA))
    XT_TRUOC = float(p.get("xt_truoc", XT_TRUOC))
    XT_KEP = p.get("xt_kep", XT_KEP)
    XT_KEP = None if XT_KEP is None else float(XT_KEP)
    XT_MOC = str(p.get("xt_moc", XT_MOC))
    global B2, B2_NANG, B2_RUT, B2_V, B2_HA, B2_GIU, B2_ARM_MIN
    B2 = bool(p.get("b2", B2))
    B2_NANG = float(p.get("b2_nang", B2_NANG))
    B2_RUT = float(p.get("b2_rut", B2_RUT))
    B2_V = float(p.get("b2_v", B2_V))
    B2_HA = float(p.get("b2_ha", B2_HA))
    B2_GIU = float(p.get("b2_giu", B2_GIU))
    B2_ARM_MIN = float(p.get("b2_arm_min", B2_ARM_MIN))
    global B2_DE, B2_DMIN, B2_NANG2, B2_MOC, B2_NGUOI
    B2_DE = float(p.get("b2_de", B2_DE))
    B2_DMIN = float(p.get("b2_dmin", B2_DMIN))
    B2_NANG2 = float(p.get("b2_nang2", B2_NANG2))
    B2_MOC = str(p.get("b2_moc", B2_MOC))
    global B3, B3_TIEN, B3_NANG, B3_V, B3_GIU, B3_MOC
    B3 = bool(p.get("b3", B3))
    B3_TIEN = float(p.get("b3_tien", B3_TIEN))
    B3_NANG = float(p.get("b3_nang", B3_NANG))
    B3_V = float(p.get("b3_v", B3_V))
    B3_GIU = float(p.get("b3_giu", B3_GIU))
    B3_MOC = str(p.get("b3_moc", B3_MOC))
    global B4, B4_TIEN, B4_DZ, B4_V, B4_BUOC, B4_GIU
    B4 = bool(p.get("b4", B4))
    B4_TIEN = float(p.get("b4_tien", B4_TIEN))
    B4_DZ = float(p.get("b4_dz", B4_DZ))
    B4_V = float(p.get("b4_v", B4_V))
    B4_BUOC = float(p.get("b4_buoc", B4_BUOC))
    B4_GIU = float(p.get("b4_giu", B4_GIU))
    global B2_MOC_DE, B2_HA_TRUOC
    B2_MOC_DE = str(p.get("b2_moc_de", B2_MOC_DE))
    B2_HA_TRUOC = bool(p.get("b2_ha_truoc", B2_HA_TRUOC))
    B2_NGUOI = dict(p.get("b2_nguoi", B2_NGUOI))
    KHE_CHON = bool(p.get("khe_chon", KHE_CHON))
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
            self.giai_doan, self.t_gd = None, 0.0     # GAP / MANG / CHUAN_BI / MAC / XONG (xem GIAI_DOAN)
            self.w_mang = None                        # goc co tay giu suot MANG moi (None = chua vao MANG moi)
            self.giu_P = None                         # vi tri 2 de giu yen trong GIU (MANG moi)
            self.moc_can = None                       # ten diem luu dang cho chay_moc.py luu (LUU_MOC)
            self.vao_lai = False                      # chay_moc.py dat True sau khi nap moc -> vao lai buoc dang lam

        def set_mode(self, m, t):
            self.mode, self.t_mode = m, t
            print(f"[PAIR] >>> {m} t={t:.1f}", flush=True)
            self.events.append(f"PAIR_{m}")

        def tinh_giai_doan(self):
            """Giai doan hien tai suy ra tu mode + buoc KF (chi doc, khong doi gi)."""
            if self.mode in ("SCAN", "GRASP", "SOLO"):
                return "GAP"
            if self.mode == "MANG":
                return "MANG"
            if self.mode == "DRESS":
                if self.ki < KF_I_CHUAN_BI:
                    return "MANG"
                return "CHUAN_BI" if self.ki < KF_I_MAC else "MAC"
            if self.mode in ("RELEASE", "DONE"):
                return "XONG"
            if self.mode == "XO_TAY":
                return "MAC"
            return self.giai_doan or "GAP"          # GIU (dress_steps): giu giai doan dang lam do

        # ---------------- MANG moi (xem MANG_BUOC) ----------------
        def kep_the_gioi(self, BP, yaw, arm):
            """Vi tri 2 kep (giua 2 dau ngon) trong toa do the gioi; khong co fingertip thi uoc theo arm + C_GRIP."""
            g = []
            for r in (0, 1):
                A = self.arms[r]
                c, s = np.cos(yaw[r]), np.sin(yaw[r])
                if getattr(A, "ftx", None) is not None:
                    g.append(BP[r, :2] + np.array([c * A.ftx - s * A.fty, s * A.ftx + c * A.fty]))
                else:
                    g.append(BP[r, :2] + (arm[r] + C_GRIP) * np.array([c, s]))
            return g

        def bat_dau_mang(self, t, BP, yaw, JP, arm):
            self.mb, self.t_mang0 = 0, t                  # buoc trong self.ds
            ds = list(MANG_BUOC_LAT if LAT_KEP else MANG_BUOC)
            if MO_KHE:
                ten_ds = [b[0] for b in ds]
                i = ten_ds.index(KHE_SAU if KHE_SAU in ten_ds else ("LAT1" if LAT_KEP else "XOAY")) + 1
                ds.insert(i, ("MO_KHE", 240))
                if KHE_DUNG:
                    ds = ds[:i + 1]
            self.ds = tuple(ds)
            self.w_mang = [np.array(JP[r, 8:11], dtype=float) for r in (0, 1)]   # co tay giu nguyen goc luc gap
            self.arm_mang = [float(arm[0]), float(arm[1])]
            g = self.kep_the_gioi(BP, yaw, arm)
            self.kep_dau = float(np.hypot(*(g[0] - g[1])))
            print(f"[MANG] bat dau: 2 kep cach nhau {self.kep_dau:.3f} m, de R0 ({BP[0,0]:.2f},{BP[0,1]:.2f}) yaw {np.degrees(yaw[0]):.0f}"
                  f" | R1 ({BP[1,0]:.2f},{BP[1,1]:.2f}) yaw {np.degrees(yaw[1]):.0f} | co tay r0={np.round(self.w_mang[0], 2)} r1={np.round(self.w_mang[1], 2)}",
                  flush=True)
            self.luu_moc("moc_gap_xong")
            self.vao_buoc_mang(t, BP, yaw, arm)

        def luu_moc(self, ten):
            """Bao chay_moc.py luu diem luu (chi khi json luu_moc: true)."""
            if LUU_MOC:
                self.moc_can = ten
                print(f"[MOC] yeu cau luu {ten}", flush=True)

        def vao_buoc_mang(self, t, BP, yaw, arm):
            """Ghi lai trang thai dau buoc (khoi cung: vi tri tuong doi 2 de, goc 2 than)."""
            self.t_mb = t
            self.m_P = [BP[0, :2].copy(), BP[1, :2].copy()]
            self.m_yaw = [float(yaw[0]), float(yaw[1])]
            self.m_D = BP[0, :2] - BP[1, :2]              # vector R1 -> R0, giu khong doi
            self.m_phi, self.m_y1 = 0.0, float(yaw[1])
            ten = self.ds[self.mb][0]
            mid = (BP[0, :2] + BP[1, :2]) / 2
            lan_y, dich_y = (LAT_LAN_Y, LAT_DICH_Y) if LAT_KEP else (MANG_LAN_Y, MANG_Y)
            if ten == "LUI":
                d = np.array([np.cos(yaw[0]) + np.cos(yaw[1]), np.sin(yaw[0]) + np.sin(yaw[1])])
                d = d / max(np.hypot(*d), 1e-6)
                self.m_dich = mid - MANG_LUI * d
            elif ten == "VE_LAN":
                self.m_dich = np.array([mid[0], lan_y])
            elif ten == "DI_NGANG":
                self.m_dich = np.array([PX, mid[1]])
            elif ten == "DI_TOI":
                self.m_dich = np.array([PX, dich_y])
            elif ten in ("LAT0", "LAT1"):
                r = int(ten[-1])
                o = 1 - r
                g = self.kep_the_gioi(BP, yaw, arm)
                self.l_r, self.l_G, self.l_Go = r, g[r].copy(), g[o].copy()
                c, sn = np.cos(yaw[r]), np.sin(yaw[r])
                v = BP[r, :2] - g[r]                      # kep -> de, trong he than robot (khong doi khi quay)
                self.l_vb = np.array([c * v[0] + sn * v[1], -sn * v[0] + c * v[1]])
                # chieu di vong: sau 90 do, de phai o phia XA kep robot kia (phia ngoai, khong quet qua ao)
                ccw = np.array([-v[1], v[0]])
                self.l_s = 1.0 if float(ccw @ (g[r] - g[o])) > 0 else -1.0
                dpsi = np.radians(LAT_YAW[r]) - float(yaw[r])
                self.l_tong = self.l_s * ((self.l_s * dpsi) % (2 * np.pi))   # tong goc phai quay (co dau)
                self.l_da, self.l_y = 0.0, float(yaw[r])
                self.l_print = -1.0
            elif ten == "MO_KHE":
                self.luu_moc("moc_xoay_xong")             # 2 robot da xoay/lat xong, dung yen (2 ben nguoi neu khe_sau DI_TOI)
                self.k_w0 = [w.copy() for w in self.w_mang]           # goc co tay goc (luc gap)
                plan = []
                n = int(round(KHE_MAX / KHE_NAC))
                for cach in KHE_CACH:
                    if cach == "dich":
                        dmax = max(float(np.max(np.abs(np.array(KHE_DICH[r], dtype=float) - self.k_w0[r]))) for r in (0, 1))
                        nd = max(1, int(np.ceil(np.degrees(dmax) / KHE_NAC - 1e-6)))
                        plan += [(cach, 100.0 * j / nd, True) for j in range(1, nd + 1)]       # % quang duong
                        continue
                    if cach.startswith("do_"):            # thu nhe 1 khop: +nac, ve 0, -nac, ve 0
                        plan += [(cach, KHE_NAC, True), (cach, 0.0, False), (cach, -KHE_NAC, True), (cach, 0.0, False)]
                        continue
                    s = -1.0 if cach.endswith("-") else 1.0
                    plan += [(cach, s * KHE_NAC * j, True) for j in range(1, n + 1)]       # len tung nac, co ghi
                    plan += [(cach, s * KHE_NAC * j, False) for j in range(n - 1, -1, -1)]  # ve 0 tung nac, khong ghi
                self.k_plan, self.k_i, self.k_tr, self.k_ket = plan, -1, None, []
                self.k_chon = None
                self.k_khop = "roll"                      # khop "vat" (vat kep quanh truc cua no); doi sau khi thu do_
                self.k_dz0 = [0.0, 0.0]
            elif ten == "XOAY":
                # chieu quay: diem giua duong di cua R0 (sau 90 do) phai nam NGUOC huong tay R1 (phia khong co ao / ban)
                d1 = np.array([np.cos(yaw[1]), np.sin(yaw[1])])
                D = self.m_D
                ccw = np.array([-D[1], D[0]])            # D quay +90 do
                self.m_s = -1.0 if float(ccw @ d1) > 0 else 1.0
            print(f"[MANG] buoc {self.mb + 1}/{len(self.ds)}: {ten} t={t:.1f}"
                  + (f" chieu={'nguoc kim dong ho' if self.m_s > 0 else 'thuan kim dong ho'}" if ten == "XOAY" else "")
                  + (f" dich=({self.m_dich[0]:.2f},{self.m_dich[1]:.2f})" if ten in ("LUI", "VE_LAN", "DI_NGANG", "DI_TOI") else "")
                  + (f" R{self.l_r} quay {np.degrees(self.l_tong):+.0f} do quanh kep ({self.l_G[0]:.2f},{self.l_G[1]:.2f})"
                     if ten in ("LAT0", "LAT1") else ""),
                  flush=True)
            self.events.append(f"MANG_{ten}")

        def khe_w(self, cach, deg):
            """Goc co tay 2 robot cho MO_KHE: pitch cung dau, roll doi xung."""
            w = [self.k_w0[0].copy(), self.k_w0[1].copy()]
            d = np.radians(deg)
            if cach == "dich":                            # deg = % quang duong toi goc tuyet doi KHE_DICH
                f = deg / 100.0
                return [w[r] + f * (np.array(KHE_DICH[r], dtype=float) - w[r]) for r in (0, 1)]
            khop = cach[3:] if cach.startswith("do_") else cach.rstrip("+-")
            if khop == "vat":
                khop = self.k_khop
            if khop in ("yaw", "roll") and cach not in ("roll",):   # doi xung: R0 +d, R1 -d (deg co dau)
                j = 0 if khop == "yaw" else 2
                w[0][j] += d
                w[1][j] -= d
                return w
            if cach in ("pitch", "ca2"):
                w[0][1] += KHE_PITCH_DAU * d
                w[1][1] += KHE_PITCH_DAU * d
            if cach in ("roll", "ca2"):
                w[0][2] += KHE_ROLL_DAU * d
                w[1][2] -= KHE_ROLL_DAU * d
            return w

        def buoc_mang(self, t, BP, yaw, JP, lift, arm, a):
            """1 buoc dieu khien MANG. Tra ve True khi xong ca giai doan."""
            ten, tmax = self.ds[self.mb]
            k = t - self.t_mb
            xong = False
            # tay: lift len MANG_LIFT, arm rut tu tu (MANG_ARM_V m/s) ve MANG_ARM; co tay giu goc luc gap
            for r in (0, 1):
                mt = max(MANG_ARM, self.arm_mang[r] - MANG_ARM_V * (t - self.t_mang0))
                lt = LAT_LIFT if (LAT_KEP and self.mb >= 3) else MANG_LIFT      # tu LAT0 tro di nang cao hon
                a[r, 3] = cmd(lt - lift[r], RATE["lift"], 3.0)
                a[r, 4] = float(np.clip(cmd(mt - arm[r], RATE["arm"], 3.0), -0.2, 0.2))
                self.arms[r].wrist(JP[r], self.w_mang[r], a)
            if ten == "NHAC":
                for r in (0, 1):                          # de dung yen tai cho
                    e = self.m_P[r] - BP[r, :2]
                    drive(a, r, 1.5 * e[0], 1.5 * e[1], yaw[r], 0.10)
                    a[r, 2] = float(np.clip(2.0 * wrap(self.m_yaw[r] - yaw[r]), -0.3, 0.3))
                xong = abs(MANG_LIFT - lift[0]) < LIFT_TOL and abs(MANG_LIFT - lift[1]) < LIFT_TOL and \
                    max(arm) < MANG_ARM + ARM_TOL
            elif ten in ("LUI", "VE_LAN", "DI_NGANG", "DI_TOI"):    # tinh tien ca khoi, khong xoay
                mid = (BP[0, :2] + BP[1, :2]) / 2
                e = self.m_dich - mid
                n = float(np.hypot(*e))
                v = e * min(1.2, MANG_V / max(n, 1e-6))
                for r, sg in ((0, 0.5), (1, -0.5)):
                    c = (mid + sg * self.m_D) - BP[r, :2]           # giu dung vi tri tuong doi
                    drive(a, r, v[0] + 1.5 * c[0], v[1] + 1.5 * c[1], yaw[r], MANG_V + 0.05)
                    a[r, 2] = float(np.clip(2.0 * wrap(self.m_yaw[r] - yaw[r]), -0.3, 0.3))
                xong = n < 0.03 and all(abs(wrap(self.m_yaw[r] - yaw[r])) < 0.05 for r in (0, 1))
            elif ten == "XOAY":                           # quay cung ca khoi quanh de R1
                self.m_phi += wrap(float(yaw[1]) - self.m_y1)
                self.m_y1 = float(yaw[1])
                phi = self.m_phi
                cp, sp = np.cos(phi), np.sin(phi)
                Dr = np.array([cp * self.m_D[0] - sp * self.m_D[1], sp * self.m_D[0] + cp * self.m_D[1]])
                tgt0 = BP[1, :2] + Dr
                e0 = tgt0 - BP[0, :2]
                ey0 = wrap(self.m_yaw[0] + phi - yaw[0])
                sai = max(float(np.hypot(*e0)), 0.5 * abs(ey0))
                con = np.pi - abs(phi)
                w = self.m_s * min(MANG_W, 0.8 * max(con, 0.0)) * float(np.clip((0.10 - sai) / 0.06, 0.0, 1.0))
                if con < 0.02:
                    w = 0.0
                # R1: xoay tai cho, giu de o cho cu
                e1 = self.m_P[1] - BP[1, :2]
                drive(a, 1, 1.5 * e1[0], 1.5 * e1[1], yaw[1], 0.10)
                a[1, 2] = float(np.clip(w, -0.4, 0.4))
                # R0: di vong (van toc tiep tuyen w x r) + sua sai vi tri, than quay cung goc
                ff = w * np.array([-Dr[1], Dr[0]])
                drive(a, 0, ff[0] + 1.5 * e0[0], ff[1] + 1.5 * e0[1], yaw[0], 0.25)
                a[0, 2] = float(np.clip(w + 2.0 * ey0, -0.5, 0.5))
                xong = con < 0.03 and float(np.hypot(*e0)) < 0.03 and abs(ey0) < 0.05
            elif ten in ("LAT0", "LAT1"):                 # robot l_r di vong quanh CHINH KEP cua no, robot kia dung giu
                r, o = self.l_r, 1 - self.l_r
                self.l_da += wrap(float(yaw[r]) - self.l_y)
                self.l_y = float(yaw[r])
                c, sn = np.cos(yaw[r]), np.sin(yaw[r])
                tgt = self.l_G + np.array([c * self.l_vb[0] - sn * self.l_vb[1], sn * self.l_vb[0] + c * self.l_vb[1]])
                e = tgt - BP[r, :2]
                g = self.kep_the_gioi(BP, yaw, arm)
                lech = float(np.hypot(*(g[r] - self.l_G)))
                con = abs(self.l_tong) - abs(self.l_da)
                sai = max(float(np.hypot(*e)), lech)
                w = self.l_s * min(LAT_W, 0.8 * max(con, 0.0)) * float(np.clip((0.08 - sai) / 0.05, 0.0, 1.0))
                if con < 0.02 or min(lift) < LAT_LIFT - 0.03:       # chua nang du cao thi chua quay
                    w = 0.0
                rr = tgt - self.l_G
                ff = w * np.array([-rr[1], rr[0]])        # van toc tiep tuyen cua de khi than quay w
                drive(a, r, ff[0] + 1.5 * e[0], ff[1] + 1.5 * e[1], yaw[r], 0.25)
                a[r, 2] = float(np.clip(w, -0.4, 0.4))
                e2 = self.m_P[o] - BP[o, :2]              # robot kia dung yen, giu goc
                drive(a, o, 1.5 * e2[0], 1.5 * e2[1], yaw[o], 0.10)
                a[o, 2] = float(np.clip(2.0 * wrap(self.m_yaw[o] - yaw[o]), -0.3, 0.3))
                if t - self.l_print >= 0.5 - 1e-6:
                    self.l_print = t
                    print(f"[LAT] R{r} t={t:.1f} da quay {np.degrees(self.l_da):+.0f}/{np.degrees(self.l_tong):+.0f} do | "
                          f"kep R{r} lech {100 * lech:.1f} cm ({g[r][0]:.2f},{g[r][1]:.2f}) | 2 kep cach nhau {np.hypot(*(g[0] - g[1])):.3f} m | "
                          f"yaw=({np.degrees(yaw[0]):.0f},{np.degrees(yaw[1]):.0f}) de{r}=({BP[r,0]:.2f},{BP[r,1]:.2f}) lift=({lift[0]:.2f},{lift[1]:.2f})",
                          flush=True)
                    if lech > LAT_CANH_BAO:
                        print(f"[LAT] CANH BAO: kep R{r} lech {100 * lech:.1f} cm > {100 * LAT_CANH_BAO:.0f} cm", flush=True)
                xong = con < 0.03 and float(np.hypot(*e)) < 0.03 and abs(LAT_LIFT - lift[0]) < LIFT_TOL and \
                    abs(LAT_LIFT - lift[1]) < LIFT_TOL
            elif ten == "MO_KHE":                         # de dung yen, nang lift, xoay co tay tung nac
                for r in (0, 1):
                    e = self.m_P[r] - BP[r, :2]
                    drive(a, r, 1.5 * e[0], 1.5 * e[1], yaw[r], 0.10)
                    a[r, 2] = float(np.clip(2.0 * wrap(self.m_yaw[r] - yaw[r]), -0.3, 0.3))
                    a[r, 3] = cmd(KHE_LIFT - lift[r], RATE["lift"], 3.0)
                xd = (int(self.arms[0].dk[0]), int(self.arms[1].dk[0]))
                g = self.kep_the_gioi(BP, yaw, arm)
                dkep = float(np.hypot(*(g[0] - g[1])))
                if self.k_i < 0:                          # 1) nang lift ca 2 cung luc, co tay giu goc goc
                    for r in (0, 1):
                        self.arms[r].wrist(JP[r], self.k_w0[r], a)
                    if all(abs(KHE_LIFT - lift[r]) < LIFT_TOL for r in (0, 1)):
                        print(f"[KHE] t={t:.1f} lift xong ({lift[0]:.2f},{lift[1]:.2f}) | xanh_dam ({xd[0]},{xd[1]}) | "
                              f"2 kep {dkep:.3f} m | co tay r0={np.round(JP[0, 8:11], 2)} r1={np.round(JP[1, 8:11], 2)}", flush=True)
                        self.k_i, self.k_tr, self.k_t0 = 0, None, t
                        self.k_dz0 = [getattr(self.arms[r], "ftdz", 0.0) for r in (0, 1)]
                        self.k_probe = {}
                        print(f"[KHE] goc 0: 2 dau ngon lech cao (trai-phai) R0 {100 * self.k_dz0[0]:+.1f} cm R1 {100 * self.k_dz0[1]:+.1f} cm", flush=True)
                        self.events.append("KHE_LIFT")
                else:
                    if self.k_i < len(self.k_plan):
                        cach, deg, ghi = self.k_plan[self.k_i]
                    else:                                 # 3) da thu het -> ve goc tot nhat
                        if self.k_chon is None and not KHE_CHON and "dich" in KHE_CACH:   # giu goc tuyet doi
                            self.k_chon = ("dich", 100.0, 0.0, 0.0, None, self.khe_w("dich", 100.0))
                            print(f"[KHE] xong -> GIU goc dich r0={np.round(self.k_chon[5][0], 2)} r1={np.round(self.k_chon[5][1], 2)}", flush=True)
                            self.k_tr, self.k_t0 = None, t
                        if self.k_chon is None and not KHE_CHON:   # khong chon: ve goc luc gap
                            self.k_chon = ("vat", 0.0, 0.0, 0.0, None, [w_.copy() for w_ in self.k_w0])
                            print(f"[KHE] thu xong -> ve goc luc gap, dung yen", flush=True)
                            self.k_tr, self.k_t0 = None, t
                        if self.k_chon is None:
                            self.k_chon = max(self.k_ket, key=lambda q: (min(q[2], q[3]), q[2] + q[3]))
                            q = self.k_chon
                            print(f"[KHE] CHON: {q[0]} {q[1]:.0f} do (xanh_dam R0 {q[2]:.0f}, R1 {q[3]:.0f}) "
                                  f"co tay r0={np.round(q[5][0], 2)} r1={np.round(q[5][1], 2)}", flush=True)
                            self.k_tr, self.k_t0 = None, t
                        cach, deg, ghi = self.k_chon[0], self.k_chon[1], False
                    w = self.khe_w(cach, deg)
                    err = max(float(self.arms[r].wrist(JP[r], w[r], a).max()) for r in (0, 1))
                    if self.k_tr is None and (err < 0.05 or t - self.k_t0 > 4.0):
                        self.k_tr, self.k_gh, self.k_sum = t, err >= 0.05, [0.0, 0.0, 0]
                    if self.k_tr is not None:
                        cho = KHE_CHO if (ghi or self.k_i >= len(self.k_plan)) else 0.3
                        if t - self.k_tr >= cho - 0.5:    # trung binh xanh_dam 0.5 s cuoi
                            self.k_sum = [self.k_sum[0] + xd[0], self.k_sum[1] + xd[1], self.k_sum[2] + 1]
                        if t - self.k_tr >= cho:
                            n_ = max(self.k_sum[2], 1)
                            m0, m1 = self.k_sum[0] / n_, self.k_sum[1] / n_
                            if self.k_i >= len(self.k_plan):
                                self.w_mang = [w[0].copy(), w[1].copy()]      # giu goc nay cho buoc sau / GIU
                                print(f"[KHE] t={t:.1f} GIU GOC TOT: {cach} {deg:.0f} do | xanh_dam ({m0:.0f},{m1:.0f}) | 2 kep {dkep:.3f} m | "
                                      f"co tay r0={np.round(JP[0, 8:11], 2)} r1={np.round(JP[1, 8:11], 2)}", flush=True)
                                self.events.append("KHE_TOT")
                                xong = True
                            else:
                                if ghi:
                                    self.k_ket.append((cach, deg, m0, m1, None, w))
                                    dz = [getattr(self.arms[r], "ftdz", 0.0) for r in (0, 1)]
                                    ten_k = cach
                                    if cach.startswith("do_"):        # thu khop: khop nao lam 2 ngon lech tren-duoi nhieu nhat = khop "vat"
                                        kh = cach[3:]
                                        self.k_probe[kh] = self.k_probe.get(kh, 0.0) + sum(abs(dz[r] - self.k_dz0[r]) for r in (0, 1))
                                        self.k_khop = max(self.k_probe, key=self.k_probe.get)
                                    elif cach.rstrip("+-") == "vat":
                                        ten_k = self.k_khop + cach[-1]
                                    print(f"[KHE] t={t:.1f} {ten_k:7s} {deg:+4.0f} do | co tay r0={np.round(JP[0, 8:11], 2)} "
                                          f"r1={np.round(JP[1, 8:11], 2)}{' (CHAM GIOI HAN?)' if self.k_gh else ''} | "
                                          f"ngon lech cao R0 {100 * dz[0]:+.1f} R1 {100 * dz[1]:+.1f} cm | "
                                          f"xanh_dam R0 {m0:.0f} R1 {m1:.0f} | 2 kep {dkep:.3f} m | lift ({lift[0]:.2f},{lift[1]:.2f})"
                                          + (f" | khop vat = {self.k_khop} {dict((k_, round(100 * v_, 1)) for k_, v_ in self.k_probe.items())}"
                                             if cach.startswith("do_") else ""), flush=True)
                                    self.events.append(f"KHE_{ten_k}_{deg:+.0f}".replace("+", "p").replace("-", "m"))
                                self.k_i, self.k_tr, self.k_t0 = self.k_i + 1, None, t
            if int(t * 20) % 20 == 0:
                g = self.kep_the_gioi(BP, yaw, arm)
                extra = f" phi={np.degrees(self.m_phi):.0f}" if ten == "XOAY" else ""
                print(f"[MANG] {ten}: 2 kep cach nhau {np.hypot(*(g[0] - g[1])):.3f} m (dau {self.kep_dau:.3f}) "
                      f"yaw=({np.degrees(yaw[0]):.0f},{np.degrees(yaw[1]):.0f}) de0=({BP[0,0]:.2f},{BP[0,1]:.2f}) "
                      f"de1=({BP[1,0]:.2f},{BP[1,1]:.2f}) lift=({lift[0]:.2f},{lift[1]:.2f}) arm=({arm[0]:.2f},{arm[1]:.2f}){extra}",
                      flush=True)
            if xong or k > tmax:
                if not xong:
                    print(f"[MANG] buoc {ten} HET GIO ({tmax} s) -> van di tiep", flush=True)
                self.mb += 1
                if self.mb >= len(self.ds):
                    g = self.kep_the_gioi(BP, yaw, arm)
                    self.kep_cuoi = float(np.hypot(*(g[0] - g[1])))
                    print(f"[MANG] XONG t={t:.1f}: 2 kep cach nhau {self.kep_cuoi:.3f} m (dau {self.kep_dau:.3f}, "
                          f"lech {100 * (self.kep_cuoi - self.kep_dau):+.1f} cm) yaw=({np.degrees(yaw[0]):.0f},{np.degrees(yaw[1]):.0f})",
                          flush=True)
                    self.events.append("MANG_XONG")
                    if "MO_KHE" not in [b[0] for b in self.ds]:
                        self.luu_moc("moc_xoay_xong")       # co MO_KHE: luu moc_mo_khe_xong sau khi do khe (GIU)
                    return True
                self.vao_buoc_mang(t, BP, yaw, arm)
            return False

        def cap_nhat_giai_doan(self, t):
            gd = self.tinh_giai_doan()
            if gd != self.giai_doan:
                self.giai_doan, self.t_gd = gd, t
                print(f"[GD] chuyen sang {gd} t={t:.1f}", flush=True)
                self.events.append(f"GD_{gd}")

        def snap(self, t, rgb, tags, info):
            if not SNAP:
                return
            if ANH_CHI and not any(tg.startswith(tuple(ANH_CHI)) for tg in tags):
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
                    self.arms[r].ftzc = float(FT[r, :, 2].mean())                 # do cao GIUA 2 dau ngon
                    self.arms[r].ftx = float(FT[r, :, 0].mean())
                    self.arms[r].fty = float(FT[r, :, 1].mean())
                    self.arms[r].ftdz = float(FT[r, 0, 2] - FT[r, 1, 2])     # ngon trai - ngon phai theo chieu cao (0 = 2 ngon nam ngang)
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
                self.arms[k].dn, self.arms[k].dt = None, None
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
                    self.arms[k].dn, self.arms[k].dt = None, None

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
                    self.luu_moc("moc_truoc_ban")
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
                    if MANG_MOI:
                        self.set_mode("MANG", t)
                        self.bat_dau_mang(t, BP, yaw, JP, arm)
                    else:
                        self.set_mode("DRESS", t)
                        print(f"[PAIR] buoc 1/{len(KF)}: {KF[0][0]}", flush=True)
                elif A0.s == "HOLD" and t - A0.hold_t > PAIR_WAIT:
                    print("[PAIR] robot_1 khong gap duoc -> robot_0 giu ao (chi diem gap)", flush=True)
                    self.set_mode("SOLO", t)

            elif self.mode == "SOLO":
                A0.step(t, JP[0], BP[0], *hd[0], cloth[0], hem[0], False, False, a)

            elif self.mode == "MANG":       # MANG moi: quay cung ca khoi quanh R1 (xem MANG_BUOC)
                if getattr(self, "vao_lai", False):        # vua nap moc: bat dau lai buoc hien tai (tham so json moi)
                    self.vao_lai = False
                    print(f"[MOC] vao lai buoc {self.ds[self.mb][0]} t={t:.1f}", flush=True)
                    self.vao_buoc_mang(t, BP, yaw, arm)
                if self.buoc_mang(t, BP, yaw, JP, lift, arm, a):
                    if DUNG_SAU_MANG:
                        print("[MANG] dung_sau_mang -> DUNG YEN GIU AO (chua lam CHUAN_BI / MAC)", flush=True)
                        self.set_mode("GIU", t)
                    else:
                        self.ki, self.t_kf = KF_I_CHUAN_BI, t      # bo qua KF MANG cu, vao CHUAN_BI
                        self.set_mode("DRESS", t)
                        print(f"[PAIR] buoc {self.ki + 1}/{len(KF)}: {KF[self.ki][0]}", flush=True)

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
                wg = self.w_mang if getattr(self, "w_mang", None) is not None else W_GIU
                A0.wrist(JP[0], wg[0], a)
                A1.wrist(JP[1], wg[1], a)
                if getattr(self, "w_mang", None) is not None:   # MANG moi: GIU DE TAI CHO (ao cang keo 2 de xich lai - lan gd2_khe)
                    if getattr(self, "giu_P", None) is None:
                        self.giu_P = [BP[0, :2].copy(), BP[1, :2].copy()]
                        self.giu_yaw = [float(yaw[0]), float(yaw[1])]
                        g = self.kep_the_gioi(BP, yaw, arm)
                        d0 = float(np.hypot(*(g[0] - g[1])))
                        if KHE_KEP is not None and d0 > float(KHE_KEP):   # nhich 2 de lai gan DEU 2 BEN theo duong noi 2 kep
                            u = (g[0] - g[1]) / max(d0, 1e-6)
                            dd = 0.5 * (d0 - float(KHE_KEP))
                            self.giu_P[0] = self.giu_P[0] - dd * u
                            self.giu_P[1] = self.giu_P[1] + dd * u
                            print(f"[KHE_DO] t={t:.1f} 2 kep {d0:.3f} m -> {float(KHE_KEP):.3f} m: moi de nhich {100 * dd:.1f} cm vao trong", flush=True)
                        self.kd_t0, self.kd_ls = (t if "MO_KHE" in [b[0] for b in getattr(self, "ds", [])] else None), []
                    for r in (0, 1):
                        e = self.giu_P[r] - BP[r, :2]
                        drive(a, r, 1.5 * e[0], 1.5 * e[1], yaw[r], 0.10)
                        a[r, 2] = float(np.clip(2.0 * wrap(self.giu_yaw[r] - yaw[r]), -0.3, 0.3))
                if getattr(self, "kd_t0", None) is not None and int(t * 20) % 10 == 0:   # do khe moi 0.5 s
                    g = self.kep_the_gioi(BP, yaw, arm)
                    dk_ = float(np.hypot(*(g[0] - g[1])))
                    kh = [do_khe(rgb[3]), do_khe(rgb[4])]
                    xd_ = (int(self.arms[0].dk[0]), int(self.arms[1].dk[0]))
                    if t - self.kd_t0 >= KHE_DO_T - 2.0:
                        self.kd_ls.append((kh[0][0], kh[0][1], kh[0][3], kh[1][0], kh[1][1], kh[1][3], xd_[0], xd_[1], dk_))
                    if int(t * 20) % 20 == 0:
                        print(f"[KHE_DO] t={t:.1f} 2 kep {dk_:.3f} m | khe dau R0 max {kh[0][0]} tv {kh[0][1]:.0f} px ({kh[0][2]} cot, {kh[0][3]} px2) "
                              f"R1 max {kh[1][0]} tv {kh[1][1]:.0f} px ({kh[1][2]} cot, {kh[1][3]} px2) | xanh_dam co tay ({xd_[0]},{xd_[1]})", flush=True)
                    if t - self.kd_t0 >= KHE_DO_T:
                        m_ = np.mean(np.array(self.kd_ls, dtype=float), axis=0) if self.kd_ls else np.zeros(9)
                        print(f"[KHE_DO] KET QUA khe_kep={KHE_KEP} | 2 kep {m_[8]:.3f} m | khe dau R0 max {m_[0]:.0f} tv {m_[1]:.1f} px dt {m_[2]:.0f} | "
                              f"R1 max {m_[3]:.0f} tv {m_[4]:.1f} px dt {m_[5]:.0f} | xanh_dam co tay ({m_[6]:.0f},{m_[7]:.0f}) (trung binh {len(self.kd_ls)} lan do)", flush=True)
                        self.events.append("KHE_DO_XONG")
                        self.luu_moc("moc_mo_khe_xong")
                        self.kd_t0 = None
                if (XO_TAY and getattr(self, "w_mang", None) is not None and getattr(self, "giu_P", None) is not None
                        and getattr(self, "kd_t0", None) is None):
                    self.set_mode("XO_TAY", t)
                    self.xt_pha, self.xt_lift, self.xt_t0 = "NANG", None, t
                    print(f"[MAC] bat dau XO_TAY t={t:.1f}: tay {XT_TAY} cao {XT_CAO} ha {XT_HA} m @ {XT_V} m/s", flush=True)
                if getattr(self, "w_mang", None) is not None and int(t * 20) % 100 == 0:
                    g = self.kep_the_gioi(BP, yaw, arm)
                    print(f"[GIU] t={t:.1f} 2 kep cach nhau {np.hypot(*(g[0] - g[1])):.3f} m "
                          f"yaw=({np.degrees(yaw[0]):.0f},{np.degrees(yaw[1]):.0f})", flush=True)

            elif self.mode == "XO_TAY":     # dua 2 tay nguoi vao khe: NANG -> TIEN -> HA -> GIU_CUOI (co tay + khoang cach 2 kep giu nguyen)
                A0.wrist(JP[0], self.w_mang[0], a)
                A1.wrist(JP[1], self.w_mang[1], a)
                for r in (0, 1):                          # de: bam giu_P (TIEN dich giu_P), KHONG quay
                    e = self.giu_P[r] - BP[r, :2]
                    drive(a, r, 1.5 * e[0], 1.5 * e[1], yaw[r], 0.10)
                    a[r, 2] = float(np.clip(2.0 * wrap(self.giu_yaw[r] - yaw[r]), -0.3, 0.3))
                g = self.kep_the_gioi(BP, yaw, arm)
                gm = 0.5 * (g[0] + g[1])
                zc = [getattr(self.arms[r], "ftzc", None) or lift[r] for r in (0, 1)]
                tay = np.array(XT_TAY, dtype=float)
                if self.xt_lift is None and XT_KEP is not None:     # dat lai khoang cach 2 kep: 2 de dich DEU 2 ben theo duong noi 2 kep
                    d0_ = float(np.hypot(*(g[0] - g[1])))
                    u_ = (g[0] - g[1]) / max(d0_, 1e-6)
                    for r in (0, 1):
                        self.giu_P[r] = self.giu_P[r] + (1 if r == 0 else -1) * 0.5 * (XT_KEP - d0_) * u_
                    print(f"[MAC] 2 kep {d0_:.3f} -> {XT_KEP:.3f} m", flush=True)
                if self.xt_lift is None:                  # lift dich de giua 2 dau ngon cao hon tay XT_CAO
                    self.xt_lift = [float(np.clip(lift[r] + tay[2] + XT_CAO - zc[r], 0.30, XT_LIFT_MAX)) for r in (0, 1)]
                    print(f"[MAC] NANG: kep z ({zc[0]:.3f},{zc[1]:.3f}) -> {tay[2] + XT_CAO:.3f} | lift dich ({self.xt_lift[0]:.3f},{self.xt_lift[1]:.3f})"
                          f"{' (CHAM LIFT_MAX)' if max(self.xt_lift) >= XT_LIFT_MAX - 1e-6 else ''}", flush=True)
                lt = list(self.xt_lift)
                k = t - self.xt_t0
                if self.xt_pha == "NANG":
                    eb_ = max(float(np.hypot(*(self.giu_P[r] - BP[r, :2]))) for r in (0, 1))
                    if XT_DUONG and (all(abs(lt[r] - lift[r]) < LIFT_TOL for r in (0, 1)) and eb_ < 0.015 or k > 10.0):
                        self.xt_hang = ["LUI", "XOAY", "NGANG", "TIEN"]
                        self.xt_pha = "KE"
                        print(f"[MAC] t={t:.1f} NANG xong: kep z ({zc[0]:.3f},{zc[1]:.3f}) | giua 2 kep ({gm[0]:.3f},{gm[1]:.3f}) "
                              f"2 kep {np.hypot(*(g[0] - g[1])):.3f} m -> {self.xt_hang}", flush=True)
                        self.events.append("MAC_NANG_XONG")
                    elif XT_DUONG:
                        pass
                    elif all(abs(lt[r] - lift[r]) < LIFT_TOL for r in (0, 1)) or k > 8.0:
                        self.xt_P0 = [self.giu_P[0].copy(), self.giu_P[1].copy()]
                        self.xt_y0 = list(self.giu_yaw)
                        self.xt_c = gm.copy()                       # tam quay = giua 2 kep
                        self.xt_pha, self.xt_t0 = "XOAY", t
                        print(f"[MAC] t={t:.1f} NANG xong: kep z ({zc[0]:.3f},{zc[1]:.3f}) | giua 2 kep ({gm[0]:.3f},{gm[1]:.3f}) "
                              f"-> XOAY ca khoi {XT_XOAY:+.0f} do quanh giua 2 kep", flush=True)
                        self.events.append("MAC_NANG_XONG")
                elif self.xt_pha == "XOAY":                       # quay CUNG: de + goc than quay cung 1 goc quanh giua 2 kep
                    ph_end = np.radians(XT_XOAY)
                    ph = float(np.sign(ph_end) * min(abs(ph_end), XT_W * k))
                    c_, s_ = np.cos(ph), np.sin(ph)
                    for r in (0, 1):
                        v_ = self.xt_P0[r] - self.xt_c
                        self.giu_P[r] = self.xt_c + np.array([c_ * v_[0] - s_ * v_[1], s_ * v_[0] + c_ * v_[1]])
                        self.giu_yaw[r] = wrap(self.xt_y0[r] + ph)
                    eb = max(float(np.hypot(*(self.giu_P[r] - BP[r, :2]))) for r in (0, 1))
                    ey = max(abs(wrap(self.giu_yaw[r] - yaw[r])) for r in (0, 1))
                    if XT_DUONG and ((abs(ph) >= abs(ph_end) and eb < 0.015 and ey < 0.03) or k > abs(ph_end) / XT_W + 8.0):
                        self.xt_pha = "KE"
                        print(f"[MAC] t={t:.1f} XOAY xong: yaw ({np.degrees(yaw[0]):.0f},{np.degrees(yaw[1]):.0f}) giua 2 kep ({gm[0]:.3f},{gm[1]:.3f})", flush=True)
                    elif (abs(ph) >= abs(ph_end) and eb < 0.015 and ey < 0.03) or k > abs(ph_end) / XT_W + 8.0:
                        d = tay[:2] + np.array(XT_LECH, dtype=float) - gm
                        self.xt_P0, self.xt_d = [self.giu_P[0].copy(), self.giu_P[1].copy()], d
                        self.xt_pha, self.xt_t0 = "TIEN", t
                        print(f"[MAC] t={t:.1f} XOAY xong: yaw ({np.degrees(yaw[0]):.0f},{np.degrees(yaw[1]):.0f}) giua 2 kep ({gm[0]:.3f},{gm[1]:.3f}) "
                              f"(tam {self.xt_c[0]:.3f},{self.xt_c[1]:.3f}) -> TIEN ({d[0]:+.3f},{d[1]:+.3f}) m", flush=True)
                        self.events.append("MAC_XOAY_XONG")
                elif self.xt_pha == "DI":                         # tinh tien CUNG ca khoi: giua 2 kep -> self.xt_dich
                    dn = float(np.hypot(*self.xt_d))
                    f_ = min(1.0, k * XT_V_DE / max(dn, 1e-6))
                    for r in (0, 1):
                        self.giu_P[r] = self.xt_P0[r] + f_ * self.xt_d
                    eb = max(float(np.hypot(*(self.giu_P[r] - BP[r, :2]))) for r in (0, 1))
                    if (f_ >= 1.0 and eb < 0.015) or k > dn / max(XT_V_DE, 1e-3) + 8.0:
                        print(f"[MAC] t={t:.1f} {self.xt_ten} xong: giua 2 kep ({gm[0]:.3f},{gm[1]:.3f}) dich ({self.xt_dich[0]:.3f},{self.xt_dich[1]:.3f}) "
                              f"lech de {100 * eb:.1f} cm{' (HET GIO - BI CHAN?)' if eb >= 0.015 else ''}", flush=True)
                        self.events.append(f"MAC_{self.xt_ten}_XONG")
                        self.xt_pha = "KE"
                if self.xt_pha == "KE":                           # lay buoc ke tiep trong hang (duong DEN NGUOI)
                    C_ = tay[:2]
                    ps_ = np.radians(XT_YAW_NGUOI)
                    fw_ = np.array([np.sin(ps_), -np.cos(ps_)])       # huong mat nguoi (the gioi)
                    ten_ = self.xt_hang.pop(0) if self.xt_hang else None
                    self.xt_t0, k = t, 0.0
                    if ten_ == "XOAY":
                        self.xt_P0 = [self.giu_P[0].copy(), self.giu_P[1].copy()]
                        self.xt_y0 = list(self.giu_yaw)
                        self.xt_c = gm.copy()
                        self.xt_pha = "XOAY"
                        print(f"[MAC] t={t:.1f} XOAY ca khoi {XT_XOAY:+.1f} do quanh giua 2 kep ({gm[0]:.3f},{gm[1]:.3f})", flush=True)
                    elif ten_ in ("LUI", "NGANG", "TIEN"):
                        if ten_ == "LUI":
                            dich_ = gm + (XT_XA - float((gm - C_) @ fw_)) * fw_
                        elif ten_ == "NGANG":
                            dich_ = C_ + XT_XA * fw_
                        else:
                            dich_ = C_ + XT_TRUOC * fw_
                        self.xt_ten, self.xt_dich = ten_, dich_
                        self.xt_P0 = [self.giu_P[0].copy(), self.giu_P[1].copy()]
                        self.xt_d = dich_ - gm
                        self.xt_pha = "DI"
                        print(f"[MAC] t={t:.1f} {ten_}: giua 2 kep ({gm[0]:.3f},{gm[1]:.3f}) -> ({dich_[0]:.3f},{dich_[1]:.3f}) "
                              f"({np.hypot(*self.xt_d):.2f} m)", flush=True)
                    else:
                        self.xt_pha = "HA" if XT_HA > 0 else "GIU_CUOI"
                        print(f"[MAC] t={t:.1f} DEN NGUOI xong: giua 2 kep ({gm[0]:.3f},{gm[1]:.3f}) z ({zc[0]:.3f},{zc[1]:.3f}) "
                              f"2 kep {np.hypot(*(g[0] - g[1])):.3f} m | de0 ({BP[0,0]:.2f},{BP[0,1]:.2f}) de1 ({BP[1,0]:.2f},{BP[1,1]:.2f}) "
                              f"yaw ({np.degrees(yaw[0]):.0f},{np.degrees(yaw[1]):.0f}) -> {self.xt_pha}", flush=True)
                        self.events.append("MAC_DEN_NGUOI")
                elif self.xt_pha == "TIEN":
                    dn = float(np.hypot(*self.xt_d))
                    f_ = min(1.0, k * XT_V_DE / max(dn, 1e-6))
                    for r in (0, 1):
                        self.giu_P[r] = self.xt_P0[r] + f_ * self.xt_d
                    eb = max(float(np.hypot(*(self.giu_P[r] - BP[r, :2]))) for r in (0, 1))
                    if (f_ >= 1.0 and eb < 0.015) or k > dn / max(XT_V_DE, 1e-3) + 8.0:
                        self.xt_pha, self.xt_t0 = "HA", t
                        print(f"[MAC] t={t:.1f} TIEN xong: giua 2 kep ({gm[0]:.3f},{gm[1]:.3f}) tay ({tay[0]:.3f},{tay[1]:.3f}) "
                              f"lech de {100 * eb:.1f} cm", flush=True)
                        self.events.append("MAC_TIEN_XONG")
                elif self.xt_pha == "HA":
                    h = min(XT_HA, XT_V * k)
                    lt = [self.xt_lift[r] - h for r in (0, 1)]
                    if h >= XT_HA and all(abs(lt[r] - lift[r]) < LIFT_TOL for r in (0, 1)) or k > XT_HA / max(XT_V, 1e-3) + 6.0:
                        self.xt_pha, self.xt_t0 = "GIU_CUOI", t
                        self.xt_lift = lt
                        print(f"[MAC] t={t:.1f} HA xong: kep z ({zc[0]:.3f},{zc[1]:.3f}) tay z {tay[2]:.3f}", flush=True)
                elif self.xt_pha == "GIU_CUOI":
                    if k >= XT_GIU:
                        self.xt_pha = "XONG"
                        print(f"[MAC] t={t:.1f} XO_TAY XONG: giua 2 kep ({gm[0]:.3f},{gm[1]:.3f}) z ({zc[0]:.3f},{zc[1]:.3f}) "
                              f"2 kep {np.hypot(*(g[0] - g[1])):.3f} m", flush=True)
                        dn_ = []
                        for im_ in (rgb[3], rgb[4]):              # diem DEN (soc ao / toc nguoi) tren camera dau
                            im_ = np.asarray(im_)[..., :3].astype(int)
                            mx_, mn_ = im_.max(-1), im_.min(-1)
                            m_ = (mx_ < 75) & (mx_ - mn_ < 25)
                            cs_ = np.nonzero(m_)[1]
                            dn_.append((int(m_.sum()), int(np.median(cs_)) if len(cs_) else -1))
                        print(f"[MAC] camera dau thay nguoi (diem den, cot giua): R0 {dn_[0]} R1 {dn_[1]}", flush=True)
                        self.events.append("XO_TAY_XONG")
                        self.luu_moc(XT_MOC)
                if B2 and self.xt_pha == "XONG":                # bat dau buoc 2 (sau khi nap moc_den_nguoi)
                    self.b2_g0 = [g[0].copy(), g[1].copy()]
                    self.b2_s = 0.0
                    self.b2_lt = [float(x) for x in self.xt_lift]
                    self.b2_tp = t
                    self.xt_pha, self.xt_t0, k = "B2_NANG", t, 0.0
                    print(f"[B2] t={t:.1f} bat dau: nang {B2_NANG} rut {B2_RUT} m @ {B2_V} m/s, de lui {B2_DE} (dung neu < {B2_DMIN} m), "
                          f"nang2 {B2_NANG2}, ha {B2_HA} | arm ({arm[0]:.3f},{arm[1]:.3f}) lift ({lift[0]:.3f},{lift[1]:.3f}) "
                          f"2 kep {np.hypot(*(g[0] - g[1])):.3f} m", flush=True)
                if B4 and self.xt_pha == "B2_DUNG":             # bat dau buoc 4 (sau khi nap moc_tay_vao_khe)
                    self.vao_lai = False
                    self.b4_P0 = [self.giu_P[0].copy(), self.giu_P[1].copy()]
                    self.b4_l0 = list(self.b2_lt)
                    self.b4_s, self.b4_moc = 0.0, 0
                    self.xt_pha, self.xt_t0, k = "B4_DI", t, 0.0
                    print(f"[B4] t={t:.1f} bat dau: doc cang tay {B4_TIEN} m, lift {B4_DZ:+.2f} m/m @ {B4_V} m/s, moc moi {B4_BUOC} m | "
                          f"lift ({lift[0]:.3f},{lift[1]:.3f}) de0 ({BP[0,0]:.2f},{BP[0,1]:.2f}) de1 ({BP[1,0]:.2f},{BP[1,1]:.2f})", flush=True)
                if self.xt_pha.startswith("B4_"):
                    ps_ = np.radians(XT_YAW_NGUOI)
                    fw_ = np.array([np.sin(ps_), -np.cos(ps_)])
                    for r in (0, 1):
                        a[r, 4] = 0.0
                    if self.xt_pha == "B4_DI":
                        dmin_, dten_ = 9.0, ""
                        for r in (0, 1):
                            for ten_p, pp_ in B2_NGUOI.items():
                                d_ = float(np.hypot(*(BP[r, :2] - np.array(pp_, dtype=float))))
                                if d_ < dmin_:
                                    dmin_, dten_ = d_, f"R{r}-{ten_p}"
                        gan_ = dmin_ < B2_DMIN
                        if not gan_:
                            self.b4_s = min(B4_TIEN, k * B4_V)
                            for r in (0, 1):
                                self.giu_P[r] = self.b4_P0[r] - self.b4_s * fw_
                        lt = [self.b4_l0[r] + B4_DZ * self.b4_s for r in (0, 1)]
                        mv_ = min(float(np.hypot(*(BP[r, :2] - self.b4_P0[r]))) for r in (0, 1))
                        if mv_ >= (self.b4_moc + 1) * B4_BUOC - 0.003:             # moi b4_buoc m (theo de THAT) luu 1 moc
                            self.b4_moc += 1
                            ten_m = f"moc_b4_{int(round(100 * self.b4_moc * B4_BUOC)):02d}"
                            print(f"[B4] t={t:.1f} {ten_m}: de da di ({100 * float(np.hypot(*(BP[0, :2] - self.b4_P0[0]))):.1f},"
                                  f"{100 * float(np.hypot(*(BP[1, :2] - self.b4_P0[1]))):.1f}) cm | lift ({lift[0]:.3f},{lift[1]:.3f}) "
                                  f"kep z ({zc[0]:.3f},{zc[1]:.3f}) | 2 kep {np.hypot(*(g[0] - g[1])):.3f} m | gan nguoi {dmin_:.3f} ({dten_})", flush=True)
                            self.events.append(f"B4_{int(round(100 * self.b4_moc * B4_BUOC)):02d}")
                            self.luu_moc(ten_m)
                        eb = max(float(np.hypot(*(self.giu_P[r] - BP[r, :2]))) for r in (0, 1))
                        if gan_ or (self.b4_s >= B4_TIEN and eb < 0.01) or k > B4_TIEN / max(B4_V, 1e-3) + 8.0:
                            if gan_:
                                for r in (0, 1):
                                    self.giu_P[r] = BP[r, :2].copy()
                            self.b4_lt = [float(x) for x in lt]
                            print(f"[B4] t={t:.1f} DI xong{' (DUNG: GAN NGUOI ' + dten_ + ')' if gan_ else ''}: s {100 * self.b4_s:.1f} cm | "
                                  f"de0 ({BP[0,0]:.2f},{BP[0,1]:.2f}) de1 ({BP[1,0]:.2f},{BP[1,1]:.2f}) | lift ({lift[0]:.3f},{lift[1]:.3f})", flush=True)
                            self.events.append("B4_DI_XONG")
                            self.xt_pha, self.xt_t0 = "B4_GIU", t
                    else:
                        lt = list(getattr(self, "b4_lt", self.b4_l0))
                        if self.xt_pha == "B4_GIU" and k >= B4_GIU:
                            self.events.append("B4_XONG")
                            self.xt_pha = "B4_DUNG"
                if B3 and self.xt_pha == "B2_DUNG":             # bat dau buoc 3 (sau khi nap moc_tay_vao_khe)
                    self.vao_lai = False
                    self.b3_P0 = [self.giu_P[0].copy(), self.giu_P[1].copy()]
                    self.b3_l0 = list(self.b2_lt)
                    self.b3_f = 0.0
                    self.xt_pha, self.xt_t0, k = "B3_TIEN", t, 0.0
                    print(f"[B3] t={t:.1f} bat dau: tien {B3_TIEN} m + nang {B3_NANG} m @ {B3_V} m/s | lift ({lift[0]:.3f},{lift[1]:.3f}) "
                          f"de0 ({BP[0,0]:.2f},{BP[0,1]:.2f}) de1 ({BP[1,0]:.2f},{BP[1,1]:.2f}) 2 kep {np.hypot(*(g[0] - g[1])):.3f} m", flush=True)
                if self.xt_pha.startswith("B3_"):
                    ps_ = np.radians(XT_YAW_NGUOI)
                    fw_ = np.array([np.sin(ps_), -np.cos(ps_)])
                    for r in (0, 1):
                        a[r, 4] = 0.0                         # arm giu nguyen
                    if self.xt_pha == "B3_TIEN":
                        dmin_, dten_ = 9.0, ""
                        for r in (0, 1):
                            for ten_p, pp_ in B2_NGUOI.items():
                                d_ = float(np.hypot(*(BP[r, :2] - np.array(pp_, dtype=float))))
                                if d_ < dmin_:
                                    dmin_, dten_ = d_, f"R{r}-{ten_p}"
                        gan_ = dmin_ < B2_DMIN
                        if not gan_:
                            self.b3_f = min(1.0, k * B3_V / max(B3_TIEN, 1e-6))
                            for r in (0, 1):
                                self.giu_P[r] = self.b3_P0[r] - self.b3_f * B3_TIEN * fw_
                        eb = max(float(np.hypot(*(self.giu_P[r] - BP[r, :2]))) for r in (0, 1))
                        lt = [self.b3_l0[r] + self.b3_f * B3_NANG for r in (0, 1)]
                        if gan_ or (self.b3_f >= 1.0 and eb < 0.01 and all(abs(lt[r] - lift[r]) < LIFT_TOL for r in (0, 1))) \
                                or k > B3_TIEN / max(B3_V, 1e-3) + 6.0:
                            if gan_:
                                for r in (0, 1):
                                    self.giu_P[r] = BP[r, :2].copy()
                            self.b3_lt = [float(x) for x in lt]
                            mv_ = [float(np.hypot(*(BP[r, :2] - self.b3_P0[r]))) for r in (0, 1)]
                            print(f"[B3] t={t:.1f} TIEN xong: tien ({100 * mv_[0]:.1f},{100 * mv_[1]:.1f}) cm{' (DUNG: GAN NGUOI)' if gan_ else ''} | "
                                  f"gan nhat {dmin_:.3f} m ({dten_}) | lift ({lift[0]:.3f},{lift[1]:.3f}) kep z ({zc[0]:.3f},{zc[1]:.3f}) | "
                                  f"de0 ({BP[0,0]:.2f},{BP[0,1]:.2f}) de1 ({BP[1,0]:.2f},{BP[1,1]:.2f}) | 2 kep {np.hypot(*(g[0] - g[1])):.3f} m", flush=True)
                            self.events.append("B3_TIEN_XONG")
                            self.xt_pha, self.xt_t0 = "B3_GIU", t
                    else:
                        lt = list(getattr(self, "b3_lt", self.b3_l0))
                        if self.xt_pha == "B3_GIU" and k >= B3_GIU:
                            print(f"[B3] t={t:.1f} XONG: kep z ({zc[0]:.3f},{zc[1]:.3f}) 2 kep {np.hypot(*(g[0] - g[1])):.3f} m | "
                                  f"de0 ({BP[0,0]:.2f},{BP[0,1]:.2f}) de1 ({BP[1,0]:.2f},{BP[1,1]:.2f})", flush=True)
                            self.events.append("B3_XONG")
                            if B3_MOC:
                                self.luu_moc(B3_MOC)
                            self.xt_pha = "B3_DUNG"
                if getattr(self, "vao_lai", False) and self.xt_pha.startswith("B2_"):    # nap tu moc (vd moc_lui_de) voi json moi
                    self.vao_lai = False
                    if self.xt_pha in ("B2_GIU", "B2_NANG2", "B2_HA"):
                        sau_ = (["B2_HA", "B2_NANG2"] if B2_HA_TRUOC else ["B2_NANG2", "B2_HA"])
                        sau_ = [x_ for x_ in sau_ if (B2_HA if x_ == "B2_HA" else B2_NANG2) > 0] + ["B2_GIU"]
                        self.xt_pha, self.xt_t0, k = sau_[0], t, 0.0
                        self.b2_tp = t
                        print(f"[B2] t={t:.1f} nap lai tu moc -> {self.xt_pha} | lift ({lift[0]:.3f},{lift[1]:.3f})", flush=True)
                if self.xt_pha.startswith("B2_"):
                    ps_ = np.radians(XT_YAW_NGUOI)
                    fw_ = np.array([np.sin(ps_), -np.cos(ps_)])
                    dtb = max(0.0, t - self.b2_tp)
                    self.b2_tp = t
                    ph_ = self.xt_pha
                    s_rut = self.b2_s
                    lt = list(self.b2_lt)

                    def b2_sau(ten_xong, t_):                   # thu tu: NANG -> RUT -> DE -> NANG2 -> HA -> GIU
                        ds_ = ["B2_NANG", "B2_RUT", "B2_DE"] + (["B2_HA", "B2_NANG2"] if B2_HA_TRUOC else ["B2_NANG2", "B2_HA"]) + ["B2_GIU"]
                        co_ = {"B2_NANG": True, "B2_RUT": True, "B2_DE": B2_DE > 0, "B2_NANG2": B2_NANG2 > 0, "B2_HA": B2_HA > 0, "B2_GIU": True}
                        for x_ in ds_[ds_.index(ten_xong) + 1:]:
                            if co_[x_]:
                                return x_
                        return "B2_GIU"

                    if ph_ == "B2_NANG":
                        lt = [self.b2_lt[r] + B2_NANG for r in (0, 1)]
                        if all(abs(lt[r] - lift[r]) < LIFT_TOL for r in (0, 1)) or k > 6.0:
                            self.b2_lt = lt
                            self.xt_pha, self.xt_t0 = b2_sau(ph_, t), t
                            print(f"[B2] t={t:.1f} NANG xong: lift ({lift[0]:.3f},{lift[1]:.3f}) kep z ({zc[0]:.3f},{zc[1]:.3f})", flush=True)
                    elif ph_ == "B2_RUT":
                        s_rut = min(B2_RUT, B2_V * k)
                        het = [arm[r] <= B2_ARM_MIN + 0.003 for r in (0, 1)]
                        if (s_rut >= B2_RUT and max(np.hypot(*(self.b2_g0[r] - s_rut * fw_ - g[r])) for r in (0, 1)) < 0.01) \
                                or all(het) or k > B2_RUT / max(B2_V, 1e-3) + 6.0:
                            self.b2_g0, self.b2_s, s_rut = [g[0].copy(), g[1].copy()], 0.0, 0.0     # giu kep tai cho tu day
                            self.xt_pha, self.xt_t0 = b2_sau(ph_, t), t
                            print(f"[B2] t={t:.1f} RUT xong: lui {100 * min(B2_RUT, B2_V * k):.1f} cm | arm ({arm[0]:.3f},{arm[1]:.3f}){' (HET ARM)' if any(het) else ''} | "
                                  f"kep R0 ({g[0][0]:.3f},{g[0][1]:.3f}) R1 ({g[1][0]:.3f},{g[1][1]:.3f}) 2 kep {np.hypot(*(g[0] - g[1])):.3f} m | "
                                  f"yaw ({np.degrees(yaw[0]):.0f},{np.degrees(yaw[1]):.0f})", flush=True)
                            self.events.append("B2_RUT_XONG")
                    elif ph_ == "B2_DE":                    # 2 de tinh tien ve phia nguoi (-fw), arm giu, dung neu de gan nguoi < B2_DMIN
                        if k == 0.0 or not hasattr(self, "b2_P0") or getattr(self, "b2_P0_t", None) != self.xt_t0:
                            self.b2_P0 = [self.giu_P[0].copy(), self.giu_P[1].copy()]
                            self.b2_P0_t = self.xt_t0
                        f_ = min(B2_DE, B2_V * k)
                        dmin_, dten_ = 9.0, ""
                        for r in (0, 1):
                            for ten_p, pp_ in B2_NGUOI.items():
                                d_ = float(np.hypot(*(BP[r, :2] - np.array(pp_, dtype=float))))
                                if d_ < dmin_:
                                    dmin_, dten_ = d_, f"R{r}-{ten_p}"
                        gan_ = dmin_ < B2_DMIN
                        if not gan_:
                            for r in (0, 1):
                                self.giu_P[r] = self.b2_P0[r] - f_ * fw_
                        eb = max(float(np.hypot(*(self.giu_P[r] - BP[r, :2]))) for r in (0, 1))
                        if gan_ or (f_ >= B2_DE and eb < 0.01) or k > B2_DE / max(B2_V, 1e-3) + 6.0:
                            if gan_:
                                for r in (0, 1):
                                    self.giu_P[r] = BP[r, :2].copy()
                            mv_ = [float(np.hypot(*(BP[r, :2] - self.b2_P0[r]))) for r in (0, 1)]
                            self.b2_g0 = [g[0].copy(), g[1].copy()]
                            self.xt_pha, self.xt_t0 = b2_sau(ph_, t), t
                            print(f"[B2] t={t:.1f} DE xong: lui ({100 * mv_[0]:.1f},{100 * mv_[1]:.1f}) cm{' (DUNG: GAN NGUOI)' if gan_ else ''} | "
                                  f"gan nhat {dmin_:.3f} m ({dten_}) | de0 ({BP[0,0]:.2f},{BP[0,1]:.2f}) de1 ({BP[1,0]:.2f},{BP[1,1]:.2f}) | "
                                  f"2 kep {np.hypot(*(g[0] - g[1])):.3f} m", flush=True)
                            self.events.append("B2_DE_XONG")
                            if B2_MOC_DE:
                                self.luu_moc(B2_MOC_DE)
                    elif ph_ in ("B2_NANG2", "B2_HA"):
                        dz_ = B2_NANG2 if ph_ == "B2_NANG2" else -B2_HA
                        h = min(abs(dz_), XT_V * k)
                        lt = [self.b2_lt[r] + np.sign(dz_) * h for r in (0, 1)]
                        if (h >= abs(dz_) and all(abs(lt[r] - lift[r]) < LIFT_TOL for r in (0, 1))) or k > abs(dz_) / max(XT_V, 1e-3) + 6.0:
                            self.b2_lt = lt
                            self.xt_pha, self.xt_t0 = b2_sau(ph_, t), t
                            print(f"[B2] t={t:.1f} {ph_[3:]} xong: lift ({lift[0]:.3f},{lift[1]:.3f}) kep z ({zc[0]:.3f},{zc[1]:.3f})", flush=True)
                            self.events.append(f"{ph_}_XONG")
                    elif ph_ == "B2_GIU" and k >= B2_GIU:
                        self.xt_pha = "B2_XONG"
                    self.b2_s = s_rut
                    for r in (0, 1):                          # kep r -> b2_g0 - s_rut * fw_: arm theo huong tay, xoay than tai cho cho phan ngang
                        if self.xt_pha == "B2_DE":
                            a[r, 4] = 0.0                     # luc de di: arm + goc than giu nguyen
                            continue
                        e_ = self.b2_g0[r] - s_rut * fw_ - g[r]
                        u_ = np.array([np.cos(yaw[r]), np.sin(yaw[r])])
                        n_ = np.array([-u_[1], u_[0]])
                        ea_ = float(e_ @ u_)
                        if arm[r] <= B2_ARM_MIN and ea_ < 0:
                            ea_ = 0.0
                        a[r, 4] = float(np.clip(cmd(ea_, RATE["arm"], 3.0), -0.3, 0.3))
                        rr_ = max(float(np.hypot(*(g[r] - BP[r, :2]))), 0.3)
                        self.giu_yaw[r] = wrap(self.giu_yaw[r] + float(np.clip(1.0 * float(e_ @ n_) / rr_, -0.3, 0.3)) * dtb)
                    if self.xt_pha == "B2_XONG":
                        ds_ = []
                        for im_ in (rgb[1], rgb[2], rgb[3], rgb[4]):    # diem DEN (soc ao nguoi): co tay R0, R1, dau R0, R1
                            im_ = np.asarray(im_)[..., :3].astype(int)
                            mx_, mn_ = im_.max(-1), im_.min(-1)
                            m_ = (mx_ < 75) & (mx_ - mn_ < 25)
                            cs_ = np.nonzero(m_)[1]
                            ds_.append((int(m_.sum()), int(np.median(cs_)) if len(cs_) else -1))
                        print(f"[B2] t={t:.1f} XONG: diem den (so, cot giua) co tay R0 {ds_[0]} R1 {ds_[1]} | dau R0 {ds_[2]} R1 {ds_[3]} | "
                              f"kep z ({zc[0]:.3f},{zc[1]:.3f}) 2 kep {np.hypot(*(g[0] - g[1])):.3f} m | arm ({arm[0]:.3f},{arm[1]:.3f}) "
                              f"de0 ({BP[0,0]:.2f},{BP[0,1]:.2f}) de1 ({BP[1,0]:.2f},{BP[1,1]:.2f})", flush=True)
                        self.events.append("B2_XONG")
                        if B2_MOC:
                            self.luu_moc(B2_MOC)
                        self.xt_pha = "B2_DUNG"
                for r in (0, 1):
                    a[r, 3] = cmd(lt[r] - lift[r], RATE["lift"], 3.0)
                if int(t * 20) % 20 == 0:
                    print(f"[MAC] t={t:.1f} {self.xt_pha} | arm ({arm[0]:.3f},{arm[1]:.3f}) | giua 2 kep ({gm[0]:.3f},{gm[1]:.3f}) z ({zc[0]:.3f},{zc[1]:.3f}) | "
                          f"lift ({lift[0]:.3f},{lift[1]:.3f}) dich ({lt[0]:.3f},{lt[1]:.3f}) | 2 kep {np.hypot(*(g[0] - g[1])):.3f} m | "
                          f"de0 ({BP[0,0]:.2f},{BP[0,1]:.2f}) de1 ({BP[1,0]:.2f},{BP[1,1]:.2f}) yaw ({np.degrees(yaw[0]):.0f},{np.degrees(yaw[1]):.0f})", flush=True)

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

            self.cap_nhat_giai_doan(t)
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
