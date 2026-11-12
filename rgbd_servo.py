"""Steps 1+2: original reverse-roll NAV, followed by RGB-D jaw alignment.

Uses public observations and bundled static robot/camera calibration. No Isaac
imports. Step 1 uses link-origin distances; step 2 adds visual gripper envelopes.
"""
import json
import time
from pathlib import Path
import numpy as np
from itertools import product, combinations

def rotation(q):
    q = np.asarray(q, float)
    norm = np.linalg.norm(q)
    if norm < 1e-8 or not np.isfinite(q).all():
        raise ValueError('Invalid quaternion')
    w, x, y, z = q / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def transform(p, q):
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = rotation(q), p
    return t


def points(t, p):
    return np.asarray(p) @ t[:3, :3].T + t[:3, 3]


def optical_mount(spec):
    eye, aim, up = (np.array(spec[k], float) for k in ('eye_mount_m', 'target_mount_m', 'up_mount'))
    forward = aim - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = np.column_stack((right, down, forward)), eye
    return t


class Kinematics:
    def __init__(self, path=None):
        self.data = json.loads(Path(path or Path(__file__).with_name('static_robot_camera.json')).read_text())
        self.names = self.data['joint_names']
        for j in self.data['joints']:
            j['_a'] = transform(j['p0'], j['q0'])
            j['_b_inv'] = np.linalg.inv(transform(j['p1'], j['q1']))

    def frames(self, q):
        values = dict(zip(self.names, q))
        out = {'base_footprint': np.eye(4)}
        pending = list(self.data['joints'])
        while pending:
            rest = []
            for j in pending:
                if j['parent'] not in out:
                    rest.append(j)
                    continue
                motion = np.eye(4)
                if j['kind'] != 'PhysicsFixedJoint':
                    axis = np.eye(3)['XYZ'.index(j['axis'])]
                    value = values[j['name']]
                    if j['kind'] == 'PhysicsPrismaticJoint':
                        motion[:3, 3] = axis * value
                    else:
                        motion[:3, :3] = rotation(np.r_[np.cos(value/2), axis*np.sin(value/2)])
                out[j['child']] = out[j['parent']] @ j['_a'] @ motion @ j['_b_inv']
            if len(rest) == len(pending):
                raise ValueError('Disconnected static robot joint tree')
            pending = rest
        return out


def unproject(depth, valid, mask, fov=70, limit=1800):
    h, w = depth.shape
    ys, xs = np.nonzero(mask & valid & np.isfinite(depth) & (depth > 0))
    if len(xs) > limit:
        idx = np.linspace(0, len(xs)-1, limit).astype(int)
        ys, xs = ys[idx], xs[idx]
    z = depth[ys, xs]
    f = w / (2*np.tan(np.deg2rad(fov)/2))
    return np.column_stack(((xs-w/2)*z/f, (ys-h/2)*z/f, z))


def fit_table(p, rng):
    """RANSAC on RGB-selected visible table, then least-squares refinement."""
    if len(p) < 50:
        return None
    best = np.zeros(len(p), bool)
    for _ in range(50):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(b-a, c-a)
        length = np.linalg.norm(n)
        if length < 1e-8:
            continue
        n /= length
        if abs(n[2]) < np.cos(np.deg2rad(12)):
            continue
        inliers = np.abs((p-a) @ n) < 0.006
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < max(40, 0.60*len(p)):
        return None
    inliers = p[best]
    center = np.mean(inliers, axis=0)
    _, _, vh = np.linalg.svd(inliers-center, full_matrices=False)
    spread=np.linalg.svd(inliers-center,compute_uv=False)/np.sqrt(len(inliers))
    if spread[1]<.008:
        return None  # A line of pixels is not a supported table surface.
    n = vh[-1]
    if n[2] < 0:
        n = -n
    residual = float(np.sqrt(np.mean(((inliers-center) @ n)**2)))
    if n[2] < np.cos(np.deg2rad(12)) or residual > 0.005:
        return None
    return n, -float(n @ center), residual


def largest_component(mask):
    """8-connected component; no learned segmentation or extra dependency."""
    unseen = mask.copy()
    best = []
    h,w = mask.shape
    for y,x in zip(*np.nonzero(mask)):
        if not unseen[y,x]:
            continue
        unseen[y,x] = False
        queue = [(int(y),int(x))]
        for cy,cx in queue:
            for dy,dx in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
                ny,nx=cy+dy,cx+dx
                if 0<=ny<h and 0<=nx<w and unseen[ny,nx]:
                    unseen[ny,nx]=False
                    queue.append((ny,nx))
        if len(queue)>len(best):
            best=queue
    out=np.zeros_like(mask)
    if best:
        yy,xx=np.array(best).T
        out[yy,xx]=True
    return out


def dilate(mask,radius=2):
    padded=np.pad(mask,radius)
    out=np.zeros_like(mask)
    for dy in range(2*radius+1):
        for dx in range(2*radius+1):
            out |= padded[dy:dy+mask.shape[0],dx:dx+mask.shape[1]]
    return out


def geometry_masks(image,masks):
    blue,yellow=masks(image)
    r,g,b=[image[...,i].astype(float) for i in range(3)]
    # Keep the old mask; add pale-yellow evidence missed by its B<110 limit.
    yellow |= (r>165)&(g>165)&(r>b+35)&(g>b+35)&(b<180)
    return blue,yellow


def hem_features(blue, yellow):
    """Return visible hem endpoints individually; never invent a cropped corner.

    Near-side/bottom hem assumption is limited to the current table approach.
    Width is compared with the local torso, not the sleeves. Missing endpoints
    do not discard the rest of the visible hem used by local tracking.
    """
    garment=largest_component(dilate(blue|yellow)) & (blue|yellow)
    yy,xx=np.nonzero(garment)
    if len(xx)<180:
        return None
    gy=np.indices(blue.shape)[0]
    candidate=yellow&garment&(gy>=yy.min()+.60*(yy.max()-yy.min()))
    hem=largest_component(dilate(candidate))&candidate
    hy,hx=np.nonzero(hem)
    if len(hx)<12 or hx.max()-hx.min()<8:
        return None
    # Follow the outer hem contour, not the filled 5cm-wide painted stripe.
    # Filled corner patches can be isotropic and have no meaningful PCA axis.
    boundary=np.zeros_like(hem)
    for column in np.unique(hx):
        rows_here=hy[hx==column]
        boundary[int(rows_here.max()),int(column)]=True
    ends=[]
    for left in (True,False):
        edge=int(hx.min() if left else hx.max())
        near=np.abs(hx-edge)<=3
        y=int(np.percentile(hy[near],90))
        sy,sx=np.nonzero(garment[max(0,y-10):min(blue.shape[0],y+3)])
        side=None if not len(sx) else int(sx.min() if left else sx.max())
        visible=edge>=3 and edge<=blue.shape[1]-4 and side is not None and abs(edge-side)<=6
        ends.append((edge,y) if visible else None)
    return dict(endpoints=ends,hem=hem,boundary=boundary,garment=garment)


def wrap(angle):
    return (angle+np.pi)%(2*np.pi)-np.pi


def yaw(q):
    R=rotation(q)
    return float(np.arctan2(R[1,0],R[0,0]))


def yaw_quat(angle):
    return np.array([np.cos(angle/2),0.,0.,np.sin(angle/2)])


def cap(vector, maximum):
    return vector*min(1.,maximum/max(np.linalg.norm(vector),1e-9))


def pair_guard(bases, velocities, measured, minimum=.40, horizon=.5):
    """Position + measured closing-velocity margin; only edits base translation.

    This is a center-distance constraint, not complete collision avoidance.
    Never forces a stationary partner to follow the other robot.
    """
    d=bases[1]-bases[0];length=float(np.linalg.norm(d))
    if length<1e-8:
        return np.zeros_like(velocities),True
    n=d/length
    closing=float((measured[1]-measured[0])@n)
    extra=min(.025,max(0.,-closing)*.18)
    limit=-(length-minimum-extra)/horizon
    relative=float((velocities[1]-velocities[0])@n)
    out=velocities.copy()
    blocked=relative<limit
    if blocked:
        # First scale inward contributions, preserve outward/tangential motion.
        inward0=max(0.,float(out[0]@n));inward1=max(0.,float(-out[1]@n))
        inward=inward0+inward1
        remove=min(inward,max(0.,limit-relative))
        if inward>1e-9:
            out[0]-=n*remove*inward0/inward
            out[1]+=n*remove*inward1/inward
        # If already inside margin, allow a small symmetric outward recovery.
        remainder=limit-float((out[1]-out[0])@n)
        if remainder>1e-6:
            correction=min(.012,remainder/2)
            out[0]-=n*correction;out[1]+=n*correction
    return out,blocked


class Step1Servo:
    """One phase machine for both robots; only public RGB-D/proprioception."""
    READY_PLANE_MAX_AGE = 3.0
    VERSION='corner-local-step1-v3-reverse-roll'
    DEADLINES={'SCAN':25.,'ACQUIRE':12.,'NAV':40.,'PREP':12.,'APPROACH':35.,
               'ROLL':18.,'VERIFY':10.,'RECOVER':8.}

    def __init__(self,masks,save_png,snap_dir,snap_every,snap_enabled=True):
        self.masks,self.save_png=masks,save_png
        self.snap_dir,self.snap_every,self.snap_enabled=snap_dir,snap_every,snap_enabled
        self.kin=Kinematics()
        if not {'joint_limits','runtime_rates'}<=self.kin.data.keys():
            raise ValueError('V3_REQUIRES_NEW_static_robot_camera.json')
        self.rates=self.kin.data['runtime_rates']
        self.rng=np.random.default_rng(7)
        self.phase='SCAN';self.since=None;self.start=None;self.events=[]
        self.plan=None;self.plan_vote=None;self.plane=None;self.plane_time=-np.inf
        self.global_measurements=[];self.local=[None,None];self.geo=[None,None]
        self.obs=None;self.t=0.;self.fault=None;self.finished=False
        self.hold_base=None;self.hold_q=None;self.resume='APPROACH'
        self.replans=0;self.recoveries=0;self.block_since=None;self.no_progress=None
        self.stable=None;self.lost_since=None;self.change_vote=None
        self.last_base=None;self.base_velocity=np.zeros((2,2));self.last_t=None
        self.last_log=-np.inf;self.last_png=-np.inf;self.last_perception=-np.inf
        self.requested=np.zeros((2,9));self.applied=np.zeros((2,9))
        self.reason='searching head images';self.diag={};self.view={}
        self.role_axis=None;self.scan_anchor=None
        self.minimum_toe=0.

    def transition(self,phase,reason):
        if phase!=self.phase:
            self.events.append(f'{self.phase}_TO_{phase}')
            self.phase=phase;self.since=self.t;self.stable=None;self.no_progress=None
        self.reason=reason

    def stop(self,reason):
        if self.fault is None:
            self.fault=reason;self.capture_hold()
        self.transition('STOP',reason);self.finished=False

    def capture_hold(self):
        if self.obs is not None:
            self.hold_base=np.array(self.obs['base_pose_world'],float).copy()
            self.hold_q=np.array(self.obs['joint_position'],float).copy()

    def geometry(self,r):
        q=np.asarray(self.obs['joint_position'],float)[r]
        bp=np.asarray(self.obs['base_pose_world'],float)[r]
        gp=np.asarray(self.obs['grasp_pose_base'],float)[r]
        fp=np.asarray(self.obs['fingertip_position_base'],float)[r]
        frames=self.kin.frames(q)
        root=transform(gp[:3],gp[3:])@np.linalg.inv(frames['grasp_center_link'])
        world=transform(bp[:3],bp[3:])
        for k,name in enumerate(('gripper_fingertip_left_link','gripper_fingertip_right_link')):
            if np.linalg.norm((root@frames[name])[:3,3]-fp[k])>.02:
                raise ValueError('FK_MISMATCH')
        return q,world,root,frames,points(world,gp[:3]),points(world,fp)

    def camera(self,index,geo):
        spec=self.kin.data['cameras'][index-1]
        T=geo[1]@geo[2]@geo[3][spec['mount_role']]@optical_mount(spec)
        return T,spec

    def cloud(self,index,geo,mask,limit=2500):
        dep=np.asarray(self.obs['depth'])[index,...,0]
        T,spec=self.camera(index,geo)
        valid=np.asarray(self.obs['depth_valid'])[index,...,0].astype(bool)
        valid=valid&np.isfinite(dep)&(dep>=spec['clip_m'][0])&(dep<=spec['clip_m'][1])
        return points(T,unproject(dep,valid,mask,spec['horizontal_fov_deg'],limit=limit))

    def read_head(self,r):
        index=3+r;geo=self.geo[r];im=np.asarray(self.obs['rgb'])[index]
        blue,yellow=geometry_masks(im,self.masks)
        features=hem_features(blue,yellow)
        if features is None:
            return None
        hem=self.cloud(index,geo,features['hem'])
        contour=self.cloud(index,geo,features['boundary'])
        body=self.cloud(index,geo,blue)
        if len(hem)<12 or len(body)<50:
            return None
        corners=[]
        for pixel in features['endpoints']:
            if pixel is None:corners.append(None);continue
            x,y=pixel;mask=np.zeros_like(blue)
            mask[max(0,y-2):y+3,max(0,x-2):x+3]=True
            sample=self.cloud(index,geo,mask&features['hem'])
            corners.append(np.median(sample,axis=0) if len(sample)>=3 else None)
        # Broad image table search constrained by observed shirt neighborhood
        # and measured height, not a narrow color ROI beside the hem.
        neutral=(np.ptp(im.astype(float),axis=2)<35)&(im.mean(axis=2)>50)&~dilate(features['garment'],3)
        table=self.cloud(index,geo,neutral)
        center=np.median(hem,axis=0)
        table=table[(np.linalg.norm(table[:,:2]-center[:2],axis=1)<.8)&
                    (table[:,2]>center[2]-.14)&(table[:,2]<center[2]+.025)]
        plane=fit_table(table,self.rng)
        if plane is not None:
            n,d,rms=plane
            if self.plane is None or (self.plane[0]@n>np.cos(np.radians(8)) and abs(self.plane[1]-d)<.06):
                self.plane=plane;self.plane_time=self.t
        if self.plane is None or self.t-self.plane_time>3.:
            return None
        n,d,rms=self.plane
        # Slightly raised fabric remains a measured point. Do not flatten its
        # elevation into the table or assume it has been grasped.
        if any(p is not None and not -.015<float(p@n+d)<.10 for p in corners):
            return None
        return dict(corners=corners,hem=hem,contour=contour,body=body,camera=index,time=self.t,
                    plane=self.plane,full=all(p is not None for p in corners))

    def consensus(self):
        full=[x for x in self.global_measurements if x is not None and x['full']]
        if not full:return None
        ordered=[]
        for x in full:
            corners=np.array(x['corners'])
            order=np.argsort(corners@self.role_axis)
            ordered.append((corners[order],x))
        if len(ordered)==2 and np.max(np.linalg.norm(ordered[0][0]-ordered[1][0],axis=1))>.03:
            self.reason='GLOBAL_VIEWS_DISAGREE';return None
        corners=np.mean([x[0] for x in ordered],axis=0)
        width=np.linalg.norm(corners[1]-corners[0])
        if not .18<width<.85:return None
        n,d,rms=self.plane
        tangents=[];outward=[]
        hem=np.vstack([x[1]['contour'] for x in ordered]);body=np.vstack([x[1]['body'] for x in ordered])
        for r,c in enumerate(corners):
            directions=[]
            for radius in (.04,.075):
                near=hem[np.linalg.norm(hem-c,axis=1)<radius]
                if len(near)<6:return None
                flat=near-np.outer(near@n+d,n)
                _,sing,vh=np.linalg.svd(flat-flat.mean(axis=0),full_matrices=False)
                if len(sing)<2 or sing[0]<1.5*max(sing[1],1e-6):
                    self.reason='AMBIGUOUS_LOCAL_HEM';return None
                t=vh[0]-n*(vh[0]@n);t/=max(np.linalg.norm(t),1e-9)
                if t@(corners[1-r]-c)<0:t=-t
                directions.append(t)
            if directions[0]@directions[1]<np.cos(np.radians(30)):
                self.reason='CURLED_CORNER_REOBSERVE';return None
            tangent=directions[0]+directions[1];tangent/=np.linalg.norm(tangent)
            chord=corners[1-r]-c;chord-=n*(chord@n);chord/=max(np.linalg.norm(chord),1e-9)
            # A hooked endpoint whose local edge runs almost perpendicular to
            # the rest of the near-side hem is ambiguous, not a new approach
            # direction. This bounds the supported cloth layouts explicitly.
            if tangent@chord<.5:
                self.reason='HOOKED_HEM_REOBSERVE';return None
            out=np.cross(tangent,n)
            if out@(np.median(body,axis=0)-c)>0:out=-out
            tangents.append(tangent);outward.append(out)
        return dict(corners=corners,tangents=np.array(tangents),outward=np.array(outward),
                    normal=n,offset=d,time=self.t,sources=[x[1]['camera'] for x in ordered])

    def local_observation(self,r):
        if self.plan is None:return None
        expected=self.plan['surface'][r]
        options=[]
        # Wrist observes ONLY the assigned patch. Head/partner head can track
        # that patch too; neither must show the other corner after planning.
        for index,observer in ((1+r,r),(3+r,r),(4-r,1-r)):
            im=np.asarray(self.obs['rgb'])[index]
            blue,yellow=geometry_masks(im,self.masks)
            T,spec=self.camera(index,self.geo[observer])
            p=points(np.linalg.inv(T),expected)
            if p[2]<spec['clip_m'][0]:continue
            f=256/(2*np.tan(np.radians(spec['horizontal_fov_deg'])/2))
            uv=128+f*p[:2]/p[2]
            if not (-10<uv[0]<266 and -10<uv[1]<266):continue
            # ROI scales in metric units, rather than a fixed fingertip box.
            rad=int(np.clip(f*.045/p[2],6,45));u,v=np.round(uv).astype(int)
            roi=np.zeros((256,256),bool)
            roi[max(0,v-rad):min(256,v+rad+1),max(0,u-rad):min(256,u+rad+1)]=True
            sample=self.cloud(index,self.geo[observer],roi&yellow)
            if len(sample)<6:continue
            distances=np.linalg.norm(sample-expected,axis=1)
            sample=sample[distances<.045]
            if len(sample)<6:continue
            distances=np.linalg.norm(sample-expected,axis=1)
            chosen=sample[np.argsort(distances)[:max(6,min(25,len(sample)))]]
            seen=np.median(chosen,axis=0)
            if np.linalg.norm(seen-expected)>.025:continue
            options.append(dict(point=seen,camera=index,time=self.t,pixels=len(sample),
                                wrist=index==1+r,error=float(np.linalg.norm(seen-expected))))
        if not options:return None
        # Prefer fresh assigned-wrist evidence; all measurements are validated
        # against the same world target and depth neighborhood.
        result=min(options,key=lambda x:(not x['wrist'],x['error']))
        result['wrist_visible']=any(x['wrist'] for x in options)
        return result

    def refresh_local_table(self):
        """Wrist needs only a nearby table patch, not a full-shirt view."""
        if self.plan is None or self.t-self.plane_time<.3:return
        for r in (0,1):
            index=1+r;im=np.asarray(self.obs['rgb'])[index]
            blue,yellow=geometry_masks(im,self.masks)
            neutral=(np.ptp(im.astype(float),axis=2)<35)&(im.mean(axis=2)>50)&~dilate(blue|yellow)
            cloud=self.cloud(index,self.geo[r],neutral)
            if not len(cloud):continue
            n,d,_=self.plane
            cloud=cloud[(np.linalg.norm(cloud[:,:2]-self.plan['surface'][r,:2],axis=1)<.25)&
                        (np.abs(cloud@n+d)<.025)]
            fitted=fit_table(cloud,self.rng)
            if fitted is not None and fitted[0]@n>np.cos(np.radians(5)):
                self.plane=fitted;self.plane_time=self.t;return

    def update(self,obs):
        required={'joint_position':(2,13),'base_pose_world':(2,7),'grasp_pose_base':(2,7),
                  'fingertip_position_base':(2,2,3),'rgb':(5,256,256,3),
                  'depth':(5,256,256,1),'depth_valid':(5,256,256,1)}
        for key,shape in required.items():
            if np.asarray(obs[key]).shape!=shape:raise ValueError(f'INPUT_SHAPE_{key}')
        for key in ('joint_position','base_pose_world','grasp_pose_base','fingertip_position_base'):
            if not np.isfinite(obs[key]).all():raise ValueError(f'NONFINITE_{key}')
        self.obs=obs;self.t=float(np.asarray(obs['simulation_time_s']).ravel()[0])
        if not np.isfinite(self.t):raise ValueError('NONFINITE_TIME')
        if self.start is None:self.start=self.t;self.since=self.t
        if self.last_t is not None and self.t<=self.last_t:raise ValueError('NONMONOTONIC_TIME_RESET_REQUIRED')
        bp=np.asarray(obs['base_pose_world'],float)
        if self.last_base is not None:
            dt=self.t-self.last_t
            self.base_velocity=.5*self.base_velocity+.5*(bp[:,:2]-self.last_base)/dt
        self.last_base=bp[:,:2].copy();self.last_t=self.t
        if self.role_axis is None:
            d=bp[1,:3]-bp[0,:3];d[2]=0;self.role_axis=d/max(np.linalg.norm(d),1e-9)
            self.scan_anchor=bp.copy()
        self.geo=[self.geometry(r) for r in (0,1)]
        for r in (0,1):
            if rotation(bp[r,3:])[2,2]<np.cos(np.radians(3.5)):
                self.stop(f'BASE_TILT_R{r}')
        # Perception runs at every public observation; it is never frozen on
        # STOP/RECOVER. Every evidence record carries its own timestamp.
        self.global_measurements=[self.read_head(r) for r in (0,1)]
        self.refresh_local_table()
        self.local=[self.local_observation(r) for r in (0,1)]

    def candidate_plan(self,measurement):
        corners=measurement['corners'];n=measurement['normal'];d=measurement['offset']
        surface=corners+.015*measurement['tangents']
        # Use the higher measured corner to reserve height for wrinkled fabric.
        height=max(0.,float(np.max(corners@n+d)))
        projected=surface-np.outer(surface@n+d,n)
        goals=projected+.035*measurement['outward']+n*(.16+height)
        bp=np.asarray(self.obs['base_pose_world'],float)
        best=None
        for toe in (0.,5.,10.,15.,20.):
            if toe<self.minimum_toe:continue
            target_bases=[];target_q=[];yaws=[]
            for r in (0,1):
                inward=-measurement['outward'][r];base_yaw=float(np.arctan2(inward[1],inward[0]))
                # Choose the small heading change that puts this base farther
                # outside the pair while its gripper reaches inward.
                possibilities=[]
                for sign in (-1,1):
                    angle=base_yaw+sign*np.radians(toe)
                    q=self.geo[r][0].copy();q[3]=.55;q[4:8]=.22/4;q[8:11]=0
                    frames=self.kin.frames(q);T=self.geo[r][2]@frames['grasp_center_link']
                    R=rotation(yaw_quat(angle));relative=R@T[:3,3]
                    base=bp[r,:3].copy();base[:2]=goals[r,:2]-relative[:2]
                    trial=q.copy();trial[3]+=.001
                    lift_axis=(R@(self.geo[r][2]@self.kin.frames(trial)['grasp_center_link'])[:3,3]-relative)/.001
                    if abs(lift_axis[2])<.8:continue
                    q[3]+=(goals[r,2]-base[2]-relative[2])/lift_axis[2]
                    lo,hi=self.kin.data['joint_limits']['lift_joint']
                    if not lo+.04<q[3]<hi-.04:continue
                    outside=self.role_axis*(-1 if r==0 else 1)
                    possibilities.append((float(base@outside),base,q,angle))
                if not possibilities:break
                _,base,q,angle=max(possibilities,key=lambda x:x[0])
                target_bases.append(base);target_q.append(q);yaws.append(angle)
            if len(target_bases)!=2:continue
            separation=np.linalg.norm(target_bases[1][:2]-target_bases[0][:2])
            if separation<.45:continue
            # Both bases must remain on the observed outside of the hem.
            if any((target_bases[r]-corners[r])@measurement['outward'][r]<.30 for r in (0,1)):continue
            predicted=[]
            for r in (0,1):
                q=target_q[r].copy();q[10]=np.pi/4*(-1 if r==0 else 1)
                f=self.kin.frames(q)
                world=transform(target_bases[r],yaw_quat(yaws[r]))@self.geo[r][2]
                predicted.append(np.array([(world@f[name])[:3,3] for name in
                    ('gripper_fingertip_left_link','gripper_fingertip_right_link')]))
            if np.min(np.linalg.norm(predicted[0][:,None]-predicted[1][None,:],axis=2))<.10:continue
            cost=toe*.02+sum(np.linalg.norm(target_bases[r][:2]-bp[r,:2]) for r in (0,1))*.1
            if best is None or cost<best[0]:
                best=(cost,dict(corners=corners.copy(),surface=surface,goals=goals,normal=n,offset=d,
                    outward=measurement['outward'].copy(),tangents=measurement['tangents'].copy(),
                    bases=np.array(target_bases),q=np.array(target_q),yaw=np.array(yaws),
                    roll=np.array([-np.pi/4,np.pi/4]),toe_degrees=toe,time=self.t,
                    predicted_base_sep=float(separation)))
        return None if best is None else best[1]

    def acquire_plan(self):
        measurement=self.consensus()
        if measurement is None:
            self.plan_vote=None;return False
        corners=measurement['corners']
        if self.plan_vote is None or np.max(np.linalg.norm(corners-self.plan_vote[1],axis=1))>.015:
            self.plan_vote=(self.t,corners.copy());return False
        if self.t-self.plan_vote[0]<.4:return False
        candidate=self.candidate_plan(measurement)
        if candidate is None:
            self.reason='NO_FEASIBLE_STANDOFF_WITHIN_20DEG';return False
        self.plan=candidate;self.plan_vote=None;self.change_vote=None;self.lost_since=None
        self.events.append('PLAN_CREATED')
        return True

    def joint_servo(self,r,action,target,arm_limit=.05):
        q=self.geo[r][0];rates=self.rates
        action[r,3]=np.clip(2.5*(target[3]-q[3])/rates['LIFT_RATE'],-.12,.12)
        action[r,4]=np.clip(2.5*(target[4:8].sum()-q[4:8].sum())/rates['ARM_RATE'],-arm_limit,arm_limit)
        action[r,5:8]=np.clip(2.5*(target[8:11]-q[8:11])/rates['WRIST_RATE'],-.10,.10)

    def base_servo(self,r,action,position,angle,speed=.16):
        bp=np.asarray(self.obs['base_pose_world'])[r];R=rotation(bp[3:])
        v=cap(1.2*(position[:2]-bp[:2])-.12*self.base_velocity[r],speed)
        action[r,:2]=R[:2,:2].T@v/self.rates['BASE_LINEAR_RATE']
        action[r,2]=np.clip(1.8*wrap(angle-yaw(bp[3:]))/self.rates['BASE_ANGULAR_RATE'],-.15,.15)

    def hold(self,action,robots=(0,1)):
        if self.hold_base is None:self.capture_hold()
        if self.hold_base is None:return
        for r in robots:
            bp=np.asarray(self.obs['base_pose_world'])[r]
            if np.linalg.norm(bp[:2]-self.hold_base[r,:2])>.15:continue
            self.base_servo(r,action,self.hold_base[r],yaw(self.hold_base[r,3:]),.015)
            self.joint_servo(r,action,self.hold_q[r],arm_limit=.03)

    def task_jacobian(self,r):
        q,world,root,frames,grip,tips=self.geo[r]
        J=np.empty((3,4));J[:,0]=world[:3,0];J[:,1]=world[:3,1]
        for col,idx in ((2,slice(4,8)),(3,3)):
            trial=q.copy();trial[idx]+=.001/4 if col==2 else .001
            T=world@root@self.kin.frames(trial)['grasp_center_link']
            J[:,col]=(T[:3,3]-grip)/.001
        return J

    def approach_action(self,r,action,roll=False):
        q=self.geo[r][0];J=self.task_jacobian(r)
        error=self.plan['goals'][r]-self.geo[r][4]
        desired=cap(1.5*error,.035)
        extension=q[4:8].sum()
        # Redundant base-forward DOF takes over before the arm saturates.
        arm_weight=.025 if extension<.045 or extension>.46 else 1.
        W=np.diag([.12,.15,arm_weight,1.])
        inverse=W@J.T@np.linalg.inv(J@W@J.T+np.eye(3)*1e-6)
        bp=np.asarray(self.obs['base_pose_world'])[r]
        R=rotation(bp[3:])
        base_pull=R[:2,:2].T@(.5*(self.plan['bases'][r,:2]-bp[:2]))
        seed=np.r_[base_pull,.5*(.22-extension),0.]
        velocity=inverse@desired+cap((np.eye(4)-inverse@J)@seed,.01)
        velocity[:2]=cap(velocity[:2],.025)
        velocity[2:]=np.clip(velocity[2:],-.045,.045)
        for k,value,lo,hi in ((2,extension,.02,.50),(3,q[3],.03,1.16)):
            if (value<=lo and velocity[k]<0) or (value>=hi and velocity[k]>0):velocity[k]=0
        action[r,[0,1,4,3]]=velocity/np.array([.616,.616,1.21,1.54])
        action[r,2]=np.clip(1.8*wrap(self.plan['yaw'][r]-yaw(bp[3:]))/2.86,-.08,.08)
        target=self.plan['q'][r,8:11].copy()
        if roll:target[2]=self.plan['roll'][r]
        action[r,5:8]=np.clip(2.2*(target-q[8:11])/5.5,-.06,.06)
        return error

    def recover(self,reason,replan=False):
        if self.phase!='RECOVER':
            self.resume=self.phase if self.phase in ('NAV','PREP','APPROACH') else 'ROLL'
            self.capture_hold();self.recoveries+=1
            if self.recoveries>4:
                self.stop('RECOVERY_LIMIT');return
            self.transition('RECOVER',reason)
        if replan and self.plan is not None:
            if self.replans>=3:
                self.stop('REPLAN_LIMIT');return
            if reason=='PERSISTENT_BASE_GUARD':self.minimum_toe=self.plan['toe_degrees']+5
            self.replans+=1;self.plan=None;self.plan_vote=None;self.change_vote=None
            # Retract above the table to restore head visibility while keeping
            # base/lift references. No lowering, cloth grasp or scene writes.
            self.hold_q[:,4:8]=.005;self.hold_q[:,8:11]=0.

    def evidence(self):
        return all(x is not None and self.t-x['time']<=.15 for x in self.local)

    def supervise_targets(self):
        if self.plan is None:return
        measured=self.consensus()
        if measured is not None:
            delta=float(np.max(np.linalg.norm(measured['corners']-self.plan['corners'],axis=1)))
            if delta>.03:
                # A solitary head can see an occlusion boundary as a new
                # endpoint. Fresh local evidence from both wrists/heads takes
                # precedence over that unsupported global change.
                if len(measured['sources'])<2 and self.evidence():
                    self.change_vote=None
                    return
                if self.change_vote is None or np.max(np.linalg.norm(measured['corners']-self.change_vote[1],axis=1))>.015:
                    self.change_vote=(self.t,measured['corners'].copy())
                if self.t-self.change_vote[0]>.45:
                    self.recover('STABLE_TARGET_CHANGE_REPLAN',replan=True)
            else:self.change_vote=None
        if self.phase in ('APPROACH','ROLL','VERIFY','READY'):
            if not self.evidence():
                self.lost_since=self.t if self.lost_since is None else self.lost_since
                # Immediately invalidate READY; brief gaps hold at current pose.
                self.finished=False;self.stable=None
                if self.t-self.lost_since>.25:self.recover('LOCAL_EVIDENCE_LOST')
            else:self.lost_since=None

    def phase_action(self,action):
        # Deadlines precede every early return, including acquisition/recovery.
        if self.phase in self.DEADLINES and self.t-self.since>self.DEADLINES[self.phase]:
            self.stop('TIMEOUT_'+self.phase)
        if self.t-self.start>135 and self.phase!='READY':self.stop('TOTAL_STEP1_TIMEOUT')
        if self.phase=='STOP':
            severe=any(x in str(self.fault) for x in ('TILT','FK','DISTANCE','CLEARANCE','NONFINITE','INPUT','EXCEPTION'))
            if not severe:self.hold(action)
            return
        if self.phase=='SCAN':
            # No per-robot HEAD_T equilibrium. Search together at low speed.
            q=np.array(self.obs['joint_position'],float)
            for r in (0,1):
                target=q[r].copy();target[3]=.55;target[4:8]=.005;target[8:11]=0
                self.joint_servo(r,action,target)
            if self.consensus() is not None:
                self.capture_hold();self.transition('ACQUIRE','hem visible');return
            for r in (0,1):
                reference=self.scan_anchor[r].copy()
                side=np.sin((self.t-self.start)*.3)*.35
                reference[:2]+=rotation(reference[3:])[:2,1]*side
                self.base_servo(r,action,reference,yaw(reference[3:]),.06)
            return
        if self.phase=='ACQUIRE':
            self.hold(action)
            if self.acquire_plan():self.transition('NAV','pair standoff planned')
            return
        if self.phase=='RECOVER':
            self.hold(action)
            if self.plan is None:
                if self.acquire_plan():self.transition('NAV','new standoff planned')
            elif self.evidence():
                measured=self.consensus()
                changed=measured is not None and np.max(np.linalg.norm(measured['corners']-self.plan['corners'],axis=1))>.03
                if not changed:self.transition(self.resume,'fresh local evidence restored')
            return
        if self.phase=='NAV':
            for r in (0,1):
                self.base_servo(r,action,self.plan['bases'][r],self.plan['yaw'][r])
                target=self.plan['q'][r].copy();target[4:8]=.005
                self.joint_servo(r,action,target)
            errors=[np.linalg.norm(np.asarray(self.obs['base_pose_world'])[r,:2]-self.plan['bases'][r,:2]) for r in (0,1)]
            angles=[abs(wrap(yaw(np.asarray(self.obs['base_pose_world'])[r,3:])-self.plan['yaw'][r])) for r in (0,1)]
            if max(errors)<.035 and max(angles)<.06:
                self.transition('PREP','both bases positioned');self.capture_hold()
            return
        if self.phase=='PREP':
            for r in (0,1):
                self.base_servo(r,action,self.plan['bases'][r],self.plan['yaw'][r],.025)
                target=self.plan['q'][r].copy();target[4:8]=.005
                self.joint_servo(r,action,target)
            ok=all(abs(self.geo[r][0][3]-self.plan['q'][r,3])<.015 and
                   np.max(np.abs(self.geo[r][0][8:11]))<.04 for r in (0,1))
            if ok:
                self.stable=self.t if self.stable is None else self.stable
                if self.t-self.stable>.3:self.transition('APPROACH','both wrists/lifts prepared')
            else:self.stable=None
            return
        if not self.evidence():
            self.hold(action);return
        roll=self.phase in ('ROLL','VERIFY','READY')
        errors=[self.approach_action(r,action,roll) for r in (0,1)]
        norms=[float(np.linalg.norm(e)) for e in errors]
        self.diag['error_xyz']=[e.tolist() for e in errors]
        self.diag['error_norm']=norms
        position_ok=max(norms)<.018
        rolls=[float(self.geo[r][0][10]) for r in (0,1)]
        roll_ok=max(abs(np.array(rolls)-self.plan['roll']))<.04
        wrist_ok=all(x['wrist_visible'] for x in self.local)
        if self.phase=='APPROACH':
            if position_ok:
                self.stable=self.t if self.stable is None else self.stable
                if self.t-self.stable>.4:self.transition('ROLL','hover stable')
            else:self.stable=None
        elif self.phase=='ROLL':
            if not position_ok and max(norms)>.04:
                action[:,7]=0
            if position_ok and roll_ok:self.transition('VERIFY','opposite 45deg rolls reached')
        else:
            if position_ok and roll_ok and wrist_ok and self.t-self.plane_time<=self.READY_PLANE_MAX_AGE:
                self.stable=self.t if self.stable is None else self.stable
                if self.t-self.stable>.6:
                    self.transition('READY','position/roll/local wrist evidence verified')
                    self.finished=True
            else:
                if self.phase=='READY':self.transition('VERIFY','ready evidence changed')
                self.finished=False;self.stable=None

    def constrain(self,action):
        self.requested=action.copy()
        bp=np.asarray(self.obs['base_pose_world'],float)
        rotations=[rotation(bp[r,3:]) for r in (0,1)]
        velocities=np.array([.616*rotations[r][:2,:2]@action[r,:2] for r in (0,1)])
        adjusted,blocked=pair_guard(bp[:,:2],velocities,self.base_velocity)
        for r in (0,1):action[r,:2]=rotations[r][:2,:2].T@adjusted[r]/.616
        distance=float(np.linalg.norm(bp[1,:2]-bp[0,:2]))
        self.diag['base_distance']=distance;self.diag['base_guard_active']=blocked
        if blocked:
            self.block_since=self.t if self.block_since is None else self.block_since
            self.stable=None;self.finished=False
            # Preserve lift/wrist preparation during NAV. During close approach
            # pause only Cartesian advance/roll until base motion is admitted.
            if self.phase in ('APPROACH','ROLL','VERIFY','READY','ALIGN_SCAN','ALIGN_MOVE','ALIGN_VERIFY','ALIGNED'):
                action[:,3:8]=0
                if self.phase=='READY':self.transition('VERIFY','base guard changed ready')
            if self.t-self.block_since>6 and self.phase not in ('STOP','RECOVER'):
                self.recover('PERSISTENT_BASE_GUARD',replan=True)
        else:self.block_since=None
        if distance<.365:
            self.stop('BASE_DISTANCE_BRAKE');action[:]=0
        tips=np.linalg.norm(self.geo[0][-1][:,None]-self.geo[1][-1][None,:],axis=2)
        if np.min(tips)<.075:
            self.stop('TIP_DISTANCE_BRAKE');action[:]=0
        if self.plan is not None and self.phase in ('APPROACH','ROLL','VERIFY','READY','ALIGN_SCAN','ALIGN_MOVE','ALIGN_VERIFY','ALIGNED'):
            n,d=self.plan['normal'],self.plan['offset']
            if min(float(np.min(g[-1]@n+d)) for g in self.geo)<.035:
                self.stop('TIP_TABLE_CLEARANCE');action[:]=0
        if self.phase=='STOP' and any(x in str(self.fault) for x in ('TILT','FK','DISTANCE','CLEARANCE','NONFINITE','INPUT','EXCEPTION')):
            action[:]=0
        # Static asset limits, with a small policy margin, are checked on every
        # phase including hold/NAV. Never modify the competition limits.
        for r in (0,1):
            q=self.geo[r][0]
            for j,col in ((3,3),(8,5),(9,6),(10,7)):
                lo,hi=self.kin.data['joint_limits'][self.kin.names[j]]
                if (q[j]<=lo+.005 and action[r,col]<0) or (q[j]>=hi-.005 and action[r,col]>0):action[r,col]=0
            extension=q[4:8].sum()
            if (extension<=.015 and action[r,4]<0) or (extension>=.505 and action[r,4]>0):action[r,4]=0
        action[:,8]=-1 # Step 1 never closes either gripper.
        action[:]=np.clip(np.nan_to_num(action),-1,1)
        self.applied=action.copy()

    def step(self,obs):
        self.events=[];self.diag={}
        self.update(obs)
        self.supervise_targets()
        action=np.zeros((2,9))
        self.phase_action(action)
        if self.phase=='RECOVER':
            self.finished=False
            action[:]=0
            self.hold(action)
        phase_before_guard=self.phase
        self.constrain(action)
        # Progress is evaluated only with fresh evidence and admitted commands.
        if self.phase in ('APPROACH','ROLL','VERIFY') and self.evidence() and not self.diag.get('base_guard_active'):
            error=max(np.linalg.norm(self.plan['goals'][r]-self.geo[r][4]) for r in (0,1))
            if error>.018:
                if self.no_progress is None:self.no_progress=(self.t,error)
                elif self.t-self.no_progress[0]>7:
                    if error>self.no_progress[1]-.003:self.recover('UNBLOCKED_NO_PROGRESS_REPLAN',replan=True)
                    self.no_progress=(self.t,error)
            else:self.no_progress=None
        else:self.no_progress=None
        if self.phase in ('RECOVER','STOP') and self.phase!=phase_before_guard:
            # Recovery was entered after the command was proposed. Stop both
            # robots this frame; check the hold command on the next frame.
            self.finished=False
            action[:,:8]=0
            action[:,8]=-1
            self.applied=action.copy()
        return action.astype(np.float32)

    def annotate(self,images,mark):
        for r in (0,1):
            if self.geo[r] is None:continue
            pts=[];colors=[]
            for measurement in self.global_measurements:
                if measurement is not None:
                    for p in measurement['corners']:
                        if p is not None:pts.append(p);colors.append((255,128,0))
            if self.plan is not None:
                pts.extend([self.plan['surface'][r],self.plan['goals'][r]])
                colors.extend([(255,0,255),(255,255,0)])
            if self.local[r] is not None:pts.append(self.local[r]['point']);colors.append((0,255,0))
            pts.extend([self.geo[r][4],*self.geo[r][-1]])
            colors.extend([(255,255,255),(0,255,255),(0,255,255)])
            for index in (1+r,3+r):
                T,spec=self.camera(index,self.geo[r]);xyz=points(np.linalg.inv(T),np.array(pts))
                f=256/(2*np.tan(np.radians(spec['horizontal_fov_deg'])/2))
                for p,color in zip(xyz,colors):
                    if p[2]>.01:
                        u,v=128+f*p[:2]/p[2]
                        if 0<=u<256 and 0<=v<256:mark(images[index],u,v,color,4)

    def folder(self,owner):
        if owner.dir is None:
            owner.dir=str(Path(self.snap_dir)/time.strftime('lan_%m%d_%H%M%S'))
            print('[ANH] luu anh vao '+owner.dir,flush=True)
        folder=Path(owner.dir);folder.mkdir(parents=True,exist_ok=True);return folder

    def finalize(self,owner,*,capture_depth=False):
        if self.t-self.last_log>=1 or self.events:
            record=dict(version=self.VERSION,time=self.t,phase=self.phase,phase_age=self.t-self.since,
                        reason=self.reason,fault=self.fault,ready=self.finished,replans=self.replans,
                        recoveries=self.recoveries,diagnostic=self.diag,
                        base=np.asarray(self.obs['base_pose_world']).tolist(),
                        local=[None if x is None else {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in x.items()} for x in self.local],
                        grip=[None if g is None else g[4].tolist() for g in self.geo],
                        plan=None if self.plan is None else {k:self.plan[k].tolist() if isinstance(self.plan[k],np.ndarray) else self.plan[k]
                              for k in ('corners','surface','goals','bases','yaw','roll','toe_degrees','predicted_base_sep')},
                        requested=self.requested.tolist(),applied=self.applied.tolist(),
                        alignment=getattr(self,'alignment_report',None),
                        alignment_sample_time=getattr(self,'alignment_report_time',None),
                        pre_roll=getattr(self,'pre_roll',None),
                        step1_done_time=getattr(self,'step1_done_time',None),
                        alignment_fresh=bool(self.t-getattr(self,'alignment_report_time',-np.inf)<=.15),
                        align_patch_quality=getattr(self,'patch_debug',None),
                        reobserve_attempts=getattr(self,'reobserve_attempts',0),
                        align_posture_search=getattr(self,'align_search_debug',None),
                        step1_complete=getattr(self,'step1_complete',False),
                        step2_complete=bool(self.phase=='ALIGNED' and self.finished),
                        step2_contract='high_air_staging_only',descent_authorized=False,
                        plane_age=None if self.plane is None else self.t-self.plane_time)
            line=json.dumps(record,allow_nan=False);print('[RGBD] '+line,flush=True)
            if self.finished:print('[RGBD] '+('BOTH_ALIGNED_STEP2' if self.phase=='ALIGNED' else 'BOTH_READY_STEP1_V3_REVERSE_ROLL'),flush=True)
            if self.snap_enabled:
                try:
                    with (self.folder(owner)/'servo_trace.jsonl').open('a',encoding='utf-8') as stream:stream.write(line+'\n')
                except OSError as exc:print('[RGBD LOG]',exc,flush=True)
            self.last_log=self.t
        if self.snap_enabled and (capture_depth or self.t-self.last_png>=self.snap_every):
            try:
                dep=np.asarray(self.obs['depth'])[...,0]
                valid=np.asarray(self.obs['depth_valid'])[...,0].astype(bool)&np.isfinite(dep)&(dep>0)
                gray=np.where(valid,255*(1-np.clip(np.nan_to_num(dep)/np.array([10,3,3,5,5])[:,None,None],0,1)),0).astype(np.uint8)
                for name,data in [('depth',gray),('depth_valid',valid.astype(np.uint8)*255)]:
                    row=np.concatenate([np.repeat(x[...,None],3,axis=2) for x in data],axis=1)
                    self.save_png(str(self.folder(owner)/f't{self.t:06.1f}_{name}.png'),row)
            except OSError as exc:print('[RGBD PNG]',exc,flush=True)
            self.last_png=self.t
        if (self.snap_enabled and hasattr(self,'ALIGN_PHASES') and
                self.phase in (*self.ALIGN_PHASES,'RECOVER') and
                (self.events or self.t-getattr(self,'last_raw',-np.inf)>=2.)):
            try:
                # Preserve optical-Z and calibration inputs, not normalized PNG
                # values, so a later audit can reproduce a failed perception frame.
                keys=('rgb','depth','depth_valid','joint_position','base_pose_world',
                      'grasp_pose_base','fingertip_position_base','simulation_time_s')
                data={k:np.asarray(self.obs[k]) for k in keys}
                data.update(phase=np.array(self.phase),version=np.array(self.VERSION))
                if hasattr(self,'branch_reference'):
                    data['branch_reference']=np.array([
                        ref if ref is not None else np.full(3,np.nan)
                        for ref in self.branch_reference
                    ])
                if self.plane is not None:data['table_plane']=np.r_[self.plane[0],self.plane[1]]
                if self.plan is not None:
                    for k in ('surface','tangents','outward'):data['plan_'+k]=self.plan[k]
                np.savez_compressed(self.folder(owner)/f't{self.t:07.2f}_observation.npz',**data)
                self.last_raw=self.t
            except OSError as exc:print('[RGBD RAW]',exc,flush=True)


def envelope_vertices(spec):
    """Corners of a padded authored VISUAL envelope, not collision geometry."""
    pad=spec.get('padding_m',.003)
    return np.array(list(product(*zip(np.array(spec['min_m'])-pad,
                                      np.array(spec['max_m'])+pad))))




def visible_yellow_boundary(blue, yellow, depth, valid, focal, camera_boxes, table_plane=None):
    """Remove contacts with nearer measured surfaces inside a gripper box."""
    exterior = ~(blue | yellow)
    adjacent = exterior & dilate(yellow, 1) & valid
    rows, cols = np.nonzero(adjacent)
    z = depth[rows, cols]
    h, w = depth.shape
    xyz = np.column_stack(((cols-w/2)*z/focal, (rows-h/2)*z/focal, z))
    gripper = np.zeros(depth.shape, dtype=bool)
    inside = np.zeros(len(xyz), dtype=bool)
    for box in camera_boxes:
        axes = box[[4, 2, 1]]-box[0]
        lengths = np.linalg.norm(axes, axis=1)
        local = (xyz-box[0]) @ (axes/lengths[:, None]).T
        inside |= np.all((local >= -.002) & (local <= lengths+.002), axis=1)
    if table_plane is not None:
        normal,offset=table_plane
        inside &= np.abs(xyz@normal+offset)>.004
    gripper[rows[inside], cols[inside]] = True

    # A yellow pixel can still have a genuine exterior neighbor on another side.
    boundary = np.zeros_like(yellow)
    for dy, dx in product((-1, 0, 1), repeat=2):
        if dy == dx == 0:
            continue
        shift = (dy, dx)
        neighbor_is_gripper = np.roll(gripper, shift, axis=(0, 1))
        nearer = np.roll(depth, shift, axis=(0, 1)) < depth-.004
        boundary |= np.roll(exterior & valid, shift, axis=(0, 1)) & ~(neighbor_is_gripper & nearer)
    boundary &= yellow
    boundary[[0,-1],:] = False
    boundary[:,[0,-1]] = False
    return boundary











def staging_region(hem, cloth, expected, tangent, outward, normal, offset):
    """Fresh coarse footprint, NOT a fitted lip or an insertion target.

    Height layers are deliberately merged only for high-air staging. Use actual
    visible samples for span; a bin's nominal width is never evidence of coverage.
    """
    hem=np.asarray(hem,float).reshape(-1,3)
    cloth=np.asarray(cloth,float).reshape(-1,3)
    hem=hem[np.isfinite(hem).all(axis=1)]
    cloth=cloth[np.isfinite(cloth).all(axis=1)]
    rel=hem-expected
    a=rel@tangent;b=rel@outward;h=hem@normal+offset
    keep=(a>-.012)&(a<.045)&(np.abs(b)<.055)&(h>-.008)&(h<.070)
    hem=hem[keep];a=a[keep];b=b[keep];h=h[keep]
    if len(hem)<8:return None,'TOO_FEW_REGION_POINTS'
    # Keep the outer visible strip; reject the side edge running into the shirt.
    outer=float(np.quantile(b,.90))
    keep=b>=outer-.010
    strip=hem[keep];aa=a[keep];bb=b[keep]
    if len(strip)<6:return None,'TOO_FEW_OUTER_POINTS'
    span=float(np.quantile(aa,.95)-np.quantile(aa,.05))
    if span<.015:return None,'REGION_SPAN'
    spread=float(np.quantile(bb,.95)-np.quantile(bb,.05))
    # Up to 20mm uncertainty is allowed ONLY at high-air staging; the planner
    # adds the measured uncertainty to its 25mm outward retreat. Wider/ambiguous
    # regions require a new view rather than silently enlarging the tolerance.
    if spread>.020:return None,'REGION_SPREAD'
    edge=expected+outward*float(np.quantile(bb,.95))
    edge=edge-normal*(edge@normal+offset)
    body=cloth[(np.linalg.norm(cloth-edge,axis=1)<.10)&((cloth-edge)@outward<-.008)]
    if len(body)<8:return None,'CLOTH_BODY_MISSING'
    return dict(edge=edge,tangent=tangent.copy(),outward=outward.copy(),
                height=0.,height_range_m=[float(h.min()),float(h.max())],
                uncertainty_m=max(.004,spread),span_m=span,
                samples=strip,hem_points=len(hem),edge_residual_m=None),None


def staging_blockers(position, heading, wrist, gap, clearance, outside):
    """The Step-2 contract: measured high-air pose, no descent authorization."""
    values=np.array([position,heading,wrist,gap,clearance,outside],float)
    if not np.isfinite(values).all():return ['NONFINITE_STAGING_GEOMETRY']
    return [name for name,ok in (
        ('POSITION',position<.010),('HEADING',heading<np.radians(15)),
        ('WRIST_ANGLE',wrist<.040),('JAW_SEPARATION',gap>=.035),
        ('HOVER_CLEARANCE',clearance>=.080),('OUTSIDE_REGION',outside>=.010)) if not ok]


class RGBDServo(Step1Servo):
    """Step 1 plus coarse high-air staging; never authorizes descent."""
    READY_PLANE_MAX_AGE = 1.0
    VERSION='corner-local-step2-staging-v1'
    ALIGN_PHASES=('ALIGN_SCAN','ALIGN_MOVE','ALIGN_VERIFY','ALIGN_REOBSERVE','ALIGNED')
    DEADLINES={**Step1Servo.DEADLINES,'ALIGN_SCAN':12.,'ALIGN_MOVE':25.,'ALIGN_VERIFY':10.}

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        spec=self.kin.data.get('gripper_visual_envelopes')
        if not spec or spec.get('schema')!=1:
            raise ValueError('STEP2_REQUIRES_NEW_STATIC_GRIPPER_ENVELOPES')
        padding=float(spec.get('padding_m',np.nan))
        required={'gripper_finger_left_link','gripper_fingertip_left_link',
                  'gripper_finger_right_link','gripper_fingertip_right_link',
                  'quick_connect_interface_link'}
        if not np.isfinite(padding) or not 0<=padding<=.02 or {b['link'] for b in spec['bodies']}!=required:
            raise ValueError('INVALID_GRIPPER_ENVELOPE_SCHEMA')
        for b in spec['bodies']:
            lo,hi=np.asarray(b['min_m']),np.asarray(b['max_m'])
            if lo.shape!=(3,) or hi.shape!=(3,) or not np.isfinite([lo,hi]).all() or np.any(hi<=lo):
                raise ValueError('INVALID_GRIPPER_ENVELOPE_BOUNDS')
        self.envelopes={b['link']:envelope_vertices(dict(b,padding_m=spec['padding_m']))
                        for b in spec['bodies']}
        self.step1_complete=False;self.align_start=None;self.align_plan=None
        self.align_vote=None;self.alignment_report=None;self.align_lost=None
        self.align_goal_baseline=None;self.align_progress=None
        self.patch_debug=[{},{}]
        self.pre_roll=[None,None]
        self.step1_done_time=None
        self.edge_reference=None
        self.edge_change_vote=None
        self.edge_change_since=None
        self.reobserve_attempts=0
        self.reobserve_robots=[]
        self.reobserve_targets=None
        self.branch_reference=[None,None]
        self.reacquiring = False
        self.reobserve_stage = None
        self.reobserve_used = np.zeros(2, dtype=int)
        self.reobserve_start_lift = None
        self.reobserve_patch_vote = None
        self.reobserve_causes = {}
        self.align_hover_floor = np.full(2, .105)

    def geometry(self,r):
        result=super().geometry(r)
        # Four millimetres is an acceptance gate, NOT a measured guarantee of
        # visual/collider/contact accuracy. Public demo residuals are reported.
        for k,name in enumerate(('gripper_fingertip_left_link','gripper_fingertip_right_link')):
            predicted=(result[2]@result[3][name])[:3,3]
            if np.linalg.norm(predicted-np.asarray(self.obs['fingertip_position_base'])[r,k])>.004:
                raise ValueError('STEP2_FK_MISMATCH_4MM')
        return result

    def bodies(self,r,q=None,base_angle=None):
        geo=self.geo[r];q=geo[0] if q is None else q
        world=geo[1].copy() if base_angle is None else self.rotated_world(r,base_angle)
        f=self.kin.frames(q)
        return {name:points(world@geo[2]@f[name],v) for name,v in self.envelopes.items()}

    def rotated_world(self,r,angle):
        world=self.geo[r][1].copy()
        current=float(np.arctan2(world[1,0],world[0,0]))
        # Change heading without discarding measured base roll/pitch.
        world[:3,:3]=rotation(yaw_quat(angle-current))@world[:3,:3]
        return world




    def patch(self,r,require_table=True):
        """Observe the assigned corner footprint with fresh wrist RGB-D.

        require_table is retained for the Step-1 pre-roll snapshot interface.
        No future landing-point coverage is asserted by this method.
        """
        if self.plane is None or self.plan is None:return None
        n,d,_=self.plane;index=1+r
        debug={index:{'stage':'region'},'outcome':'WRIST_EDGE_MISSING'}
        self.patch_debug[r]=debug
        im=np.asarray(self.obs['rgb'])[index]
        blue,yellow=geometry_masks(im,self.masks)
        camera,spec=self.camera(index,self.geo[r])
        depth=np.asarray(self.obs['depth'])[index,...,0]
        valid=np.asarray(self.obs['depth_valid'])[index,...,0].astype(bool)
        valid &= np.isfinite(depth)&(depth>=spec['clip_m'][0])&(depth<=spec['clip_m'][1])
        focal=depth.shape[1]/(2*np.tan(np.deg2rad(spec['horizontal_fov_deg'])/2))
        boundary=visible_yellow_boundary(blue,yellow,depth,valid,focal,
            [points(np.linalg.inv(camera),box) for box in self.bodies(r).values()],
            (camera[:3,:3].T@n,float(camera[:3,3]@n+d)))
        debug[index]['occluded_boundary_pixels']=int(np.count_nonzero(
            (yellow&dilate(~(blue|yellow),1)&~boundary)[1:-1,1:-1]))
        hem=self.cloud(index,self.geo[r],boundary,limit=16000)
        cloth=self.cloud(index,self.geo[r],blue|yellow,limit=16000)
        result,reason=staging_region(hem,cloth,self.plan['surface'][r],
            self.plan['tangents'][r],self.plan['outward'][r],n,d)
        debug[index].update(raw_boundary=len(hem),reject_reason=reason)
        if result is None:return None
        debug.update(outcome='STAGING_REGION_VISIBLE',span_m=result['span_m'],
            uncertainty_m=result['uncertainty_m'],height_range_m=result['height_range_m'])
        debug[index]['stage']='accepted_region'
        result.update(time=self.t,camera=index,wrist=True,support_points=0,
                      support=np.empty((0,3)),support_ages=np.empty(0))
        return result

    def orientation(self,r,q,angle):
        bodies=self.bodies(r,q,angle);n=self.plane[0]
        tips=[bodies['gripper_fingertip_'+side+'_link'].mean(0) for side in ('left','right')]
        lower=int(np.argmin(np.array(tips)@n));side=('left','right')[lower]
        gap=float(abs((tips[1]-tips[0])@n))
        world=self.rotated_world(r,angle)
        f=self.kin.frames(q);T=world@self.geo[r][2]
        rear=(T@f['quick_connect_interface_link'])[:3,3]
        axis=(tips[0]+tips[1])/2-rear;axis-=n*(axis@n)
        if np.linalg.norm(axis)<.05:return None
        axis/=np.linalg.norm(axis)
        return dict(bodies=bodies,axis=axis,lower=side,gap=gap)

    def make_alignment(self,patches):
        targets=[]
        self.align_search_debug=[]
        for r,p in enumerate(patches):
            counts=dict(postures=0,heading=0,translation=0,certificate=0,sweep=0)
            self.align_search_debug.append(counts)
            n,d,_=self.plane;q=self.geo[r][0];angle=yaw(np.asarray(self.obs['base_pose_world'])[r,3:])
            best=None
            # Small finite posture search. Base rotation is a last resort due to
            # its larger swept volume. No arbitrary fixed-world pitch axis.
            for base_deg in (0.,-6.,6.):
                for yaw_deg,pitch_deg in product((0.,-8.,8.),(0.,-6.,6.)):
                    trial=q.copy();trial[8]+=np.radians(yaw_deg);trial[9]+=np.radians(pitch_deg)
                    if any(not self.kin.data['joint_limits'][self.kin.names[j]][0]+.03<trial[j]<
                               self.kin.data['joint_limits'][self.kin.names[j]][1]-.03 for j in (8,9,10)):continue
                    info=self.orientation(r,trial,angle+np.radians(base_deg))
                    counts['postures']+=1
                    if info is None or info['gap']<.035:continue
                    error=float(np.arccos(np.clip(info['axis']@(-p['outward']),-1,1)))
                    if error>np.radians(15):continue
                    counts['heading']+=1
                    allp=np.vstack(list(info['bodies'].values()))
                    lip=info['bodies']['gripper_fingertip_'+info['lower']+'_link'].mean(0)
                    # Align lower pad along the assigned edge; retract until
                    # ALL gripper boxes have >=25mm plus region uncertainty outside clearance. Never
                    # treat the grasp center as the lower pad contact point.
                    delta=-p['tangent']*float((lip-p['edge'])@p['tangent'])
                    delta+=p['outward']*(.025+p['uncertainty_m']-float(np.min((allp+delta-p['edge'])@p['outward'])))
                    delta+=n*(self.align_hover_floor[r]-float(np.min((allp+delta)@n+d)))
                    if np.linalg.norm(delta)>.13:continue
                    counts['translation']+=1
                    # Predict moved grasp center using the same FK posture.
                    world=self.rotated_world(r,angle+np.radians(base_deg))
                    grip=(world@self.geo[r][2]@self.kin.frames(trial)['grasp_center_link'])[:3,3]
                    moved={k:v+delta for k,v in info['bodies'].items()}
                    # Certify ONLY the high staging endpoint.
                    clearance=float(np.min(np.vstack(list(moved.values()))@n+d))
                    if clearance<.080:continue
                    cert=dict(scope='staging_only',valid=True,descent_m=0.,
                              descent_authorized=False)
                    counts['certificate']+=1
                    # Check intermediate posture envelopes along an idealized
                    # compensated path. Online measured/predicted guards below
                    # additionally cover controller lag; no descent is executed.
                    current=self.geo[r][4];sweep_ok=True
                    for fraction in (.25,.5,.75):
                        sq=q+fraction*(trial-q);sa=angle+fraction*np.radians(base_deg)
                        sb=self.bodies(r,sq,sa)
                        sw=self.rotated_world(r,sa)
                        sg=(sw@self.geo[r][2]@self.kin.frames(sq)['grasp_center_link'])[:3,3]
                        translation=current+fraction*(grip+delta-current)-sg
                        if np.min((np.vstack(list(sb.values()))+translation)@n+d)<.065:
                            sweep_ok=False;break
                    if not sweep_ok:continue
                    counts['sweep']+=1
                    cost=np.linalg.norm(delta)+.06*error+.002*(abs(yaw_deg)+abs(pitch_deg))+.008*abs(base_deg)
                    if best is None or cost<best[0]:
                        best=(cost,dict(goal=grip+delta,q=trial,base_yaw=angle+np.radians(base_deg),
                            patch=p,lower_jaw=info['lower'],heading_error_rad=error,certificate=cert,
                            hover_envelopes=moved))
            if best is None:return None
            targets.append(best[1])
        # Pair clearance for conservative envelope boxes at both proposed poses.
        a=np.vstack(list(targets[0]['hover_envelopes'].values()))
        b=np.vstack(list(targets[1]['hover_envelopes'].values()))
        # Orientation-independent separating axis: center-to-center.
        axis=targets[1]['goal']-targets[0]['goal'];axis/=np.linalg.norm(axis)
        if float(np.min(b@axis)-np.max(a@axis))<.075:return None
        return targets

    def alignment_action(self,r,action):
        target=self.align_plan[r];q=self.geo[r][0];bp=np.asarray(self.obs['base_pose_world'])[r]
        error=target['goal']-self.geo[r][4]
        # Include induced Cartesian motion from wrist AND base angular rates.
        angular=np.r_[np.clip(1.5*wrap(target['base_yaw']-yaw(bp[3:])), -.04,.04),
                       np.clip(1.7*(target['q'][8:11]-q[8:11]),-.08,.08)]
        Jang=np.empty((3,4));eps=1e-4
        Jang[:,0]=np.cross(np.array([0.,0.,1.]),self.geo[r][4]-bp[:3])
        for col,j in enumerate((8,9,10),1):
            trial=q.copy();trial[j]+=eps
            T=self.geo[r][1]@self.geo[r][2]@self.kin.frames(trial)['grasp_center_link']
            Jang[:,col]=(T[:3,3]-self.geo[r][4])/eps
        J=self.task_jacobian(r);extension=q[4:8].sum()
        W=np.diag([.06,.12,.03 if extension<.05 or extension>.46 else 1.,1.])
        inv=W@J.T@np.linalg.inv(J@W@J.T+np.eye(3)*1e-6)
        velocity=inv@(cap(1.4*error,.018)-Jang@angular)
        scale=min(1., .012/max(np.linalg.norm(velocity[:2]),1e-9),
                  .025/max(np.max(np.abs(velocity[2:])),1e-9))
        velocity*=scale
        angular*=scale
        action[r,[0,1,4,3]]=velocity/np.array([.616,.616,1.21,1.54])
        action[r,2]=angular[0]/2.86;action[r,5:8]=angular[1:]/5.5
        return float(np.linalg.norm(error))

    def recover(self,reason,replan=False):
        in_align=self.phase in self.ALIGN_PHASES
        if self.phase=='ALIGNED':
            self.align_start=self.t
        super().recover(reason,replan)
        if in_align:
            self.align_plan = None
            self.align_vote = None
            self.align_lost = None
            self.edge_change_vote = None
            self.edge_change_since = None
            self.reobserve_patch_vote = None
            self.reacquiring = True
        if in_align and self.phase=='RECOVER' and not replan:self.resume='ALIGN_SCAN'
        if replan:
            self.align_plan=None;self.align_vote=None;self.pre_roll=[None,None]
            self.branch_reference=[None,None]
            self.reacquiring = False
            self.align_hover_floor[:] = .105

    def check_edge_change(self,patches):
        # The plan remains fixed during convergence. Only meaningful planar
        # movement or loss of the planned outside margin triggers a new plan.
        n,d,_=self.plane
        drift=[];unsafe=[]
        for target,patch in zip(self.align_plan,patches):
            delta=patch['edge']-target['patch']['edge']
            drift.append(float(np.linalg.norm(delta-n*(delta@n))))
            verts=np.vstack(list(target['hover_envelopes'].values()))
            margin=float(np.min((verts-patch['edge'])@patch['outward']))-patch['uncertainty_m']
            unsafe.append(margin<.010)
        self.diag['region_change']=dict(planar_drift_m=drift,unsafe_target=unsafe)
        changed=max(drift)>.015 or any(unsafe)
        if not changed:
            self.edge_change_vote=None;self.edge_change_since=None
            return 'STABLE'
        edges=np.array([p['edge'] for p in patches])
        if self.edge_change_since is None:
            self.edge_change_since=self.t;self.capture_hold()
        if self.t-self.edge_change_since>1.2:return 'CONFIRMATION_TIMEOUT'
        if self.edge_change_vote is None or np.max(np.linalg.norm(edges-self.edge_change_vote[1],axis=1))>.006:
            self.edge_change_vote=(self.t,edges)
            return 'WAIT_CONFIRMATION'
        return 'REPLAN' if self.t-self.edge_change_vote[0]>=.25 else 'WAIT_CONFIRMATION'

    def record_align_evidence(self,patches):
        self.diag['align_evidence']={
            'missing_patch_robots':[r for r,p in enumerate(patches) if p is None],
            'local_evidence_ok':self.evidence(),'plane_fresh':bool(self.t-self.plane_time<=self.READY_PLANE_MAX_AGE),
        }

    def begin_reobserve(self, robots):
        """Assess new evidence before selecting any bounded viewpoint motion."""
        if not robots:
            return False
        self.capture_hold()
        self.reobserve_robots = sorted(set(robots))
        self.reobserve_causes = {
            r: self.patch_debug[r].get('outcome', 'UNKNOWN') for r in robots
        }
        self.reobserve_stage = 'ASSESS'
        self.reobserve_episode_moved = set()
        self.reobserve_stage_time = self.t
        self.reobserve_patch_vote = None
        self.reobserve_start_lift = self.hold_q[:, 3].copy()
        self.reacquiring = True
        self.align_plan = None
        self.align_vote = None
        self.align_lost = None
        self.edge_change_vote = None
        self.edge_change_since = None
        self.finished = False
        self.transition('ALIGN_REOBSERVE', 'reobserve assigned staging regions')
        return True

    def confirm_reacquisition(self, patches):
        """Require a continuous, stable pair of fresh patches before replanning."""
        valid = (all(p is not None for p in patches) and self.evidence()
                 and self.t-self.plane_time <= self.READY_PLANE_MAX_AGE)
        if not valid:
            self.reobserve_patch_vote = None
            return False
        edges = np.array([p['edge'] for p in patches])
        vote = self.reobserve_patch_vote
        if vote is None or np.max(np.linalg.norm(edges-vote[1], axis=1)) > .006:
            self.reobserve_patch_vote = (self.t, edges)
            return False
        if self.t-vote[0] < .35:
            return False
        self.branch_reference = [p['edge'].copy() for p in patches]
        self.reacquiring = False
        self.align_plan = None
        self.align_vote = None
        self.align_lost = None
        self.align_progress = None
        self.edge_reference = None
        self.edge_change_vote = None
        self.edge_change_since = None
        self.capture_hold()
        self.transition('ALIGN_SCAN', 'both fresh staging regions reacquired')
        return True

    def start_viewpoint_segment(self, patches):
        """Choose affected robots only; one 5mm segment per robot per recovery episode."""
        missing = [r for r, p in enumerate(patches) if p is None]
        movable = []
        for r in missing:
            debug = self.patch_debug[r]
            wrist = debug.get(1+r, {})
            occlusion_hint = wrist.get('occluded_boundary_pixels', 0) >= 3
            reason = debug.get('outcome')
            if reason not in ('WRIST_EDGE_MISSING', 'TABLE_PATCH_INSUFFICIENT'):
                continue
            if not occlusion_hint or self.reobserve_used[r] >= 4 or r in self.reobserve_episode_moved:
                continue
            upper = self.kin.data['joint_limits']['lift_joint'][1]
            if self.geo[r][0][3]+.005 > upper-.01:
                continue
            n, d, _ = self.plane
            if np.min(np.vstack(list(self.bodies(r).values()))@n+d) < .065:
                continue
            movable.append(r)
        if not movable or not self.evidence():
            return False
        if self.t-self.plane_time > self.READY_PLANE_MAX_AGE:
            return False
        self.capture_hold()
        self.reobserve_robots = movable
        self.reobserve_episode_moved.update(movable)
        for r in movable:
            self.reobserve_causes[r] = self.patch_debug[r].get("outcome", "UNKNOWN")
        self.reobserve_start_lift = self.hold_q[:, 3].copy()
        self.reobserve_targets = self.reobserve_start_lift.copy()
        self.reobserve_targets[movable] += .005
        self.reobserve_plane = (self.plane[0].copy(), self.plane[1])
        self.reobserve_stage = 'MOVE'
        self.reobserve_stage_time = self.t
        self.reobserve_counted = np.zeros(2, dtype=bool)
        self.reobserve_patch_vote = None
        self.events.append('REOBSERVE_MOVE_START')
        return True

    def reobserve_action(self, action):
        self.hold(action)
        patches = [self.patch(r) for r in (0, 1)]
        self.record_align_evidence(patches)
        self.diag['reobserve'] = {
            'stage': self.reobserve_stage, 'robots': self.reobserve_robots,
            'causes': self.reobserve_causes,
            'segments_used': self.reobserve_used.tolist(),
            'patch_outcomes': [d.get('outcome') for d in self.patch_debug],
        }
        if self.t-self.since > 8.:
            self.recover('REOBSERVE_TIMEOUT')
            return
        if self.reobserve_stage != 'MOVE':
            if self.confirm_reacquisition(patches):
                action[:] = 0
                self.hold(action)
                return
            if self.t-self.reobserve_stage_time < .6:
                return
            # A feasible patch may need its complete fresh-evidence vote.
            if self.reobserve_patch_vote is not None:
                return
            if self.start_viewpoint_segment(patches):
                return
            reason = ('REOBSERVE_NO_GAIN' if self.reobserve_stage == 'SETTLE'
                      else 'REOBSERVE_NO_SUPPORTED_MOTION')
            self.recover(reason)
            return

        normal, offset = self.reobserve_plane
        changed = (normal@self.plane[0] < np.cos(np.radians(2)) or
                   np.max(np.abs(self.plan['surface']@(normal-self.plane[0])
                                 + offset-self.plane[1])) > .004)
        if changed or self.t-self.plane_time > 3.:
            self.recover('REOBSERVE_PLANE_UNCERTAIN')
            return
        if not self.evidence():
            self.recover('REOBSERVE_LOCAL_LOST')
            return
        moved = np.array([self.geo[r][0][3] for r in (0,1)])-self.reobserve_start_lift
        newly_moved = [r for r in self.reobserve_robots
                       if moved[r] > .0005 and not self.reobserve_counted[r]]
        self.reobserve_used[newly_moved] += 1
        self.reobserve_attempts += len(newly_moved)
        self.reobserve_counted[newly_moved] = True
        errors = [self.reobserve_targets[r]-self.geo[r][0][3] for r in self.reobserve_robots]
        if any(error < -.002 for error in errors):
            self.recover('REOBSERVE_LIFT_OVERSHOOT')
            return
        # Evidence has returned: stop at the current height and verify while held.
        visible = all(p is not None for p in patches)
        reached = all(abs(error) < .001 for error in errors)
        if visible or reached:
            for r in self.reobserve_robots:
                self.align_hover_floor[r] += max(0., float(moved[r]))
            self.capture_hold()
            self.reobserve_stage = 'SETTLE'
            self.reobserve_stage_time = self.t
            self.reobserve_patch_vote = None
            self.events.append('REOBSERVE_MOVE_END')
            action[:] = 0
            self.hold(action)
            return
        if self.t-self.reobserve_stage_time > 2.5:
            self.recover('REOBSERVE_LIFT_NO_PROGRESS')
            return
        for r, error in zip(self.reobserve_robots, errors):
            action[r, 3] = np.clip(3.*error, 0., .010)/self.rates['LIFT_RATE']

    def phase_action(self,action):
        if self.phase=='ALIGN_REOBSERVE':
            if self.t-self.align_start>50.:
                self.stop('TOTAL_ALIGN_TIMEOUT')
                return
            self.reobserve_action(action)
            return
        if self.phase=='RECOVER' and self.resume=='ALIGN_SCAN' and self.plan is not None:
            self.hold(action)
            if self.t-self.since>8.:
                self.stop('TIMEOUT_RECOVER');return
            patches=[self.patch(r) for r in (0,1)]
            self.record_align_evidence(patches)
            self.confirm_reacquisition(patches)
            return
        if self.phase not in self.ALIGN_PHASES:
            previous=self.phase
            super().phase_action(action)
            if previous=='APPROACH' and self.phase=='ROLL':
                self.pre_roll=[None,None]
                for r in (0,1):
                    patch=self.patch(r,require_table=False)
                    self.pre_roll[r]=None if patch is None else {
                        'edge':patch['edge'].tolist(),'tangent':patch['tangent'].tolist(),
                        'outward':patch['outward'].tolist(),'time':self.t,
                        'support_points':patch['support_points']}
            if self.phase=='READY' and self.finished:
                if not self.step1_complete:
                    self.step1_done_time=self.t
                    print(f'[STEP1_DONE] t={self.t:.3f}',flush=True)
                self.step1_complete=True;self.finished=False;self.align_start=self.t
                self.branch_reference=[None,None]
                self.align_plan=None;self.capture_hold()
                self.transition('ALIGN_SCAN','step1 complete; plan high-air staging only')
            return
        if self.phase!='ALIGNED' and self.t-self.align_start>50.:
            self.stop('TOTAL_ALIGN_TIMEOUT');return
        if self.phase in self.DEADLINES and self.t-self.since>self.DEADLINES[self.phase]:
            self.stop('TIMEOUT_'+self.phase);return
        self.finished=False
        patches=[self.patch(r) for r in (0,1)]
        self.record_align_evidence(patches)
        if any(x is None for x in patches) or not self.evidence() or self.t-self.plane_time>self.READY_PLANE_MAX_AGE:
            self.edge_change_vote=None
            self.edge_change_since=None
            self.align_vote=None
            if self.align_lost is None:self.capture_hold()
            self.hold(action);self.stable=None
            if self.phase=='ALIGNED':
                self.align_start=self.t
                self.transition('ALIGN_VERIFY','alignment evidence lost')
            self.align_lost=self.t if self.align_lost is None else self.align_lost
            if self.t-self.align_lost>.5:
                missing=[r for r,p in enumerate(patches) if p is None]
                if not self.begin_reobserve(missing):self.recover('ALIGN_RGBD_LOST')
            return
        self.align_lost=None
        if self.phase=='ALIGN_SCAN':
            self.hold(action)
            edges=np.array([x['edge'] for x in patches])
            if self.align_vote is None or np.max(np.linalg.norm(edges-self.align_vote[1],axis=1))>.006:
                self.align_vote=(self.t,edges);return
            if self.t-self.align_vote[0]<.4:return
            candidate=self.make_alignment(patches)
            if candidate is None:self.reason='NO_FEASIBLE_JAW_APPROACH';return
            self.align_plan=candidate;self.align_progress=None;self.capture_hold()
            self.edge_reference=np.array([p['edge'] for p in patches])
            self.edge_change_vote=None
            self.edge_change_since=None
            self.transition('ALIGN_MOVE','coarse region and high-air envelopes planned')
            return
        edge_state=self.check_edge_change(patches)
        if edge_state!='STABLE':
            self.hold(action)
            self.stable=None
            self.align_progress=None
            if self.phase=='ALIGNED':
                self.align_start=self.t
                self.transition('ALIGN_VERIFY','edge change under observation')
            if edge_state=='CONFIRMATION_TIMEOUT':
                self.recover('EDGE_CONFIRMATION_TIMEOUT')
            if edge_state=='REPLAN':
                self.align_plan=None;self.align_vote=None
                self.transition('ALIGN_SCAN','local edge changed; remeasure')
            return
        errors=[self.alignment_action(r,action) for r in (0,1)]
        reports=[];valid=True
        for r,p in enumerate(patches):
            n,d,_=self.plane
            info=self.orientation(r,self.geo[r][0],yaw(np.asarray(self.obs['base_pose_world'])[r,3:]))
            if info is None:
                self.recover('STAGING_ORIENTATION_UNDEFINED');return
            verts=np.vstack(list(info['bodies'].values()))
            heading=float(np.arccos(np.clip(info['axis']@(-p['outward']),-1,1)))
            angular=float(np.max(abs(self.geo[r][0][8:11]-self.align_plan[r]['q'][8:11])))
            clearance=float(np.min(verts@n+d))
            outside=float(np.min((verts-p['edge'])@p['outward']))-p['uncertainty_m']
            blockers=staging_blockers(errors[r],heading,angular,info['gap'],clearance,outside)
            valid &= not blockers
            reports.append(dict(robot=r,scope='high_air_staging',lower_jaw=info['lower'],
                position_error_m=errors[r],heading_error_deg=float(np.degrees(heading)),
                wrist_error_rad=angular,pad_vertical_separation_m=info['gap'],
                hover_clearance_m=clearance,outside_region_margin_m=outside,
                region_span_m=p['span_m'],region_uncertainty_m=p['uncertainty_m'],
                region_anchor=p['edge'].tolist(),blockers=blockers,
                descent_authorized=False,insertion_certified=False))
        self.alignment_report=reports
        self.alignment_report_time=self.t
        if self.phase=='ALIGN_MOVE' and max(errors)>.010:
            metric=max(errors)
            if self.align_progress is None:self.align_progress=(self.t,metric)
            elif self.t-self.align_progress[0]>5:
                if metric>self.align_progress[1]-.002:
                    self.recover('ALIGN_NO_PROGRESS');return
                self.align_progress=(self.t,metric)
        if valid:
            if self.phase=='ALIGNED':
                # transition() clears the acquisition timer. A verified steady
                # pose must not re-wait 0.5s on the next otherwise valid frame.
                self.finished=True
            else:
                if self.phase=='ALIGN_MOVE':self.transition('ALIGN_VERIFY','staging pose reached; verify fresh region and clearance')
                self.stable=self.t if self.stable is None else self.stable
                if self.t-self.stable>.5:
                    self.transition('ALIGNED','both high-air staging poses stable; Step3 must remeasure before descent')
                    self.finished=True
        else:
            self.stable=None
            if self.phase=='ALIGN_VERIFY':
                self.transition('ALIGN_MOVE','correct staging pose')
            if self.phase=='ALIGNED':
                self.align_start=self.t
                self.transition('ALIGN_VERIFY','staging acceptance changed')

    def constrain(self,action):
        super().constrain(action)
        if self.phase in self.ALIGN_PHASES:
            # Geometry guard at measured pose plus one action interval. Base
            # center guard above is inherited unchanged, including NAV behavior.
            for r in (0,1):
                n,d,_=self.plane
                if float(np.min(np.vstack(list(self.bodies(r).values()))@n+d))<.050:
                    self.stop('ALIGN_ENVELOPE_TABLE_CLEARANCE');action[:]=0;break
                q=self.geo[r][0].copy();dt=.10
                q[3]+=action[r,3]*1.54*dt;q[4:8]+=action[r,4]*1.21*dt/4
                q[8:11]+=action[r,5:8]*5.5*dt
                angle=yaw(np.asarray(self.obs['base_pose_world'])[r,3:])+action[r,2]*2.86*dt
                predicted=np.vstack(list(self.bodies(r,q,angle).values()))
                shift=self.geo[r][1][:3,:2]@(action[r,:2]*.616*dt)
                if float(np.min((predicted+shift)@n+d))<.055:
                    action[r,:8]=0;self.finished=False;self.stable=None
                    self.recover('ALIGN_PREDICTED_CLEARANCE');break
            # Guard may have removed Cartesian compensation during rotation;
            # never continue wrist/base rotation in that circumstance.
            if self.diag.get('base_guard_active'):
                action[:,2:8]=0;self.finished=False;self.stable=None
                if self.phase=='ALIGNED':
                    self.align_start=self.t
                    self.transition('ALIGN_VERIFY','pair guard invalidated alignment')
            a=np.vstack(list(self.bodies(0).values()));b=np.vstack(list(self.bodies(1).values()))
            axis=self.geo[1][4]-self.geo[0][4];axis/=max(np.linalg.norm(axis),1e-9)
            gap=float(np.min(b@axis)-np.max(a@axis))
            self.diag['gripper_envelope_pair_gap_m']=gap
            if gap<.045:
                self.stop('ALIGN_GRIPPER_DISTANCE_BRAKE');action[:]=0
        self.applied=action.copy()

    def annotate(self,images,mark):
        super().annotate(images,mark)
        if self.align_plan is None:return
        for r in (0,1):
            p=self.align_plan[r];pts=[p['patch']['edge'],p['goal']]
            pts.extend(np.vstack(list(self.bodies(r).values())))
            for index in (1+r,3+r):
                T,spec=self.camera(index,self.geo[r]);xyz=points(np.linalg.inv(T),np.array(pts))
                f=256/(2*np.tan(np.radians(spec['horizontal_fov_deg'])/2))
                for i,x in enumerate(xyz):
                    if x[2]>.01:
                        u,v=128+f*x[:2]/x[2]
                        if 0<=u<256 and 0<=v<256:mark(images[index],u,v,(255,80,255) if i<2 else (50,220,255),2)


