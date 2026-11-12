"""Passive, episode-local mapping from the public PhyRC policy observation."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import minimum_filter, distance_transform_edt
from costmap_memory import OccupancyMemory, GridSpec, detect_gaps
from traversal_memory import TraversalMemory

HERE = Path(__file__).resolve().parent
if (HERE / '.deps').is_dir():
    sys.path.insert(0,str(HERE/'.deps'))
# Reuse robot self-geometry modules already developed in the user's V2 workspace.
# No object pose, cloth particle, audit, or simulator scene ground truth is read.
geometry_parent=os.getenv('V2_GEOMETRY_PARENT')
candidates=([Path(geometry_parent)] if geometry_parent else [])+[HERE,*HERE.parents[:3]]
for parent in candidates:
    if (parent/'v2_exchange').is_dir():
        sys.path.insert(0,str(parent));break
try:
    from embreex.mesh_construction import TriangleMesh
    from embreex.rtcore_scene import EmbreeScene
    from v2_exchange.core import forward_kinematics as fk
    from v2_exchange.core.robot_meshes import ROOT, load_colliders
    _GEOMETRY_IMPORT_ERROR = None
except ImportError as exc:
    TriangleMesh = EmbreeScene = fk = ROOT = load_colliders = None
    _GEOMETRY_IMPORT_ERROR = exc

PROJECT = Path(fk.__file__).resolve().parents[1] if fk is not None else HERE
CONTRACT_PATH = PROJECT / "config/policy_interface.json"
CONFIG = json.loads(CONTRACT_PATH.read_bytes()) if CONTRACT_PATH.exists() else {"cameras": []}
CAMS = [s["id"] for s in CONFIG["cameras"]]
RES = .05
ORIGIN = np.array([-4.5, -3.5, 0.])  # one-metre left buffer for R0 base footprint
SHAPE = (160, 140, 5)
CENTERS = (np.indices(SHAPE).reshape(3,-1).T + .5) * RES + ORIGIN
STEP = 4
# 64x64 samples per camera. Subpixel offsets rotate across observation frames;
# this fills persistent blind lattice gaps WITHOUT increasing rays/frame.
SELF_TOL = .035

def optical_mount(s):
    eye = np.array(s["eye_mount_m"], float)
    forward = np.array(s["target_mount_m"]) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, s["up_mount"])
    right /= np.linalg.norm(right)
    tr = np.eye(4)
    tr[:3,:3] = np.column_stack((right, np.cross(forward,right), forward))
    tr[:3,3] = eye
    return tr


class Geometry:
    def __init__(self):
        if _GEOMETRY_IMPORT_ERROR is not None:
            raise RuntimeError("Isaac/PhyRC collision geometry dependencies are required") from _GEOMETRY_IMPORT_ERROR
        if not CONTRACT_PATH.exists():
            raise FileNotFoundError(CONTRACT_PATH)
        usd = PROJECT / "assets/stretch_4.usd"
        self.joints = fk.read_usd_joints(usd)
        mapping = json.loads((PROJECT / "config/stretch4_dof_mapping.json").read_bytes())
        self.names = [mapping[k]["dof_names"] for k in ("/World/Stretch4","/World/Stretch4_2")]
        meshes, _, _ = load_colliders(usd, self.joints)
        self.meshes = [(link, v @ tr[:3,:3].T + tr[:3,3], tri) for _,link,v,tri,tr in meshes]
        self.faces = []
        offset = 0
        for _ in range(2):
            for _, vertices, faces in self.meshes:
                self.faces.append(faces+offset)
                offset += len(vertices)
        self.faces = np.vstack(self.faces).astype("i")

    def at(self, obs):
        bases = np.asarray(obs["base_pose_world"],float).reshape(2,7)
        joints = np.asarray(obs["joint_position"],float).reshape(2,13)
        if not (np.isfinite(bases).all() and np.isfinite(joints).all()):
            raise ValueError("invalid public base or joint state")
        world, vertices = [], []
        for i in range(2):
            base = fk.mat(bases[i,:3], bases[i,3:])
            local = fk.forward_graph(self.joints, ROOT, dict(zip(self.names[i],joints[i])))
            links = {name:base@tr for name,tr in local.items()}
            world.append(links)
            for name,v,_ in self.meshes:
                tr = links[name]
                vertices.append(v@tr[:3,:3].T+tr[:3,3])
        scene = EmbreeScene()
        mesh = TriangleMesh(scene, vertices=np.vstack(vertices).astype("f"), indices=self.faces)
        return world, scene, mesh


def floor_plane(xyz, eligible):
    """RANSAC constrained to a near-horizontal plane near the public world ground datum."""
    pts = xyz[eligible & (np.abs(xyz[:,:,2]) < .09)]
    if len(pts) < 15:
        return None
    rng = np.random.default_rng(21)
    pts = pts[::max(1,len(pts)//600)]
    triples = pts[rng.integers(0,len(pts),(48,3))]
    normals = np.cross(triples[:,1]-triples[:,0], triples[:,2]-triples[:,0])
    norms = np.linalg.norm(normals,axis=1)
    use = (norms>.004) & (np.abs(normals[:,2])>.985*norms)
    if not np.any(use):
        return None
    normals = normals[use]/norms[use,None]
    normals *= np.sign(normals[:,2,None])
    offsets = np.sum(triples[use,0]*normals,axis=1)
    counts = (np.abs(pts@normals.T-offsets)<.013).sum(axis=0)
    k = int(counts.argmax())
    return (normals[k], offsets[k]) if counts[k]>=max(12,.55*len(pts)) else None


def atomic_npz(path, **values):
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as fp:
        np.savez_compressed(fp,**values)
    os.replace(temp,path)


class Memory:
    def __init__(self):
        self.votes = np.zeros((4,*SHAPE),np.uint32)  # free, surface, self/occlusion, cloth
        self.floor = np.zeros((2*SHAPE[0], 2*SHAPE[1]),np.uint32)
        self.low = np.zeros(SHAPE[:2],bool)
        self.frames = []

    def add(self, frame, votes, floor, low):
        self.votes += votes
        self.floor += floor
        self.low |= low
        self.frames.append(frame)

    def save(self, path, volume=None, base_visits=None, traversal=None):
        free, hit, robot, cloth = self.votes
        support = (self.floor.reshape(SHAPE[0],2,SHAPE[1],2).min(axis=(1,3))>=2) & ~self.low
        labels = np.zeros(SHAPE,np.uint8)
        labels[free>=2] = 1
        labels[hit>0] = 2
        labels[(hit>0)&(free>0)] = 3
        # A bottom return may be reconciled only with independently measured floor.
        labels[:,:,0][support & (hit[:,:,0]>0)] = 1
        labels[(robot>0)|(cloth>0)] = 3
        reason = (labels==0).any(2).astype("B") | ((labels==2).any(2).astype("B")*2)
        reason |= (labels==3).any(2).astype("B")*4 | (~support).astype("B")*8
        clearance = distance_transform_edt(np.pad(reason==0,1))[1:-1,1:-1]*RES - np.sqrt(2)*RES - .215 - .02
        cost = np.full(SHAPE[:2],255,"B")
        allowed = clearance>0
        cost[allowed] = np.ceil(253*np.exp(-clearance[allowed]/.2)).astype("B")
        payload=dict(cost=cost, reason=reason, labels=labels, support=support, votes=self.votes,
                     floor_subcells=self.floor, low_returns=self.low, frames=self.frames,
                     origin=ORIGIN, resolution=RES)
        if base_visits is not None:
            payload["base_visit_count_2d"] = np.asarray(base_visits, dtype=np.uint16)
        if traversal is not None:
            payload.update(traversal.export())
        if volume is not None:
            # Keep legacy fields for export_map; new fields are explicitly named.
            payload.update(volume.export())
            payload.update(volume.project_base(support, output_res=RES))
            # Gap candidates go to JSON, not an object-dtype NPZ field.
        atomic_npz(path, **payload)
        return int(allowed.sum())


class Mapper:
    def __init__(self, directory, geometry, masks, source, review_regions=None):
        self.out = Path(directory)
        self.out.mkdir(parents=True,exist_ok=False)
        (self.out/"evidence").mkdir()
        self.geometry, self.masks = geometry, masks
        self.review_regions = review_regions or {}
        self.full, self.post = Memory(), Memory()
        spec=GridSpec(origin=tuple(float(x) for x in ORIGIN), resolution=.10,
                      shape=(SHAPE[0]//2, SHAPE[1]//2, 24))
        self.full_3d, self.post_3d = OccupancyMemory(spec), OccupancyMemory(spec)
        self.last_static_rays = []
        self.observation_index = 0
        # Record full-rate observed base poses, not just each 12th RGB-D update.
        # Traversal footprints are evidence of presence, NOT collision-free space.
        self.traversal = TraversalMemory(ORIGIN[:2], RES, SHAPE[:2])
        self.base_visit_count_2d = self.traversal.base_visit_count_2d
        self.last_pose_frame = -1
        self.last = -1
        self.last_garment_evidence = []  # public head RGB-D only, current frame
        self.last_time = -np.inf
        self.grasp = None
        self.record = dict(source=source, cameras=CAMS, frames=[], snapshots=[], planner_allowed=False)

    def record_pose(self, obs, frame):
        """Full-rate pose stream independently of expensive RGB-D map_stride."""
        if frame == self.last_pose_frame:
            return
        if 'base_pose_world' not in obs:
            return  # offline image-only test fixture; explicitly no pose evidence
        t = float(np.asarray(obs['simulation_time_s']).item())
        self.traversal.observe(frame, t, obs['base_pose_world'])
        self.last_pose_frame = frame

    def observe(self, obs, frame):
        t = float(np.asarray(obs["simulation_time_s"]).item())
        if frame<=self.last or not np.isfinite(t) or t<=self.last_time:
            raise ValueError("timestamps and frame indices must increase within one episode")
        # Direct mapper test calls remain supported; PolicyV2 records every frame.
        self.record_pose(obs, frame)
        world, scene, mesh = self.geometry.at(obs)
        votes = np.zeros((4,*SHAPE),bool)
        floor = np.zeros((2*SHAPE[0],2*SHAPE[1]),bool)
        low = np.zeros(SHAPE[:2],bool)
        rows, stats = [], []
        head_garment_evidence = []
        static_rays = []
        # Visit each of 16 subpixel offsets across consecutive mapping updates.
        # All arrays (RGB, D, mask, rays and voxel reprojection) share the offset.
        sample_u = int(self.observation_index % STEP)
        sample_v = int((self.observation_index // STEP) % STEP)
        U,V = np.meshgrid(np.arange(sample_u,256,STEP),
                          np.arange(sample_v,256,STEP))
        for cid,s in enumerate(CONFIG["cameras"]):
            name = s["id"]
            if name=="overview":
                continue
            cam = world[int(name[6])]["/stretch/"+s["mount_role"]]@optical_mount(s)
            f = 128/np.tan(np.deg2rad(s["horizontal_fov_deg"])/2)
            dirs = np.stack(((U-128)/f,(V-128)/f,np.ones_like(U)),axis=2)@cam[:3,:3].T
            near, far = s["clip_m"]
            # Start rays at the camera near plane; tfar is optical Z in this parameterization.
            ray = dirs.reshape(-1,3).astype("f")
            pred = scene.run((cam[:3,3]+near*ray).astype("f"),ray,output=1)["tfar"].reshape(U.shape)+near
            rgb = np.asarray(obs["rgb"][cid])
            d = np.asarray(obs["depth"][cid]).reshape(256,256)[sample_v::STEP,sample_u::STEP]
            valid = np.asarray(obs["depth_valid"][cid]).reshape(256,256)[sample_v::STEP,sample_u::STEP].astype(bool)
            valid &= np.isfinite(d)&(d>=near)&(d<=far)
            detected_self = valid & (np.abs(d-pred)<=SELF_TOL)
            occluded = valid & (d>pred+SELF_TOL)
            env = valid & ~detected_self & ~occluded
            blue,yellow = self.masks(rgb[sample_v::STEP,sample_u::STEP])
            garment = env & (blue|yellow)
            xyz = cam[:3,3] + dirs*np.where(valid,d,0)[:,:,None]
            # Only head cameras; world-coordinate cloth candidates from *public*
            # RGB-D already stripped of robot mesh returns. Record pre-grasp
            # candidates; PolicyV2 decides when it is safe to use this evidence.
            if name.endswith('_head'):
                cloth_points = xyz[garment & (xyz[:,:,2] >= .08) & (xyz[:,:,2] <= 1.8)]
                if len(cloth_points) >= 8:
                    center = np.median(cloth_points,axis=0)
                    if np.isfinite(center).all():
                        head_garment_evidence.append(dict(
                            camera=name,points=int(len(cloth_points)),
                            garment_xyz_world=center.tolist(),
                            frame=int(frame),time_s=t))
            # Normals across valid, continuous, non-robot surface patches.
            normals = np.cross(xyz[2:,1:-1]-xyz[:-2,1:-1],xyz[1:-1,2:]-xyz[1:-1,:-2])
            nz = np.linalg.norm(normals,axis=2)
            flat = np.zeros(U.shape,bool)
            flat[1:-1,1:-1] = (env[2:,1:-1]&env[:-2,1:-1]&env[1:-1,2:]&env[1:-1,:-2]
                               &(nz>1e-8)&(np.abs(normals[:,:,2])>.96*nz))
            static = env & ~garment
            # 3D rays terminate at their observed *static* endpoint.
            # Never extrapolate behind a robot, cloth, invalid depth, or occlusion.
            static_rays.append((cam[:3,3].copy(),xyz[static].reshape(-1,3).copy(),cid))
            plane = floor_plane(xyz,flat&static)
            # RGB-D floor evidence: independent near-horizontal normals and public
            # world-ground datum. This is a measured local ground *candidate*,
            # not a semantic ground-truth floor or a guarantee of base access.
            # A missing global RANSAC plane must not discard all valid local floor returns.
            datum = flat & static & (np.abs(xyz[:,:,2]) <= .025)
            ground = datum if plane is None else (flat & static & (np.abs(xyz@plane[0]-plane[1]) < .015) & (np.abs(xyz[:,:,2]) <= .045))
            sub = np.floor((xyz[:,:,:2]-ORIGIN[:2])/(RES/2)).astype(int)
            inside = ((sub>=0)&(sub<np.array(floor.shape))).all(2)
            floor[tuple(sub[ground&inside].T)] = True
            low_returns = env & inside & (xyz[:,:,2]>.015)&(xyz[:,:,2]<=.05)&~ground
            low[tuple((sub[low_returns]//2).T)] = True
            optical = (CENTERS-cam[:3,3])@cam[:3,:3]
            z = optical[:,2]
            pixel=optical[:,:2]*(f/np.maximum(z,1e-5))[:,None]+128
            uv=np.rint((pixel-np.array([sample_u,sample_v]))/STEP).astype(int)
            ids = np.flatnonzero((z>=near)&(z<=far)&((uv>=0)&(uv<64)).all(1))
            u,v = uv[ids].T
            depth = d[v,u]
            end = np.minimum(np.where(valid,d,0),pred)
            clear = minimum_filter(end,3,mode="constant",cval=0)[v,u] > z[ids]+.05
            close = np.abs(depth-z[ids])<=.05
            uncertain = (detected_self|occluded)[v,u] & (np.abs(pred[v,u]-z[ids])<=.05)
            for layer,sel in enumerate([clear,close&static[v,u],uncertain,close&garment[v,u]]):
                votes[layer].reshape(-1)[ids[sel]]=True
            torso = np.zeros(U.shape,bool)
            annotation = self.review_regions.get((frame,name))
            if annotation and hashlib.sha256(rgb.tobytes()).hexdigest()==annotation[0]:
                l,r,top,bottom=annotation[1]
                torso = (U>=l)&(U<=r)&(V>=top)&(V<=bottom)&static
            sel = inside & env & (xyz[:,:,2]>.12) & (xyz[:,:,2]<1.9)
            rows.append(np.column_stack((np.full(sel.sum(),cid),U[sel],V[sel],xyz[sel],rgb[sample_v::STEP,sample_u::STEP][sel],
                                         flat[sel],garment[sel],torso[sel])).astype("f"))
            stats.append(dict(camera=name, valid=int(valid.sum()),self_returns=int(detected_self.sum()),
                              depth_behind_mesh=int(occluded.sum()),cloth_color=int(garment.sum()),
                              floor_pixels=int((ground&inside).sum()),
                              floor_model='LOCAL_GROUND_DATUM' if plane is None else 'RANSAC_WITH_WORLD_DATUM',
                              static_depth_rays=int(static.sum()),
                              sample_offset_uv=[sample_u,sample_v]))
        atomic_npz(self.out/"evidence"/f"{frame}.npz",samples=np.vstack(rows),time_s=t,
                   rgb_preview=np.asarray(obs["rgb"])[:,::2,::2,:3])
        self.full.add(frame,votes,floor,low)
        for origin,points,cid in static_rays:
            self.full_3d.update_rays(origin,points,cid,t)
        if self.grasp is not None:
            self.post.add(frame,votes,floor,low)
            for origin,points,cid in static_rays:
                self.post_3d.update_rays(origin,points,cid,t)
        self.last_garment_evidence = head_garment_evidence
        self.last_static_rays = static_rays
        self.last,self.last_time=frame,t
        self.observation_index += 1
        self.last_votes=(votes,floor,low)
        self.tips=[w["/stretch/grasp_center_link"].tolist() for w in world]
        self.record["frames"].append(dict(frame=frame,time_s=t,cameras=stats))

    def handoff(self, obs, frame, source):
        if self.grasp is not None:
            raise ValueError("duplicate handoff")
        if self.last!=frame:
            self.observe(obs,frame)
        self.grasp=frame
        self.post.add(frame,*self.last_votes)
        for origin,points,cid in self.last_static_rays:
            self.post_3d.update_rays(origin,points,cid,self.last_time)
        self.record["handoff"]=dict(frame=frame,time_s=self.last_time,source=source,grasp_fk_world=self.tips,
                                    base_pose_world=np.asarray(obs["base_pose_world"]).reshape(2,7).tolist(),
                                    physical_grasp_verified=False)
        atomic_npz(self.out/"handoff_wrist.npz",rgb=np.asarray(obs["rgb"])[1:3],depth=np.asarray(obs["depth"])[1:3],
                   depth_valid=np.asarray(obs["depth_valid"])[1:3])
        self.snapshot("handoff")

    def current_base_map(self):
        """Strict fail-closed map: the same safety policy as before this patch."""
        support = (self.full.floor.reshape(SHAPE[0],2,SHAPE[1],2).min(axis=(1,3))>=2) & ~self.full.low
        return self.full_3d.project_base(support,output_res=RES)

    def current_review_map(self):
        """Uncertified RGB-D evidence / traversability hypothesis, plan_only only.

        Uses the actual accumulated free-ray and occupied-voxel evidence,
        measured floor support and near-floor obstacles; does not convert
        UNKNOWN into certified FREE. May be used for OFFLINE hypothesis searches
        but NEVER for real/experimental movement or safety qualification.
        """
        counts=self.full.floor.reshape(SHAPE[0],2,SHAPE[1],2).sum(axis=(1,3))
        strict_support=(self.full.floor.reshape(SHAPE[0],2,SHAPE[1],2).min(axis=(1,3))>=2) & ~self.full.low
        strict=self.full_3d.project_base(strict_support,output_res=RES)
        return self.full_3d.project_base_review(strict, floor_votes=counts,
                 near_floor_obstacle=self.full.low,output_res=RES)

    def snapshot(self, name):
        counts={}
        volume_stats={}
        gap_candidates={}
        for prefix,mem,volume in (("full",self.full,self.full_3d),("post",self.post,self.post_3d)):
            if mem.frames:
                counts[prefix]=mem.save(self.out/f"{name}_{prefix}.npz",volume=volume,
                                        base_visits=self.base_visit_count_2d,
                                        traversal=self.traversal)
                derived=volume.project_base((mem.floor.reshape(SHAPE[0],2,SHAPE[1],2).min(axis=(1,3))>=2)&~mem.low,
                                            output_res=RES)
                volume_stats[prefix]=dict(known_free_cells=int(derived["base_known_free_2d"].sum()),
                                          unknown_cells=int(derived["base_unknown_2d"].sum()),
                                          occupied_cells=int(derived["base_occupied_2d"].sum()),
                                          voxel_height_m=volume.spec.shape[2]*volume.spec.resolution,
                                          camera_observation_updates=volume.num_updates)
                gap_candidates[prefix]=detect_gaps(derived["base_known_free_2d"],
                      derived["base_occupied_2d"], ORIGIN[:2],RES,max_candidates=10)
        self.record["snapshots"].append(dict(name=name,frame=self.last,allowed_base_cells=counts,
                        occupancy_3d=volume_stats))
        self.record["gap_candidates_latest"]=gap_candidates
        gap_path=self.out/f"{name}_gap_candidates.json"
        gap_temp=gap_path.with_suffix(".tmp")
        gap_temp.write_text(json.dumps(dict(frame=self.last,source="observed_up_to_frame",
                                             candidates=gap_candidates),indent=2),encoding="utf-8")
        os.replace(gap_temp,gap_path)
        path=self.out/"report.json"
        temp=path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.record,indent=2),encoding="utf-8")
        os.replace(temp,path)
