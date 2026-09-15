"""Directional, continuous teleop compliance; never pauses the simulation.

The strain envelope is a controller operating limit, not a new cloth material.
An elastic edge traction estimate supplies the resisting direction, not a
calibrated force sensor. Relieving and tangential commands remain available.
"""
import os
from contextlib import contextmanager
from contextvars import ContextVar
import numpy as np
import torch
import warp as wp
from pxr import Gf, Usd, UsdGeom


_measurement_batch = ContextVar('cloth_measurement_batch', default=None)


@contextmanager
def cloth_control_batch():
    """Reuse global strain only between controllers with no intervening physics.

    The scope ends before simulation_app.update(). Never cache by frame number:
    diagnostic callers may reuse a number after stepping or restoring a slot.
    """
    token = _measurement_batch.set({})
    try:
        yield
    finally:
        _measurement_batch.reset(token)


class ClothLoad:
    def __init__(self, cloth):
        mesh=UsdGeom.Mesh(cloth.prim)
        xf=UsdGeom.Xformable(cloth.prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        p=np.asarray([xf.Transform(Gf.Vec3d(*map(float,v))) for v in mesh.GetPointsAttr().Get()])
        t=np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
        a=p[t[:,1]]-p[t[:,0]];b=p[t[:,2]]-p[t[:,0]]
        length=np.linalg.norm(a,axis=1);x=(a*b).sum(1)/length
        y=np.linalg.norm(np.cross(a,b),axis=1)/length
        if np.any(length*y<=0):raise ValueError('degenerate rest triangles')
        area=length*y/2
        raw=np.concatenate([t[:,[0,1]],t[:,[1,2]],t[:,[2,0]]])
        edges,rev=np.unique(np.sort(raw,axis=1),axis=0,return_inverse=True)
        weights=np.bincount(rev,weights=np.tile(area/3,3),minlength=len(edges))
        device=cloth.get_world_positions(clone=False).device
        self.t=torch.as_tensor(t,device=device)
        self.inv=torch.as_tensor(np.column_stack([1/length,1/y,x/(length*y)]),device=device,dtype=torch.float32)
        self.area=torch.as_tensor(area/area.sum(),device=device,dtype=torch.float32)
        self.edges=torch.as_tensor(edges,device=device)
        restlen=np.linalg.norm(p[edges[:,0]]-p[edges[:,1]],axis=1)
        self.length=torch.as_tensor(restlen,device=device,dtype=torch.float32)
        self.weight=torch.as_tensor(weights/restlen,device=device,dtype=torch.float32)

    def measure(self, cloth):
        batch = _measurement_batch.get()
        key = (id(self), id(cloth))
        if batch is not None and key in batch:
            _, _, p, levels = batch[key]
            return p, levels.copy()
        p=cloth.get_world_positions(clone=False)[0]
        a=p[self.t[:,1]]-p[self.t[:,0]];b=p[self.t[:,2]]-p[self.t[:,0]]
        u=a*self.inv[:,0,None];v=b*self.inv[:,1,None]-a*self.inv[:,2,None]
        aa=(u*u).sum(1);bb=(v*v).sum(1);ab=(u*v).sum(1)
        strain=torch.sqrt(torch.clamp((aa+bb+torch.sqrt((aa-bb)**2+4*ab*ab))/2,min=0))-1
        values,order=torch.sort(strain)
        cumulative=self.area[order].cumsum(0)
        indices=torch.searchsorted(cumulative,torch.tensor([.95,.99],device=p.device)).clamp(max=len(values)-1)
        levels = values[indices].detach().cpu().numpy()
        if batch is not None:
            batch[key] = (self, cloth, p, levels)
        return p, levels.copy()

    def traction(self, p, rig):
        st=rig['state'];grab=st['grabbed'];idx=grab[1]
        mask=st.get('_grab_anchor_mask')
        # Topology is fixed while a grasp is held. Rebuilding the full V/E
        # selection mask every control tick was O(V+E), although only K edges
        # across the 48-node grip contribute. Build once; evaluate those K
        # edges on subsequent ticks. Positions and forces are NEVER cached.
        key=(id(self),id(grab),idx._version,grab[2]._version,id(mask),
             mask._version if mask is not None else None,
             id(self.edges),self.edges._version,id(self.length),self.length._version,
             id(self.weight),self.weight._version)
        cache=st.get('_traction_boundary')
        if cache is None or cache[0]!=key:
            if mask is None:
                chosen=idx[torch.argsort(grab[2].norm(dim=1))[:48]]
            else:chosen=idx[mask]
            selected=torch.zeros(len(p),device=p.device,dtype=torch.bool);selected[chosen]=True
            ea,eb=self.edges.T
            crossing=selected[ea]^selected[eb]
            ea=ea[crossing];eb=eb[crossing]
            first=selected[ea];anchor=torch.where(first,ea,eb);free=torch.where(first,eb,ea)
            # Owners prevent id reuse after release/reset. Tensor versions
            # also invalidate in-place changes to the mask or rest data.
            cache=(key,(self,grab,mask,self.edges,self.length,self.weight),
                   anchor,free,self.length[crossing],self.weight[crossing])
            st['_traction_boundary']=cache
        _,_,anchor,free,rest_length,weight=cache
        d=p[anchor]-p[free];length=d.norm(dim=1).clamp(min=1e-9)
        tension=(length/rest_length-1).clamp(min=0)*weight
        forces=d*(tension/length)[:,None]
        tf=rig['robot']._articulation_view._physics_view.get_link_transforms()[0,rig['grasp_link_idx']]
        center=tf[:3]
        force=forces.sum(0);torque=torch.cross(p[anchor]-center,forces,dim=1).sum(0)
        wrench=torch.cat([force,torque]).detach().cpu().numpy()
        norm=np.linalg.norm(wrench[:3])
        if norm<1e-9:
            # Rare slack/isolated attachment: direction away from the fabric's
            # centre still permits backing off instead of locking every axis.
            wrench[:3]=(center-p.mean(0)).detach().cpu().numpy();norm=np.linalg.norm(wrench[:3])
        return wrench/max(norm,1e-9)


def constrain_drive(rig, cloths, before, indices, positions, linear, angular, dt):
    """Filter only commands that increase load; leave gripping/release untouched."""
    st=rig['state'];grab=st.get('grabbed')
    if grab is None or os.environ.get('STRETCH4_CONTINUOUS_CLOTH','1')=='0':
        st['_cloth_compliance']={'scale':1.0,'p95':0.0,'p99':0.0}
        return positions,linear,angular
    cloth=cloths[grab[0]]
    model=getattr(cloth,'_continuous_load',None)
    if model is None:
        model=ClothLoad(cloth);cloth._continuous_load=model
    p,levels=model.measure(cloth)
    p95,p99=map(float,levels)
    # 2026-09-11 tuning: restrict pulling at 90% of the previous strain levels.
    # These are control thresholds, not a constitutive strain/failure limit.
    start=float(os.environ.get('STRETCH4_COMPLIANCE_START','.315'))
    end=float(os.environ.get('STRETCH4_COMPLIANCE_END','.45'))
    local=float(os.environ.get('STRETCH4_COMPLIANCE_LOCAL','.72'))
    local_start=float(os.environ.get('STRETCH4_COMPLIANCE_LOCAL_START','.495'))
    if not (0<start<end<local and 0<local_start<local):
        raise ValueError('invalid compliance strain envelope')
    history=st.get('_compliance_history')
    growth=np.zeros(2)
    if history is not None and history[0] is grab:
        growth=.65*history[2]+.35*(levels-history[1])/max(dt,1e-6)
    st['_compliance_history']=(grab,levels.copy(),growth)
    predicted=levels+.07*np.maximum(growth,0)
    risk=float(np.clip(max((predicted[0]-start)/(end-start),(predicted[1]-local_start)/(local-local_start)),0,1))
    scale=1-risk*risk*(3-2*risk)
    st['_cloth_compliance']={'scale':scale,'p95':p95,'p99':p99}
    if risk<=0:return positions,linear,angular
    wrench=model.traction(p,rig)
    normal=wrench[:2];n2=float(normal@normal)
    if n2>1e-8:
        loading=float(linear[:2]@normal)
        if loading>0:linear[:2]-=(1-scale)*loading/n2*normal
    if angular[2]*wrench[5]>1e-6:angular[2]*=scale
    # Actual tensor Jacobian includes six root DOFs for a floating base.
    robot=rig['robot'];view=robot._articulation_view._physics_view
    jac=view.get_jacobians()[0,rig['grasp_link_idx']].detach().cpu().numpy()
    offset=jac.shape[-1]-robot.num_dof
    gradient=wrench@jac[:,offset:]
    actual=robot.get_joint_positions().detach().cpu().numpy()
    # Small, benign strain must not switch on a hard 6 mm position-error cap.
    # Fade target anti-windup in only in the upper half of the load envelope.
    blend=float(np.clip((risk-.5)/.5,0,1))
    antiwindup=blend*blend*(3-2*blend)
    st['_cloth_compliance']['antiwindup']=antiwindup
    positions=positions.copy()
    for i,joint in enumerate(indices[:-2]):
        delta=positions[i]-before[i]
        if delta*gradient[joint]>1e-8:positions[i]=before[i]+scale*delta
        # Discard accumulated outward drive error under load. Otherwise a key
        # held against the cloth keeps winding up a large position command.
        error=positions[i]-actual[joint]
        allowance=(.006 if i<5 else .04)*scale
        if antiwindup>0 and error*gradient[joint]>1e-8 and abs(error)>allowance:
            limited=actual[joint]+np.sign(error)*allowance
            positions[i]+=(limited-positions[i])*antiwindup
    st['lift']=float(positions[0]);st['arm']=float(positions[1:5].sum())
    for name,value in zip(('yaw','pitch','roll'),positions[5:8]):st[name]=float(value)
    # Match the base command integrators to the admitted velocity to prevent
    # a stored full-speed request from jumping out when the tension relaxes.
    quat=robot.get_world_pose()[1].detach().cpu().numpy();w,x,y,z=quat
    forward=np.array([1-2*(y*y+z*z),2*(x*y+z*w)]);forward/=max(np.linalg.norm(forward),1e-9)
    right=np.array([forward[1],-forward[0]])
    st['base_fwd']=float(linear[:2]@forward);st['base_strafe']=float(linear[:2]@right)
    st['base_turn']=float(angular[2])
    return positions,linear,angular


def set_base_target(rig, linear, angular):
    rig['state']['_base_drive_target']=(linear.copy(),angular.copy())


@wp.kernel
def _base_wrench(velocity: wp.array2d(dtype=wp.float32),
                 target: wp.array(dtype=wp.float32), mass: float,
                 force: wp.array3d(dtype=wp.float32),
                 torque: wp.array3d(dtype=wp.float32)):
    body = wp.tid()
    f = wp.vec3(0.0)
    t = wp.vec3(0.0)
    if body == 0:
        acceleration = wp.vec3((target[0] - velocity[0, 0]) / 0.04,
                               (target[1] - velocity[0, 1]) / 0.04, 0.0)
        acceleration *= wp.min(6.0 / wp.max(wp.length(acceleration), 1.0e-6), 1.0)
        f = mass * acceleration
        angular_error = wp.vec3(target[3] - velocity[0, 3],
                                target[4] - velocity[0, 4], target[5] - velocity[0, 5])
        t = mass * 0.18 * angular_error / 0.07
        t *= wp.min(40.0 / wp.max(wp.length(t), 1.0e-6), 1.0)
    for axis in range(3):
        force[0, body, axis] = f[axis]
        torque[0, body, axis] = t[axis]


def apply_base_drive(rig, dt):
    """Same bounded wrench servo, fused into one GPU kernel per robot/step.

    Target upload is reused for all physics steps of the same control tick.
    The cache is rebuilt when reset creates a new PhysX articulation view.
    """
    if 'robot' not in rig:return
    target=rig['state'].get('_base_drive_target')
    if target is None:return
    robot=rig['robot'];view=robot._articulation_view._physics_view
    if not robot._articulation_view.is_physics_handle_valid():return
    if os.environ.get('STRETCH4_BASE_FORCE_CONTROL','1')=='0':
        value=np.concatenate(target)[None,:]
        robot._articulation_view.set_velocities(torch.as_tensor(value,device=robot.get_linear_velocity().device,dtype=torch.float32))
        return
    velocity=view.get_root_velocities()
    cache=rig.get('_base_force_cache')
    if cache is None or cache['view'] is not view:
        masses=view.get_masses();mass=float(masses.sum().item())
        force=torch.zeros((1,masses.shape[1],3),device=velocity.device)
        torque=torch.zeros_like(force)
        cache=dict(view=view,mass=mass,force=force,torque=torque,
                   force_wp=wp.from_torch(force),torque_wp=wp.from_torch(torque),
                   ids=torch.tensor([0],dtype=torch.int32,device=velocity.device),target=None)
        rig['_base_force_cache']=cache
    if cache['target'] is not target:
        cache['target']=target
        cache['target_tensor']=torch.as_tensor(np.concatenate(target),device=cache['force'].device,dtype=torch.float32)
        cache['target_wp']=wp.from_torch(cache['target_tensor'])
    # Share PyTorch's stream so tensor-view reads and force application see the kernel writes.
    with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(velocity.device))):
        wp.launch(_base_wrench,dim=cache['force'].shape[1],inputs=[
            wp.from_torch(velocity),cache['target_wp'],cache['mass'],cache['force_wp'],cache['torque_wp']])
    view.apply_forces_and_torques_at_position(cache['force'],cache['torque'],None,cache['ids'],True)
