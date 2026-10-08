"""Isolated HoST narrow-range experiment; original tasks/assets are never edited."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,random,re,time,types
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
import mujoco
import warp as wp
ROOT=Path('/home/huy/minipi_getup');OUT=ROOT/'results/host/host_supine_narrow_v1'
ORIG=ROOT/'src/minipi_getup/host/assets/pi_12dof_host.xml'
DONOR=ROOT/'src/minipi_getup/asset_zoo/robots/hightorque_minipi/xmls/cl_pai.xml'
ASSET=OUT/'assets/pi_12dof_host_narrow.xml'
START=datetime.fromisoformat('2026-10-08T04:40:21+00:00').timestamp()
TRAIN_CUTOFF=START+25*60
TOTAL_CUTOFF=START+30*60
os.environ.setdefault('MPLCONFIGDIR','/tmp/host-supine-narrow-mpl')
wp.config.kernel_cache_dir='/tmp/host-supine-narrow-warp'
torch.set_num_threads(4)
from minipi_getup.host.config import PiCfg,PiCfgPPO,class_to_dict
import minipi_getup.host.env as hm

def dump(name,obj):
 (OUT/name).write_text(json.dumps(obj,indent=2,default=str))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def seed(value=1):random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)
def use_asset():hm.ASSET_XML=ASSET

def prepare():
 import xml.etree.ElementTree as ET
 assert not ASSET.exists(),'Do not overwrite an existing experiment asset'
 ASSET.parent.mkdir(parents=True,exist_ok=True)
 old=ORIG.read_text();d=ET.parse(DONOR);ranges={j.attrib['name']:j.attrib['range']for j in d.findall('.//joint')if 'range'in j.attrib}
 changes=[]
 def replace(match):
  name=match.group(2);assert name in ranges,name
  changes.append({'joint':name,'original':match.group(4),'narrow':ranges[name]})
  return match.group(1)+name+match.group(3)+ranges[name]+match.group(5)
 new=re.sub(r'(<joint name=")([^"]+)("[^>]*range=")([^"]+)("[^>]*/>)',replace,old)
 assert len(changes)==12
 meshdir=(ORIG.parent/ET.fromstring(old).find('compiler').attrib['meshdir']).resolve()
 new=re.sub(r'meshdir="[^"]+"',f'meshdir="{meshdir}"',new,count=1)
 ASSET.write_text(new)
 a=mujoco.MjModel.from_xml_path(str(ORIG));b=mujoco.MjModel.from_xml_path(str(ASSET));diff=[]
 for key in dir(a):
  try:v=getattr(a,key);w=getattr(b,key)
  except Exception:continue
  if isinstance(v,np.ndarray)and v.shape==w.shape and not np.array_equal(v,w):diff.append(key)
 assert diff==['jnt_range'],diff
 assert all(np.array_equal(a.geom_size,b.geom_size)for _ in [0])
 manifest={'experiment':'host_supine_narrow_v1','request_start_utc':datetime.fromtimestamp(START,timezone.utc).isoformat(),'train_cutoff_utc':datetime.fromtimestamp(TRAIN_CUTOFF,timezone.utc).isoformat(),'total_cutoff_utc':datetime.fromtimestamp(TOTAL_CUTOFF,timezone.utc).isoformat(),'original_asset':str(ORIG),'original_sha256':sha(ORIG),'donor_sha256':sha(DONOR),'narrow_asset_sha256':sha(ASSET),'compiled_model_fields_changed':diff,'changes':changes,'meshdir_note':'Packaging path made absolute to the same original mesh files; compiled geometry/inertia/control unchanged.','training_from_scratch':True,'seed':1,'num_envs':4096,'only_physical_change':'jnt_range','no_position_or_velocity_projection':True}
 dump('manifest.json',manifest)
 # Isolated stop stress: native MuJoCo, same timestep/armature, gravity off to isolate stops.
 spec=mujoco.MjSpec.from_file(str(ASSET));spec.option.timestep=.0025;spec.option.gravity=[0,0,0]
 for j in list(spec.joints):
  if j.name in ranges:
   ac=spec.add_actuator(name='probe_'+j.name,target=j.name,trntype=mujoco.mjtTrn.mjTRN_JOINT)
   ac.gainprm[0]=1;ac.gear[0]=1
 m=spec.compile();results=[]
 for ji in range(1,m.njnt):
  n=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,ji)
  for sign in (-1,1):
   dt=mujoco.MjData(m);dt.qpos[2]=1;lo,hi=m.jnt_range[ji];qa=m.jnt_qposadr[ji];va=m.jnt_dofadr[ji];peak=0.;qmin=0.;qmax=0.
   for t in range(800):
    if t%2==0:dt.ctrl[:]=-.2*dt.qvel[6:];dt.ctrl[ji-1]=np.clip(sign*20-.2*dt.qvel[va],-20,20)
    mujoco.mj_step(m,dt);qmin=min(qmin,float(dt.qpos[qa]));qmax=max(qmax,float(dt.qpos[qa]));peak=max(peak,lo-dt.qpos[qa],dt.qpos[qa]-hi)
   results.append({'joint':n,'direction':sign,'min':qmin,'max':qmax,'overshoot_rad':peak,'final':float(dt.qpos[qa])})
 assert all(abs(r['min'])<2.5 and abs(r['max'])<2.5 for r in results if 'hip_'in r['joint'])
 # Feasible target motions: no collision forces within nominal standing/squat path.
 normal=[];k=mujoco.MjModel.from_xml_path(str(ASSET))
 for t in np.linspace(0,1,31):
  dt=mujoco.MjData(k);dt.qpos[2]=1
  for ji in range(1,k.njnt):
   n=mujoco.mj_id2name(k,mujoco.mjtObj.mjOBJ_JOINT,ji);q=(-.8 if 'hip_pitch'in n else 1.3 if 'calf'in n else -.5 if 'ankle_pitch'in n else 0)*t
   dt.qpos[k.jnt_qposadr[ji]]=q;assert k.jnt_range[ji,0]<=q<=k.jnt_range[ji,1]
  mujoco.mj_forward(k,dt);normal.append({'path_fraction':float(t),'self_contacts':dt.ncon,'joint_limit_constraints':int(sum(dt.efc_type[:dt.nefc]==mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT))})
 assert not any(x['joint_limit_constraints']for x in normal)
 dump('validation.json',{'compiled_ranges':{mujoco.mj_id2name(b,mujoco.mjtObj.mjOBJ_JOINT,j):b.jnt_range[j].tolist()for j in range(1,b.njnt)},'stress_stop_tests':results,'max_stop_overshoot_rad':max(r['overshoot_rad']for r in results),'normal_crouch_path':normal,'original_asset_unchanged':sha(ORIG)==manifest['original_sha256'],'stress_scope':'2 s each direction at 20 Nm minus original damping, gravity off, same .01 armature and .0025 timestep; native MuJoCo stop test; no state projection. Not training.'})
 print('VALIDATED only compiled difference jnt_range; peak stop overshoot',max(r['overshoot_rad']for r in results),flush=True)

def train():
 import subprocess
 idle=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
 assert not idle,f'GPU busy, not starting: {idle}'
 assert torch.cuda.is_available()
 assert time.time()<TRAIN_CUTOFF-60
 use_asset();seed()
 from minipi_getup.host_rl.on_policy_runner import OnPolicyRunner
 cfg=PiCfg();pcfg=PiCfgPPO();pcfg.runner.run_name='host_supine_narrow_v1';pcfg.runner.save_interval=50
 env=hm.LeggedRobot_Pi(cfg,4096,'cuda:0')
 # Verify loaded GPU model before any training.
 expected=mujoco.MjModel.from_xml_path(str(ASSET)).jnt_range[1:]
 assert np.allclose(env.robot.data.joint_pos_limits[0].cpu().numpy(),expected)
 assert env.joint_vel_cap is None and env.limit_solref is None
 dump('config.json',{'env_cfg':class_to_dict(cfg),'train_cfg':class_to_dict(pcfg),'asset':str(ASSET),'num_envs':4096,'seed':1})
 logdir=OUT/'logs';logdir.mkdir(exist_ok=False)
 runner=OnPolicyRunner(env,cfg,class_to_dict(pcfg),str(logdir),device='cuda:0')
 stream=(OUT/'training.csv').open('w',newline='',buffering=1)
 fields=['iteration','elapsed_total_s','base_height','head_height','assisted_logged_success','max_qdot','max_torque','action_scale','pull_force','task','regu','style','target']
 writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();oldlog=runner.log
 class DeadlineReached(Exception):pass
 def log(self,locs):
  oldlog(locs);m=self.last_port_metrics;it=locs['it'];row={'iteration':it,'elapsed_total_s':time.time()-START}
  keys={'base_height':'Robot/base_height_mean','head_height':'Robot/head_height_mean','assisted_logged_success':'Episode_end/success','max_qdot':'Robot/joint_vel_abs_max','max_torque':'Robot/torque_abs_max','action_scale':'Curriculum/action_rescale_mean','pull_force':'Curriculum/force_mean','task':'RewardGroup/task','regu':'RewardGroup/regu','style':'RewardGroup/style','target':'RewardGroup/target'}
  row.update({k:float(m.get(v,float('nan')))for k,v in keys.items()});writer.writerow(row)
  dump('status.json',{'phase':'training','last_completed_update':it,'elapsed_total_s':time.time()-START,'last_metrics':row})
  duration=locs['collection_time']+locs['learn_time']
  if time.time()+max(15,duration*2)>=TRAIN_CUTOFF:
   self.current_learning_iteration=it+1
   self.save(str(logdir/f'model_{it+1}_final.pt'),it=it+1)
   self.writer.flush();dump('training_end.json',{'reason':'wall-clock deadline after completed PPO update','iteration':it+1,'checkpoint':self.last_saved,'elapsed_total_s':time.time()-START,'total_env_steps':self.tot_timesteps,'original_asset_unchanged':sha(ORIG)==json.loads((OUT/'manifest.json').read_text())['original_sha256']})
   raise DeadlineReached()
 runner.log=types.MethodType(log,runner)
 try:runner.learn(12000,init_at_random_ep_len=pcfg.runner.init_at_random_ep_len)
 except DeadlineReached:print('CONTROLLED STOP checkpoint persisted:',runner.last_saved,flush=True)
 finally:
  stream.close()
  if runner.writer:runner.writer.flush();runner.writer.close()
 if not (OUT/'training_end.json').exists():dump('training_end.json',{'reason':'completed iteration budget','iteration':runner.current_learning_iteration,'checkpoint':runner.last_saved,'elapsed_total_s':time.time()-START})

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','train']);args=a.parse_args()
 if args.mode=='prepare':prepare()
 else:train()
