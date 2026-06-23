import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation as Rot
from dynamo_table import load_dataset

PATH = "/mnt/storage/data3/users/wen-lu/tau/warp/position_8_5/dynamo/dynamo_project/abp_align_eo/"
PX, RISE, TWIST = 19.36, 4.75, 0.7
ds = load_dataset(PATH, "dynamo", None, TWIST, RISE, PX, write_temp=False)
rate = ds.model_rate

def cmean(d): return np.degrees(np.angle(np.exp(1j*np.radians(d)).mean()))
def cstd(d):
    r = np.abs(np.exp(1j*np.radians(d)).mean()); return np.degrees(np.sqrt(-2*np.log(max(r,1e-12))))

print(f"rate={rate:+.4f} deg/A\n")
print(" fid    N  nMaj nMin  phi0Maj phi0Min  offset  minSpread  S_angle |S.n|  zMaj   zMin")
offsets=[]; sang=[]; sdotn=[]
for fil in ds.filaments:
    if not fil.fittable or fil.n < 12: continue
    n = fil.axis
    D = Rot.from_euler('ZXZ', fil.eulers, degrees=True)
    z = D.as_matrix()[:, :, 2]
    pol = np.sign(z @ n)
    maj = pol == np.sign(np.sum(pol)); minr = ~maj
    if minr.sum() < 4 or maj.sum() < 4:
        # still report
        pass
    phi, pos = fil.phi, fil.pos
    phi0M = cmean(((phi[maj]-rate*pos[maj]+180)%360)-180)
    if minr.sum()>=3:
        phi0m = cmean(((phi[minr]-rate*pos[minr]+180)%360)-180)
        sprd  = cstd(((phi[minr]-rate*pos[minr]+180)%360)-180)
        off   = ((phi0m-phi0M+180)%360)-180
        # de-screwed relative rotation between the two polarity registers
        Dt = Rot.from_rotvec((-np.radians(rate*pos))[:,None]*n[None,:]) * D
        S = Dt[maj].mean() * Dt[minr].mean().inv()
        rv = S.as_rotvec(); ang=np.degrees(norm(rv)); axn=abs((rv/(norm(rv)+1e-12))@n)
        zM=(z[maj]@n).mean(); zm=(z[minr]@n).mean()
        offsets.append(off); sang.append(ang); sdotn.append(axn)
        print(f"{fil.fid:4d} {fil.n:4d} {maj.sum():4d} {minr.sum():4d}  {phi0M:7.1f} {phi0m:7.1f} {off:7.1f}  "
              f"{sprd:7.1f}   {ang:6.1f} {axn:5.2f}  {zM:+.2f}  {zm:+.2f}")
    else:
        print(f"{fil.fid:4d} {fil.n:4d} {maj.sum():4d} {minr.sum():4d}  {phi0M:7.1f}    --      --       --       --    --   {(z[maj]@n).mean():+.2f}    --")

offsets=np.array(offsets)
print(f"\nfilaments with a real 2nd (flipped) register: {len(offsets)}")
print(f"flip offset across filaments: mean={cmean(offsets):.1f}  std={cstd(offsets):.1f} deg   "
      f"values={np.round(np.sort(offsets),0).tolist()}")
print(f"symmetry-op angle: median={np.median(sang):.1f} deg   |axis.n| median={np.median(sdotn):.2f}")
