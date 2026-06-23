import os; os.environ["QT_QPA_PLATFORM"]="offscreen"
import numpy as np
from PyQt6 import QtWidgets
from dynamo_table import load_dataset
from selection_store import SelectionStore
from plot_common import ModelParams, effective_phi
from detail_window import DetailWindow
P="/mnt/storage/data3/users/wen-lu/tau/warp/position_8_5/dynamo/dynamo_project/abp_align_eo/"
S="/tmp/claude-193122/-home-wlchung-LBEM-invest-helical-F-3D/2971e26b-8026-4193-869d-3d22f7491404/scratchpad"
app=QtWidgets.QApplication([]); ds=load_dataset(P,"dynamo",None,0.7,4.75,19.36,write_temp=False); rate=ds.model_rate
def onblack(f,store,grp):
    p=effective_phi(f,store); r=((p[grp]-(rate*f.pos[grp]+f.phi0)+180)%360)-180; return int((np.abs(r)<=20).sum())
for fid in (80,15,6):
    f=next(x for x in ds.filaments if x.fid==fid)
    store=SelectionStore(os.path.join(S,"remove_list.txt")); store.clear_flips()
    dw=DetailWindow(f,ModelParams(ds),store,gl_enabled=False); dw._set_mode(True)
    dsr=f.phi-rate*f.pos; pinkmask=f.flipped&(np.abs(((dsr-f.phi0_flip+180)%360)-180)<=20) if np.isfinite(f.phi0_flip) else np.zeros(f.n,bool)
    grp=np.where(pinkmask)[0]; gtags=[int(f.tags[i]) for i in grp]
    dw.flip_staged=set(gtags)
    b=onblack(f,store,grp); dw._commit_flip(); a=onblack(f,store,grp)   # tilt once
    dw._commit_flip(); c=store.flip_count()                            # tilt again (no re-stage)
    print(f"fil{fid}: pink n={len(grp)}  tilt->black {b}->{a}/{len(grp)}   tilt-again flips={c} (->0 expected)")
    # sequence tilt-rot-tilt-rot returns to original
    store.clear_flips(); dw.flip_staged=set(gtags)
    for op in (dw._commit_flip,dw._commit_rot_flip,dw._commit_flip,dw._commit_rot_flip): op()
    print(f"        tilt-rot-tilt-rot -> flips={store.flip_count()} (0 expected)")
