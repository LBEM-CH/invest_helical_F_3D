import os; os.environ["QT_QPA_PLATFORM"]="offscreen"
import numpy as np
from PyQt6 import QtWidgets
from dynamo_table import load_dataset
from selection_store import SelectionStore
from plot_common import ModelParams
from detail_window import DetailWindow
P="/mnt/storage/data3/users/wen-lu/tau/warp/position_8_5/dynamo/dynamo_project/abp_align_eo/"
S="/tmp/claude-193122/-home-wlchung-LBEM-invest-helical-F-3D/2971e26b-8026-4193-869d-3d22f7491404/scratchpad"
app=QtWidgets.QApplication([]); ds=load_dataset(P,"dynamo",None,0.7,4.75,19.36,write_temp=False)
f=next(x for x in ds.filaments if x.fid==15); store=SelectionStore(os.path.join(S,"remove_list.txt"))
assert store.flip_count()==0, "store not clean!"
dw=DetailWindow(f,ModelParams(ds),store,gl_enabled=False); dw._set_mode(True)
grp=[int(f.tags[i]) for i in np.where(f.flipped)[0]]
seq=[("tilt",dw._commit_flip),("rot",dw._commit_rot_flip),("tilt",dw._commit_flip),("rot",dw._commit_rot_flip)]
counts=[]
for name,op in seq:
    dw.flip_staged=set(grp); op(); counts.append(store.flip_count())
print(f"clean start; tilt,rot,tilt,rot -> counts {counts}; ends at original (0): {store.flip_count()==0}")
# also: tilt twice
store.clear_flips()
dw.flip_staged=set(grp); dw._commit_flip(); c1=store.flip_count()
dw.flip_staged=set(grp); dw._commit_flip(); c2=store.flip_count()
print(f"tilt,tilt -> {c1},{c2}; back to original: {c2==0}")
