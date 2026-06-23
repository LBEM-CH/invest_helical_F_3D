import os; os.environ["QT_QPA_PLATFORM"]="offscreen"
import numpy as np
from PyQt6 import QtWidgets
from dynamo_table import load_dataset
from selection_store import SelectionStore
from plot_common import ModelParams, effective_phi
from overview_window import OverviewWindow
P="/mnt/storage/data3/users/wen-lu/tau/warp/position_8_5/dynamo/dynamo_project/abp_align_eo/"
S="/tmp/claude-193122/-home-wlchung-LBEM-invest-helical-F-3D/2971e26b-8026-4193-869d-3d22f7491404/scratchpad"
app=QtWidgets.QApplication([]); ds=load_dataset(P,"dynamo",None,0.7,4.75,19.36,write_temp=False); rate=ds.model_rate
store=SelectionStore(os.path.join(S,"remove_list.txt")); store.clear(); store.clear_flips()
ov=OverviewWindow(ds,store,ModelParams(ds),gl_enabled=False)
N=sum(f.n for f in ds.filaments)
print(f"buttons present: autoexcl={hasattr(ov,'btn_autoexcl')} invert={hasattr(ov,'btn_invert')}")
# off-black BEFORE flipping
def offblack():
    b=0
    for f in ds.filaments:
        if not (f.fittable and np.isfinite(f.phi0)): continue
        r=((effective_phi(f,store)-(rate*f.pos+f.phi0)+180)%360)-180; b+=int((np.abs(r)>20).sum())
    return b
print(f"off-black before flip: {offblack()}")
ov._auto_flip_all()
print(f"off-black after auto-flip-all: {offblack()}")
ov._auto_exclude()
m1=store.count()
print(f"after auto-exclude: marked={m1}  (should ~= off-black after flip)")
ov._invert_selection()
print(f"after invert: marked={store.count()}  (={N}-{m1}? -> {N-m1})  total segs={N}")
