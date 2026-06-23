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
f=next(x for x in ds.filaments if x.fid==6); store=SelectionStore(os.path.join(S,"remove_list.txt")); store.clear_flips()
dw=DetailWindow(f,ModelParams(ds),store,gl_enabled=False); dw._set_mode(True)
grp=[int(f.tags[i]) for i in np.where(f.flipped)[0]][:5]
dw.flip_staged=set(grp); dw._commit_flip()                       # tilt
print("after manual tilt:", [store.get_state(t) for t in grp][:3], "(expect (1,0))")
dw.flip_staged=set(grp); dw._commit_rot_flip()                   # + rot -> both
print("after + rot     :", [store.get_state(t) for t in grp][:3], "(expect (1,1))")
dw.flip_staged=set(grp); dw._commit_flip()                       # toggle tilt off -> rot only
print("after tilt again:", [store.get_state(t) for t in grp][:3], "(expect (0,1))")
