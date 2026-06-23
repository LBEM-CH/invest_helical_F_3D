import os; os.environ["QT_QPA_PLATFORM"]="offscreen"
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from PyQt6 import QtWidgets
from dynamo_table import load_dataset
from selection_store import SelectionStore
from plot_common import ModelParams
from overview_window import OverviewWindow
P="/mnt/storage/data3/users/wen-lu/tau/warp/position_8_5/dynamo/dynamo_project/abp_align_eo/"
S="/tmp/claude-193122/-home-wlchung-LBEM-invest-helical-F-3D/2971e26b-8026-4193-869d-3d22f7491404/scratchpad"
app=QtWidgets.QApplication([]); ds=load_dataset(P,"dynamo",None,0.7,4.75,19.36,write_temp=False)
store=SelectionStore(os.path.join(S,"remove_list.txt")); store.clear(); store.clear_flips()
ov=OverviewWindow(ds,store,ModelParams(ds),gl_enabled=False)
def polflip_count():  # flipped segs whose polarity differs from original (= tilt or both)
    c=0
    for f in ds.filaments:
        if not f.fittable: continue
        for i,t in enumerate(f.tags):
            a=store.get_flip(int(t))
            if a is None: continue
            z=Rot.from_euler('ZXZ',np.asarray(a,float),degrees=True).as_matrix()[:,2]@f.axis
            if np.sign(z)!=np.sign(f.polarity[i]): c+=1
    return c
ov.btn_autoflip.click()
print(f"after auto-flip-all: flips={store.flip_count()}  polarity-flipped(tilt/both)={polflip_count()}")
ov.btn_resumetilt.click()
print(f"after resume tilt:   flips={store.flip_count()}  polarity-flipped(tilt/both)={polflip_count()} (should be 0)")
