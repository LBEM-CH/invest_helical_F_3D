import os; os.environ["QT_QPA_PLATFORM"]="offscreen"
from PyQt6 import QtWidgets
from dynamo_table import load_dataset
from selection_store import SelectionStore
from plot_common import ModelParams
from overview_window import OverviewWindow
P="/mnt/storage/data3/users/wen-lu/tau/warp/position_8_5/dynamo/dynamo_project/abp_align_eo/"
S="/tmp/claude-193122/-home-wlchung-LBEM-invest-helical-F-3D/2971e26b-8026-4193-869d-3d22f7491404/scratchpad"
app=QtWidgets.QApplication([]); ds=load_dataset(P,"dynamo",None,0.7,4.75,19.36,write_temp=False)
store=SelectionStore(os.path.join(S,"remove_list.txt")); store.clear_flips()
ov=OverviewWindow(ds,store,ModelParams(ds),gl_enabled=False)
def tally():
    t=r=b=0
    for f in ds.filaments:
        for tg in f.tags:
            tt,rr=store.get_state(int(tg)); t+=tt; r+=rr; b+=(tt and rr)
    return dict(tilt=t,rot=r,both_purple=b)
ov.btn_rotall.setChecked(True);  print("rot ON :", tally(), "(both_purple must be 0)")
ov.btn_tiltall.setChecked(True); print("tilt ON:", tally(), "(both_purple must be 0)")
ov.btn_rotall.setChecked(False); print("rot OFF:", tally(), "(rot->0, tilt kept)")
