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
store=SelectionStore(os.path.join(S,"remove_list.txt")); store.clear_flips()
ov=OverviewWindow(ds,store,ModelParams(ds),gl_enabled=False)
def counts():
    t=r=b=0
    for f in ds.filaments:
        for tag in f.tags:
            tt,rr=store.get_state(int(tag))
            t+=tt; r+=rr; b+= (tt and rr)
    return t,r,b
print("buttons:", ov.btn_tiltall.text(), "|", ov.btn_rotall.text(), "| checkable:", ov.btn_tiltall.isCheckable(), ov.btn_rotall.isCheckable())
ov.btn_tiltall.setChecked(True)
print("tilt ON :", "tilt/rot/both bits =", counts(), " flips=", store.flip_count())
ov.btn_rotall.setChecked(True)
print("rot  ON :", "tilt/rot/both bits =", counts(), " flips=", store.flip_count())
# check file has 6 columns + bits
with open(store.flip_path) as fh: lines=[l for l in fh if not l.startswith('#')]
print("flip file sample row:", lines[0].strip(), "| ncols=", len(lines[0].split()))
# round-trip
store2=SelectionStore(os.path.join(S,"remove_list.txt"))
import random
tg=int(store.flip_tags()[0])
print("round-trip state for tag",tg,":", store.get_state(tg),"->",store2.get_state(tg), "angles match:", np.allclose(store.get_flip(tg), store2.get_flip(tg)))
ov.btn_tiltall.setChecked(False)
print("tilt OFF:", "tilt/rot/both bits =", counts(), "(tilt should be 0, rot kept)")
