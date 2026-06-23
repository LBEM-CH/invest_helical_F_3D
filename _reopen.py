import os; os.environ["QT_QPA_PLATFORM"]="offscreen"
import numpy as np
from PyQt6 import QtWidgets
from dynamo_table import load_dataset
from selection_store import SelectionStore
from plot_common import ModelParams
from overview_window import OverviewWindow
P="/mnt/storage/data3/users/wen-lu/tau/warp/position_8_5/dynamo/dynamo_project/abp_align_eo/"
S="/tmp/claude-193122/-home-wlchung-LBEM-invest-helical-F-3D/2971e26b-8026-4193-869d-3d22f7491404/scratchpad"
RM=os.path.join(S,"remove_list.txt")
app=QtWidgets.QApplication([])
def snapshot(store):
    return {int(t): store._flips[int(t)] for t in store.flip_tags()}
# --- session 1: apply tilt-all + rot-all, save ---
ds=load_dataset(P,"dynamo",None,0.7,4.75,19.36,write_temp=False)
st1=SelectionStore(RM); st1.clear_flips()
ov1=OverviewWindow(ds,st1,ModelParams(ds),gl_enabled=False)
print("on open: tilt btn checked?", ov1.btn_tiltall.isChecked(), " rot btn checked?", ov1.btn_rotall.isChecked(), " flips=", st1.flip_count())
ov1.btn_tiltall.setChecked(True); ov1.btn_rotall.setChecked(True)
snap1=snapshot(st1); print("session1 after tilt+rot ON: flips=", len(snap1))
# --- session 2: REOPEN (fresh dataset + store loads flipped_list.txt) ---
ds2=load_dataset(P,"dynamo",None,0.7,4.75,19.36,write_temp=False)
st2=SelectionStore(RM)                       # loads flipped_list.txt
ov2=OverviewWindow(ds2,st2,ModelParams(ds2),gl_enabled=False)
print("reopen: flips loaded=", st2.flip_count(), " | resume matches session1:", snapshot(st2)==snap1)
print("reopen: tilt btn checked?", ov2.btn_tiltall.isChecked(), " (should be False/off)")
# click flip-all AGAIN -> must NOT double-flip
ov2.btn_tiltall.setChecked(True); ov2.btn_rotall.setChecked(True)
snap2=snapshot(st2)
same = snap2==snap1
print("after clicking tilt+rot ON again: flips=", len(snap2), " | identical to session1 (no double-flip):", same)
