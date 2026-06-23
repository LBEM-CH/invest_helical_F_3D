import os; os.environ["QT_QPA_PLATFORM"]="offscreen"
import numpy as np
from PyQt6 import QtWidgets
from dynamo_table import load_dataset
from selection_store import SelectionStore
from plot_common import ModelParams
from overview_window import OverviewWindow
P="/mnt/storage/data3/users/wen-lu/tau/warp/position_8_5/dynamo/dynamo_project/abp_align_eo/"
S="/tmp/claude-193122/-home-wlchung-LBEM-invest-helical-F-3D/2971e26b-8026-4193-869d-3d22f7491404/scratchpad"
RM=os.path.join(S,"remove_list.txt"); app=QtWidgets.QApplication([])
ds=load_dataset(P,"dynamo",None,0.7,4.75,19.36,write_temp=False)
st1=SelectionStore(RM); st1.clear_flips(); ov1=OverviewWindow(ds,st1,ModelParams(ds),gl_enabled=False)
ov1.btn_tiltall.setChecked(True); ov1.btn_rotall.setChecked(True)
s1={int(t):st1._flips[int(t)] for t in st1.flip_tags()}
st2=SelectionStore(RM)                       # reopen: load flipped_list.txt
s2={int(t):st2._flips[int(t)] for t in st2.flip_tags()}
tags_match = set(s1)==set(s2)
bits_match = all(s1[t][:2]==s2[t][:2] for t in s1)
maxang = max(max(abs(s1[t][2+k]-s2[t][2+k]) for k in range(3)) for t in s1)
print(f"same tags: {tags_match}   same (tilt,rot) bits: {bits_match}   max angle diff loaded-vs-memory: {maxang:.5f}°")
print("=> mismatch is purely 4-decimal file rounding" if tags_match and bits_match and maxang<0.001 else "=> REAL mismatch!")
