import os; os.environ["QT_QPA_PLATFORM"]="offscreen"
import numpy as np
from PyQt6 import QtWidgets
from dynamo_table import load_dataset
from selection_store import SelectionStore
from plot_common import ModelParams
from overview_window import OverviewWindow
P="/mnt/storage/data3/users/wen-lu/tau/warp/position_8_5/dynamo/dynamo_project/abp_align_eo/"
S="/tmp/claude-193122/-home-wlchung-LBEM-invest-helical-F-3D/2971e26b-8026-4193-869d-3d22f7491404/scratchpad"
app=QtWidgets.QApplication([]); ds=load_dataset(P,"dynamo",None,0.7,4.75,19.36,write_temp=False)
N=sum(f.n for f in ds.filaments)
store=SelectionStore(os.path.join(S,"remove_list.txt")); store.clear(); store.clear_flips()
ov=OverviewWindow(ds,store,ModelParams(ds),gl_enabled=False)
ov._auto_exclude(False)                 # simulate the bool that clicked emits
print(f"_auto_exclude(False): marked={store.count()} / {N}  (was marking ALL {N} before fix)")
store.clear()
ov.btn_autoexcl.click()                 # the real signal path
print(f"button.click():       marked={store.count()} / {N}")
print("FIXED" if 0 < store.count() < N else "STILL BROKEN")
