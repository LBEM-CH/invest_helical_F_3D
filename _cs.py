import os; os.environ["QT_QPA_PLATFORM"]="offscreen"
from PyQt6 import QtWidgets
from dynamo_table import load_dataset
from selection_store import SelectionStore
from plot_common import ModelParams
from overview_window import OverviewWindow
P="/mnt/storage/data3/users/wen-lu/tau/warp/position_8_5/dynamo/dynamo_project/abp_align_eo/"
S="/tmp/claude-193122/-home-wlchung-LBEM-invest-helical-F-3D/2971e26b-8026-4193-869d-3d22f7491404/scratchpad"
app=QtWidgets.QApplication([]); ds=load_dataset(P,"dynamo",None,0.7,4.75,19.36,write_temp=False)
store=SelectionStore(os.path.join(S,"remove_list.txt")); store.clear()
ov=OverviewWindow(ds,store,ModelParams(ds),gl_enabled=False)
ov.btn_autoexcl.click(); print(f"after auto-exclude: marked={store.count()}")
ov.btn_clearsel.click(); print(f"after clear selection ({ov.btn_clearsel.text()}): marked={store.count()} (0 expected)")
print("row:", " | ".join([ov.btn_tiltall.text(), ov.btn_rotall.text(), ov.btn_autoexcl.text(), ov.btn_clearsel.text()]))
