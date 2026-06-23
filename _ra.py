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
ov.btn_autoflip.click(); print(f"after auto-flip all: flips={store.flip_count()}")
ov.btn_resumeflip.click(); print(f"after resume all:    flips={store.flip_count()} (0 expected)")
print(f"row: auto-flip all | {ov.btn_resumeflip.text()} | {ov.btn_autoexcl.text()} | {ov.btn_invert.text()}")
