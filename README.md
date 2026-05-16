cd "/home/user/capstone/capstone_design-ui/capstone_design-main(1)/capstone_design-main/main"

DETECTION_BACKEND_URL=https://detection-demo-production.up.railway.app \
DETECTION_TRANSPORT=websocket \
DETECTION_LOG_SEND=1 \
python3 pruned_yolo_cctv.py
