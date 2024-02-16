from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2
import numpy as np
from pathlib import Path
from boxmot import DeepOCSORT

tracker = DeepOCSORT(
    model_weights=Path('./models/osnet_x0_25_msmt17.pt'),  # which ReID model to use
    device='cpu',
    fp16=False,
)

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path='./models/yolov8n.pt',
    confidence_threshold=0.5,
    device="cpu",  # or 'cuda:0'
)

vid = cv2.VideoCapture(0)
color = (0, 0, 255)  # BGR
thickness = 2
fontscale = 0.5

while True:
    ret, im = vid.read()

    # get sliced predictions
    result = get_sliced_prediction(
        im,
        detection_model,
        slice_height=256,
        slice_width=256,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )
    dets = np.zeros((len(result.object_prediction_list), 6), dtype=np.float32)

    for ind, object_prediction in enumerate(result.object_prediction_list):
        bbox = np.array(object_prediction.bbox.to_xyxy(), dtype=np.float32)
        
        # Check if object_prediction.score has a value attribute
        if hasattr(object_prediction.score, 'value'):
            score_value = object_prediction.score.value
        else:
            # If not, handle it accordingly based on the actual structure
            score_value = float(object_prediction.score)  # Replace this line based on the actual structure
        
        dets[ind, :4] = bbox
        dets[ind, 4] = score_value
        dets[ind, 5] = object_prediction.category.id

    tracks = tracker.update(dets, im)  # --> (x, y, x, y, id, conf, cls, ind)

    if tracks.shape[0] != 0:
        xyxys = tracks[:, 0:4].astype('int')  # float64 to int
        ids = tracks[:, 4].astype('int')  # float64 to int
        confs = tracks[:, 5].round(decimals=2)
        clss = tracks[:, 6].astype('int')  # float64 to int
        inds = tracks[:, 7].astype('int')  # float64 to int

        # print bboxes with their associated id, cls, and conf
        for xyxy, id, conf, cls in zip(xyxys, ids, confs, clss):
            im = cv2.rectangle(
                im,
                (xyxy[0], xyxy[1]),
                (xyxy[2], xyxy[3]),
                color,
                thickness
            )
            cv2.putText(
                im,
                f'id: {id}, conf: {conf}, c: {cls}',
                (xyxy[0], xyxy[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontscale,
                color,
                thickness
            )

    # show image with bboxes, ids, classes, and confidences
    cv2.imshow('frame', im)

    # break on pressing q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
