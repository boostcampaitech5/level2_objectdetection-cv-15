from ensemble_boxes import nms, soft_nms, weighted_boxes_fusion
import pandas as pd
import os
from tqdm import tqdm

# codebook 내부 ensemble 폴더에 들어있는 csv파일들을 모두 가져옴
file_list = os.listdir("./ensemble")
csv_list = []
for file_name in file_list :
    if file_name.endswith('.csv'):
        csv_list.append(pd.read_csv(os.path.join("./ensemble", file_name), keep_default_na=False))

# sum할 pd를 그냥 첫번째 파일에서 형식을 가져온 후 그곳에 덮어씀
csv_sum = pd.read_csv(os.path.join("./ensemble", file_list[0]), keep_default_na=False)

# WBF hyperparameter 
iou_thr = 0.5
skip_box_thr = 0.0001

for img_num in tqdm(range(csv_sum.shape[0])) :
    boxes = []
    scores = []
    labels = []
    for csv in csv_list :
        box = []
        score = []
        label = []
        predictions = csv.loc[img_num]["PredictionString"].split(" ")
        predictions.pop()
        objects = [predictions[n:n+6] for n in range(0, len(predictions), 6)]
        for obj in objects :
            base_box = []
            label.append(int(obj[0]))
            score.append(float(obj[1]))
            base_box.append(float(obj[2]) / 1024)
            base_box.append(float(obj[3]) / 1024)
            base_box.append(float(obj[4]) / 1024)
            base_box.append(float(obj[5]) / 1024)
            box.append(base_box)
        if box != []:
            boxes.append(box)
        if score != []:
            scores.append(score)
        if label != []:
            labels.append(label)
    result_box, result_score, result_label = weighted_boxes_fusion(boxes, scores, labels, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    # result_box, result_score, result_label = soft_nms(boxes, scores, labels, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)
    prediction_string = ""
    result_box *= 1024
    for i in range(len(result_label)):
        prediction_string += str(int(result_label[i])) + " "
        prediction_string += str(result_score[i]) + " "
        prediction_string += str(result_box[i][0]) + " " + str(result_box[i][1]) + " " + str(result_box[i][2]) + " " + str(result_box[i][3]) + " "

    csv_sum.loc[img_num]["PredictionString"] = prediction_string
csv_sum.to_csv("/ensemble_WBF_withRetina.csv", index=None)