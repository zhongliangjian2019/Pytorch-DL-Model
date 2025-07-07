"""
@brief 分类模型测试与评估

*主要评价指标
    (1) 指标类：准确率/精确率/召回率/F1-Score
    (2) 曲线类：PR曲线/ROC曲线/DR曲线

*文件数据格式：
    文件格式: txt
    文件数据：
    第1行：记录分类标签, 例如 label0, label1, ...
    第2行及以后：记录信息, 字段1：文件名（非文件全路径）, 字段2：真实类别标签(str), 字段3：预测类别标签(str),
                        字段4及以后依次为每个类别的预测得分
    例如：
        第2行：filename_0, y_true, y_pre, score_0, score_1, score_2, ...
        ......
        第n行：filename_0, y_true, y_pre, score_0, score_1, score_2, ...
"""

import cv2
import numpy as np
import os
import shutil
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn
import matplotlib.pyplot as plt

class ClassifyTestor(object):
    """分类模型推理"""
    def __init__(self, model_path: str, input_shape: tuple, labels: list):
        self.model = cv2.dnn.readNetFromONNX(model_path)
        self.in_channel = input_shape[0]
        self.in_size = input_shape[1:]
        self.datas = {"labels": labels,
                      "filename": [],
                      "y_true": [],
                      "y_pred": [],
                      "scores": []}

    def Inference(self, image: np.ndarray):
        """Model inference"""
        input_blob = cv2.dnn.blobFromImage(image, 1.0/255.0, self.in_size, swapRB=True)
        self.model.setInput(input_blob)
        output_blob = self.model.forward()
        return output_blob

    def Preprocess(self, image_path: str):
        """Model preprocess"""
        # Read image
        image = cv2.imdecode(np.fromfile(image_path, np.uint8), 1)

        # Check channel number
        if image.ndim == 2 and self.in_channel == 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Check input size
        if image.shape[0] != self.in_size[0] or image.shape[1] != self.in_size[1]:
            image = cv2.resize(image, self.in_size)

        return image
    def Postprocess(self, output_blob: np.ndarray):
        """
        @brief Model post-process
        @param output_blob: model output result, shape is (N, 2), N is batch size, 2 is class_index and class_score
        """
        # output = BTF.softmax(output_blob, axis=1)
        output = output_blob
        y_pred_idx = np.argmax(output, axis=1)[0]
        scores = output[0, :]
        return y_pred_idx, scores.tolist()

    def Predict(self, image_path: str):
        """Get classification result"""
        image = self.Preprocess(image_path)
        predict = self.Inference(image)
        y_pred, scores = self.Postprocess(predict)
        return y_pred, scores

    def GetResult(self, data_dir: str):
        """Get predict result"""
        img_paths = []
        cls_labels = []
        with open(os.path.join(data_dir, 'test.txt'), 'r', encoding='utf-8') as file:
            for line in file.readlines():
                label, path = line.split(';')
                img_paths.append(path.rstrip('\n'))
                cls_labels.append(int(label))

        cls_names = self.datas['labels']
        for y_true, file in zip(cls_labels, img_paths):
            y_pred, scores = self.Predict(file)
            self.datas["filename"].append(file)
            self.datas["y_true"].append(cls_names[y_true])
            self.datas["y_pred"].append(cls_names[y_pred])
            self.datas["scores"].append(scores)

        save_path = os.path.join(data_dir, "result.txt")
        with open(save_path, 'w') as file:
            file.write(','.join(self.datas['labels']) + '\n')
            for filename, y_true, y_pred, scores in zip(self.datas["filename"], self.datas["y_true"],
                                                        self.datas["y_pred"], self.datas["scores"]):
                scores_str = ",".join(["%.2f" % (score) for score in scores])
                file.write("%s,%s,%s,%s\n" % (filename, y_true, y_pred, scores_str))

class Evaluater(object):
    """分类模型评估器"""
    def __init__(self):
        self.output_dir = None
        self.datas = None
        self.confusion_matrix = None

    def setEvaluater(self, predict_file: str, output_dir: str):
        """设置评估器"""
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.datas = self.getDataInformation(predict_file)
        self.confusion_matrix = self.getConfusionMatrix(self.datas)

    def getDataInformation(self, predict_file: str):
        """从文件解析数据信息"""
        datas = {"labels": [],
                  "filename": [],
                  "y_true": [],
                  "y_pred": [],
                  "scores": []}
        with open(predict_file, 'r') as file:
            line = file.readline().strip('\n')
            datas["labels"] = line.split(',')
            line = file.readline().strip('\n')
            while line:
                infor = line.split(',')
                datas["filename"].append(infor[0])
                datas["y_true"].append(infor[1])
                datas["y_pred"].append(infor[2])
                datas["scores"].append([float(score) for score in infor[3:]])
                line = file.readline().strip('\n')
        return datas

    def getConfusionMatrix(self, datas):
        """获取混淆矩阵"""
        return confusion_matrix(y_true=datas["y_true"], y_pred=datas["y_pred"], labels=datas["labels"])

    def getPrecisionScore(self):
        """获取分类精确率"""
        scores = {}
        for idx, pos_label in enumerate(self.datas["labels"]):
            scores[pos_label] = self.confusion_matrix[idx, idx] / self.confusion_matrix[:, idx].sum()
        return scores

    def getRecallScore(self):
        """获取分类召回率"""
        scores = {}
        for idx, pos_label in enumerate(self.datas["labels"]):
            scores[pos_label] = self.confusion_matrix[idx, idx] / self.confusion_matrix[idx, :].sum()
        return scores

    def getAccuracyScore(self):
        """获取分类准确度"""
        return np.trace(self.confusion_matrix) / np.sum(self.confusion_matrix)

    def getF1Score(self, precisions: dict, recalls: dict):
        """获取F1得分"""
        scores = {}
        for label in self.datas['labels']:
            scores[label] = 2 * (precisions[label] * recalls[label]) / (precisions[label] + recalls[label])
        return scores

    def paintPrecisionRecallCurve(self):
        """绘制精度-召回曲线"""
        pr_infos = {}
        for idx, label in enumerate(self.datas["labels"]):
            probas_pred = np.array(self.datas["scores"])
            precision, recall, thresholds = precision_recall_curve(
                y_true=self.datas["y_true"], probas_pred=probas_pred[:, idx], pos_label=label)
            pr_infos[label] = (precision, recall, thresholds)

        # Precision Curve
        plt.figure()
        for pos_label in self.datas["labels"]:
            precision, recall, thresholds = pr_infos[pos_label]
            thresh = thresholds.tolist()
            thresh.append(1.0)
            plt.plot(thresh, precision, label=pos_label)
        plt.title("Precision Curve")
        plt.xlabel("threshold")
        plt.ylabel("precision")
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "precision_curve.png"))

        # Recall Curve
        plt.figure()
        for pos_label in self.datas["labels"]:
            precision, recall, thresholds = pr_infos[pos_label]
            thresh = thresholds.tolist()
            thresh.append(1.0)
            plt.plot(thresh, recall, label=pos_label)
        plt.title("Recall Curve")
        plt.xlabel("threshold")
        plt.ylabel("recall")
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "recall_curve.png"))

        # PR Curve
        plt.figure()
        for pos_label in self.datas["labels"]:
            precision, recall, thresholds = pr_infos[pos_label]
            plt.plot(recall, precision, label=pos_label)
        plt.title("PR Curve")
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "pr_curve.png"))

    def paintRocCurve(self):
        """绘制ROC曲线"""
        plt.figure()
        for idx, label in enumerate(self.datas["labels"]):
            probas_pred = np.array(self.datas["scores"])
            fpr, tpr, _ = roc_curve(y_true=self.datas["y_true"], y_score=probas_pred[:, idx], pos_label=label)
            auc_score = auc(fpr, tpr)
            auc_infor = label + " AUC(%.4f)" % (auc_score)
            plt.plot(fpr, tpr, label=auc_infor)

        plt.title("ROC Curve")
        plt.xlabel("fpr")
        plt.ylabel("tpr")
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        save_path = os.path.join(self.output_dir, "roc_curve.png")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)

    def paintConfusionMatrix(self):
        """绘制混淆矩阵"""
        plt.figure()
        seaborn.heatmap(self.confusion_matrix, annot=True, fmt='d',
                        xticklabels=self.datas["labels"], yticklabels=self.datas["labels"])
        plt.xlabel("predicted")
        plt.ylabel("actual")
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(self.output_dir, "confusion_matrix.png"))

    def mistakeClassifySave(self):
        """错误分类图像保存"""
        labels = self.datas["labels"]
        indexs = {labels[i]: i for i in range(len(labels))}
        for label in labels:
            save_dir = os.path.join(self.output_dir, label + "_to_other")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        for file, y_true, y_pred, scores in zip(self.datas["filename"], self.datas["y_true"], self.datas["y_pred"], self.datas["scores"]):
            if y_true == y_pred:
                continue
            src_path = file
            dst_path = os.path.join(self.output_dir, y_true + "_to_other", y_pred + "_%.2f_" % (scores[indexs[y_pred]]) + os.path.basename(file))
            shutil.copyfile(src_path, dst_path)

    def run(self):
        """运行评估器"""
        # 获取关键测量指标
        accuracy_score = self.getAccuracyScore()

        precision_scores = self.getPrecisionScore()

        recall_scores = self.getRecallScore()

        f1_scores = self.getF1Score(precision_scores, recall_scores)

        # 指标输出
        metric_file = os.path.join(self.output_dir, "metric.txt")
        with open(metric_file, mode='w') as file:
            file.write("Model Metric Result\n")
            file.write("Accuracy: %.4f\n" % (accuracy_score))
            for label in self.datas["labels"]:
                file.write("Name: %s, Precision: %.4f, Recall: %.4f, F1-score: %.4f\n" %
                           (label, precision_scores[label], recall_scores[label], f1_scores[label]))

        # 绘制关键曲线
        self.paintConfusionMatrix()

        self.paintPrecisionRecallCurve()

        self.paintRocCurve()

        # 错误分类保存
        self.mistakeClassifySave()


if __name__ == '__main__':
    data_dir = '../data'
    model_path = '../export/mbv3_small.onnx'
    input_shape = (3, 224, 224)
    class_labels = ["class1", "class2", "class3"]

    # 运行模型推理
    testor = ClassifyTestor(model_path=model_path, input_shape=input_shape, labels=class_labels)
    testor.GetResult(data_dir=data_dir)

    # 进行模型评估
    output_dir = os.path.join(data_dir, 'metrics_result')
    evaluate = Evaluater()
    evaluate.setEvaluater(os.path.join(data_dir, 'result.txt'), output_dir)
    evaluate.run()










