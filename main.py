# %%
import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# ===============================
# 🧱 配置参数区域
# ===============================
DATASET_URL = "andrewmvd/face-mask-detection"  # kaggle 数据集名称
DATASET_DIR = "dataset"        # 解压数据集的路径
ANNOT_DIR = os.path.join(DATASET_DIR, "annotations")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")

OUTPUT_DIR = "dataset/output"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")
YAML_PATH = "voc.yaml"

CLASS_NAMES = ["with_mask", "without_mask","mask_weared_incorrect"]
TRAIN_RATIO = 0.8

# ===============================
# 📥 下载并解压数据集
# ===============================
def download_dataset(dataset_url, output_dir):
    try:
        import kagglehub
    except ImportError:
        print("❌ 请先安装 kagglehub：pip install kagglehub")
        return

    if os.path.exists(output_dir):
        print(f"✅ 数据集已存在：{os.path.abspath(output_dir)}")
        return

    print("🚀 正在下载数据集中...")
    downloaded_path = kagglehub.dataset_download(dataset_url)
    shutil.copytree(downloaded_path, output_dir)
    shutil.rmtree(downloaded_path)
    print(f"✅ 数据集下载完成：{os.path.abspath(output_dir)}")

# ===============================
# 🔁 VOC → YOLO 格式转换函数
# ===============================
def convert_voc_to_yolo(xml_file, yolo_save_dir, class_names):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    img_width = int(root.find("size/width").text)
    img_height = int(root.find("size/height").text)
    label_lines = []

    for obj in root.findall("object"):
        cls_name = obj.find("name").text
        if cls_name not in class_names:
            continue
        cls_id = class_names.index(cls_name)

        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        # 转换为归一化中心坐标 + 宽高
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        label_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # 保存为 .txt 文件
    txt_file = Path(yolo_save_dir) / (Path(xml_file).stem + ".txt")
    with open(txt_file, "w") as f:
        f.write("\n".join(label_lines))

# ===============================
# 🔀 划分数据集并执行转换
# ===============================
def prepare_dataset(train_ratio):
    # 创建输出文件夹
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    # 加载所有标注文件和对应图像
    xml_files = sorted(Path(ANNOT_DIR).glob("*.xml"))
    img_files = [Path(IMAGES_DIR) / (xml.stem + ".png") for xml in xml_files]

    # 验证文件是否存在
    for img in img_files:
        if not img.exists():
            raise FileNotFoundError(f"未找到图像文件：{img}")

    # 打乱并划分数据集
    data = list(zip(xml_files, img_files))
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    train_data, test_data = data[:split_idx], data[split_idx:]

    def process(data_list, target_dir):
        for xml_path, img_path in data_list:
            shutil.copy(xml_path, target_dir)
            shutil.copy(img_path, target_dir)
            convert_voc_to_yolo(xml_path, target_dir, CLASS_NAMES)

    process(train_data, TRAIN_DIR)
    process(test_data, TEST_DIR)

    print(f"✅ 数据集划分完成，训练集数量：{len(train_data)}，测试集数量：{len(test_data)}")

# ===============================
# 📝 生成 YOLO YAML 配置文件
# ===============================
def create_yaml(path, train_dir, val_dir, class_names):
    with open(path, "w") as f:
        f.write(f"train: {os.path.abspath(train_dir)}\n")
        f.write(f"val: {os.path.abspath(val_dir)}\n\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")
        f.write(f"weights: [1.0, 2.0, 4.0]  # 根据各类别样本量设置逆权重\n")
        f.write(f"sample_weights: True       # 启用加权随机采样\n")
    print(f"📄 已生成配置文件：{path}")

# ===============================
# 🚀 主执行入口
# ===============================

download_dataset(DATASET_URL, DATASET_DIR)
prepare_dataset(TRAIN_RATIO)
create_yaml(YAML_PATH, TRAIN_DIR, TEST_DIR, CLASS_NAMES)


# %%
# !pip install ultralytics
# !nvidia-smi
from ultralytics import YOLO
device = 'cuda'  # 使用GPU训练,可选cuda或cpu

model = YOLO("baseModel/yolov8n.pt")  # 使用预训练模型
model.train(
    data="voc.yaml",
    device=0 if device == "cuda" else "cpu",
    epochs=100,                  # 增加总训练轮次
    batch=32,                    # 减小batch size以适应更大分辨率
    imgsz=800,                   # 优化显存使用（800->640）
    optimizer="AdamW",
    lr0=0.0005,                    # 降低初始学习率
    lrf=0.005,                    # 余弦退火最终学习率
    warmup_epochs=3,   # 新增学习率预热
    weight_decay=0.05,           # 添加权重衰减防止过拟合

    # 数据增强强化
    augment=True,
    hsv_h=0.3,                   # 增强色调扰动
    hsv_s=0.6,                   # 增强饱和度扰动
    translate=0.2,               # 增大平移幅度
    scale=0.5,                   # 扩大缩放范围
    shear=0.3,                   # 增大剪切幅度
    mosaic=1.0,                  # 全程开启mosaic
    close_mosaic=15,             # 最后15个epoch关闭mosaic稳定训练

    # 损失函数调整
    cls=3.0,                     # 增大分类损失权重
    box=1.5,                     # 增大框回归损失权重
    dfl=1.5,                     # 增大点框损失权重
    # fl_gamma=2.0,                # 聚焦困难样本（类似focal loss）

    # obj=1.5,                     # 适当增大目标存在损失权重
    copy_paste=0.3,              # 新增复制粘贴增强（特别针对少数类）
    mixup=0.2,       # 新增MixUp增强（默认未启用）
    # 类别平衡策略
    # 需在data.yaml中添加：
    #   weights: [1.0, 1.0, 2.0, 4.0]  # 按类别样本量倒数设置权重
    #   sample_weights: True           # 启用样本加权采样

    # 训练控制
    patience=15,                  # 延长早停观察期
    dropout=0.3,                 # 添加Dropout正则化
    amp=True,                    # 保持混合精度训练
    pretrained=True,
    save=True,
    exist_ok=True,
)

# %%
# 预测输出
import os
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import matplotlib.pyplot as plt

# ------------ 全局配置 ------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "runs/detect/train/weights/best.pt"
model = YOLO(MODEL_PATH)
INPUT_PATH = "dataset/output/test/"  # 输入路径,可以为图片,视频,文件夹,摄像头编号
# INPUT_PATH = "dataset/output/video/test.mp4"  # 输入路径,可以为图片,视频,文件夹,摄像头编号
# INPUT_PATH=0

SAVE = True  # 是否保存预测结果
OUTPUT_PATH = "predict/"  # 预测结果保存路径

# ------------ 工具函数 ------------
def draw_boxes_pil(image, results):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls)
        conf = float(box.conf)
        label = f"{model.names[cls_id]} {conf:.2f}"

        text_bbox = font.getbbox(label)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill="red")
        draw.text((x1, y1 - text_h), label, fill="white", font=font)

    return image

def save_image(image, save_path, origin_path=None):
    if os.path.isdir(save_path):
        filename = os.path.basename(origin_path)
        save_path = os.path.join(save_path, filename)
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)
    print(f"✅ 已保存图片: {save_path}")

# ------------ 单图预测 ------------
def predict_image(image_path, save=False, save_path=None):
    image = Image.open(image_path).convert("RGB")
    results = model.predict(image_path, imgsz=640, device=DEVICE)
    image = draw_boxes_pil(image, results)

    plt.imshow(image)
    plt.axis("off")
    plt.title("预测结果")
    plt.show()

    if save and save_path:
        save_image(image, save_path, origin_path=image_path)

# ------------ 视频预测 ------------
def predict_video(video_path, save=False, save_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 视频文件无法打开")
        return

    if save:
        if os.path.isdir(save_path):
            filename = os.path.basename(video_path)
            save_path = os.path.join(save_path, f"{os.path.splitext(filename)[0]}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps, w, h = cap.get(5), int(cap.get(3)), int(cap.get(4))
        out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, imgsz=640, device=DEVICE)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls)
            conf = float(box.conf)
            label = f"{model.names[cls_id]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("预测中 - 按 Q 退出", frame)
        if save:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if save:
        out.release()
        print(f"✅ 已保存视频: {save_path}")
    cv2.destroyAllWindows()

# ------------ 文件夹批量图片 ------------
def predict_folder(folder_path, save=False, output_dir=None):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                img_path = os.path.join(root, file)
                image = Image.open(img_path).convert("RGB")
                results = model.predict(img_path, imgsz=640, device=DEVICE)
                image = draw_boxes_pil(image, results)

                if save and output_dir:
                    rel_path = os.path.relpath(img_path, folder_path)
                    save_path = os.path.join(output_dir, rel_path)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    image.save(save_path)

    if save:
        print(f"✅ 文件夹预测完成，结果已保存至: {output_dir}")

# ------------ 摄像头实时预测 ------------
def predict_camera(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头 {index}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, imgsz=640, device=DEVICE)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls)
            conf = float(box.conf)
            label = f"{model.names[cls_id]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("摄像头预测 - 按 Q 退出", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ------------ 总入口函数 ------------
def run_predict(path, save=False, save_path=None):
    if isinstance(path, int):
        predict_camera(index=path)
    elif os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            predict_image(path, save, save_path)
        elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
            predict_video(path, save, save_path)
    elif os.path.isdir(path):
        predict_folder(path, save, save_path)
    else:
        print("❌ 无效路径，请确认输入正确的图片/视频/文件夹/摄像头编号")

# ------------ 示例调用 ------------
run_predict(INPUT_PATH, SAVE, OUTPUT_PATH)      # 预测输出


