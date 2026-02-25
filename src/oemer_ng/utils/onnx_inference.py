#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import onnxruntime as ort


def load_onnx_model(model_path, verbose=True):
    """加载ONNX模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    if verbose:
        print(f"[1/4] 加载ONNX模型: {model_path}")
    
    # 创建ONNX Runtime会话
    providers = ['CPUExecutionProvider']
    if ort.get_device() == 'GPU':
        providers.insert(0, 'CUDAExecutionProvider')
    
    session = ort.InferenceSession(model_path, providers=providers)
    
    # 获取模型输入输出信息
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    
    if verbose:
        print(f"      模型加载成功")
        print(f"      输入节点: {input_name}")
        print(f"      输出节点: {output_names}")
    
    return session, input_name, output_names


def preprocess_image(image_path, target_size=640):
    """预处理图像用于ONNX推理"""
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    
    # 保存原始尺寸
    original_shape = img.shape[:2]
    
    # Resize到目标尺寸，保持宽高比
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 创建目标尺寸的画布并居中放置
    img_padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    img_padded[top:top+new_h, left:left+new_w] = img_resized
    
    # 转换为RGB并归一化
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    
    # 转换为NCHW格式 (batch, channels, height, width)
    img_transposed = np.transpose(img_norm, (2, 0, 1))
    img_batch = np.expand_dims(img_transposed, axis=0)
    
    return img_batch, original_shape, (scale, top, left)


def postprocess_outputs(outputs, original_shape, preprocess_params, conf_threshold=0.25, iou_threshold=0.45):
    """后处理ONNX模型输出"""
    scale, pad_top, pad_left = preprocess_params
    orig_h, orig_w = original_shape
    
    # outputs通常是 [1, num_boxes, 85] 或 [1, 84, num_boxes] 格式
    # 85 = x, y, w, h, objectness + 80 classes
    output = outputs[0]
    
    # 如果是[1, 84, 8400]格式，转置为[1, 8400, 84]
    if len(output.shape) == 3 and output.shape[1] < output.shape[2]:
        output = np.transpose(output, (0, 2, 1))
    
    output = output[0]  # 移除batch维度
    
    boxes = []
    for detection in output:
        # YOLOv8格式: [x_center, y_center, width, height, class_scores...]
        if len(detection) < 5:
            continue
        
        x_center, y_center, width, height = detection[:4]
        class_scores = detection[4:]
        
        # 获取最高分数的类别
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]
        
        if confidence < conf_threshold:
            continue
        
        # 转换坐标：从模型空间到原始图像空间
        x1 = (x_center - width / 2 - pad_left) / scale
        y1 = (y_center - height / 2 - pad_top) / scale
        x2 = (x_center + width / 2 - pad_left) / scale
        y2 = (y_center + height / 2 - pad_top) / scale
        
        # 限制在图像范围内
        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        x2 = max(0, min(x2, orig_w))
        y2 = max(0, min(y2, orig_h))
        
        boxes.append({
            'xyxy': np.array([x1, y1, x2, y2]),
            'cls': class_id,
            'conf': confidence
        })
    
    # NMS (非极大值抑制)
    if len(boxes) > 0:
        boxes = non_max_suppression(boxes, iou_threshold)
    
    return boxes


def calculate_iou(box1, box2):
    """计算两个框的IoU"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    
    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def non_max_suppression(boxes, iou_threshold):
    """非极大值抑制"""
    if len(boxes) == 0:
        return []
    
    # 按置信度排序
    boxes = sorted(boxes, key=lambda x: x['conf'], reverse=True)
    
    keep = []
    while len(boxes) > 0:
        current = boxes.pop(0)
        keep.append(current)
        
        boxes = [
            box for box in boxes
            if box['cls'] != current['cls'] or
            calculate_iou(current['xyxy'], box['xyxy']) < iou_threshold
        ]
    
    return keep


def detect_objects(session, input_name, output_names, image_path, conf=0.25, iou=0.45, imgsz=640):
    """使用ONNX模型检测对象"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    print(f"\n[2/4] 检测对象...")
    
    # 预处理
    input_data, original_shape, preprocess_params = preprocess_image(image_path, imgsz)
    
    # 推理
    outputs = session.run(output_names, {input_name: input_data})
    
    # 后处理
    boxes = postprocess_outputs(outputs, original_shape, preprocess_params, conf, iou)
    
    print(f"      检测到 {len(boxes)} 个对象")
    
    return boxes


def merge_overlapping_boxes(boxes, iou_threshold=0.3):
    """合并重叠的边界框"""
    if len(boxes) == 0:
        return []
    
    box_list = []
    for box in boxes:
        x1, y1, x2, y2 = box['xyxy']
        cls_id = int(box['cls'])
        conf = float(box['conf'])
        box_list.append([float(x1), float(y1), float(x2), float(y2), cls_id, conf])
    
    box_list.sort(key=lambda x: x[5], reverse=True)
    
    merged = []
    used = [False] * len(box_list)
    
    for i in range(len(box_list)):
        if used[i]:
            continue
        
        current = box_list[i]
        x1, y1, x2, y2, cls_id, conf = current
        overlapping = [current]
        
        for j in range(i + 1, len(box_list)):
            if used[j]:
                continue
            
            other = box_list[j]
            ox1, oy1, ox2, oy2, other_cls, other_conf = other
            
            if cls_id != other_cls:
                continue
            
            iou = calculate_iou([x1, y1, x2, y2], [ox1, oy1, ox2, oy2])
            
            if iou > iou_threshold:
                overlapping.append(other)
                used[j] = True
        
        if len(overlapping) > 1:
            all_x1 = [b[0] for b in overlapping]
            all_y1 = [b[1] for b in overlapping]
            all_x2 = [b[2] for b in overlapping]
            all_y2 = [b[3] for b in overlapping]
            
            merged_box = [
                min(all_x1), min(all_y1), max(all_x2), max(all_y2),
                cls_id, max([b[5] for b in overlapping])
            ]
            merged.append(merged_box)
        else:
            merged.append(current)
        
        used[i] = True
    
    return merged


def extract_and_paste_regions(image_path, merged_boxes, save_path, verbose=True):
    """将检测区域切出并贴到白色背景"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    
    h, w = img.shape[:2]
    white_canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    if verbose:
        print(f"\n[3/4] 提取和粘贴 {len(merged_boxes)} 个区域...")
    
    for i, box in enumerate(merged_boxes):
        x1, y1, x2, y2, cls_id, conf = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        region = img[y1:y2, x1:x2].copy()
        white_canvas[y1:y2, x1:x2] = region
        
        if verbose:
            print(f"      区域 {i+1}: ({x1}, {y1}) -> ({x2}, {y2}), 类别: {cls_id}, 置信度: {conf:.3f}")
    
    cv2.imwrite(save_path, white_canvas)
    if verbose:
        print(f"\n[4/4] 已保存清理图像: {save_path}")
    
    return white_canvas


def cleanup_image_with_onnx(
    image: np.ndarray,
    session,
    input_name: str,
    output_names: list,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    merge_iou_threshold: float = 0.3,
    image_size: int = 640,
    verbose: bool = False
) -> np.ndarray:
    """
    使用 ONNX 模型清理图像的包装函数
    
    Args:
        image: 输入图像 (RGB 格式)
        session: ONNX Runtime 会话
        input_name: 模型输入节点名称
        output_names: 模型输出节点名称列表
        conf_threshold: 置信度阈值
        iou_threshold: NMS 的 IoU 阈值
        merge_iou_threshold: 合并框的 IoU 阈值
        image_size: ONNX 推理的图像尺寸
        verbose: 是否打印详细信息
        
    Returns:
        清理后的图像 (RGB 格式)
    """
    try:
        # 转换 RGB 到 BGR
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 保存临时图像
        temp_input = 'temp_onnx_input.png'
        temp_output = 'temp_onnx_output.png'
        cv2.imwrite(temp_input, image_bgr)
        
        # 预处理
        input_data, original_shape, preprocess_params = preprocess_image(temp_input, image_size)
        
        # 推理
        outputs = session.run(output_names, {input_name: input_data})
        
        # 后处理
        boxes = postprocess_outputs(
            outputs, original_shape, preprocess_params,
            conf_threshold, iou_threshold
        )
        
        if verbose:
            print(f"      检测到 {len(boxes)} 个原始框")
        
        # 合并重叠框
        merged_boxes = merge_overlapping_boxes(boxes, merge_iou_threshold)
        
        if verbose:
            print(f"      合并后剩余 {len(merged_boxes)} 个区域")
        
        # 提取并粘贴到白色背景
        cleaned_bgr = extract_and_paste_regions(
            temp_input, merged_boxes, temp_output, verbose=verbose
        )
        
        # 转换回 RGB
        cleaned_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)
        
        # 清理临时文件
        try:
            os.remove(temp_input)
            os.remove(temp_output)
        except:
            pass
        
        if verbose:
            print(f"      ONNX 清理完成")
        
        return cleaned_rgb
        
    except Exception as e:
        if verbose:
            print(f"      ONNX 清理失败: {e}")
        # 清理临时文件
        for temp_file in ['temp_onnx_input.png', 'temp_onnx_output.png']:
            try:
                os.remove(temp_file)
            except:
                pass
        return image


def main():
    """主函数：ONNX模型推理 -> 图像清理"""
    print("=" * 70)
    print("ONNX模型推理 + 图像清理")
    print("=" * 70)
    
    # 配置参数
    INPUT_IMAGE = 'qcwq.png'
    MODEL_PATH = 'src\\resize_model\\best.onnx'
    OUTPUT_IMAGE = 'qcwq_clean.png'
    CONFIDENCE_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45
    MERGE_IOU_THRESHOLD = 0.3
    IMAGE_SIZE = 640
    
    try:
        # 检查输入文件
        if not os.path.exists(INPUT_IMAGE):
            print(f"\n错误: 输入文件不存在: {INPUT_IMAGE}")
            print("请修改 INPUT_IMAGE 变量指向有效的图像文件")
            return
        
        if not os.path.exists(MODEL_PATH):
            print(f"\n错误: 模型文件不存在: {MODEL_PATH}")
            print("请修改 MODEL_PATH 变量指向有效的ONNX模型文件")
            return
        
        # 步骤1: 加载ONNX模型
        session, input_name, output_names = load_onnx_model(MODEL_PATH)
        
        # 步骤2: 检测对象
        boxes = detect_objects(session, input_name, output_names, INPUT_IMAGE, 
                           CONFIDENCE_THRESHOLD, IOU_THRESHOLD, IMAGE_SIZE)
        
        print(f"      检测到 {len(boxes)} 个原始框")
        
        # 步骤3: 合并重叠框
        merged_boxes = merge_overlapping_boxes(boxes, MERGE_IOU_THRESHOLD)
        print(f"      合并后剩余 {len(merged_boxes)} 个区域")
        
        # 步骤4: 提取并粘贴到白色背景
        extract_and_paste_regions(INPUT_IMAGE, merged_boxes, OUTPUT_IMAGE)
        
        # 输出结果
        print("\n" + "=" * 70)
        print("处理完成！")
        print("=" * 70)
        print(f"✓ 清理图像已保存: {OUTPUT_IMAGE}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
