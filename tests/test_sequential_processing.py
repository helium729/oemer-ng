#!/usr/bin/env python3
"""
测试：先后使用 onnx_inference.py 和 preprocessing.py 处理图片
"""

import sys
from pathlib import Path

# 添加 src 目录到 Python 路径
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

import cv2
import numpy as np
from oemer_ng.utils.onnx_inference import (
    load_onnx_model,
    preprocess_image as onnx_preprocess,
    postprocess_outputs,
    merge_overlapping_boxes,
    extract_and_paste_regions
)
from oemer_ng.utils.preprocessing import ImagePreprocessor


def main():
    """主函数：先后顺序处理"""
    print("=" * 70)
    print("步骤1: ONNX 清理")
    print("步骤2: 预处理（Resize + 归一化）")
    print("=" * 70)
    
    # 配置参数
    INPUT_IMAGE = 'qcwq.png'
    ONNX_MODEL_PATH = 'src\\resize_model\\best.onnx'
    ONNX_OUTPUT = 'qcwq_onnx_clean.png'
    FINAL_OUTPUT = 'qcwq_final.png'
    
    # ONNX 参数
    CONFIDENCE_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45
    MERGE_IOU_THRESHOLD = 0.3
    IMAGE_SIZE = 640
    
    # 预处理参数
    TARGET_SIZE = (512, 512)
    
    try:
        # 检查输入文件
        if not Path(INPUT_IMAGE).exists():
            print(f"\n❌ 错误: 输入文件不存在: {INPUT_IMAGE}")
            return
        
        if not Path(ONNX_MODEL_PATH).exists():
            print(f"\n❌ 错误: 模型文件不存在: {ONNX_MODEL_PATH}")
            return
        
        # ============================================================
        # 步骤1: ONNX 清理
        # ============================================================
        print(f"\n{'='*70}")
        print("步骤 1/2: ONNX 清理")
        print(f"{'='*70}")
        
        # 加载 ONNX 模型
        print(f"\n[1.1] 加载 ONNX 模型: {ONNX_MODEL_PATH}")
        session, input_name, output_names = load_onnx_model(ONNX_MODEL_PATH)
        
        # 预处理
        print(f"\n[1.2] 预处理图像...")
        input_data, original_shape, preprocess_params = onnx_preprocess(
            INPUT_IMAGE, IMAGE_SIZE
        )
        print(f"      原始尺寸: {original_shape}")
        print(f"      缩放比例: {preprocess_params[0]:.4f}")
        
        # 推理
        print(f"\n[1.3] ONNX 推理...")
        outputs = session.run(output_names, {input_name: input_data})
        print(f"      推理完成")
        
        # 后处理
        print(f"\n[1.4] 后处理检测结果...")
        boxes = postprocess_outputs(
            outputs, original_shape, preprocess_params,
            CONFIDENCE_THRESHOLD, IOU_THRESHOLD
        )
        print(f"      检测到 {len(boxes)} 个原始框")
        
        # 合并重叠框
        print(f"\n[1.5] 合并重叠框...")
        merged_boxes = merge_overlapping_boxes(boxes, MERGE_IOU_THRESHOLD)
        print(f"      合并后剩余 {len(merged_boxes)} 个区域")
        
        # 提取并粘贴到白色背景
        print(f"\n[1.6] 提取并粘贴到白色背景...")
        cleaned_image = extract_and_paste_regions(
            INPUT_IMAGE, merged_boxes, ONNX_OUTPUT
        )
        print(f"      已保存: {ONNX_OUTPUT}")
        print(f"      清理后尺寸: {cleaned_image.shape}")
        
        # ============================================================
        # 步骤2: 预处理（Resize + 归一化）
        # ============================================================
        print(f"\n{'='*70}")
        print("步骤 2/2: 预处理（Resize + 归一化）")
        print(f"{'='*70}")
        
        # 创建预处理器
        print(f"\n[2.1] 创建预处理器...")
        preprocessor = ImagePreprocessor(
            target_size=TARGET_SIZE,
            normalize=True
        )
        print(f"      目标尺寸: {TARGET_SIZE}")
        print(f"      归一化: True")
        
        # 读取 ONNX 清理后的图像
        print(f"\n[2.2] 读取 ONNX 清理后的图像...")
        cleaned_bgr = cv2.imread(ONNX_OUTPUT)
        cleaned_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)
        print(f"      图像尺寸: {cleaned_rgb.shape}")
        
        # 执行预处理
        print(f"\n[2.3] 执行预处理...")
        preprocessed = preprocessor.preprocess(cleaned_rgb, return_tensor=False)
        print(f"      预处理完成")
        print(f"      输出形状: {preprocessed.shape}")
        print(f"      输出类型: {preprocessed.dtype}")
        print(f"      输出范围: [{preprocessed.min():.2f}, {preprocessed.max():.2f}]")
        
        # 保存最终结果
        print(f"\n[2.4] 保存最终结果...")
        
        # 转换回 uint8
        if preprocessed.dtype == np.float32 or preprocessed.dtype == np.float64:
            preprocessed_uint8 = ((preprocessed * 255.0).clip(0, 255)).astype(np.uint8)
        else:
            preprocessed_uint8 = preprocessed
        
        # 从 (C, H, W) 转换回 (H, W, C)
        if len(preprocessed_uint8.shape) == 3 and preprocessed_uint8.shape[0] == 3:
            preprocessed_uint8 = np.transpose(preprocessed_uint8, (1, 2, 0))
        
        cv2.imwrite(FINAL_OUTPUT, cv2.cvtColor(preprocessed_uint8, cv2.COLOR_RGB2BGR))
        print(f"      已保存: {FINAL_OUTPUT}")
        
        # ============================================================
        # 输出结果汇总
        # ============================================================
        print(f"\n{'='*70}")
        print("处理完成！")
        print(f"{'='*70}")
        print(f"\n处理流程:")
        print(f"  1. 输入图像: {INPUT_IMAGE}")
        print(f"  2. ONNX 清理: {ONNX_OUTPUT}")
        print(f"  3. 预处理: {FINAL_OUTPUT}")
        print(f"\n图像尺寸变化:")
        print(f"  原始: {cv2.imread(INPUT_IMAGE).shape}")
        print(f"  ONNX 清理后: {cleaned_image.shape}")
        print(f"  最终: {preprocessed_uint8.shape}")
        print(f"{'='*70}")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
