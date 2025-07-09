import os
import cv2
from ultralytics import YOLO
from pathlib import Path


class GearClassificationTester:
    def __init__(self, model_path="best.pt"):
        """
        初始化齿轮缺陷分类测试器
        
        Args:
            model_path: 模型文件路径
        """
        try:
            self.model = YOLO(model_path)
            print(f"✅ 成功加载模型: {model_path}")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            # 如果best.pt不存在，尝试使用预训练模型
            print("尝试使用预训练模型...")
            self.model = YOLO('yolo11s-cls.pt')
        
        # 定义类别名称
        self.class_names = {
            0: 'indentation',  # 压痕
            1: 'pitting',      # 点蚀
            2: 'scuffing',     # 擦伤
            3: 'spalling'      # 剥落
        }
    
    def test_single_image(self, image_path):
        """
        对单张图像进行分类测试
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            dict: 包含预测结果的字典
        """
        try:
            # 进行预测
            results = self.model(image_path)
            
            # 获取预测结果
            result = results[0]
            
            # 获取最高概率的类别
            probs = result.probs.data.cpu().numpy()
            top_class = probs.argmax()
            confidence = probs[top_class]
            
            return {
                'image_path': image_path,
                'predicted_class': self.class_names.get(top_class, f'Class_{top_class}'),
                'confidence': float(confidence),
                'all_probabilities': {
                    self.class_names.get(i, f'Class_{i}'): float(prob) 
                    for i, prob in enumerate(probs)
                }
            }
            
        except Exception as e:
            return {
                'image_path': image_path,
                'error': str(e)
            }
    
    def test_directory(self, test_dir="test"):
        """
        对测试目录下的所有图像进行分类测试
        
        Args:
            test_dir: 测试目录路径
        """
        test_path = Path(test_dir)
        
        if not test_path.exists():
            print(f"❌ 测试目录不存在: {test_path}")
            return
        
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # 获取所有图像文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(test_path.glob(f'*{ext}'))
            image_files.extend(test_path.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"❌ 在 {test_path} 目录下未找到图像文件")
            return
        
        print(f"📁 找到 {len(image_files)} 张图像，开始测试...")
        print("=" * 80)
        
        results = []
        correct_predictions = 0
        total_predictions = 0
        
        for image_file in image_files:
            result = self.test_single_image(str(image_file))
            results.append(result)
            
            if 'error' in result:
                print(f"❌ {image_file.name}: {result['error']}")
                continue
            
            # 显示预测结果
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            
            print(f"📷 {image_file.name}")
            print(f"   预测类别: {predicted_class}")
            print(f"   置信度: {confidence:.4f}")
            
            # 显示所有类别的概率
            print("   所有类别概率:")
            for class_name, prob in result['all_probabilities'].items():
                print(f"      {class_name}: {prob:.4f}")
            
            # 如果图像文件名包含真实标签，计算准确率
            true_label = self._extract_true_label(image_file.name)
            if true_label:
                total_predictions += 1
                if predicted_class.lower() == true_label.lower():
                    correct_predictions += 1
                    print(f"   ✅ 预测正确 (真实标签: {true_label})")
                else:
                    print(f"   ❌ 预测错误 (真实标签: {true_label})")
            
            print("-" * 40)
        
        # 显示总体统计
        print("\n📊 测试统计:")
        print(f"总图像数: {len(image_files)}")
        print(f"成功预测: {len([r for r in results if 'error' not in r])}")
        print(f"预测失败: {len([r for r in results if 'error' in r])}")
        
        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions
            print(f"准确率: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
        
        return results
    
    def _extract_true_label(self, filename):
        """
        从文件名中提取真实标签
        
        Args:
            filename: 图像文件名
            
        Returns:
            str: 真实标签，如果无法提取则返回None
        """
        filename_lower = filename.lower()
        for class_name in self.class_names.values():
            if class_name.lower() in filename_lower:
                return class_name
        return None
    
    def test_with_visualization(self, image_path, save_result=False):
        """
        测试单张图像并可视化结果
        
        Args:
            image_path: 图像路径
            save_result: 是否保存结果图像
        """
        result = self.test_single_image(image_path)
        
        if 'error' in result:
            print(f"❌ 测试失败: {result['error']}")
            return
        
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ 无法读取图像: {image_path}")
            return
        
        # 在图像上添加预测结果
        predicted_class = result['predicted_class']
        confidence = result['confidence']
        
        # 添加文本
        text = f"{predicted_class}: {confidence:.3f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255)
        thickness = 2
        
        # 获取文本大小
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # 添加背景矩形
        cv2.rectangle(image, (10, 10), (text_size[0] + 20, text_size[1] + 20), (0, 0, 0), -1)
        
        # 添加文本
        cv2.putText(image, text, (15, text_size[1] + 15), font, font_scale, color, thickness)
        
        # 显示图像
        cv2.imshow('Gear Defect Classification', image)
        print(f"📷 显示图像: {Path(image_path).name}")
        print(f"   预测: {predicted_class} (置信度: {confidence:.4f})")
        print("   按任意键继续...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 保存结果图像
        if save_result:
            output_path = f"result_{Path(image_path).name}"
            cv2.imwrite(output_path, image)
            print(f"💾 结果已保存到: {output_path}")


def main():
    """主函数"""
    print("🚀 齿轮缺陷分类测试程序")
    print("=" * 50)
    
    # 创建测试器
    tester = GearClassificationTester()
    
    # 测试选项
    print("\n请选择测试模式:")
    print("1. 测试单张图像")
    print("2. 测试整个目录")
    print("3. 测试单张图像(带可视化)")
    
    try:
        choice = input("\n请输入选择 (1-3): ").strip()
        
        if choice == "1":
            # 测试单张图像
            image_path = input("请输入图像路径: ").strip()
            if os.path.exists(image_path):
                result = tester.test_single_image(image_path)
                print("\n测试结果:")
                if 'error' in result:
                    print(f"❌ {result['error']}")
                else:
                    print(f"预测类别: {result['predicted_class']}")
                    print(f"置信度: {result['confidence']:.4f}")
            else:
                print("❌ 图像文件不存在")
        
        elif choice == "2":
            # 测试整个目录
            test_dir = input("请输入测试目录路径 (默认: test): ").strip()
            if not test_dir:
                test_dir = "test"
            tester.test_directory(test_dir)
        
        elif choice == "3":
            # 带可视化的测试
            image_path = input("请输入图像路径: ").strip()
            if os.path.exists(image_path):
                save_result = input("是否保存结果图像? (y/n): ").strip().lower() == 'y'
                tester.test_with_visualization(image_path, save_result)
            else:
                print("❌ 图像文件不存在")
        
        else:
            print("❌ 无效选择")
    
    except KeyboardInterrupt:
        print("\n\n👋 程序已退出")
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")


if __name__ == "__main__":
    main()
