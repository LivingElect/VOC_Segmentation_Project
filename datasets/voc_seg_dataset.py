import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

# ==========================================
# 🎨 工业级色卡标准：VOC2012 官方 RGB 字典
# 包含了 20 种物体 + 1 种背景 (索引 0)
# ==========================================
VOC_COLORMAP = [
    (0, 0, 0),         # 索引 0:  background (背景) - 纯黑
    (128, 0, 0),       # 索引 1:  aeroplane (飞机) - 暗红
    (0, 128, 0),       # 索引 2:  bicycle (自行车) - 暗绿
    (128, 128, 0),     # 索引 3:  bird (鸟) - 暗黄
    (0, 0, 128),       # 索引 4:  boat (船) - 暗蓝
    (128, 0, 128),     # 索引 5:  bottle (瓶子) - 紫色
    (0, 128, 128),     # 索引 6:  bus (公交车) - 蓝绿
    (128, 128, 128),   # 索引 7:  car (汽车) - 灰色
    (64, 0, 0),        # 索引 8:  cat (猫) - 深红棕
    (192, 0, 0),       # 索引 9:  chair (椅子) - 亮红
    (64, 128, 0),      # 索引 10: cow (牛) - 橄榄绿
    (192, 128, 0),     # 索引 11: diningtable (餐桌) - 亮黄棕
    (64, 0, 128),      # 索引 12: dog (狗) - 深紫
    (192, 0, 128),     # 索引 13: horse (马) - 亮紫
    (64, 128, 128),    # 索引 14: motorbike (摩托车) - 灰蓝
    (192, 128, 128),   # 索引 15: person (人) - 浅粉红
    (0, 64, 0),        # 索引 16: potted plant (盆栽) - 深绿
    (128, 64, 0),      # 索引 17: sheep (羊) - 棕色
    (0, 192, 0),       # 索引 18: sofa (沙发) - 亮绿
    (128, 192, 0),     # 索引 19: train (火车) - 黄绿
    (0, 64, 128)       # 索引 20: tv/monitor (电视/显示器) - 深海军蓝
]

class VOCSegDataset(Dataset):
    def __init__(self, root_dir, split='train', image_size=256, transform=None):
        """
        root_dir: VOCdevkit/VOC2012 的根目录
        split: 'train' 或 'val'，用于读取不同的划分文件
        image_size: 统一缩放的尺寸
        transform: 仅应用于原图的数据增强（如 ColorJitter）
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.transform = transform
        
        # 核心目录定位
        self.img_dir = os.path.join(root_dir, 'JPEGImages')
        self.mask_dir = os.path.join(root_dir, 'SegmentationClass')
        
        # 1. 根据 split 读取对应的 txt 文件，获取图片名单
        txt_path = os.path.join(root_dir, 'ImageSets', 'Segmentation', f'{split}.txt')
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"❌ 找不到划分文件: {txt_path}。请确保 VOC 数据集完整！")
            
        with open(txt_path, 'r') as f:
            self.file_names = [x.strip() for x in f.readlines()]
            
        # ==========================================
        # 🧠 大厂级黑科技：极其暴力的 O(1) 颜色查表法
        # 将 [R, G, B] 压平成一个唯一的整数索引，消除嵌套 for 循环！
        # ==========================================
        self.colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
        for i, color in enumerate(VOC_COLORMAP):
            # 哈希公式：(R * 256 + G) * 256 + B
            idx = (color * 256 + color) * 256 + color
            self.colormap2label[idx] = i
            
        # 🚨 极其关键的免死金牌：处理白色模糊边界
        # VOC 中的物体边界常有 的白线，代表“人类也分不清”。
        # 我们将其 ID 设为 255，配合 train.py 中的 ignore_index=255，让网络免受惩罚。
        border_idx = (224 * 256 + 224) * 256 + 192
        self.colormap2label[border_idx] = 255

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        
        # ==========================================
        # 1. 物理读取
        # ==========================================
        img_path = os.path.join(self.img_dir, file_name + '.jpg')
        mask_path = os.path.join(self.mask_dir, file_name + '.png')
        
        # 面具图必须显式转换为 RGB，防止 PIL 默认以“调色板(P)”模式读取
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')
        
        # ==========================================
        # 2. 极其严苛的空间隔离操作
        # ==========================================
        # 原图缩放：使用平滑的双线性插值 (Bilinear)，保留视觉质感
        image = TF.resize(image, (self.image_size, self.image_size), 
                          interpolation=TF.InterpolationMode.BILINEAR)
        # 面具缩放：🚨 绝对只能用最近邻插值 (Nearest)！否则会产生字典外的不明杂色，导致代码直接崩溃！
        mask = TF.resize(mask, (self.image_size, self.image_size), 
                         interpolation=TF.InterpolationMode.NEAREST)
        
        # ==========================================
        # 3. 施加光影魔法 (仅对原图)
        # ==========================================
        if self.transform is not None:
            # ColorJitter 等颜色增强，只会改变图像色调，不会改变物理空间位置，所以极其安全
            image = self.transform(image)
            
        # 转为 Tensor (同时会将像素值从 0~255 归一化到 0.0~1.0)
        image_tensor = TF.to_tensor(image) 
        
        # ==========================================
        # 4. 颜色降维打击：RGB -> Class ID
        # ==========================================
        # 将 PIL Mask 转为 NumPy 数组 [H, W, 3]
        mask_np = np.array(mask, dtype=np.int32)
        
        # 瞬间计算全图所有像素的哈希索引矩阵 [H, W]
        idx_matrix = (mask_np[:, :, 0] * 256 + mask_np[:, :, 1]) * 256 + mask_np[:, :, 2]
        
        # 💥 查表法降临：直接根据索引矩阵提取出 0~20 (及 255) 的 ID 矩阵
        mask_tensor = self.colormap2label[idx_matrix]
        
        # 最终出货：
        # image_tensor:, float32
        # mask_tensor: , int64 (long)
        return image_tensor, mask_tensor

# ==========================================
# 🛠️ 质检车间：跑起来看看数据对不对
# ==========================================
if __name__ == '__main__':
    print("🛠️ 正在启动数据清洗管线质检...")
    
    # 请确保相对路径正确，假设终端运行在 VOC_Segmentation_Project 根目录下
    try:
        test_dataset = VOCSegDataset(root_dir='./data/VOCdevkit/VOC2012', split='train')
        img, mask = test_dataset
        
        print(f"✅ 数据集加载成功！共包含 {len(test_dataset)} 张训练图片。")
        print(f"✅ 原图 Tensor 形状: {img.shape}, 数据类型: {img.dtype}")
        print(f"✅ 面具 Tensor 形状: {mask.shape}, 数据类型: {mask.dtype}")
        print(f"🔍 随机抽检的一层面具中，包含的有效类别 ID 有: {torch.unique(mask).tolist()}")
        print("🎉 完美！数据管线畅通无阻，哈希映射极速完成！")
        
    except FileNotFoundError as e:
        print(e)
        print("💡 提示：请检查你的 ./data/VOCdevkit/VOC2012 路径是否真的有数据。")