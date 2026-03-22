import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler # 🌟 4070S 专属满血驱动

from datasets.voc_seg_dataset import VOCSegDataset
# ⚠️ 这里使用我之前写的纯血 U-Net 类名。如果你的文件名是 fcn_unet，类名是 UNet，请保持如下：
from models.fcn_unet import UNet

# ==========================================
# 🌟 大厂级核心：全局混淆矩阵计算器 (真正正确的 mIoU 算法)
# 彻底取代之前粗暴的 Batch 求平均，实现完美的像素级统计
# ==========================================
class Evaluator:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        # 创建一个 N x N 的全 0 矩阵记录所有像素的预测去向
        self.mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def update(self, preds, targets):
        # 🛡️ 过滤掉标签是 255 的模糊白边，不让它们污染成绩
        valid_mask = (targets != 255)
        preds = preds[valid_mask]
        targets = targets[valid_mask]
        
        # 极速一维化混淆矩阵算子 (完全无 for 循环)
        n = self.num_classes
        inds = n * targets + preds
        self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n).cpu()

    def get_miou_and_acc(self):
        h = self.mat.float()
        # 像素准确率: 对角线之和(猜对的像素) / 总像素
        acc = torch.diag(h).sum() / (h.sum() + 1e-15)
        
        # 交并比 IoU = TP / (TP + FP + FN)
        iou = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h) + 1e-15)
        # mIoU 是忽略 NaN 后的 21 个类的 IoU 求平均
        miou = torch.nanmean(iou)
        
        return miou.item(), acc.item()

    def reset(self):
        # 每个 Epoch 结束后清零矩阵
        self.mat.zero_()

# ==========================================
# 🚀 工业级主控台
# ==========================================
def main():
    # 1. 加载超参数配置文件
    with open('configs/voc_seg.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 2. 🚨 空间对齐防线：坚决移除 RandomHorizontalFlip，只保留颜色抖动
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ])
    
    # 3. 组装数据管道
    # 注意：为了让 config 生效，需要确保你在 Dataset 内部或这里正确处理了 transform 传入
    train_dataset = VOCSegDataset(
        root_dir=config['data']['root'], 
        image_size=config['data']['img_size']
        # 如果 Dataset 的 __init__ 支持 transform 参数，可以加上 transform=transform
    )
    
    val_dataset = VOCSegDataset(
        root_dir=config['data']['root'], 
        image_size=config['data']['img_size']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        drop_last=True # 🛡️ 极其重要：训练时丢弃凑不够一个 batch 的图，防止 BatchNorm 崩溃
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    # 4. 点火计算引擎
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 炼丹炉点火！当前引擎: {device} | 已激活 AMP 自动混合精度加速")
    
    model = UNet(n_channels=3, n_classes=config['model']['num_classes']).to(device)
    
    # 5. 极其致命的 255 免疫护盾
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 🌟 4070S 满血驱动器
    scaler = GradScaler()
    
    # 🌟 实例化我们手写的真·全局评估器
    evaluator = Evaluator(num_classes=config['model']['num_classes'])
    
    best_miou = 0.0
    epochs = config['training']['epochs']
    
    # ==========================================
    # 🔄 训练与验证死循环
    # ==========================================
    for epoch in range(epochs):
        
        # --- [阶段 A：残酷的梯队训练] ---
        model.train()
        train_loss = 0.0
        evaluator.reset() # 每次新 Epoch 必须清空历史成绩
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device, dtype=torch.long)
            
            optimizer.zero_grad()
            
            # AMP 混合精度上下文
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            # 缩放梯度并反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            # 记录预测结果至混淆矩阵
            pred = torch.argmax(outputs, dim=1)
            evaluator.update(pred, masks)
            
            if batch_idx % 20 == 0:
                print(f"  [Train] Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        # 结算 Train 全局指标
        train_miou, train_acc = evaluator.get_miou_and_acc()
        train_loss /= len(train_loader)
        
        # --- [阶段 B：严格的闭卷验证] ---
        model.eval()
        val_loss = 0.0
        evaluator.reset() # 验证集也要用全新的矩阵
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device, dtype=torch.long)
                
                # 验证集也开启 AMP，节省显存，加速验证
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    
                val_loss += loss.item()
                
                # 记录验证集预测结果至混淆矩阵
                pred = torch.argmax(outputs, dim=1)
                evaluator.update(pred, masks)
        
        # 结算 Val 全局指标
        val_miou, val_acc = evaluator.get_miou_and_acc()
        val_loss /= len(val_loader)
        
        # ==========================================
        # 📊 数据大屏与金丹入库
        # ==========================================
        print(f"\n🟢 Epoch [{epoch+1}/{epochs}] 结算报告:")
        print(f"  ➔ Train | Loss: {train_loss:.4f} | mIoU: {train_miou:.4f} | Acc: {train_acc:.4f}")
        print(f"  ➔ Val   | Loss: {val_loss:.4f}   | mIoU: {val_miou:.4f}   | Acc: {val_acc:.4f}")
        print("-" * 50)
        
        if val_miou > best_miou:
            best_miou = val_miou
            os.makedirs('checkpoints', exist_ok=True)
            save_path = 'checkpoints/seg_best_model.pth'
            torch.save(model.state_dict(), save_path)
            print(f"🏆 刷新记录！成功保存新一代绝世金丹，当前最高 mIoU: {best_miou:.4f}\n")

if __name__ == '__main__':
    main()