import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 🧱 核心积木：双重卷积块
# ==========================================
class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) * 2
    无论是在编码器(提取特征)还是解码器(恢复分辨率)，这都是最基础的运算单元。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # 🌟 工业级细节：因为后面接了 BatchNorm，这里的 bias 必须设为 False，节省显存和计算量
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# ==========================================
# 📉 编码器积木：下采样
# ==========================================
class Down(nn.Module):
    """Maxpool 下采样 + 双重卷积
    作用：将图片长宽缩小一半，提取更深层的语义特征，扩大感受野。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# ==========================================
# 📈 解码器积木：上采样与跳跃连接
# ==========================================
class Up(nn.Module):
    """转置卷积上采样 + 跳跃连接 (Skip Connection) + 双重卷积
    作用：将深层的低分辨率语义图放大，并融合浅层的高清边缘图。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 1. 转置卷积 (Deconvolution)：负责将长宽放大 2 倍，同时通道数减半
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # 2. 融合卷积：接收拼接后的特征，再次进行特征提取
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # 第一步：将深层特征 x1 放大
        x1 = self.up(x1)
        
        # 第二步：🌟 极其致命的跳跃连接 (Skip Connection)
        # x2 是当初下采样时保留的“高清边缘底片”
        # 将放大后的 x1 和高清底片 x2 在通道维度 (dim=1) 强行拼接！
        x = torch.cat([x2, x1], dim=1)
        
        # 第三步：过卷积层融合信息
        return self.conv(x)

# ==========================================
# 🎯 像素输出层
# ==========================================
class OutConv(nn.Module):
    """最后的输出层：使用 1x1 卷积完成降维打击"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        # 1x1 卷积不改变图像长宽，只负责把特征通道数压缩到我们需要分类的类别数 (例如 21 类)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# ==========================================
# 🧠 发动机总装：U-Net
# ==========================================
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=21):
        """
        n_channels: 输入图片的通道数 (RGB 为 3)
        n_classes: 需要预测的类别数 (VOC2012 是 20 类物体 + 1 类背景 = 21)
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # --- 左半边：特征提取 (Encoder) ---
        # 假设输入是 [Batch, 3, 256, 256]
        self.inc = DoubleConv(n_channels, 64)       # -> [Batch, 64, 256, 256]
        self.down1 = Down(64, 128)                  # -> [Batch, 128, 128, 128]
        self.down2 = Down(128, 256)                 # -> [Batch, 256, 64, 64]
        self.down3 = Down(256, 512)                 # -> [Batch, 512, 32, 32]
        
        # 谷底 (Bottleneck)
        self.down4 = Down(512, 1024)                # -> [Batch, 1024, 16, 16]

        # --- 右半边：细节还原 (Decoder) ---
        # Up(1024, 512) 的内部逻辑：输入 1024 放大变 512，拼接对岸过来的 512，总和 1024，再过卷积变 512
        self.up1 = Up(1024, 512)                    # -> [Batch, 512, 32, 32]
        self.up2 = Up(512, 256)                     # -> [Batch, 256, 64, 64]
        self.up3 = Up(256, 128)                     # -> [Batch, 128, 128, 128]
        self.up4 = Up(128, 64)                      # -> [Batch, 64, 256, 256]
        
        # --- 最终制裁 ---
        self.outc = OutConv(64, n_classes)          # -> [Batch, 21, 256, 256]

    def forward(self, x):
        # 1. 深入敌阵，疯狂下采样，同时把每一层的清晰底片保存下来 (x1, x2, x3, x4)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 2. 绝地反击，疯狂上采样，每一层都找当年的高清底片拼接来恢复边缘
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # 3. 吐出像素级判决书
        logits = self.outc(x)
        return logits

# ==========================================
# 🛠️ 车间点火测试
# ==========================================
if __name__ == '__main__':
    print("🛠️ 正在对 U-Net 发动机进行出厂检验...")
    # 模拟一张 256x256 的马路照片
    dummy_input = torch.randn(2, 3, 256, 256) 
    
    # 实例化 21 分类的发动机
    model = UNet(n_channels=3, n_classes=21)
    
    # 前向传播测试
    dummy_output = model(dummy_input)
    
    print(f"✅ 输入张量形状: {dummy_input.shape}")
    print(f"✅ 输出张量形状: {dummy_output.shape}")
    
    if dummy_output.shape == (2, 21, 256, 256):
        print("🎉 检验通过！维度完美对齐，可以接入 train.py 启动炼丹！")
    else:
        print("❌ 警告：维度不匹配，请检查代码！")