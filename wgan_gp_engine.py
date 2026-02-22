import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- 1. 定义生成器 (Generator) ---
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, output_dim),
            nn.Tanh() # 假设数据已归一化到 [-1, 1] 或 [0, 1]
        )

    def forward(self, z):
        return self.model(z)

# --- 2. 定义判别器 (Critic/Discriminator) ---
class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, 1) # 输出一个分数，而不是概率
        )

    def forward(self, img):
        return self.model(img)

# --- 3. WGAN-GP 核心训练函数 ---
def train_wgan_gp(minority_data, epochs=500, latent_dim=10, device='cpu'):
    """
    针对少数类数据进行增强
    minority_data: 少数类的特征矩阵 (Tensor)
    """
    input_dim = minority_data.shape[1]
    gen = Generator(latent_dim, input_dim).to(device)
    critic = Critic(input_dim).to(device)
    
    optimizer_G = optim.Adam(gen.parameters(), lr=0.0001, betas=(0.5, 0.9))
    optimizer_C = optim.Adam(critic.parameters(), lr=0.0001, betas=(0.5, 0.9))
    
    lambda_gp = 10 # 梯度惩罚系数
    
    for epoch in range(epochs):
        # (1) 训练判别器 (Critic)
        for _ in range(5): # WGAN要求Critic训练次数多于Generator
            real_samples = minority_data[torch.randint(0, len(minority_data), (64,))].to(device)
            z = torch.randn(64, latent_dim).to(device)
            fake_samples = gen(z).detach()
            
            # 计算梯度惩罚 (Gradient Penalty)
            alpha = torch.rand(64, 1).to(device)
            interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
            d_interpolates = critic(interpolates)
            fake = torch.ones(64, 1).to(device)
            gradients = torch.autograd.grad(
                outputs=d_interpolates, inputs=interpolates,
                grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
            
            # 判别器损失：尽可能拉大真假样本的分数差
            loss_C = critic(fake_samples).mean() - critic(real_samples).mean() + gp
            optimizer_C.zero_grad()
            loss_C.backward()
            optimizer_C.step()

        # (2) 训练生成器 (Generator)
        z = torch.randn(64, latent_dim).to(device)
        gen_samples = gen(z)
        loss_G = -critic(gen_samples).mean()
        
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()
        
    return gen # 返回训练好的生成器

# --- 4. 整合进你的数据流 ---
def augment_data(X_train, y_train, target_count=5000):
    """
    X_train: 原始训练特征 (numpy)
    y_train: 原始训练标签 (numpy)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    new_X, new_y = [X_train], [y_train]
    
    # 找出样本量少的类别（例如 MSCI）
    classes, counts = np.unique(y_train, return_counts=True)
    for cls, count in zip(classes, counts):
        if count < target_count:
            print(f"正在为类别 {cls} 生成增强样本 (当前: {count} -> 目标: {target_count})")
            cls_data = torch.FloatTensor(X_train[y_train == cls])
            # 训练该类的生成器
            gen = train_wgan_gp(cls_data, epochs=200, device=device)
            # 生成新数据
            z = torch.randn(target_count - count, 10).to(device)
            with torch.no_grad():
                synthetic_data = gen(z).cpu().numpy()
            new_X.append(synthetic_data)
            new_y.append(np.full(target_count - count, cls))
            
    return np.vstack(new_X), np.concatenate(new_y)