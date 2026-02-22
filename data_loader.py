import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_classif
import warnings

warnings.filterwarnings('ignore')


class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, output_dim),
            nn.Sigmoid() # 配合 MinMaxScaler，将输出限制在 [0, 1]
        )
    def forward(self, z):
        return self.model(z)

class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.model(x)

def train_wgan_gp_internal(minority_data, epochs=100, latent_dim=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = minority_data.shape[1]
    gen = Generator(latent_dim, input_dim).to(device)
    critic = Critic(input_dim).to(device)
    
    optimizer_G = optim.Adam(gen.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizer_C = optim.Adam(critic.parameters(), lr=1e-4, betas=(0.5, 0.9))
    
    minority_data = minority_data.to(device)
    
    for epoch in range(epochs):
        for _ in range(5):
            idx = torch.randint(0, len(minority_data), (min(64, len(minority_data)),))
            real = minority_data[idx]
            z = torch.randn(len(real), latent_dim).to(device)
            fake = gen(z).detach()
            
            alpha = torch.rand(len(real), 1).to(device)
            interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
            d_interpolates = critic(interpolates)
            gradients = torch.autograd.grad(
                outputs=d_interpolates, inputs=interpolates,
                grad_outputs=torch.ones_like(d_interpolates),
                create_graph=True, retain_graph=True
            )[0]
            gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
            
            loss_C = critic(fake).mean() - critic(real).mean() + gp
            optimizer_C.zero_grad(); loss_C.backward(); optimizer_C.step()
            
        z = torch.randn(64, latent_dim).to(device)
        fake = gen(z)
        loss_G = -critic(fake).mean()
        optimizer_G.zero_grad(); loss_G.backward(); optimizer_G.step()
        
    return gen


def load_and_process_data(file_path, batch_size=64, apply_wgan=True):
    print(f"--- 正在加载数据集: {file_path} ---")
    data = pd.read_csv(file_path)


    data['result'] = data['result'].astype(str).str.strip("b'").str.replace("'", "")
    

    data = data[data["result"].isin(['0', '2', '3', '4', '6'])]
    data['result'] = data['result'].astype(int)
    data = data.dropna().drop_duplicates()


    X = data.drop(columns=['result']).select_dtypes(include=[np.number])
    y = data['result']

    constant_columns = [col for col in X.columns if X[col].nunique() <= 1]
    if constant_columns:
        print(f"已过滤常量特征: {constant_columns}")
        X = X.drop(columns=constant_columns)


    f_values, _ = f_classif(X, y)
    importance_df = pd.DataFrame({'Feature': X.columns, 'F_Value': f_values}).sort_values(by='F_Value', ascending=False)
    print("\n[特征重要性前10名]:\n", importance_df.head(10))


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    if apply_wgan:
        print("\n--- 正在执行 WGAN-GP 数据增强 (仅限训练集) ---")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        final_X_train = [X_train_scaled]
        final_y_train = [y_train.values]
        
        classes, counts = np.unique(y_train, return_counts=True)

        target_count = int(max(counts) * 0.5) 

        for cls, count in zip(classes, counts):
            if count < target_count:
                print(f"正在增强类别 {cls}: {count} -> {target_count}")
                cls_data = torch.FloatTensor(X_train_scaled[y_train == cls])
                gen = train_wgan_gp_internal(cls_data, epochs=150)
                z = torch.randn(target_count - count, 10).to(device)
                with torch.no_grad():
                    samples = gen(z).cpu().numpy()
                final_X_train.append(samples)
                final_y_train.append(np.full(target_count - count, cls))
        
        X_train_final = np.vstack(final_X_train)
        y_train_final = np.concatenate(final_y_train)
    else:
        X_train_final = X_train_scaled
        y_train_final = y_train.values

    X_train_tensor = torch.tensor(X_train_final, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1)
    

    num_classes = len(np.unique(y))
    unique_y = sorted(np.unique(y))
    y_map = {val: i for i, val in enumerate(unique_y)}
    
    y_train_mapped = np.array([y_map[v] for v in y_train_final])
    y_test_mapped = np.array([y_map[v] for v in y_test])

    Y_train_tensor = torch.nn.functional.one_hot(torch.tensor(y_train_mapped).long(), num_classes).float()
    Y_test_tensor = torch.nn.functional.one_hot(torch.tensor(y_test_mapped).long(), num_classes).float()


    train_loader = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor), batch_size=batch_size, shuffle=True)
    
    print(f"\n处理完成！最终训练样本数: {len(X_train_final)}, 特征数: {X_train_final.shape[1]}")
    
    return train_loader, X_test_tensor, Y_test_tensor, X_train_final.shape[1], num_classes


