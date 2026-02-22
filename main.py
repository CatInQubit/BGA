import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 导入自定义模块
from model import BiLSTMWithResidualGatedAttention
from data_loader import load_and_process_data

# === 1. 全局配置参数 (保持不变) ===
FILE_PATH = r'D:/2024技术学习/项目/OTIT项目/焦锐健论文/data/gas_final.arff.csv' 
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 64

# 定义类别名称 (对齐 Edge-IIoT 工业子集)
TARGET_NAMES = ['Normal', 'CMRI', 'MSCI', 'MPCI', 'DoS']

# ==========================================================
# 模式切换开关：
# 'Main'        -> 【主实验模式】保留所有绘图：Loss、混淆矩阵、门控热图。
# 'Ablation'    -> 【消融实验模式】保留原有 6 指标对比大表输出。
# 'Sensitivity' -> 【敏感性分析模式】保留维度与头数 F1 趋势汇总。
# 'Robustness'  -> 【新增：抗噪测试】专门产出论文 Table 4.4 的对比数据。
MODE = 'Main' 
# ==========================================================

def plot_heatmap(model, X_test, y_true, device):
    """
    【原有核心功能】：生成 Normal vs Attack 的可解释性对比热图。
    """
    model.eval()
    try:
        # 寻找测试集中第一个正常样本和第一个攻击样本
        normal_idx = np.where(y_true == 0)[0][0]
        attack_idx = np.where(y_true != 0)[0][0] 
        attack_label = TARGET_NAMES[y_true[attack_idx]]
    except IndexError:
        print("未能在测试集中找到对比样本，跳过热图。")
        return

    indices = [normal_idx, attack_idx]
    titles = [f'Normal Traffic (Class: Normal)', f'Malicious Attack (Class: {attack_label})']
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 5))
    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = X_test[idx].unsqueeze(0).to(device)
            _, gate_weights = model(sample)
            if gate_weights is not None:
                weight_data = gate_weights.squeeze().cpu().numpy().reshape(1, -1)
                sns.heatmap(weight_data, ax=axes[i], cmap='YlGnBu', cbar=True)
                axes[i].set_title(titles[i], fontsize=12)
                axes[i].set_yticks([])
    plt.tight_layout()
    plt.savefig('attention_heatmap.png', dpi=300)
    print(">>> 门控热图已保存为 'attention_heatmap.png'")
    plt.show()

def run_experiment_engine(variant_name='BGA', show_plots=True, hidden_dim=64, num_heads=4, return_model=False):
    """
    【通用运行引擎】：整合了原有所有指标采集和绘图逻辑，增加模型返回接口。
    """
    # 加载数据 (data_loader 内部会打印 ANOVA 排名和 WGAN 进度)
    train_loader, X_test_tensor, Y_test_tensor, input_dim, output_dim = load_and_process_data(FILE_PATH, BATCH_SIZE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    model = BiLSTMWithResidualGatedAttention(
        input_dim, output_dim, 
        hidden_dim=hidden_dim, 
        num_heads=num_heads, 
        variant=variant_name
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 训练循环 (Main模式详细输出) ---
    loss_history = []
    if show_plots:
        print(f"\n--- 启动训练变体: {variant_name} (Hidden:{hidden_dim}, Heads:{num_heads}) ---")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels.argmax(dim=1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        if show_plots:
            print(f'Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}')

    # --- 评估阶段 ---
    model.eval()
    X_test_device = X_test_tensor.to(device)
    start_time = time.time()
    with torch.no_grad():
        y_pred_raw, gate_weights = model(X_test_device)
        inf_time = (time.time() - start_time) / len(X_test_tensor) * 1000 # ms
        y_pred_labels = y_pred_raw.argmax(dim=1).cpu().numpy()
    
    y_true_indices = Y_test_tensor.argmax(dim=1).cpu().numpy()

    # 关键修复：加入 TARGET_NAMES 映射，防止打印错乱
    acc = accuracy_score(y_true_indices, y_pred_labels)
    report_dict = classification_report(y_true_indices, y_pred_labels, target_names=TARGET_NAMES, output_dict=True, digits=4)
    weighted_avg = report_dict['weighted avg']

    # --- 可视化 (仅在 Main 模式触发) ---
    if show_plots:
        print(f"\n[Overall Accuracy]: {acc:.4f}")
        print("\n[Detailed Classification Report]:")
        print(classification_report(y_true_indices, y_pred_labels, target_names=TARGET_NAMES, digits=4))
        
        # 1. 原有 Loss 曲线
        plt.figure(figsize=(8, 4))
        plt.plot(loss_history, color='royalblue', label='Training Loss', linewidth=2)
        plt.title(f'Loss Convergence ({variant_name})')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True); plt.legend(); plt.show()

        # 2. 原有 混淆矩阵
        cm = confusion_matrix(y_true_indices, y_pred_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES)
        plt.title(f'Confusion Matrix ({variant_name})')
        plt.xlabel('Predicted'); plt.ylabel('True'); plt.show()

        # 3. 原有 门控热图
        if variant_name == 'BGA':
            plot_heatmap(model, X_test_device, y_true_indices, device)

    stats = {
        "variant": variant_name, "hidden_dim": hidden_dim, "heads": num_heads,
        "precision": weighted_avg['precision'], "recall": weighted_avg['recall'],
        "f1": weighted_avg['f1-score'], "acc": acc, 
        "params": f"{total_params/1000:.1f}K", "latency": inf_time
    }

    if return_model:
        return stats, model, X_test_tensor, Y_test_tensor
    return stats

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if MODE == 'Main':
        # 执行原有主实验逻辑
        run_experiment_engine(variant_name='BGA', show_plots=True)
    
    elif MODE == 'Ablation':
        # 执行原有消融实验大表逻辑
        print(">>> 启动消融对比实验 (静默模式)...")
        results = []
        for v in ['BaseLSTM', 'BiLSTM', 'BiLSTM+MHA', 'BGA']:
            res = run_experiment_engine(variant_name=v, show_plots=False)
            results.append(res)
        
        print("\n" + "="*95)
        print(f"{'Variant':<15} | {'Prec':<7} | {'Recall':<7} | {'F1-Score':<8} | {'Acc':<7} | {'Params':<8} | {'Latency'}")
        print("-" * 95)
        for r in results:
            print(f"{r['variant']:<15} | {r['precision']:<7.4f} | {r['recall']:<7.4f} | {r['f1']:<8.4f} | {r['acc']:<7.4f} | {r['params']:<8} | {r['latency']:.4f}ms")
        print("="*95)

    elif MODE == 'Sensitivity':
        # 执行原有参数敏感性分析
        print(">>> 启动超参数敏感性分析...")
        dim_stats = []
        for d in [32, 64, 128]:
            res = run_experiment_engine(variant_name='BGA', show_plots=False, hidden_dim=d)
            dim_stats.append((d, res['f1']))
            print(f"Dimension {d} 完成测试.")
        
        # 输出汇总 (用于画敏感性折线图)
        print("\n" + "="*40)
        print("【敏感性分析汇总 (Weighted F1)】")
        for d, f in dim_stats: print(f"Hidden Dim {d:<3} : F1 = {f:.4f}")
        print("="*40)

    elif MODE == 'Robustness':
        # 【新增功能】：产出论文 Table 4.4 针对加密噪音的鲁棒性实验数据
        print(">>> 启动抗噪鲁棒性应力测试 (Table 4.4)...")
        # 修改 main.py 里的 Robustness 模式
        noise_levels = [0.0, 0.002, 0.005, 0.01] # 无、低、中、高噪声
        robust_data = {}

        for v_name in ['BiLSTM+MHA', 'BGA']: # 对比普通注意力与门控注意力
            print(f"\n1. 正在准备基准模型: {v_name}")
            stats, trained_model, x_test_raw, y_test_raw = run_experiment_engine(
                variant_name=v_name, show_plots=False, return_model=True
            )
            
            y_true = y_test_raw.argmax(dim=1).numpy()
            v_scores = []
            
            for level in noise_levels:
                # 注入模拟随机噪声 (Gaussian Noise)
                noise = torch.randn_like(x_test_raw) * level
                x_noisy = (x_test_raw + noise).to(device)
                
                trained_model.eval()
                with torch.no_grad():
                    preds_raw, _ = trained_model(x_noisy)
                    preds = preds_raw.argmax(dim=1).cpu().numpy()
                
                # 采集加权 F1
                rep = classification_report(y_true, preds, target_names=TARGET_NAMES, output_dict=True)
                v_scores.append(rep['weighted avg']['f1-score'])
                print(f"   - 噪声水平 {level:<4} 测试完成. F1: {v_scores[-1]:.4f}")
            
            robust_data[v_name] = v_scores

        # 打印符合论文格式的最终表格数据
        print("\n" + "="*75)
        print(f"{'Model Variant':<15} | {'None(0.0)':<10} | {'Low(0.002)':<10} | {'Mid(0.005)':<10} | {'High(0.01)'}")
        print("-" * 75)
        for name, scores in robust_data.items():
            print(f"{name:<15} | {scores[0]:<10.4f} | {scores[1]:<10.4f} | {scores[2]:<10.4f} | {scores[3]:.4f}")
        print("="*75)
   