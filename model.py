import torch
import torch.nn as nn

class GatedMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, use_gating=True):
        super(GatedMultiheadAttention, self).__init__()
        self.use_gating = use_gating
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.gate = nn.Linear(embed_dim, embed_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. Multi-head Attention
        attn_output, _ = self.mha(x, x, x)
        
        # 2. Gating Logic
        if self.use_gating:
            gate_value = self.sigmoid(self.gate(x))
        else:
            # 消融实验：BiLSTM + MHA 模式，门控强行设为 1
            gate_value = torch.ones_like(attn_output)
            
        # 3. Gated Residual Connection (保持原有逻辑)
        gated_output = gate_value * attn_output + (1 - gate_value) * x
        return self.layer_norm(gated_output), gate_value

class BiLSTMWithResidualGatedAttention(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=1, num_heads=4, variant='BGA'):
        """
        variant 参数说明:
        - 'BGA': 完整模型 (默认, 功能不变)
        - 'BiLSTM+MHA': 双向LSTM + 普通注意力 (无门控)
        - 'BiLSTM': 仅双向LSTM (无注意力层)
        - 'BaseLSTM': 仅单向LSTM (无注意力层)
        """
        super(BiLSTMWithResidualGatedAttention, self).__init__()
        self.variant = variant
        
        # 判定是否双向 (只有 BaseLSTM 是单向)
        is_bidirectional = False if variant == 'BaseLSTM' else True
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=is_bidirectional)
        
        # 计算 LSTM 输出维度
        self.lstm_out_dim = hidden_dim * 2 if is_bidirectional else hidden_dim
        
        # 判定是否需要注意力层
        if variant in ['BGA', 'BiLSTM+MHA']:
            use_gating = True if variant == 'BGA' else False
            self.gated_attn = GatedMultiheadAttention(embed_dim=self.lstm_out_dim, num_heads=num_heads, use_gating=use_gating)
        
        self.fc = nn.Linear(self.lstm_out_dim, output_dim)

    def forward(self, x):
        # Step 1: LSTM 编码
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, lstm_out_dim)

        # Step 2: 根据变体决定是否经过注意力层
        if self.variant in ['BGA', 'BiLSTM+MHA']:
            # 保持原有的 Gated Attention + Residual 逻辑
            attn_output, gate_weights = self.gated_attn(lstm_out)
            # 残差连接 (对应你原代码的 attn_output + lstm_out)
            mid_out = attn_output + lstm_out 
        else:
            # 消融实验：直接跳过注意力层
            mid_out = lstm_out
            gate_weights = None

        # Step 3: 全局平均池化 (保持原有逻辑)
        pooled_out = mid_out.mean(dim=1) 
        output = self.fc(pooled_out)

        return output, gate_weights