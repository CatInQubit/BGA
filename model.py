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
        attn_output, _ = self.mha(x, x, x)
        if self.use_gating:
            gate_value = self.sigmoid(self.gate(x))
        else:
            gate_value = torch.ones_like(attn_output)
            
        gated_output = gate_value * attn_output + (1 - gate_value) * x
        return self.layer_norm(gated_output), gate_value

class BiLSTMWithResidualGatedAttention(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=1, num_heads=4, variant='BGA'):

        super(BiLSTMWithResidualGatedAttention, self).__init__()
        self.variant = variant
        

        is_bidirectional = False if variant == 'BaseLSTM' else True
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=is_bidirectional)
        

        self.lstm_out_dim = hidden_dim * 2 if is_bidirectional else hidden_dim
        

        if variant in ['BGA', 'BiLSTM+MHA']:
            use_gating = True if variant == 'BGA' else False
            self.gated_attn = GatedMultiheadAttention(embed_dim=self.lstm_out_dim, num_heads=num_heads, use_gating=use_gating)
        
        self.fc = nn.Linear(self.lstm_out_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, lstm_out_dim)


        if self.variant in ['BGA', 'BiLSTM+MHA']:
            attn_output, gate_weights = self.gated_attn(lstm_out)
            mid_out = attn_output + lstm_out 
        else:
            mid_out = lstm_out
            gate_weights = None

        pooled_out = mid_out.mean(dim=1) 
        output = self.fc(pooled_out)


        return output, gate_weights
