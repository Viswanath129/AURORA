import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2, kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTMExpert(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=64, kernel_size=3):
        super(ConvLSTMExpert, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv_lstm = ConvLSTMCell(in_channels, hidden_dim, kernel_size)
        
        # Output layer
        self.conv_out = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=0),
            nn.Dropout2d(p=0.5) # MC Dropout for uncertainty
        )
        
    def forward(self, x, future_steps=1):
        """
        x: (B, T, C, H, W)
        """
        b, t, c, h, w = x.size()
        hidden = self.conv_lstm.init_hidden(b, (h, w))
        
        # Process input sequence
        for step in range(t):
            hidden = self.conv_lstm(x[:, step], hidden)
            
        predictions = []
        
        # Predict future
        # In a real seq2seq, we might feed output back as input. 
        # For 1-step, we just take the hidden state.
        
        last_h, last_c = hidden
        
        # We want uncertainty. In MC Dropout, we run multiple passes.
        # But 'forward' usually does one pass.
        # We will return the mean prediction here for training, 
        # and handle MC sampling in an external 'predict_with_uncertainty' method or logic.
        
        out = torch.sigmoid(self.conv_out(last_h))
        return out

    def predict_with_uncertainty(self, x, num_samples=10):
        self.train() # Enable dropout
        preds = []
        for _ in range(num_samples):
            with torch.no_grad():
                preds.append(self.forward(x).cpu())
        
        preds = torch.stack(preds) # (Samples, B, C, H, W)
        mean_pred = preds.mean(dim=0)
        var_pred = preds.var(dim=0)
        
        return mean_pred, var_pred

if __name__ == "__main__":
    model = ConvLSTMExpert(in_channels=1, hidden_dim=16)
    dummy_input = torch.randn(2, 4, 1, 64, 64) # B, T, C, H, W
    out = model(dummy_input)
    print(f"Output shape: {out.shape}")
    
    mu, sigma = model.predict_with_uncertainty(dummy_input, num_samples=5)
    print(f"MC Mean shape: {mu.shape}, Variance shape: {sigma.shape}")
