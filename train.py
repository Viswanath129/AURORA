import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from optical_flow import OpticalFlowExpert
from convlstm import ConvLSTMExpert
from diffusion import DiffusionExpert
from morphology import MorphologyExpert
from routing_net import RoutingNetwork

# Settings
BATCH_SIZE = 8
EPOCHS = 10 
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data():
    if not os.path.exists('sequences.pth'):
        print("sequences.pth not found. Run build_sequences.py first.")
        return None
    
    data = torch.load('sequences.pth')
    dataset = TensorDataset(data['inputs'], data['targets'])
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader

def train_experts(train_loader):
    print("Initializing AURORA Experts...")
    
    # Path 1: Advection (Non-trainable)
    expert_flow = OpticalFlowExpert()
    
    # Path 2: Morphology (Trainable)
    expert_morph = MorphologyExpert().to(DEVICE)
    opt_morph = torch.optim.Adam(expert_morph.parameters(), lr=LR)
    
    # Path 3: Emergence (Trainable)
    expert_diff = DiffusionExpert().to(DEVICE)
    opt_diff = torch.optim.Adam(expert_diff.parameters(), lr=LR)
    
    # Path 4: Temporal (Trainable)
    expert_lstm = ConvLSTMExpert().to(DEVICE)
    opt_lstm = torch.optim.Adam(expert_lstm.parameters(), lr=LR)
    
    criterion = nn.MSELoss()
    
    print("Training Experts (Paths 2, 3, 4)...")
    expert_morph.train()
    expert_diff.train()
    expert_lstm.train()
    
    for epoch in range(5):
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_dim = y.unsqueeze(1) # (B, 1, H, W)
            
            # 1. Train Morphology (Takes last frame)
            # x is (B, 4, H, W) -> last frame is x[:, -1]
            p_morph, _ = expert_morph(x)
            loss_morph = criterion(p_morph, y_dim)
            
            opt_morph.zero_grad()
            loss_morph.backward()
            opt_morph.step()
            
            # 2. Train Diffusion (Takes sequence)
            p_diff, _ = expert_diff(x)
            loss_diff = criterion(p_diff, y_dim)
            
            opt_diff.zero_grad()
            loss_diff.backward()
            opt_diff.step()
            
            # 3. Train LSTM (Takes sequence with channel dim)
            x_seq = x.unsqueeze(2)
            p_lstm = expert_lstm(x_seq)
            loss_lstm = criterion(p_lstm, y_dim)
            
            opt_lstm.zero_grad()
            loss_lstm.backward()
            opt_lstm.step()
            
            total_loss += loss_morph.item() + loss_diff.item() + loss_lstm.item()
            
        print(f"Epoch {epoch+1} Experts Loss: {total_loss:.4f}")
        
    return expert_flow, expert_morph, expert_diff, expert_lstm

def train_routing(expert_flow, expert_morph, expert_diff, expert_lstm, train_loader):
    print("Training Pixel-Wise Routing Network...")
    routing_model = RoutingNetwork().to(DEVICE)
    optimizer = torch.optim.Adam(routing_model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    # Freeze experts
    expert_morph.eval()
    expert_diff.eval()
    expert_lstm.eval()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_dim = y.unsqueeze(1)
            
            # Get Expert Predictions
            with torch.no_grad():
                # Path 1: Advection
                p_flow, u_flow = expert_flow.predict(x)
                p_flow, u_flow = p_flow.to(DEVICE), u_flow.to(DEVICE)
                
                # Path 2: Morphology
                p_morph, u_morph = expert_morph(x)
                
                # Path 3: Emergence
                p_diff, u_diff = expert_diff(x)
                
                # Path 4: Temporal
                x_seq = x.unsqueeze(2)
                p_lstm, u_lstm = expert_lstm.predict_with_uncertainty(x_seq, num_samples=3)
                p_lstm, u_lstm = p_lstm.to(DEVICE), u_lstm.to(DEVICE)
            
            # Train Router
            # Input includes current frame x[:, -1]
            curr_frame = x[:, -1].unsqueeze(1)
            
            final_pred, _ = routing_model(
                p_flow, u_flow, 
                p_morph, u_morph,
                p_diff, u_diff, 
                p_lstm, u_lstm,
                curr_frame
            )
            
            loss = criterion(final_pred, y_dim)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Routing Epoch {epoch+1} Loss: {total_loss:.4f}")
        
    # Save everything
    torch.save(expert_morph.state_dict(), 'expert_morph.pth')
    torch.save(expert_diff.state_dict(), 'expert_diff.pth')
    torch.save(expert_lstm.state_dict(), 'expert_lstm.pth')
    torch.save(routing_model.state_dict(), 'routing_net.pth')
    print("AURORA models saved successfully.")

if __name__ == "__main__":
    loaders = load_data()
    if loaders:
        train_l, val_l = loaders
        # Step 1: Train component experts
        ex_flow, ex_morph, ex_diff, ex_lstm = train_experts(train_l)
        # Step 2: Train the routing brain
        train_routing(ex_flow, ex_morph, ex_diff, ex_lstm, train_l)
