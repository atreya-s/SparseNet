#Making UNet Model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class SparseAutoencoder(nn.Module):
    def __init__(self, in_channels, sparsity_lambda=1e-4, sparsity_target=0.05):
        super().__init__()
        self.sparsity_lambda = sparsity_lambda
        self.sparsity_target = sparsity_target

        # Modified encoder/decoder for 2D input
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 1)
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 1)
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        sparsity_loss = self.sparsity_penalty(encoded)
        return decoded, sparsity_loss

    def sparsity_penalty(self, encoded):
        # Calculate mean activation across all dimensions except channels
        rho_hat = torch.mean(encoded, dim=[0, 2, 3])
        rho = torch.tensor(self.sparsity_target).to(encoded.device)
        
        
        epsilon = 1e-8
        rho_hat = torch.clamp(rho_hat, min=epsilon, max=1-epsilon)
        
        
        kl_div = rho * torch.log(rho/rho_hat) + (1-rho) * torch.log((1-rho)/(1-rho_hat))
        return self.sparsity_lambda * torch.sum(kl_div)

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], sparsity_target=0.05, beta=1e-3):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.sparse_autoencoders = nn.ModuleList()
        self.beta = beta

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            self.sparse_autoencoders.append(SparseAutoencoder(feature, sparsity_target=sparsity_target))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 3, feature))  # Modified to account for concatenated sparse output

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        sparse_outputs = []
        total_sparsity_loss = 0

        # Down path
        for down, autoencoder in zip(self.downs, self.sparse_autoencoders):
            x = down(x)
            skip_connections.append(x)
            
            # Apply sparse autoencoder
            sparse_out, sparsity_loss = autoencoder(x)
            sparse_outputs.append(sparse_out)
            total_sparsity_loss += sparsity_loss
            
            x = self.pool(x)

        x = self.bottleneck(x)
        
        # Reverse lists for up path
        skip_connections = skip_connections[::-1]
        sparse_outputs = sparse_outputs[::-1]

        # Up path
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            sparse_output = sparse_outputs[idx // 2]

           
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
                sparse_output = TF.resize(sparse_output, size=skip_connection.shape[2:])

            # Concatenate skip connection and sparse output
            concat_skip = torch.cat((skip_connection, x, sparse_output), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x), self.beta * total_sparsity_loss
    

def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    
    preds, sparsity_loss = model(x)
    
    
    print("Input shape:", x.shape)
    print("Output shape:", preds.shape)
    print("Sparsity loss:", sparsity_loss.item())
    
    
    assert preds.shape == x.shape, f"Shape mismatch: input {x.shape} vs output {preds.shape}"
    
    print("Test passed successfully!")

if __name__ == "__main__":
    test()
