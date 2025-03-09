import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class CNN_RNN_Regression(nn.Module):
    def __init__(self, cnn_backbone='resnet18', lstm_hidden_size=128, lstm_num_layers=1, dropout_p=0.5):
        super(CNN_RNN_Regression, self).__init__()
        
        # Load a pretrained CNN (transfer learning) and remove its final fc layer.
        if cnn_backbone == 'resnet18':
            cnn = models.resnet18(pretrained=True)
            self.feature_dim = cnn.fc.in_features  # typically 512 for resnet18
        elif cnn_backbone == 'resnet34':
            cnn = models.resnet34(pretrained=True)
            self.feature_dim = cnn.fc.in_features
        else:
            raise ValueError("Unsupported CNN backbone")
            
        # Remove the final fully-connected layer.
        modules = list(cnn.children())[:-1]  # remove last fc layer
        self.cnn_backbone = nn.Sequential(*modules)
        
        # Freeze lower layers if needed (optional):
        # for param in self.cnn_backbone.parameters():
        #     param.requires_grad = False
        
        # Define LSTM to aggregate slice features (we assume each patient scan is a sequence of slices)
        self.lstm = nn.LSTM(input_size=self.feature_dim,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            batch_first=True, 
                            bidirectional=False)
        
        # A dropout layer for regularization
        self.dropout = nn.Dropout(dropout_p)
        
        # Final regression head
        self.fc = nn.Linear(lstm_hidden_size, 1)  # Output a single continuous value
        
    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, channels, H, W)
        """
        batch_size, seq_len, C, H, W = x.size()
        
        # Reshape to process each slice individually through CNN backbone:
        x = x.view(batch_size * seq_len, C, H, W)  # (B * seq_len, C, H, W)
        features = self.cnn_backbone(x)  # output shape: (B * seq_len, feature_dim, 1, 1)
        features = features.view(batch_size, seq_len, self.feature_dim)  # (B, seq_len, feature_dim)
        
        # Optionally, apply dropout to features here
        features = self.dropout(features)
        
        # Pass sequence through LSTM
        lstm_out, (hn, cn) = self.lstm(features)  # lstm_out: (B, seq_len, hidden_size)
        
        # Use the final hidden state of LSTM (last time step)
        # Alternatively, you can use pooling (mean or max) over time
        final_feature = lstm_out[:, -1, :]  # (B, hidden_size)
        
        # Final regression output
        output = self.fc(final_feature)  # (B, 1)
        return output

# Example usage:
if __name__ == '__main__':
    # Create a dummy input: batch of 2 patients, each with 10 CT slices of size 224x224 with 3 channels.
    dummy_input = torch.randn(2, 10, 3, 224, 224)
    model = CNN_RNN_Regression(cnn_backbone='resnet18')
    pred = model(dummy_input)
    print("Output shape:", pred.shape)  # Should be (2, 1)
