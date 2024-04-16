import torch
import torch.nn as nn
import torch.nn.functional as F
from imagenet import get_model
from torch.utils.checkpoint import checkpoint



class SimpleMIL(nn.Module):
    def __init__(self, backbone_name, instance_hidden_size=32, hidden_size=64, num_classes=2, pretrained_backbone=True):
        super(SimpleMIL, self).__init__()
        self.instance_hidden_size = instance_hidden_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.mode = 'attention'
        self.backbone = get_model(backbone_name, instance_hidden_size, pretrained_backbone)

        for param in self.backbone.parameters():
            param.requires_grad = True

        self.mil_classifier = nn.Sequential(
            # nn.ReLU(),
            nn.Linear(self.instance_hidden_size, 2),
        )
        self.bn = nn.Sequential(
            nn.BatchNorm1d(self.instance_hidden_size)
        )

        self.attention = nn.Sequential(
            nn.Linear(self.instance_hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1)
        )

        if self.mode == 'transformer':
            self.embedder = nn.Sequential(
                nn.Linear(1, self.instance_hidden_size, bias=True)
            )
            self.transformer = nn.Sequential(
                nn.TransformerEncoder(nn.TransformerEncoderLayer(self.instance_hidden_size, 2, self.instance_hidden_size), num_layers=2),
            )


    def forward(self, x):
        batch_size = x.shape[0]
        mil_size = x.shape[1]
        temp_shape = [-1] + list(x.shape[2:])
        # reshape to push all images through at once
        x_view = x.view(temp_shape)
        features = self.backbone(x_view)
        # now split up again
        split_shape = [batch_size, mil_size, -1]
        features_view = features.view(split_shape)
        all_classifications = []
        for b in range(batch_size):
            features_single = features_view[b]
            if self.mode == 'attention':
                attention = self.attention(features_single)
                attention = torch.transpose(attention, 1, 0)
                attention_weights = F.softmax(attention, dim=1)
                aggregated_features = torch.mm(attention_weights, features_single)
                mil_classification = self.mil_classifier(aggregated_features)
            elif self.mode == 'max':
                aggregated_features, _ = torch.max(features_single, dim=0, keepdim=True)
                mil_classification = self.mil_classifier(aggregated_features)
            elif self.mode == 'transformer':
                class_vec = self.embedder(torch.zeros(1, 1).to(features_single.device))
                features_transformer = torch.cat([class_vec, features_single]).view([1, -1, features_single.shape[-1]])
                transformed = self.transformer(features_transformer)
                mil_classification = self.mil_classifier(transformed[:,0])
            else:
                aggregated_features = torch.mean(features_single, dim=0, keepdim=True)
                mil_classification = self.mil_classifier(aggregated_features)
            all_classifications.append(mil_classification)
        final_classification = torch.cat(all_classifications, dim=0)
        return final_classification, final_classification