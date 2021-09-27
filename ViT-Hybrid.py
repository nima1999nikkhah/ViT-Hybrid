import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import math
from efficientnet_pytorch import EfficientNet
from icecream import ic
import json
from PIL import Image
from einops.layers.torch import Rearrange

parser = argparse.ArgumentParser(description='ClassifierHybrid')

parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='Number of epochs to train (default: 1000)')
parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                    help='Learning rate (default: 1e-2)')
parser.add_argument('--img_size', type=int, default=260,
                    help='Image size as input (default: 260)')
parser.add_argument('--d_model', type=int, default=16,
                    help='Dimension of embeddings (default: 768)')
parser.add_argument('--d_mlp', type=int, default=256,
                    help='Dimension of MLP inside transformer (default: 3072)')
parser.add_argument('--num_heads', type=int, default=8,
                    help='Number of heads (default: 8)')
parser.add_argument('--num_layers', type=int, default=8,
                    help='Number of layers (default: 8)')
parser.add_argument('--num_classes', type=int, default=196,
                    help='Number of classes (default: 196)')
parser.add_argument('--num_patches', type=int, default=9,
                    help='Number of patches (default: 9)')
parser.add_argument('--patch_size', type=int, default=3,
                    help='Patche sizes (default: 3)')
parser.add_argument('--num_channels', type=int, default=1408,
                    help='Number of channels after backbone (default: 1408)')
parser.add_argument('--data_json', type=str, default='tmpe.json',
                    help='The root of the json file')
parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                    help='Input batch size for training (default: 10)')

args = parser.parse_args()

class vit_data(Dataset):

    def __init__(self, pth, transform_image, img_size):

        with open(pth, 'r') as f:
            self.dlist = json.load(f)

        self.trnsfrm = transform_image
        self.img_size = img_size

    def __len__(self):

        return len(self.dlist)

    def __getitem__(self, indx):

        image_ = Image.open(self.dlist[indx][0])

        img_ = image_.resize((self.img_size,self.img_size))

        img = self.trnsfrm(img_)

        clss = torch.tensor(self.dlist[indx][1])

        return (img, clss)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError("Embedding dimension {0}\
                should be divisible by number of heads {1}".format(embed_dim, num_heads))
        self.proj_dim = embed_dim // num_heads
        self.query_mat = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.query_mat.weight)
        nn.init.zeros_(self.query_mat.bias)
        self.key_mat = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.key_mat.weight)
        self.value_mat = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.value_mat.weight)
        self.combine_mat = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.combine_mat.weight)

    def attention(self, query, key, value):

        score = torch.matmul(query, torch.transpose(key, 2, 3))
        d_key = key.size(-1)
        scaled_score = score / math.sqrt(d_key)
        s = nn.Softmax(dim=-1)
        weights = s(scaled_score)
        output = torch.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = torch.reshape(x, (batch_size, -1, self.num_heads, self.proj_dim))
        return torch.transpose(x, 1, 2)

    def forward(self, inputs):

        batch_size = inputs.size(0)

        query = self.query_mat(inputs)
        key = self.key_mat(inputs)
        value = self.value_mat(inputs)

        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = torch.transpose(attention, 1, 2)
        concat_attention = torch.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_mat(concat_attention)

        return output

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),

            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, embed_dim),

            nn.Dropout(dropout_rate)
        )

        self.layernorm1 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.init_weights()

    def init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, inputs):
        x = self.layernorm1(inputs)
        x = self.attn(x)
        x = self.dropout1(x)
        x = x + inputs
        y = self.layernorm2(x)
        y = self.mlp(x)

        out = x + y

        return out

class ViT(nn.Module):
    def __init__(self, img_size, channels, patch_size, num_layers,
        num_classes, d_model, num_heads, d_mlp, dropout_rate=0.1):
        super(ViT, self).__init__()

        num_patches = (img_size // patch_size) ** 2
        self.patch_dim = channels * patch_size ** 2
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = torch.rand(size=(1, num_patches + 1, d_model), requires_grad=True)

        self.dropout = nn.Dropout(dropout_rate)

        self.class_embedding = torch.zeros(size=(1, 1, d_model), requires_grad=True)

        self.patch_projection = nn.Linear(self.patch_dim, d_model)
        self.encoder_layers = [TransformerBlock(d_model, num_heads, d_mlp, dropout_rate)
                for _ in range(num_layers)]
        self.layernorm = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.mlp_head = nn.Linear(d_model, num_classes)
        nn.init.zeros_(self.mlp_head.weight)


    def extract_patches(self, images):
        batch_size = images.size(0)

        patches = Rearrange('b c (h p) (w q) -> b (c p q) h w', p=self.patch_size, q=self.patch_size)

        patches = patches(images)
        patches = torch.reshape(patches, (batch_size, -1, self.patch_dim))

        return patches

    def forward(self, x):
        batch_size = x.size(0)
        patches = self.extract_patches(x)
        x = self.patch_projection(patches)

        class_embedding = self.class_embedding.expand(batch_size, 1, self.d_model)
        x = torch.cat((class_embedding, x), dim=1)

        x = x + self.pos_embedding
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.layernorm(x)

        output = self.mlp_head(x[:, 0])
        return output

def main():

    transform_image = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    dtst = vit_data(pth=args.data_json, transform_image=transform_image, img_size=args.img_size)
    dtld = DataLoader(dtst, batch_size=args.batch_size)

    vision_transformer = ViT(img_size=args.num_patches,
                             channels=args.num_channels,
                             patch_size=args.patch_size,
                             num_layers=args.num_layers,
                             num_classes=args.num_classes,
                             d_model=args.d_model,
                            num_heads=args.num_heads,
                              d_mlp=args.d_mlp)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(vision_transformer.parameters(), lr=args.lr, betas=(0.5, 0.999))

    my_model = EfficientNet.from_pretrained('efficientnet-b2')
    for param in my_model.parameters():
        param.requires_grad = False

    for epoch in range(args.epochs):
        for i, data in enumerate(dtld, 0):

            inpt, label = data

            optimizer.zero_grad()

            features = my_model.extract_features(inpt)

            output = vision_transformer(features)

            flag = (torch.argmax(output,dim=1)==label)

            sm = torch.sum(flag)

            error = criterion(output, label)

            error.backward()

            optimizer.step()

if __name__ == "__main__":
    main()
