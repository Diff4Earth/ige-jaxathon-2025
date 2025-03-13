import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


################################################################################
# Code for UNETSWE
################################################################################
class DoubleConv2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        num_groups=8,
        norm_type="BatchNorm2d",
        activation="SiLU",
    ):
        super(DoubleConv2D, self).__init__()

        norm_layer = {
            "BatchNorm2d": nn.BatchNorm2d(out_channels),
            "GroupNorm": nn.GroupNorm(num_groups, out_channels),
        }
        activation_layer = {
            "ReLU": nn.ReLU(inplace=True),
            "SiLU": nn.SiLU(inplace=True),
        }

        if norm_type not in norm_layer:
            raise NotImplementedError(
                f"Normalization type {norm_type} is not implemented"
            )
        if activation not in activation_layer:
            raise NotImplementedError(
                f"Activation function {activation} is not implemented"
            )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            norm_layer[norm_type],
            activation_layer[activation],
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            norm_layer[norm_type],
            activation_layer[activation],
        )

    def forward(self, x):
        return self.conv(x)


################################################################################
# Code for UNETSWE
################################################################################
class UNET2D(torch.nn.Module):
    def __init__(
        self,
        name,
        condi_net,
        num_layers,
        in_channels,
        out_channels,
        features,
        kernel_size=3,
        stride=1,
        padding=1,
        norm_type="BatchNorm2d",
        activation="SiLU",
    ):
        super(UNET2D, self).__init__()
        assert condi_net in [True, False, None], "condi_net should be a boolean value"
        self.name = name
        self.condi_net = condi_net
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.norm_type = norm_type
        self.activation = activation

        self.dynamics = nn.Conv2d(
            in_channels[0], int(features[0] / 2), kernel_size, stride, padding
        )
        self.statics = nn.Conv2d(
            in_channels[1], int(features[0] / 2), kernel_size, stride, padding
        )

        # Encoder (downsampling) blocks
        self.encoder = nn.ModuleList()
        for i in range(num_layers):
            in_ch = features[i - 1] if i > 0 else features[0]
            out_ch = features[i]
            self.encoder.append(
                DoubleConv2D(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    num_groups=out_ch // 4,
                    norm_type=self.norm_type,
                    activation=self.activation,
                )
            )

        # Pooling layer for downsampling reducing half of the size
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Upsampling layer
        self.up = nn.ModuleList()
        for i in range(num_layers - 1, 0, -1):
            self.up.append(
                nn.ConvTranspose2d(
                    features[i], features[i - 1], kernel_size=2, stride=2
                )
            )

        # Decoder (upsampling) blocks
        self.decoder = nn.ModuleList()
        for i in range(num_layers - 1, 0, -1):
            self.decoder.append(
                DoubleConv2D(
                    in_channels=features[i],
                    out_channels=features[i - 1],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    num_groups=features[i - 1] // 4,
                    norm_type=self.norm_type,
                    activation=self.activation,
                )
            )

        # Output layer
        if len(out_channels) == 1:
            self.output = nn.Conv2d(
                features[0], out_channels[0], kernel_size, stride, padding
            )
        else:
            print("Multiple output channels")
            self.output = nn.ModuleList()
            for i in range(len(out_channels)):
                self.output.append(
                    nn.Conv2d(
                        features[0], out_channels[i], kernel_size, stride, padding
                    )
                )

    def forward(self, x, parameters=None, **kwargs):
        H = kwargs["H"]
        mask = kwargs["mask"]
        x_dynamics = self.dynamics(x)
        x_statics = self.statics(torch.cat([H, mask], dim=1))
        x = torch.cat([x_dynamics, x_statics], dim=1)

        # Encoder
        skip_connections = []
        for i in range(self.num_layers):
            if i == 0:
                x = self.encoder[i](x)
            else:
                x = self.encoder[i](self.pool(x))
            skip_connections.append(x)

        # Decoder
        skip_connections = skip_connections[::-1][
            1:
        ]  # reverse the list and remove the first element
        for i in range(self.num_layers - 1):
            x = self.up[i](x)
            # Ensure the dimensions match by cropping or padding
            if x.size(2) != skip_connections[i].size(2) or x.size(
                3
            ) != skip_connections[i].size(3):
                diffY = skip_connections[i].size(2) - x.size(2)
                diffX = skip_connections[i].size(3) - x.size(3)
                # Calculate the padding
                pad_left = diffX // 2
                pad_right = diffX - pad_left
                pad_top = diffY // 2
                pad_bottom = diffY - pad_top
                # Apply padding
                x = F.pad(
                    x,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=0,
                )
            # Concatenate the skip connection
            x = torch.cat([x, skip_connections[i]], dim=1)
            x = self.decoder[i](x)

        dx_dt = self.output(x)
        return dx_dt

        # x = self.decoder1(torch.cat([x4, self.up(x5)], dim=1))
        # x = self.decoder2(torch.cat([x3, self.up(x)], dim=1))
        # x = self.decoder3(torch.cat([x2, self.up(x)], dim=1))
        # x = self.decoder4(torch.cat([x1, self.up(x)], dim=1))
        # x = self.decoder5(x)
        # dx_dt = self.output(x)
        # return dx_dt
