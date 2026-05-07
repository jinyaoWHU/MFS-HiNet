import torch
import torch.nn as nn
import torch.nn.functional as F 
from hyptorch import hynn,pmath
import torch.utils.checkpoint as checkpoint


class SSConv(nn.Module):
    '''
    Spectral-Spatial Convolution
    '''
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(SSConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            groups=out_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.LeakyReLU()
        self.BN=nn.BatchNorm2d(in_ch)
        
    
    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.Act1(out)
        out = self.depth_conv(out)
        out = self.Act2(out)
        return out


class BackboneHi(nn.Module):

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        num_layers: int,
        curvature,
        kernel_size: int
    ):
        super().__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.kernel_size = kernel_size

        feature_dim = 32
        hidden_dim = 64
        poincare_dim = 128

        self.dropout_rate = 0.0

        # ======================================
        # Hyperbolic projection
        # ======================================
        self.to_poincare = hynn.ToPoincare(
            c=curvature,
            train_c=True,
            train_x=True,
            ball_dim=feature_dim,
            riemannian=True,
            clip_r=None
        )

        # ======================================
        # Spectral denoising module
        # ======================================
        self.spectral_denoise = nn.Sequential()

        for layer_idx in range(num_layers):

            in_channels = (
                input_channels
                if layer_idx == 0
                else feature_dim
            )

            self.spectral_denoise.add_module(
                f"bn_{layer_idx}",
                nn.BatchNorm2d(in_channels)
            )

            self.spectral_denoise.add_module(
                f"conv_{layer_idx}",
                nn.Conv2d(
                    in_channels,
                    feature_dim,
                    kernel_size=1
                )
            )

            self.spectral_denoise.add_module(
                f"act_{layer_idx}",
                nn.LeakyReLU()
            )

        # ======================================
        # Spatial-spectral feature extraction
        # ======================================
        self.spatial_spectral_branch = nn.Sequential()

        for layer_idx in range(num_layers):

            self.spatial_spectral_branch.add_module(
                f"ssconv_{layer_idx}",
                SSConv(
                    feature_dim,
                    feature_dim,
                    kernel_size=kernel_size
                )
            )

        # ======================================
        # Euclidean classifier
        # ======================================
        self.classifier = nn.Sequential(

            nn.Linear(128, hidden_dim),

            nn.Dropout(self.dropout_rate),

            nn.Linear(hidden_dim, feature_dim),

            nn.Dropout(self.dropout_rate),

            nn.Linear(feature_dim, num_classes)
        )

        # ======================================
        # Hyperbolic linear layer
        # ======================================
        self.hyperbolic_linear = hynn.HypLinear(
            input_channels,
            poincare_dim,
            c=0.5,
            bias=True
        )

        # ======================================
        # Output projection
        # ======================================
        self.output_layer = nn.Linear(
            feature_dim,
            num_classes
        )

    def forward(self, x: torch.Tensor):

        height, width, bands = x.shape

        # [H,W,C] -> [1,C,H,W]
        x = x.permute(2, 0, 1).unsqueeze(0)

        # Spectral denoising
        denoise_feature = self.spectral_denoise(x)

        # Spatial-spectral feature extraction
        spatial_feature = self.spatial_spectral_branch(
            denoise_feature
        )

        # [1,C,H,W] -> [H,W,C]
        spatial_feature = (
            spatial_feature
            .squeeze(0)
            .permute(1, 2, 0)
        )

        # Flatten spatial dimension
        spatial_feature = spatial_feature.reshape(
            height * width,
            -1
        )

        # Hyperbolic projection
        hyperbolic_feature, curvature = self.to_poincare(
            spatial_feature
        )

        return hyperbolic_feature, curvature

class FeatureExtractor(nn.Module):

    def __init__(
        self,
        input_channels,
        num_layers,
        feature_dim,
        kernel_size
    ):
        super().__init__()

        self.denoise = nn.Sequential()

        for layer_idx in range(num_layers):

            in_channels = (
                input_channels
                if layer_idx == 0
                else feature_dim
            )

            self.denoise.add_module(
                f"bn_{layer_idx}",
                nn.BatchNorm2d(in_channels)
            )

            self.denoise.add_module(
                f"conv_{layer_idx}",
                nn.Conv2d(
                    in_channels,
                    feature_dim,
                    kernel_size=1
                )
            )

            self.denoise.add_module(
                f"act_{layer_idx}",
                nn.LeakyReLU()
            )

        self.feature_branch = nn.Sequential()

        for layer_idx in range(num_layers):

            self.feature_branch.add_module(
                f"ssconv_{layer_idx}",
                SSConv(
                    feature_dim,
                    feature_dim,
                    kernel_size=kernel_size
                )
            )

    def forward(self, x):

        h, w, _ = x.shape

        x = x.permute(2, 0, 1).unsqueeze(0)

        x = self.denoise(x)

        x = self.feature_branch(x)

        x = x.squeeze(0).permute(1, 2, 0)

        x = x.reshape(h * w, -1)

        return x
    
class MFSCN(nn.Module):

    def __init__(
        self,
        input_channels,
        local_class_num,
        global_class_num,
        num_layers,
        kernel_size
    ):
        super().__init__()

        feature_dim = 32
        dropout_rate = 0.0

        # ==================================
        # Three feature branches
        # ==================================
        self.euc_branch = FeatureExtractor(
            input_channels,
            num_layers,
            feature_dim,
            kernel_size
        )

        self.hyperbolic_branch = FeatureExtractor(
            input_channels,
            num_layers,
            feature_dim,
            kernel_size
        )

        self.global_branch = FeatureExtractor(
            input_channels,
            num_layers,
            feature_dim,
            kernel_size
        )

        # ==================================
        # Hyperbolic modules
        # ==================================
        self.to_poincare = hynn.ToPoincare(
            c=1,
            train_c=True,
            train_x=True,
            ball_dim=feature_dim,
            riemannian=False
        )

        self.hyperbolic_linear = hynn.HypLinear(
            feature_dim,
            feature_dim,
            c=0.5,
            bias=False
        )

        self.hyperbolic_classifier = hynn.HyperbolicMLR(
            feature_dim,
            local_class_num,
            c=0.5
        )

        # ==================================
        # Classification heads
        # ==================================
        self.euc_classifier = nn.Linear(
            feature_dim,
            local_class_num
        )

        self.global_classifier = nn.Linear(
            feature_dim,
            global_class_num
        )

        self.fusion_classifier = nn.Linear(
            feature_dim * 3,
            local_class_num
        )

    def forward(self, x, curvature):

        # ==================================
        # Euclidean branch
        # ==================================
        euc_feature = self.euc_branch(x)

        euc_output = self.euc_classifier(
            euc_feature
        )

        # ==================================
        # Hyperbolic branch
        # ==================================
        hyper_feature = self.hyperbolic_branch(x)

        hyper_feature, curvature = self.to_poincare(
            hyper_feature
        )

        hyper_output = self.hyperbolic_linear(
            hyper_feature,
            curvature
        )

        hyper_output = self.hyperbolic_classifier(
            hyper_output,
            curvature
        )

        # ==================================
        # Global branch
        # ==================================
        global_feature = self.global_branch(x)

        global_output = self.global_classifier(
            global_feature
        )

        # ==================================
        # Fusion
        # ==================================
        fusion_feature = torch.cat(
            (
                euc_feature,
                hyper_feature,
                global_feature
            ),
            dim=1
        )

        fusion_output = self.fusion_classifier(
            fusion_feature
        )

        return (
            euc_feature,
            hyper_feature,
            global_feature,
            curvature,
            euc_output,
            hyper_output,
            global_output,
            fusion_output
        )

