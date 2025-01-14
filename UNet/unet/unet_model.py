""" Full assembly of the parts to form the complete network """

from .unet_parts import *



""" 2 down """
# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=False, filterBase =16):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.inc = (DoubleConv(n_channels, filterBase))
#         self.down1 = (Down(filterBase * 1, filterBase * 2))
#         factor = 2 if bilinear else 1
#         self.down2 = (Down(filterBase * 2, filterBase * 4 // factor))
#         self.up1 = (Up(filterBase * 4, filterBase * 2 // factor, bilinear))
#         self.up2 = (Up(filterBase * 2, filterBase * 1, bilinear))
#         self.outc = (OutConv(filterBase, n_classes))

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x = self.up1(x3, x2)
#         x = self.up2(x, x1)
#         logits = self.outc(x)
#         return logits

#     def use_checkpointing(self):
#         self.inc = torch.utils.checkpoint(self.inc)
#         self.down1 = torch.utils.checkpoint(self.down1)
#         self.down2 = torch.utils.checkpoint(self.down2)
#         self.up1 = torch.utils.checkpoint(self.up1)
#         self.up2 = torch.utils.checkpoint(self.up2)
#         self.outc = torch.utils.checkpoint(self.outc)


""" 3 down """
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, filterBase = 32):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, filterBase))
        self.down1 = (Down(filterBase * 1, filterBase * 2))
        self.down2 = (Down(filterBase * 2, filterBase * 4))
        factor = 2 if bilinear else 1
        self.down3 = (Down(filterBase * 4, filterBase * 8 // factor))
        self.up1 = (Up(filterBase * 8, filterBase * 4 // factor, bilinear))
        self.up2 = (Up(filterBase * 4, filterBase * 2 // factor, bilinear))
        self.up3 = (Up(filterBase * 2, filterBase * 1, bilinear))
        self.outc = (OutConv(filterBase, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.outc = torch.utils.checkpoint(self.outc)

""" 4 down """
# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=False, filterBase = 64):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.inc = (DoubleConv(n_channels, filterBase))
#         self.down1 = (Down(filterBase * 1, filterBase * 2))
#         self.down2 = (Down(filterBase * 2, filterBase * 4))
#         self.down3 = (Down(filterBase * 4, filterBase * 8))
#         factor = 2 if bilinear else 1
#         self.down4 = (Down(filterBase * 8, filterBase * 16 // factor))
#         self.up1 = (Up(filterBase * 16, filterBase * 8 // factor, bilinear))
#         self.up2 = (Up(filterBase * 8, filterBase * 4 // factor, bilinear))
#         self.up3 = (Up(filterBase * 4, filterBase * 2 // factor, bilinear))
#         self.up4 = (Up(filterBase * 2, filterBase * 1, bilinear))
#         self.outc = (OutConv(filterBase, n_classes))

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits

#     def use_checkpointing(self):
#         self.inc = torch.utils.checkpoint(self.inc)
#         self.down1 = torch.utils.checkpoint(self.down1)
#         self.down2 = torch.utils.checkpoint(self.down2)
#         self.down3 = torch.utils.checkpoint(self.down3)
#         self.down4 = torch.utils.checkpoint(self.down4)
#         self.up1 = torch.utils.checkpoint(self.up1)
#         self.up2 = torch.utils.checkpoint(self.up2)
#         self.up3 = torch.utils.checkpoint(self.up3)
#         self.up4 = torch.utils.checkpoint(self.up4)
#         self.outc = torch.utils.checkpoint(self.outc)




# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=False):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.inc = (DoubleConv(n_channels, 64))
#         self.down1 = (Down(64, 128))
#         self.down2 = (Down(128, 256))
#         self.down3 = (Down(256, 512))
#         factor = 2 if bilinear else 1
#         self.down4 = (Down(512, 1024 // factor))
#         self.up1 = (Up(1024, 512 // factor, bilinear))
#         self.up2 = (Up(512, 256 // factor, bilinear))
#         self.up3 = (Up(256, 128 // factor, bilinear))
#         self.up4 = (Up(128, 64, bilinear))
#         self.outc = (OutConv(64, n_classes))

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits

#     def use_checkpointing(self):
#         self.inc = torch.utils.checkpoint(self.inc)
#         self.down1 = torch.utils.checkpoint(self.down1)
#         self.down2 = torch.utils.checkpoint(self.down2)
#         self.down3 = torch.utils.checkpoint(self.down3)
#         self.down4 = torch.utils.checkpoint(self.down4)
#         self.up1 = torch.utils.checkpoint(self.up1)
#         self.up2 = torch.utils.checkpoint(self.up2)
#         self.up3 = torch.utils.checkpoint(self.up3)
#         self.up4 = torch.utils.checkpoint(self.up4)
#         self.outc = torch.utils.checkpoint(self.outc)