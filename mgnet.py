import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F

class MgIte(nn.Module):
    def __init__(self, A, B):
        super().__init__()
        self.A = A
        self.B = B        
        self.bn1 =nn.BatchNorm2d(A.weight.size(0))
        self.bn2 =nn.BatchNorm2d(B.weight.size(0))

    def forward(self, out):
        u, f = out
        u = u + F.relu(self.bn2(self.B(F.relu(self.bn1((f-self.A(u)))))))
        out = (u, f)
        return out

    
class MgRestriction(nn.Module):
    def __init__(self, A_old, A_conv, Pi_conv, R_conv):
        super().__init__()
        self.A_old = A_old
        self.A_conv = A_conv
        self.Pi_conv = Pi_conv
        self.R_conv = R_conv

        self.bn1 = nn.BatchNorm2d(Pi_conv.weight.size(0))
        self.bn2 = nn.BatchNorm2d(A_old.weight.size(0))

    def forward(self, out):
        u_old, f_old = out
        u = F.relu(self.bn1(self.Pi_conv(u_old)))
        f = F.relu(self.bn2(self.R_conv(f_old-self.A_old(u_old)))) + self.A_conv(u)
        out = (u,f)
        return out


class MgNet(nn.Module):
    def __init__(self, args,num_classes):
        super().__init__()
        self.num_iteration = args.num_ite
        self.args = args
        
        # inilization layer
        if args.dataset == 'mnist':
            self.num_channel_input=1
        else:
            self.num_channel_input=3
        self.conv1 = nn.Conv2d(self.num_channel_input, args.num_channel_f, kernel_size=3, stride=1,\
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(args.num_channel_f)        

        A_conv = nn.Conv2d(args.num_channel_u, args.num_channel_f, kernel_size=3, stride=1, padding=1, bias=False)
        if not args.wise_B:
            B_conv = nn.Conv2d(args.num_channel_f, args.num_channel_u, kernel_size=3,stride=1, padding=1, bias=False)

        layers = []
        for l, num_iteration_l in enumerate(self.num_iteration):
            for i in range(num_iteration_l):
                if args.wise_B:
                    B_conv = nn.Conv2d(args.num_channel_f, args.num_channel_u, kernel_size=3,\
                                       stride=1, padding=1, bias=False)
                
                layers.append(MgIte(A_conv, B_conv))

            setattr(self, 'layer'+str(l), nn.Sequential(*layers))


            if l < len(self.num_iteration)-1:
                A_old = A_conv
                #B_old = B_conv
                A_conv = nn.Conv2d(args.num_channel_u, args.num_channel_f, kernel_size=3,\
                                   stride=1, padding=1, bias=False)
                if not args.wise_B:
                    B_conv = nn.Conv2d(args.num_channel_f, args.num_channel_u, kernel_size=3,\
                                       stride=1, padding=1, bias=False)
                Pi_conv = nn.Conv2d(args.num_channel_u, args.num_channel_u, kernel_size=3,\
                                    stride=2, padding=1, bias=False)
                R_conv = nn.Conv2d(args.num_channel_f, args.num_channel_u, kernel_size=3, \
                                   stride=2, padding=1, bias=False)
                layers= [MgRestriction(A_old, A_conv, Pi_conv, R_conv)]
                
        
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(args.num_channel_u ,num_classes)

    def forward(self, u):
        f = F.relu(self.bn1(self.conv1(u)))
        if torch.cuda.is_available():
            u = torch.zeros(f.size(),device=torch.device('cuda'))
        else:
            u = torch.zeros(f.size())
        out = (u, f)

        for l in range(len(self.num_iteration)):
            out = getattr(self, 'layer'+str(l))(out)
        u,f = out
        u = self.pooling(u)
        u = u.view(u.shape[0], -1)
        u = self.fc(u)
        return u