import numpy as np
import torch
import torch.nn as nn
from conv2d import Conv2d

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5),
                                                   attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # permute之后若想用view聚合维度必须使用contiguous
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..

        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


# https://github.com/pytorch/pytorch/issues/3867#issuecomment-407663012
# only support stride == 1
class Conv2dSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka,kb,ka,kb)),
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        )
    def forward(self, x):
        return self.net(x)


class TwoD_Attention_Layer(nn.Module):
    def __init__(self, n = 64,c =64,dropout=0.1):
        super(TwoD_Attention_Layer, self).__init__()

        self.n = n
        self.c = c
        self.convq = Conv2d(n,c,3,1).cuda()
        self.convk = Conv2d(n,c,3,1).cuda()
        self.convv = Conv2d(n,c,3,1).cuda()
        self.conv = Conv2d(2*c,n,3,1).cuda()
        self.bnq = nn.BatchNorm2d(c).cuda()
        self.bnk = nn.BatchNorm2d(c).cuda()
        self.bnv = nn.BatchNorm2d(c).cuda()
        self.bn = nn.BatchNorm2d(n).cuda()
        self.SA_time = ScaledDotProductAttention(temperature=1., attn_dropout=dropout)
        self.SA_freq = ScaledDotProductAttention(temperature=1., attn_dropout=dropout)

        self.final_conv1 = Conv2d(n,c,3).cuda()
        self.final_conv1_act = nn.ReLU().cuda()
        self.final_conv2 = Conv2d(c,c,3).cuda()
        self.bnf1 = nn.BatchNorm2d(c).cuda()
        self.bnf2 = nn.BatchNorm2d(c).cuda()
        self.act = nn.ReLU().cuda()

    def forward(self, inputs):
        '''
        :param inputs: B*n*T*D
        :return: B*n*T*D
        '''
        residual = inputs # B*T*D*n

        q = self.bnq(self.convq(inputs)) # B*c*T*D
        k = self.bnk(self.convk(inputs))
        v = self.bnv(self.convv(inputs))
        sz_b, c, len_q, d_q = q.size()
        _, c, len_k, d_k = k.size()
        _, c, len_v, d_v = v.size()

        q_time = q.view(-1, len_q, d_q)
        k_time = k.view(-1, len_k, d_k)
        v_time = v.view(-1, len_v, d_v)

        q_fre = q.permute(0, 1, 3, 2).contiguous().view(-1,d_q,len_q)
        k_fre = k.permute(0, 1, 3, 2).contiguous().view(-1,d_k,len_k)
        v_fre = v.permute(0, 1, 3, 2).contiguous().view(-1,d_v,len_v)

        self.SA_time.temperature = np.power(d_q,0.5)
        scaled_attention_time, attention_weights_time = self.SA_time(
            q_time, k_time, v_time, None)  # (B*c)*T*D
        self.SA_freq.temperature = np.power(len_q,0.5)
        scaled_attention_fre, attention_weights_fre = self.SA_freq(
            q_fre, k_fre, v_fre, None)  # (B*c)*D*T

        scaled_attention_time = scaled_attention_time.view(sz_b, c, len_q, d_q)

        scaled_attention_fre = scaled_attention_fre.view(sz_b, c, d_q ,len_q)
        scaled_attention_fre = scaled_attention_fre.permute(0, 1, 3, 2)

        out = torch.cat((scaled_attention_time,scaled_attention_fre),dim=1) # B*2c*T*D

        out = self.bn(self.conv(out)) + residual  # B*n*T*D

        final_out = self.bnf1(self.final_conv1_act(self.final_conv1(out)))
        final_out = self.bnf2(self.final_conv2(final_out))

        final_out = self.act(final_out + out)

        return final_out


class Pre_Net(nn.Module):
    def __init__(self,d_mel, d_model=512, num_M=2, n=3, c=64,dropout=0.1):
        super(Pre_Net, self).__init__()
        self.d_mel = d_mel
        self.num_M = num_M
        self.d_model = d_model
        self.n = n
        self.c = c

        self.downsample = Conv2d(n,c,3,2)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.ReLU()

        self.downsample2 = Conv2d(c,c,3,2)
        self.bn2 = nn.BatchNorm2d(c)
        self.act2 = nn.ReLU()

        self.TwoD_layers = [TwoD_Attention_Layer(c, c,dropout=dropout) for _ in range(num_M)]

        self.linear = nn.Linear(d_mel*c//4, d_model)

    def forward(self, inputs):
        '''
        :param inputs: N x Ti x (D_mel*3)
        :return: B*T*d_model
        '''
        B, T, D = inputs.size()
        inputs = inputs.view(B,T,3,-1).permute(0,2,1,3) #N x 3 x Ti x D_mel

        out = self.bn(self.downsample(inputs))
        out = self.bn2(self.downsample2(out))
        # print('downsample.shape:', out.shape)

        for i in range(self.num_M):
            out = self.TwoD_layers[i](out)

        B, c, T, D = out.size()

        out = out.permute(0,2,1,3).contiguous().view(B, T, -1) # B*T*(D*c)

        out = self.linear(out) # B*T*d_model

        return out

if __name__=='__main__':
    inputs = torch.randn(16,100,240)
    prenet = Pre_Net(80,n=3)
    out = prenet(inputs)
    print(out.size())

    a = torch.randn(1,1,6)
    print(a)
    print(a.view(1,1,2,3))