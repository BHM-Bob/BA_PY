import torch
import torch.nn as nn
import spacy

SEQ_MAX_LEN = 522

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [batch size, seq len, hid dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        #x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        
        #x = [batch size, seq len, hid dim]
        
        return x

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0, r'hid_dim % n_heads != 0'
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, head dim , key len ]
        #V = [batch size, n heads, value len, head dim]
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 3, 1)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)                        
        
        #energy = [batch size, n heads, query len, key len]
        energy = torch.matmul(Q, K) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        #attention = [batch size, n heads, query len, key len]
        attention = torch.softmax(energy, dim = -1)                
                
        #x = [batch size, n heads, query len, head dim]
        x = torch.matmul(self.dropout(attention), V)        
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len] 
                
        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        
        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        return src

class Encoder(nn.Module):
    def __init__(self,  input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length = SEQ_MAX_LEN):
        super().__init__()
        self.device = device
        
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device) 
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len]
        #src_mask = [batch size, 1, 1, src len]
        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        #pos = [batch size, src len]
        
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        #src = [batch size, src len, hid dim]
        
        for layer in self.layers:
            src = layer(src, src_mask)
            
        #src = [batch size, src len, hid dim]
            
        return src

class DecoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        # 因为是Seq2Seq模型，在decoder时需要之前时间段的predict信息，所以这里无论是训练还是推理，都会掺入trg
        # 只不过训练时掺入的是真正的trg，推理时掺入的是之前预测的trg
        # 所以在推理时需要防止trg信息泄露
        
        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
        
        #self attention
        # _trg = [batch size, trg len, hid dim]
        # _trg在trg len维度：i: 第0 ~ i token做MHSA
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        
        #dropout, residual connection and layer norm
        # 由于在推理时trg是从<sos>token开始送decoder，然后得到第一个解码token（predicted token），故这里只要做到trg的len与scr的len分离即可
        # 但是对于训练，这里需要做到在并行predict时，某一时间的predict过程不能掺入对应traget信息。
        # TODO : 但是这里的残差似乎是忽视了self.self_attention(trg, trg, trg, trg_mask)对trg_mask的努力，
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
            
        #trg = [batch size, trg len, hid dim]
            
        #encoder attention
        # attention: [N, Lt, Ls]
        # 其中有: x = [N, Lt, Ls] @ [N, Ls, D] = [N, Lt, D]
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        
        #dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
                    
        #trg = [batch size, trg len, hid dim]
        
        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        
        #dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        # 训练时模型输入的src是由<sos>和<eos>包围的，trg有<sos>开头但无<eos>结尾。
        # 所以第一个predicted token参考的trg是<sos>（如果没有trg信息泄露的话），最后的token可能是预测出的<eos>。
        # 而label是双包围的trg去掉了开头的<sos>，所以模型需要预测出<eos>，并且第一个token也是符合的。
        return trg, attention

class Decoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device,
                 max_length = SEQ_MAX_LEN):
        super().__init__()
        
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout, 
                                                  device)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
                
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        #pos = [batch size, trg len]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
                            
        #trg = [batch size, trg len, hid dim]
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
                
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)        
        
        #output = [batch size, trg len, output dim]
        output = self.fc_out(trg)
        
        return output, attention

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        
        #src = [batch size, src len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_trg_mask(self, trg):
        #trg = [batch size, trg len]
        
        #trg_pad_mask = [batch size, 1, 1, trg len]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        trg_len = trg.shape[1]
        
        # torch.tril取下三角矩阵
        #trg_sub_mask = [trg len, trg len]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
            
        #trg_mask = [batch size, 1, trg len, trg len]
        trg_mask = trg_pad_mask & trg_sub_mask
        
        return trg_mask

    def forward(self, src, trg):
        
        #src = [batch size, src len]
        #trg = [batch size, trg len]
                
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        
        enc_src = self.encoder(src, src_mask)
        
        #enc_src = [batch size, src len, hid dim]
                
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return output, attention


def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 50):
    
    model.eval()
        
    if isinstance(sentence, str):
        nlp = spacy.load('de_core_news_sm')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
        
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    # [1, Ls]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    # [1, 1, 1, Ls]
    src_mask = model.make_src_mask(src_tensor)
    
    with torch.no_grad():
        # [1, Ls, D]
        enc_src = model.encoder(src_tensor, src_mask)
        
    # [1, ]
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    # NOTE: 让模型从sos预测出第一个token，送入预测的trg包含sos，而计入模型预测的tokens不包含sos，从而利用sos错开一位
    for i in range(max_len):
        # [1, 1] >>> [1, 2] >>> [1, Lt]
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        # [1, 1, 1, 1] >>> [1, 1, 2, 2] >>> [1, 1, Lt, Lt]
        trg_mask = model.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
            # trg: [1, 1] => [1, 1, D] >>> [1, 2, D] >>> [1, Lt, D]
            # enc_src: [1, Ls, D]
            # trg_mask: [1, 1, 1, 1] >>> [1, 1, 2, 2] >>> [1, 1, Lt, Lt]
            # src_mask: [1, 1, 1, Ls]
            # attention = [1, 1, D] @ [1, D, Ls] = [1, 1, Ls] >>> [1. 2. Ls] >>> [1, Lt, Ls]
            # output = attention @ [1, Ls, D] => [1, 1, D] >>> [1, 2, D] >>> [1, Lt, D]
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        # [1, 1] >>> [1, 2] >>> [1, Lt]
        pred_token = output.argmax(2)[:,-1].item()
        # [2, ] >>> [3. ] >>> [Lt, ]
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
    return trg_tokens[1:], attention