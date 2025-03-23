import torch
import torch.nn as nn



class StyleModel(nn.Module):
    def __init__(self,emb_size,table_input_size):
        super(StyleModel,self).__init__()
        self.emb_processor1 = nn.Linear(emb_size,256)
        self.emb_processor2 = nn.Linear(256,128)
        self.emb_processor3 = nn.Linear(128,64)
        self.batchNorm_emb = nn.BatchNorm1d(64)
        #
        self.table_processor1 = nn.Linear(table_input_size,128)
        self.table_processor2 = nn.Linear(128,64)
        self.table_processor3 = nn.Linear(64,32)
        self.batchNorm_table = nn.BatchNorm1d(32)
        #
        self.combined_processor1 = nn.Linear(64+32,64)
        self.output = nn.Linear(64,2)
        #
        self.dropout = nn.Dropout(p=0.2)
        self.silu = nn.SiLU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self,embedding,table_input):
        emb_out = self.silu(self.emb_processor1(embedding))
        emb_out = self.silu(self.emb_processor2(emb_out))
        emb_out = self.silu(self.emb_processor3(emb_out))
        emb_out = self.batchNorm_emb(emb_out)
        emb_out = self.dropout(emb_out)
        #
        table_out = self.silu(self.table_processor1(table_input))
        table_out = self.silu(self.table_processor2(table_out))
        table_out = self.silu(self.table_processor3(table_out))
        table_out = self.batchNorm_table(table_out)
        table_out = self.dropout(table_out)
        #
        out = torch.cat((emb_out, table_out), dim=1)
        out = self.silu(self.combined_processor1(out))
        out = self.relu(self.output(out))
        return out
        





