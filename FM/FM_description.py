## FM model 類別
class FactorizationMachineModel(torch.nn.Module):

    ## 初始化變數類別
    def __init__(self, field_dims, embed_dim):

        # 呼叫父類別的初始化變數
        super().__init__()

        # 計算 Embedding (vx) 的部分
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)

        # 計算線性的部分
        self.linear = FeaturesLinear(field_dims)

        # 利用 Embedding (vx) 的結果計算二次項的部分
        self.fm = FactorizationMachine(reduce_sum=True)

    ## 前向傳遞的類別
    def forward(self, x):

        # 和線性項跟二次項作相加
        x = self.linear(x) + self.fm(self.embedding(x))

        # 將結果作二值化(1, 0)
        return torch.sigmoid(x.squeeze(1))
