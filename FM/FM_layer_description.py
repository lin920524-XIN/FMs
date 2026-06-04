## 二次項的類別
class FactorizationMachine(torch.nn.Module):

    ## 初始化變數類別
    def __init__(self, reduce_sum=True):

        # 呼叫父類別的初始化變數
        super().__init__()

        # 是否要依照 Embedding 的維度方向作加總
        self.reduce_sum = reduce_sum

    ## 前向傳遞的類別
    def forward(self, x):

        # 相加後平方項
        square_of_sum = torch.sum(x, dim=1) ** 2

        # 平方後相加項
        sum_of_square = torch.sum(x ** 2, dim=1)

        # 相加後平方和平方後相加的結果
        ix = square_of_sum - sum_of_square

        # 如果要依照 Embedding 的維度方向作加總
        if self.reduce_sum:

            # 依照第二維度的方向作元素加總
            ix = torch.sum(ix, dim=1, keepdim=True)

        # 將結果除以二，即可計算出二次項的部分
        return 0.5 * ix

## Embedding 項的類別
class FeaturesEmbedding(torch.nn.Module):

    ## 初始化變數類別
    def __init__(self, field_dims, embed_dim):

        # 呼叫父類別的初始化變數
        super().__init__()

        # 建立隱向量矩陣的 Embedding 張量
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)

        # 建立索引的偏移列表
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

        # 設定隱向量矩陣的 Embedding 張量的初始化數值設定
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    ## 前向傳遞的類別
    def forward(self, x):

        # 先將輸入資料的索引做偏移
        x = x + x.new_tensor(self.offsets).unsqueeze(0)

        # 再經過隱向量矩陣的 Embedding 張量的查表，即可獲得 vx
        return self.embedding(x)

## 線性項的類別
class FeaturesLinear(torch.nn.Module):

    ## 初始化變數類別
    def __init__(self, field_dims, output_dim=1):

        # 呼叫父類別的初始化變數
        super().__init__()

        # 建立線性權重的 Embedding 張量
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)

        # 建立全域偏置的參數
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))

        # 建立索引的偏移列表
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    ## 前向傳遞的類別
    def forward(self, x):

        # 先將輸入資料的索引做偏移
        x = x + x.new_tensor(self.offsets).unsqueeze(0)

        # 先經過線性權重的的 Embedding 張量的查表後，加入全域偏置後即可得到線性的部分
        return torch.sum(self.fc(x), dim=1) + self.bias
