# 訓練資料的數量
DATASET_SIZE = 10000

# 測試資料的數量
TEST_NUM = 50000

# 亂數種子
RANDOM_SEED = 42

# 特徵數量
VAR_NUM = 32

# 潛在維度維度
K = VAR_NUM

# 訓練次數
NUM_EPOCH = 1000

# 學習率
LEARNING_RATE = 1e-3

# 批次大小
BATCH_SIZE = 512

# 特徵狀態維度
FIELD_DIMS = [2] * VAR_NUM

## 權重跟交互作用的變數生成函數
def custom_bias_generator(var_num):

    # 用均勻分布生成 -1~1，"特徵數量" 個隨機數值作為權重
    return np.random.uniform(-1, 1, var_num)

# 用 dimod 內建函數生成隨機 bqm 模型
bqm = dimod.generators.gnm_random_bqm(
  
  # 有 "VAR_NUM" 個特徵數量
  variables=VAR_NUM,

  # 計算特徵交互作用的數量
  num_interactions=VAR_NUM*(VAR_NUM-1)/2,

  # 變數的型別(二值化，0或1)
  vartype=dimod.BINARY,
  
  # 用自訂的變數生成函數來生成所有變數
  bias_generator=custom_bias_generator)

# 隨機生成 "DATASET_SIZE" 筆，數值為 0或1 的整數，單筆資料中有 "VAR_NUM" 個特徵的訓練資料
xs_train = np.random.randint(0, 2, (DATASET_SIZE, VAR_NUM), dtype=np.int8)

# 依照先前建立的 bqm ，計算所有訓練資料所對應的能量值(輸出標籤)
ys_train = np.array([bqm.energy(x) for x in xs_train], dtype=np.float64)

# 隨機生成 "DATASET_SIZE" 筆，數值為 0或1 的整數，單筆資料中有 "VAR_NUM" 個特徵的的測試資料
xs_test = np.random.randint(0, 2, (TEST_NUM, VAR_NUM), dtype=np.int8)

# 依照先前建立的 bqm ，計算所有測試資料所對應的能量值(輸出標籤)
ys_test = np.array([bqm.energy(x) for x in xs_test], dtype=np.float64)

# 判斷是否可以用 CUDA 來訓練資料，若否則使用 cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 輸出最終訓練的裝置名稱
print(f"Using device: {device}")

# 生成一個特徵狀態維度是"field_dims"，潛在維度維度為"K"的初始的 FM 模型
model_py = FactorizationMachineModel(field_dims=FIELD_DIMS, embed_dim=K)

# 將 FM 模型放入指定裝置中
model_py.to(device)

# 將訓練資料轉換成 numpy 的整數格式
x_tensor = torch.from_numpy(xs_train).long()

# 將訓練資料所對應的能量值轉換成 numpy 的浮數點格式
y_tensor = torch.from_numpy(ys_train).float()

# 將訓練資料跟其對應的能量值打包成 TensorDataset
train_dataset = TensorDataset(x_tensor, y_tensor)

# 在利用 DataLoader ，依照 "BATCH_SIZE" 的大小來作訓練資料的批次打包
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 設定 MSE 為訓練的損失函數
criterion = nn.MSELoss()

# 設定優化器的種類(Adam)，依照 LEARNING_RATE 的數值來設定學習率，並針對 "model_py.parameters()" 的數值來作梯度計算與數值更新
optimizer = torch.optim.Adam(model_py.parameters(), lr=LEARNING_RATE)

# 將 FM 模型設定成訓練模式(會自動計算梯度值)
model_py.train()

## 依照 "NUM_EPOCH" 的值來作反覆的 FM 模型的完整訓練
for epoch in range(NUM_EPOCH):

    # 每一次的完整訓練前先重置總損失值
    total_loss = 0

    ## 每次都從 "train_loader" 中抓取一個批次訓練資料，反覆進行參數的數值更新直到所有批次資料都有被訓練過
    for i, (inputs, labels) in enumerate(train_loader):

        # 將單一批次資料的訓練輸入資料跟標籤放入裝置中
        inputs, labels = inputs.to(device), labels.to(device)

        # 把輸入資料放入模型作計算，得到輸出預測值
        outputs = model_py(inputs)

        # 依照標籤跟輸出預測值來作損失的計算
        loss = criterion(outputs.squeeze(), labels)

        # 在用優化器來作權重梯度的計算前，需先清空先前可能計算的梯度數值
        optimizer.zero_grad()

        # 依照損失來反向計算各個權重的梯度值
        loss.backward()

        # 依照各個權重的梯度值來更新權重參數
        optimizer.step()

        # 在每一次對批次資料作訓練後，累加其損失值
        total_loss += loss.item()

    ## 在每 100 次的完整訓練
    if (epoch + 1) % 100 == 0:

        # 顯示輸出單次完整訓練中，單筆批次資料的平均損失值
        print(f'Epoch [{epoch+1}/{NUM_EPOCH}], Loss: {total_loss / len(train_loader):.4f}')

## 建立將 FM 參數轉換成 bqm 的函式
def fm_to_bqm(model, var_num, device='cpu'):

    # 將模型設定成評估模式
    model.eval()

    # 將模型放入指定裝置中
    model.to(device)

    ## 建立預測函數，可以計算輸入資料放入模型會有甚麼輸出
    def predict(x_np):

        # 先將輸入資料轉換成 torch 的張量型別，定壓縮成一個長張量，最後傳入裝置中
        x = torch.tensor(x_np, dtype=torch.long).unsqueeze(0).to(device)

        ## 在評估模式中，不用自動計算梯度值
        with torch.no_grad():

            # 把訓練資料放入模型後，提取預測的輸出標籤數值
            return model(x).item()

    # 為了要提取全域偏置，我們須需要一個全部數值都是0的輸入資料
    base = np.zeros(var_num, dtype=np.int64)

    # 利用全部數值都是0的輸入資料，放入模型後即可計算出全域偏置
    b = predict(base)

    # 為了要求的每一參數的權重，我們需要利用對角矩陣來來提取每一特徵的單獨權重
    weight = base + np.eye(var_num, dtype=np.int64)

    # 每次都提取單一特徵所對應的列向量，並將其放入模型後，將結果扣掉全域偏置即可得到其特徵所對應的權重值
    h = f_ei = np.array([predict(weight[i]) for i in range(var_num)]) - b

    # 先建立一個空的矩陣來記錄二次項的參數
    Q = np.zeros((var_num, var_num))

    # 針對每一個特徵作交互作用的參數提取
    for i in range(var_num):

        # 在單一特徵中，除了自己本身之外，都對每個其餘特徵作提取(因為交互作用是相對的，故兩兩特徵只要提取一次)
        for j in range(i + 1, var_num):

            # 先建立一個全部數值都是0的輸入資料
            x_ij = base.copy()

            # 把輸入資料的第 "i" 個數值設為1
            x_ij[i] = 1

            # 把輸入資料的第 "j" 個數值設為1
            x_ij[j] = 1

            # 將輸入資料放入模型作預測後，扣除各自的權重及全域偏置後，即可得到第 "i" 個跟第 "j" 的特徵的交互作用，並將結果放入二次項矩陣
            Q[i, j] = predict(x_ij) - h[i] - h[j] - b
      
    # 為了要符合 dimod 套件的格式(字典)，提取所有權重並轉成 {索引: 權重} 的格式
    h_dict = {i: float(h[i]) for i in range(var_num)}

    # 為了要符合 dimod 套件的格式(字典)，提取所有特徵交互作用並轉成 {(索引i, 索引j): 交互權重} 的格式
    Q_dict = {(i, j): float(Q[i, j]) for i in range(var_num) for j in range(i+1, var_num)}

    # 將轉換成字典的權重跟特徵交互作用，以及浮數點的全域偏置，打包成 dimod 套件的 bqm
    bqm_pred = dimod.BinaryQuadraticModel(h_dict, Q_dict, float(b), dimod.BINARY)

    # 輸出 FM 的 bqm 模型跟二次 Q 矩陣(嚴格上三角)
    return bqm_pred, Q

# 利用函式，將先全訓練完的模型作 bqm 的轉換，並抓取 bqm 模型跟二次 Q 矩陣(嚴格上三角)
bqm_pytorch, Q_pytorch = fm_to_bqm(model_py, VAR_NUM)

# 設定繪製 Q 矩陣的畫布大小
fig, ax = plt.subplots(figsize=(10, 8))

# 先將嚴格上三角的 Q 矩陣作自身與轉置的疊加，取得對稱的 Q 矩陣後，依照冷暖色調將矩陣繪製成圖片，並透過插值方法使每一格的邊界分明
im = ax.imshow(Q_pytorch + Q_pytorch.T, cmap='coolwarm', interpolation='nearest')

# 設定圖片的顏色條
cbar = fig.colorbar(im, ax=ax)

# 替顏色條作文字的說明
cbar.set_label('Interaction Strength')

# 設定圖片的標題
ax.set_title("Mapped Symmetric Q Matrix (32 Variables) from torchfm")

# 設定 x 軸方向的標籤
ax.set_xlabel("Variable Index")

# 設定 y 軸方向的標籤
ax.set_ylabel("Variable Index")

# 依照打包過後，FM 的 bqm 模型，計算原始測試資料所對應的預測能量值
ys_pred_pytorch = np.array([bqm_pytorch.energy(x) for x in xs_test], dtype=np.float64)

# 先依照原始測試資料的實際能量值大小作排序，並記錄其對應的索引值
sorted_indices = np.argsort(ys_test)

# 依照對應的索引值，將原始測試資料的實際能量值依照數值的大小作由小到大的排序
ys_sorted = ys_test[sorted_indices]

# 同樣依照對應的索引值，將原始測試資料所對應的預測能量值依照數值的大小作由小到大的排序
ys_pred_pt_sorted = ys_pred_pytorch[sorted_indices]

# 設定實際及預測能量對應的畫布大小
plt.figure(figsize=(12, 7))

# 依照排序後的實際能量作折線圖的繪製，並將顏色設定成藍色，標籤為"真實能量"，曲線寬度為 4pt，不透明度為0.8
plt.plot(ys_sorted, color="blue", label="True Energy", linewidth=4, alpha=0.8)

# 依照排序後的預測能量作折線圖的繪製，並將顏色設定成紅色，標籤為"預測能量"，並以虛線來繪製
plt.plot(ys_pred_pt_sorted, color="red", label="Predicted Energy (torchfm)", linestyle='-')

# 設定折線圖的標題
plt.title("Comparison of True and PyTorch Predicted Energies", fontsize=16)

# 設定 x 軸方向的標籤，字體大小設定為 16pt
plt.xlabel("Sorted Sample Index", fontsize=16)

# 設定 y 軸方向的標籤，字體大小設定為 16pt
plt.ylabel("Energy", fontsize=16)

# 設定要顯示所有折線的標籤
plt.legend(fontsize=16)

# 設定要顯示網格
plt.grid(True)

# 設定直接顯示圖片
plt.show()
