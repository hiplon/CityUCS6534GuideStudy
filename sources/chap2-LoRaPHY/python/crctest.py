def whiten(data, whitening_seq):
    # 确保数据和白化序列具有相同的长度
    if len(data) != len(whitening_seq):
        raise ValueError("Data and whitening sequence must have the same length")

    # 初始化输出数据
    data_w = bytearray(len(data))

    # 执行白化操作，逐字节进行位异或
    for i in range(len(data)):
        data_w[i] = data[i] ^ whitening_seq[i]

    return data_w

# 示例数据和白化序列（假设它们有相同的长度）
data = bytearray([0x01, 0x02, 0x03, 0x04, 0x05])
whitening_seq = bytearray([0xAA, 0xBB, 0xCC, 0xDD, 0xEE])

# 调用白化函数
whitened_data = whiten(data, whitening_seq)

# 打印白化后的数据
print("Whitened Data:", whitened_data)