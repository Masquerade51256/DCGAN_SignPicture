import xlsxwriter

# 在"model.py"中329-331行添加了输出模型代价值到"loss.txt"的代码
# 本模块功能为将记录在"loss.txt"中的代价值写入excel表格中，并找出总代价最小值
# 运行前需要在当前python环境中安装xlswriter包
# 运行完成后请使用excel工具打开"loss.xls"表格，B5单元格内为总代价的最小值
# 训练集重新训练或更换训练及之前请先清空"loss.txt"中的内容，因为model中使用add模式写入

workbook = xlsxwriter.Workbook('loss.xlsx')
worksheet = workbook.add_worksheet()

fp = open("loss.txt", 'r')
re = fp.readline()
col = 0
worksheet.write(0, col, "epoch")
worksheet.write(1, col, "d_loss")
worksheet.write(2, col, "g_loss")
worksheet.write(3, col, "loss")
worksheet.write(4, col, "loss_min")

while re:
    loss = re.split(', ')
    d_loss = loss[0]
    g_loss = loss[1]
    d_loss = d_loss.replace('d_loss:', '')
    g_loss = g_loss.replace('g_loss:', '')
    g_loss = g_loss.replace('\n', '')

    worksheet.write(0, col + 1, int(col / 78))  # epoch
    worksheet.write(1, col + 1, float(d_loss))  # 判别代价
    worksheet.write(2, col + 1, float(g_loss))  # 生成代价
    worksheet.write(3, col + 1, float(d_loss) + float(g_loss))  # 求和，即总代价值

    print(col + 1)
    col += 1
    re = fp.readline()

# 计算最小值
worksheet.write(4, 1, '=MIN(4:4)')
fp.close()
workbook.close()
