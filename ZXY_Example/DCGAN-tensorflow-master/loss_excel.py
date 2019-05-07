import xlsxwriter


workbook=xlsxwriter.Workbook('loss.xlsx')
worksheet=workbook.add_worksheet()

fp = open("loss.txt", 'r')
re = fp.readline()
col = 0
worksheet.write(1,col,"d_loss")
worksheet.write(2,col,"g_loss")
worksheet.write(3,col,"x")
worksheet.write(4,col,"x_min")
while re :
    loss = re.split(', ')
    d_loss = loss[0]
    g_loss = loss[1]
    d_loss = d_loss.replace('d_loss:', '')
    g_loss = g_loss.replace('g_loss:', '')
    g_loss = g_loss.replace('\n', '')

    worksheet.write(0,col+1,int(col/78))
    worksheet.write(1,col+1,float(d_loss))
    worksheet.write(2,col+1,float(g_loss))
    worksheet.write(3,col+1,abs(float(d_loss)+float(g_loss)))

    print( col+1 )
    col += 1
    re = fp.readline()
worksheet.write(4,1,'=MIN(4:4)')
fp.close()
workbook.close()