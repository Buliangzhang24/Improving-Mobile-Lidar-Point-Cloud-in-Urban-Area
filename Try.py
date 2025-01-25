import laspy

# 读取LAS文件
las = laspy.read("your_point_cloud.las")

# 输出头部信息
print(las.header)
