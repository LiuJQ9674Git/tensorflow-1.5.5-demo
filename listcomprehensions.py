# 列表解析 基本语法
# [expression for iter_val in iterable]
# [expression for iter_val in iterable if cond_expr]
li = []
for i in range(1, 11):
    li.append(i*2)
print(li)

# 用列表解析式实现如下：
li = [i*2 for i in range(1, 11)]
print(li)

# 筛选条件
li = [i*2 for i in range(1, 11) if i*2 > 10]
print(li)

# 嵌套循环
li1 = ['A', 'B', 'C']
li2 = ['1', '2', '3']
li3 = []
for m in li1:
    for n in li2:
        li3.append((m,n))
print(li3)

# 列表解析式实现如下：
li1 = ['A', 'B', 'C']
li2 = ['1', '2', '3']
li3 = [(m,n) for m in li1 for n in li2]
print(li3)

# 字典解析式
a = {'language1':'python', 'language2':'java','language3':'c'}
b = {}
for key, value in a.items():
    if key == 'language1':
        b[key] = value
print(b)

a = {'language1':'python', 'language2':'java','language3':'c'}
b = {key: value for key, value in a.items() if key == 'language1'}
print(b)

# 集合解析式
a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
b = set()
for i in a:
    if i > 5:
        b.add(i)
print(b)

a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
b = {i for i in a if i > 5}
print(b)

