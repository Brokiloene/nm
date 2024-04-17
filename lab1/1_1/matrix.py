

class Matrix:
    def __init__(self, data):
        self._rows = len(data)
        self._cols = len(data[0])
        self._data = data
    
    @property
    def rows(self):
        return self._rows

    @property
    def cols(self):
        return self._cols
    
    @property
    def data(self):
        return self._data
    
    def __str__(self):
        return '\n'.join(str(row) for row in self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def __mul__(self, other):
        if self.cols != other.rows:
            raise ValueError("rows != cols")

        newdata = [[0 for row in range(other.cols)] for col in range(self.rows)]
        for i in range(self.rows):
            for j in range(other.cols):
                for k in range(self.cols):
                    newdata[i][j] += self.data[i][k]*other.data[k][j]
        return Matrix(newdata)
    
    
    
m1 = Matrix([[1], [2], [3]])
m2 = Matrix([[1,2,3]])


print(m1)
print(m2)

print(m1 * m2)


# Реализовать алгоритм LU -  разложения матриц (с выбором главного элемента) в виде программы. 
# Используя разработанное программное обеспечение, решить систему линейных алгебраических уравнений (СЛАУ). 
# Для матрицы СЛАУ вычислить определитель и обратную матрицу. 

# 20
# 7x1 + 8x2 + 4x3 - 6x4 = -126
# -x1 + 6x2 - 2x3 - 6x4 = -42
# 2x1 + 9x2 + 6x3 - 4x4 = -115
# 5x1 + 9x2 +  x3 +  x4 = -67

# прогнать вручную с вычитаниями
# также сделать перестановки
# добавить машинные эпсилон для сравнения дробей (мб взять из utf-8?)