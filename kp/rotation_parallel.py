import json
from typing import Tuple
from pathlib import Path

import numpy as np

from multiprocessing import Pool, cpu_count


def matrix_multiply_chunk(args):
    A_chunk, B = args
    # Проверяем размерность
    if A_chunk.shape[1] != B.shape[0]:
        raise ValueError("Несогласованные размеры блока матрицы для умножения.")
    
    # Векторизированное умножение блока
    return np.matmul(A_chunk, B)

def parallel_matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.shape[1] != B.shape[0]:
        raise ValueError("Несогласованные размеры матриц для умножения.")
    
    # Количество потоков ограничено количеством строк
    num_workers = min(cpu_count(), A.shape[0])
    chunk_size = max(1, A.shape[0] // num_workers)

    # Разделяем A на блоки строк
    A_chunks = [A[i:i + chunk_size] for i in range(0, A.shape[0], chunk_size)]

    with Pool(processes=num_workers) as pool:
        results = pool.map(matrix_multiply_chunk, [(chunk, B) for chunk in A_chunks])
    
    return np.vstack(results)

def process_row(args):
    row, row_idx = args
    row = np.asarray(row).flatten()  # Преобразуем строку в одномерный массив
    return sum(row[j]**2 for j in range(len(row)) if row_idx != j)

def parallel_matrix_norm(A: np.ndarray) -> float:
    n = A.shape[0]
    num_workers = min(cpu_count(), n)
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_row, [(np.asarray(A[i]).flatten(), i) for i in range(n)])
    
    # Суммируем квадраты внедиагональных элементов из всех строк
    total_sum = sum(results)
    return np.sqrt(total_sum)

def find_max_in_row(args):
    row, row_idx = args
    row = np.asarray(row).flatten()
    max_value = float('-inf')
    max_col = -1
    
    for j in range(len(row)):
        if row_idx == j:  # Пропускаем диагональные элементы
            continue
        if np.abs(row[j]) > np.abs(max_value) or max_value == float('-inf'):
            max_value = row[j]
            max_col = j
    
    return (max_value, row_idx, max_col)

def parallel_find_max(A: np.ndarray):
    n = A.shape[0]
    num_workers = min(cpu_count(), n)
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(find_max_in_row, [(A[i], i) for i in range(n)])
    
    # Находим глобальный максимум среди локальных
    max_result = max(results, key=lambda x: abs(x[0]))
    cur_max = max_result[0]
    l = max_result[1]
    k = max_result[2]
    
    return cur_max, l, k

np.set_printoptions(precision=4, floatmode="fixed", suppress=True)

def matrix_norm(A: np.matrix) -> float:
    n = A.shape[0]
    res = np.sqrt( sum(A[i,j]**2 if i < j else 0 for i in range(n) for j in range(n)) )
    return res

def naive_matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # Проверка совместимости размеров матриц
    if A.shape[1] != B.shape[0]:
        raise ValueError("Несогласованные размеры матриц: количество столбцов A должно совпадать с количеством строк B.")
    
    # Определение размеров результирующей матрицы
    m, n = A.shape
    p = B.shape[1]
    
    C = np.zeros((m, p))
    
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    
    return C

def rotation_method(A: np.matrix, EPS) -> Tuple[np.array, np.matrix]:
    n = A.shape[0]

    converged = False
    num = 1
    U = np.eye(n)
    while not converged:
        # cur_max = 0
        # l, k = 0, 0
        # for i in range(n):
        #     for j in range(n):
        #         if i == j:
        #             continue
        #         if np.abs(A[i, j]) > abs(cur_max):
        #             cur_max = A[i, j]
        #             l = i
        #             k = j
        _, l, k = parallel_find_max(A)

        phi = 0.5 * np.arctan(2*A[l, k] / (A[l, l]-A[k, k]))
        U_k = np.eye(n)
        U_k[l, l] = np.cos(phi)
        U_k[k, k] = np.cos(phi)
        U_k[l, k] = -np.sin(phi)
        U_k[k, l] = np.sin(phi)

        # print (U_k.T.shape, A.shape, U_k.shape)
        # print(A.shape, U_k.T.shape)
        A = parallel_matrix_multiply(U_k.T, A)
        A = parallel_matrix_multiply(A, U_k)
        U = parallel_matrix_multiply(U, U_k)

        # A = naive_matrix_multiply(U_k.T, A)
        # A = naive_matrix_multiply(A, U_k)
        # U = naive_matrix_multiply(U, U_k)
        # A = U_k.T @ A @ U_k
        # U @= U_k

        # print(f"Iteration №{num}\nU_{num}=\n", U_k, f"\nA_{num}=\n", A)

        norm = matrix_norm(A)
        # norm = parallel_matrix_norm(A)
        # norm = matrix_norm(A)
        # print(f"Iteration №{num}; ║A_{num}║ = {norm}, EPS = {EPS}")
        num += 1
        if norm < EPS:
            converged = True
    
    return np.diag(A).copy(), U

import time
def solve(inputfile):
    with open(inputfile, "r") as f:
        data = json.load(f)

        A = np.matrix(data["A"], dtype=np.float32)
        EPS = data["EPS"]

    start_time = time.time()
    eigenvalues, eigenvectors = rotation_method(A, EPS)
    print(f"Rotation {time.time() - start_time} seconds")

    eigenvalues.sort()

    print("\nCalculated eignevalues using Rotation Method: ", eigenvalues)    
    print("Calculated eignevectors using Rotation Method:\n", eigenvectors)

    start_time = time.time()
    numpy_eigenvalues, numpy_eigenvectors = np.linalg.eig(A)
    print(f"Numpy eigen {time.time() - start_time} seconds")

    numpy_eigenvalues.sort()
    print("\nCalculated eignevalues using NumPy function:", numpy_eigenvalues)
    print("Calculated eignevectors using NumPy function:\n", numpy_eigenvectors)

cur_dir = Path(__file__).parent
solve(cur_dir / "input35.json" )
