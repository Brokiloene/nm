import numpy as np
import json

def generate_symmetric_matrix(size: int, eps: float) -> dict:
    A = np.random.randint(-10, 10, (size, size))
    A = (A + A.T) / 2
    print(A)
    A_list = A.tolist()
    
    matrix_data = {
        "A": A_list,
        "EPS": eps
    }
    
    return matrix_data

if __name__ == "__main__":
    size = 75
    eps = 0.1
    
    matrix_data = generate_symmetric_matrix(size, eps)
    with open('input75.json', 'w') as f:
        json.dump(matrix_data, f, indent=4)
    
    print("Симметричная матрица сохранена в symmetric_matrix.json")
