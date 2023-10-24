import numpy as np

def main():
    while True:
        print("Menú:")
        print("1. Matriz Inversa 2x2")
        print("2. Matriz Inversa 3x3")
        print("3. Multiplicación de Matrices")
        print("4. Sistema de Ecuaciones Lineales")
        print("5. Salir")

        choice = input("Elija una opción (1/2/3/4/5): ")

        if choice == '1':
            inverse_2x2()
        elif choice == '2':
            inverse_3x3()
        elif choice == '3':
            matrix_multiplication()
        elif choice == '4':
            system_of_equations()
        elif choice == '5':
            break
        else:
            print("Opción no válida. Por favor, elija una opción válida.")

def inverse_2x2():
    # Solicita al usuario ingresar valores para la matriz 2x2
    matrix_2x2 = np.empty((2, 2), dtype=float)
    for i in range(2):
        for j in range(2):
            matrix_2x2[i, j] = float(input(f"Ingrese el valor en la fila {i + 1}, columna {j + 1} de la matriz: "))

    try:
        inverse_matrix_2x2 = np.linalg.inv(matrix_2x2)
        print("Matriz 2x2:")
        print(matrix_2x2)
        print("Matriz Inversa 2x2:")
        print(inverse_matrix_2x2)
    except np.linalg.LinAlgError:
        print("Matriz singular (sin inversa)")

def inverse_3x3():
    # Solicita al usuario ingresar valores para la matriz 3x3
    matrix_3x3 = np.empty((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            matrix_3x3[i, j] = float(input(f"Ingrese el valor en la fila {i + 1}, columna {j + 1} de la matriz: "))

    try:
        inverse_matrix_3x3 = np.linalg.inv(matrix_3x3)
        print("Matriz 3x3:")
        print(matrix_3x3)
        print("Matriz Inversa 3x3:")
        print(inverse_matrix_3x3)
    except np.linalg.LinAlgError:
        print("Matriz singular (sin inversa)")

def matrix_multiplication():
    # Ingresa las dimensiones de la matriz A
    rows_A = int(input("Ingrese el número de filas de la matriz A: "))
    cols_A = int(input("Ingrese el número de columnas de la matriz A: "))

    # Ingresa las dimensiones de la matriz B
    rows_B = int(input("Ingrese el número de filas de la matriz B: "))
    cols_B = int(input("Ingrese el número de columnas de la matriz B: "))

    if cols_A != rows_B:
        print("Las dimensiones de las matrices no son compatibles para la multiplicación.")
    else:
        # Solicita al usuario ingresar valores para la matriz A
        matrix_A = np.empty((rows_A, cols_A), dtype=float)
        for i in range(rows_A):
            for j in range(cols_A):
                matrix_A[i, j] = float(input(f"Ingrese el valor en la fila {i + 1}, columna {j + 1} de la matriz A: "))

        # Solicita al usuario ingresar valores para la matriz B
        matrix_B = np.empty((rows_B, cols_B), dtype=float)
        for i in range(rows_B):
            for j in range(cols_B):
                matrix_B[i, j] = float(input(f"Ingrese el valor en la fila {i + 1}, columna {j + 1} de la matriz B: "))

        # Realiza la multiplicación de matrices
        result_matrix = np.dot(matrix_A, matrix_B)

        print("Matriz A:")
        print(matrix_A)
        print("Matriz B:")
        print(matrix_B)
        print("Resultado de la multiplicación:")
        print(result_matrix)

def system_of_equations():
    # Ingresa las dimensiones de la matriz del sistema
    rows = int(input("Ingrese el número de filas de la matriz del sistema: "))
    cols = int(input("Ingrese el número de columnas de la matriz del sistema: "))

    # Inicializa la matriz del sistema y el vector b
    matrix = np.empty((rows, cols), dtype=float)
    b = np.empty(rows, dtype=float)

    # Ingresa los datos de la matriz del sistema
    for i in range(rows):
        for j in range(cols):
            matrix[i, j] = float(input(f"Ingrese el valor en la fila {i + 1}, columna {j + 1} de la matriz: "))

    # Ingresa los datos del vector b
    for i in range(rows):
        b[i] = float(input(f"Ingrese el valor del lado derecho (b) en la fila {i + 1}: "))

    print("Seleccione el método de solución:")
    print("1. Método de Gauss")
    print("2. Método de Cramer")

    method_choice = input("Elija el método (1/2): ")


    if method_choice == '1':
        solution_gauss(matrix, b)
    elif method_choice == '2':
        solution_cramer(matrix, b)
    else:
        print("Opción no válida. Por favor, elija una opción válida.")
    
    # Verifica los tres casos posibles
    if isinstance(solution_gauss, str) and isinstance(solution_cramer, str):
        print("ii. Sin Solución")
    elif np.array_equal(solution_gauss, solution_cramer):
        print("i. Soluciones Únicas")
    #else:
        #print("iii. Soluciones Infinitas")

def solution_gauss(matrix, b):
    n = len(matrix)
    
    # Comprueba si la matriz es singular
    if np.linalg.det(matrix) == 0:
        print("El sistema tiene soluciones infinitas")
        return
    
    x = np.zeros(n, dtype=float)
    
    # Aplicar eliminación de Gauss para convertir la matriz a una forma triangular superior
    for i in range(n):
        for j in range(i + 1, n):
            factor = matrix[j, i] / matrix[i, i]
            b[j] -= factor * b[i]
            matrix[j, i:] -= factor * matrix[i, i:]

    # Resolver el sistema desde la última fila hacia arriba
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(matrix[i, i + 1:], x[i + 1:])) / matrix[i, i]

    print("Solucion utilizando el método de Gauss-Jordan:")
    print(x)

def solution_cramer(matrix, b):
    n = len(matrix)
    solutions = []

    for i in range(n):
        matrix_copy = matrix.copy()
        matrix_copy[:, i] = b
        det_matrix = np.linalg.det(matrix_copy)

        if abs(det_matrix) < 1e-10:
            solutions.append("0")
        else:
            x = det_matrix / np.linalg.det(matrix)
            solutions.append(x)

    print("Solución utilizando la Regla de Cramer:")
    print(solutions)

if __name__ == "__main__":
    main()
