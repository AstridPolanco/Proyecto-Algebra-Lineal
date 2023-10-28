import tkinter as tk
from tkinter import messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def inverse_2x2():

    result_label.config(text="Resultados Matriz Inversa 2x2:")
    result_text.delete("1.0", tk.END)  # Limpiar el área de resultados

    # Solicita al usuario ingresar valores para la matriz 2x2
    matrix_2x2 = np.empty((2, 2), dtype=float)
    for i in range(2):
        for j in range(2):
            value = simpledialog.askfloat("Ingresar Valor", f"Ingrese el valor en la fila {i + 1}, columna {j + 1} de la matriz:")
            if value is not None:
                matrix_2x2[i, j] = value

    try:
        inverse_matrix_2x2 = np.linalg.inv(matrix_2x2)
        result_text.insert(tk.END, f"Matriz 2x2:\n{matrix_2x2}\n\n")
        result_text.insert(tk.END, f"Matriz Inversa 2x2:\n{inverse_matrix_2x2}")
    except np.linalg.LinAlgError:
        messagebox.showerror("Error", "Matriz singular (sin inversa)")

def inverse_3x3():
    result_label.config(text="Resultados Matriz Inversa 3x3:")
    result_text.delete("1.0", tk.END)  # Limpiar el área de resultados

    # Solicita al usuario ingresar valores para la matriz 3x3
    matrix_3x3 = np.empty((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            value = simpledialog.askfloat("Ingresar Valor", f"Ingrese el valor en la fila {i + 1}, columna {j + 1} de la matriz:")
            if value is not None:
                matrix_3x3[i, j] = value

    try:
        inverse_matrix_3x3 = np.linalg.inv(matrix_3x3)
        result_text.insert(tk.END, f"Matriz 3x3:\n{matrix_3x3}\n\n")
        result_text.insert(tk.END, f"Matriz Inversa 3x3:\n{inverse_matrix_3x3}")
    except np.linalg.LinAlgError:
        messagebox.showerror("Error", "Matriz singular (sin inversa)")

def matrix_multiplication():

    result_label.config(text="Resultados:")
    result_text.delete("1.0", tk.END)  # Limpiar el área de resultados
    # Ingresa las dimensiones de la matriz A
    rows_A = simpledialog.askinteger("Dimensiones", "Ingrese el número de filas de la matriz A: ")
    cols_A = simpledialog.askinteger("Dimensiones", "Ingrese el número de columnas de la matriz A: ")

    # Ingresa las dimensiones de la matriz B
    rows_B = simpledialog.askinteger("Dimensiones", "Ingrese el número de filas de la matriz B: ")
    cols_B = simpledialog.askinteger("Dimensiones", "Ingrese el número de columnas de la matriz B: ")

    if rows_A is None or cols_A is None or rows_B is None or cols_B is None:
        return

    if cols_A != rows_B:
        messagebox.showerror("Error", "Las dimensiones de las matrices no son compatibles para la multiplicación.")
        return
    
    else:

        # Solicita al usuario ingresar valores para la matriz A
        matrix_A = np.empty((rows_A, cols_A), dtype=float)
        for i in range(rows_A):
            for j in range(cols_A):
                value = simpledialog.askfloat("Ingresar Valor", f"Ingrese el valor en la fila {i + 1}, columna {j + 1} de la matriz A:")
                if value is not None:
                   matrix_A[i, j] = value

        # Solicita al usuario ingresar valores para la matriz B
        matrix_B = np.empty((rows_B, cols_B), dtype=float)
        for i in range(rows_B):
            for j in range(cols_B):
                value = simpledialog.askfloat("Ingresar Valor", f"Ingrese el valor en la fila {i + 1}, columna {j + 1} de la matriz B:")
                if value is not None:
                   matrix_B[i, j] = value

        # Realiza la multiplicación de matrices
        result_matrix = np.dot(matrix_A, matrix_B)

        result_text.insert(tk.END, f"Matriz A:\n{matrix_A}\n\n")
        result_text.insert(tk.END, f"Matriz B:\n{matrix_B}\n\n")
        result_text.insert(tk.END, f"Resultado de la multiplicación:\n{result_matrix}")

def system_of_equations():

    result_label.config(text="Resultados:")
    result_text.delete("1.0", tk.END)  # Limpiar el área de resultados

    # Ingresa las dimensiones de la matriz del sistema
    rows = simpledialog.askinteger("Ingrese el número de filas de la matriz del sistema", "Número de filas: ")
    cols = simpledialog.askinteger("Ingrese el número de columnas de la matriz del sistema", "Número de columnas: ")

    if rows is None or cols is None:
        return

    # Inicializa la matriz del sistema y el vector b
    matrix = np.empty((rows, cols), dtype=float)
    b = np.empty(rows, dtype=float)

    # Ingresa los datos de la matriz del sistema
    for i in range(rows):
        for j in range(cols):
            value = simpledialog.askfloat("Ingresar Valor", f"Ingrese el valor en la fila {i + 1}, columna {j + 1} de la matriz:")
            if value is not None:
                matrix[i, j] = value

    # Ingresa los datos del vector b
    for i in range(rows):
        value = simpledialog.askfloat("Ingresar Valor", f"Ingrese el valor del lado derecho (b) en la fila {i + 1}:")
        if value is not None:
            b[i] = value

    # Mostrar el método de solución
    method_choice = messagebox.askquestion("Método de Solución", "¿Desea utilizar el Método de Gauss para resolver el sistema?")

    if method_choice == "yes":
        solution_gauss(matrix, b)
    else:
        solution_cramer(matrix, b)

    # Calcular el tipo de solución
    solution_type = classify_solutions(matrix, b)

    if solution_type == "unique":
        result_text.insert(tk.END, "El sistema tiene una solución única.\n")
        #result_text.insert(tk.END, f"Solución: {solution_gauss(matrix, b)}\n")
    elif solution_type == "infinite":
        result_text.insert(tk.END, "El sistema tiene soluciones infinitas.\n")
    else:
        result_text.insert(tk.END, "El sistema no tiene solución.\n")
    if solution_type != "no_solution":
        plot_linear_equations(matrix, b)

def classify_solutions(matrix, b):
    if np.linalg.matrix_rank(matrix) == np.linalg.matrix_rank(np.column_stack((matrix, b))) == len(matrix[0]):
        return "unique"  # Solución única
    elif np.linalg.matrix_rank(matrix) == np.linalg.matrix_rank(np.column_stack((matrix, b))) < len(matrix[0]):
        return "infinite"  # Soluciones infinitas
    else:
        return "no_solution"  # No hay solución
    
def solution_gauss(matrix, b):
    n = len(matrix)
    
    # Comprueba si la matriz es singular
    if np.linalg.det(matrix) == 0:
        result_text.insert(tk.END, "Sistema sin solución\n")
    else:
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

    result_text.insert(tk.END, "Solución utilizando el Método de Gauss-Jordan:\n")
    result_text.insert(tk.END, f"x = {x}\n")

def solution_cramer(matrix, b):
    n = len(matrix)
    # Comprueba si la matriz es singular
    if np.linalg.det(matrix) == 0:
        result_text.insert(tk.END, "Sistema sin solución\n")
    else:
        x = np.zeros(n, dtype=float)

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
    
    result_text.insert(tk.END, "Solución utilizando la Regla de Cramer:\n")
    result_text.insert(tk.END, f"x = {solutions}\n")

# La función plot_linear_equations graficará las ecuaciones
def plot_linear_equations(matrix, b):
    fig, ax = plt.subplots()
    x = np.linspace(-10, 10, 400)  # Valores de x para el gráfico

    # Graficar las ecuaciones
    for i in range(len(matrix)):
        y = (b[i] - matrix[i, 0] * x) / matrix[i, 1]
        ax.plot(x, y, label=f'Ecuación {i + 1}')

    # Establecer etiquetas y leyenda
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    # Mostrar el gráfico
    plt.grid(True)
    plt.axhline(0, color='black', lw=1)
    plt.axvline(0, color='black', lw=1)
    plt.title('Gráfico de Ecuaciones')
    #plt.show()
    
    # Agrega la gráfica a una ventana emergente
    window = tk.Toplevel(root)
    window.title("Gráfico")
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.get_tk_widget().pack()
        
def display_result(label_text, matrix):
    result_text.insert(tk.END, f"{label_text}\n{matrix}\n\n")

def clear_results():
    result_label.config(text="Resultados:")
    result_text.delete("1.0", tk.END)  # Limpiar el área de resultados

root = tk.Tk()
root.title("Calculadora de Matrices")
# Crear etiquetas para resultados
result_label = tk.Label(root, text="Resultados:", font=("Arial Black", 25), bg="#E0FFFF")
result_label.pack()

result_text = tk.Text(root, height=10, width=50, relief="sunken", bd="5", bg="white")
result_text.pack()

# Establecer estilos y dimensiones
root.geometry("600x400")
root.configure(bg="#E0FFFF", bd= 8, cursor="hand1", relief="ridge")
root.resizable(0,0)

# Crear un menú
menu = tk.Menu(root)
root.config(menu=menu)

# Crear un menú desplegable para las operaciones
operations_menu = tk.Menu(menu, tearoff=0)
menu.add_cascade(label="Operaciones", menu=operations_menu)
operations_menu.add_command(label="Matriz Inversa 2x2", command=inverse_2x2)
operations_menu.add_command(label="Matriz Inversa 3x3", command=inverse_3x3)
operations_menu.add_command(label="Multiplicación de Matrices", command=matrix_multiplication)
operations_menu.add_command(label="Sistema de Ecuaciones", command=system_of_equations)

# Botón para limpiar resultados
btn_clear_results = tk.Button(root, text="Limpiar Resultados", command=clear_results, bg="#191970", fg="white", width= 15, height=2, cursor="hand1", relief="raised", bd="5")
btn_clear_results.pack(pady=10)

# Botón para salir
btn_exit = tk.Button(root, text="Salir", command=root.destroy, bg="red", fg="white", width=15, height=2, cursor="hand1", relief="raised", bd="5")
btn_exit.pack(pady=30)

root.mainloop()