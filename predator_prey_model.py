"""
Модель "Жертва-Хищник" с управлением
=====================================

Реализация математической модели взаимодействия популяций жертв и хищников
с возможностью управления для стабилизации системы.
"""

import numpy as np
from typing import Tuple, Callable, Optional
import matplotlib.pyplot as plt


class PredatorPreyModel:
    """
    Класс для моделирования системы жертва-хищник с управлением.
    
    Система дифференциальных уравнений:
    ẋ₁ = a₁x₁ - β₁x₁x₂ + u(x₁,x₂)  # динамика жертв
    ẋ₂ = -a₂x₂ + β₂x₁x₂            # динамика хищников
    """
    
    def __init__(self, a1: float, a2: float, beta1: float, beta2: float):
        """
        Инициализация модели с параметрами.
        
        Args:
            a1: коэффициент роста жертв
            a2: коэффициент убыли хищников
            beta1: коэффициент воздействия хищников на жертв
            beta2: коэффициент воздействия жертв на хищников
        """
        self.a1 = a1
        self.a2 = a2
        self.beta1 = beta1
        self.beta2 = beta2
        
        # Целевое значение для стабилизации
        self.x2_target = None
        self.control_function = None
        
    def set_target(self, x2_star: float):
        """Установка целевого значения популяции хищников."""
        self.x2_target = x2_star
        
    def set_control(self, control_func: Callable[[float, float], float]):
        """Установка функции управления."""
        self.control_function = control_func
        
    def dynamics_without_control(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Правые части системы без управления (классическая модель Лотки-Вольтерра).
        
        Args:
            t: время
            state: вектор состояния [x1, x2]
            
        Returns:
            производные [ẋ1, ẋ2]
        """
        x1, x2 = state
        
        dx1_dt = self.a1 * x1 - self.beta1 * x1 * x2
        dx2_dt = -self.a2 * x2 + self.beta2 * x1 * x2
        
        return np.array([dx1_dt, dx2_dt])
    
    def dynamics_with_control(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Правые части системы с управлением.
        
        Args:
            t: время
            state: вектор состояния [x1, x2]
            
        Returns:
            производные [ẋ1, ẋ2]
        """
        x1, x2 = state
        
        # Управляющее воздействие
        u = self.control_function(x1, x2) if self.control_function else 0
        
        dx1_dt = self.a1 * x1 - self.beta1 * x1 * x2 + u
        dx2_dt = -self.a2 * x2 + self.beta2 * x1 * x2
        
        return np.array([dx1_dt, dx2_dt])
    
    def equilibrium_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Вычисление точек равновесия системы без управления.
        
        Returns:
            (trivial_point, nontrivial_point): тривиальная и нетривиальная точки равновесия
        """
        # Тривиальная точка равновесия
        trivial = np.array([0, 0])
        
        # Нетривиальная точка равновесия
        x1_eq = self.a2 / self.beta2
        x2_eq = self.a1 / self.beta1
        nontrivial = np.array([x1_eq, x2_eq])
        
        return trivial, nontrivial
    
    def compute_desired_equilibrium(self, x2_star: float) -> float:
        """
        Вычисление желаемого значения x1 для заданного x2*.
        
        Args:
            x2_star: целевое значение популяции хищников
            
        Returns:
            соответствующее значение x1
        """
        # Из условия равновесия: ẋ2 = 0 => -a2*x2 + β2*x1*x2 = 0
        # При x2 ≠ 0: x1 = a2/β2
        return self.a2 / self.beta2
    
    def phase_portrait_data(self, x1_range: Tuple[float, float], 
                           x2_range: Tuple[float, float], 
                           num_points: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Генерация данных для построения фазового портрета.
        
        Args:
            x1_range: диапазон значений x1
            x2_range: диапазон значений x2
            num_points: количество точек по каждой оси
            
        Returns:
            (X1, X2, U, V): сетки координат и компоненты векторного поля
        """
        x1 = np.linspace(x1_range[0], x1_range[1], num_points)
        x2 = np.linspace(x2_range[0], x2_range[1], num_points)
        X1, X2 = np.meshgrid(x1, x2)
        
        U = np.zeros_like(X1)
        V = np.zeros_like(X2)
        
        for i in range(num_points):
            for j in range(num_points):
                derivatives = self.dynamics_without_control(0, [X1[i,j], X2[i,j]])
                U[i,j] = derivatives[0]
                V[i,j] = derivatives[1]
        
        return X1, X2, U, V


def euler_method(dynamics_func: Callable, initial_state: np.ndarray, 
                t_span: Tuple[float, float], dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Численное решение ОДУ методом Эйлера.
    
    Args:
        dynamics_func: функция правых частей системы
        initial_state: начальные условия
        t_span: временной интервал (t0, tf)
        dt: шаг интегрирования
        
    Returns:
        (t, y): массивы времени и решения
    """
    t0, tf = t_span
    t = np.arange(t0, tf + dt, dt)
    n_steps = len(t)
    n_vars = len(initial_state)
    
    y = np.zeros((n_steps, n_vars))
    y[0] = initial_state
    
    for i in range(1, n_steps):
        dy_dt = dynamics_func(t[i-1], y[i-1])
        y[i] = y[i-1] + dt * dy_dt
    
    return t, y


if __name__ == "__main__":
    # Пример использования
    print("Модель жертва-хищник с управлением")
    print("=" * 40)
    
    # Создание модели с типичными параметрами
    model = PredatorPreyModel(a1=1.0, a2=0.5, beta1=0.2, beta2=0.1)
    
    # Вычисление точек равновесия
    trivial, nontrivial = model.equilibrium_points()
    print(f"Тривиальная точка равновесия: {trivial}")
    print(f"Нетривиальная точка равновесия: {nontrivial}")
    
    # Начальные условия
    initial_state = np.array([8.0, 4.0])
    print(f"Начальные условия: x1(0) = {initial_state[0]}, x2(0) = {initial_state[1]}") 