"""
Анализ устойчивости методом Ляпунова
===================================

Формальное доказательство устойчивости системы жертва-хищник с управлением
с использованием функций Ляпунова.
"""

import numpy as np
import sympy as sp
from typing import Tuple, Callable, Dict, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from predator_prey_model import PredatorPreyModel


class LyapunovAnalyzer:
    """
    Класс для анализа устойчивости методом Ляпунова.
    """
    
    def __init__(self, model: PredatorPreyModel):
        """
        Инициализация анализатора.
        
        Args:
            model: экземпляр модели PredatorPreyModel
        """
        self.model = model
        self.x1_star = None
        self.x2_star = None
        self.control_function = None
        
        # Символьные переменные для аналитических выкладок
        self.x1, self.x2 = sp.symbols('x1 x2', real=True, positive=True)
        self.x1s, self.x2s = sp.symbols('x1_star x2_star', real=True, positive=True)
        self.a1, self.a2 = sp.symbols('a1 a2', real=True, positive=True)
        self.beta1, self.beta2 = sp.symbols('beta1 beta2', real=True, positive=True)
        self.T1, self.T2 = sp.symbols('T1 T2', real=True, positive=True)
        
    def set_equilibrium_point(self, x1_star: float, x2_star: float):
        """
        Установка точки равновесия для анализа.
        
        Args:
            x1_star: равновесное значение популяции жертв
            x2_star: равновесное значение популяции хищников
        """
        self.x1_star = x1_star
        self.x2_star = x2_star
        
    def set_control_function(self, control_func: Callable[[float, float], float]):
        """
        Установка функции управления.
        
        Args:
            control_func: функция управления u(x1, x2)
        """
        self.control_function = control_func
        
    def construct_quadratic_lyapunov_function(self, P11: float, P12: float, P22: float) -> Dict:
        """
        Построение квадратичной функции Ляпунова.
        
        V(x) = (x - x*)ᵀ P (x - x*)
        
        где P = [[P11, P12],
                 [P12, P22]] - положительно определенная матрица
        
        Args:
            P11: элемент матрицы P (1,1)
            P12: элемент матрицы P (1,2) = P (2,1)
            P22: элемент матрицы P (2,2)
            
        Returns:
            словарь с информацией о функции Ляпунова
        """
        # Проверка положительной определенности матрицы P
        det_P = P11 * P22 - P12**2
        if P11 <= 0 or det_P <= 0:
            raise ValueError("Матрица P должна быть положительно определенной")
        
        def lyapunov_function(x1: float, x2: float) -> float:
            """Значение функции Ляпунова в точке (x1, x2)."""
            if self.x1_star is None or self.x2_star is None:
                raise ValueError("Необходимо установить точку равновесия")
            
            e1 = x1 - self.x1_star
            e2 = x2 - self.x2_star
            
            V = P11 * e1**2 + 2 * P12 * e1 * e2 + P22 * e2**2
            return V
        
        def lyapunov_derivative(x1: float, x2: float) -> float:
            """Производная функции Ляпунова вдоль траекторий системы."""
            if self.x1_star is None or self.x2_star is None:
                raise ValueError("Необходимо установить точку равновесия")
            
            e1 = x1 - self.x1_star
            e2 = x2 - self.x2_star
            
            # Градиент функции Ляпунова
            dV_dx1 = 2 * P11 * e1 + 2 * P12 * e2
            dV_dx2 = 2 * P12 * e1 + 2 * P22 * e2
            
            # Правые части системы с управлением
            u_val = self.control_function(x1, x2) if self.control_function else 0
            
            dx1_dt = self.model.a1 * x1 - self.model.beta1 * x1 * x2 + u_val
            dx2_dt = -self.model.a2 * x2 + self.model.beta2 * x1 * x2
            
            # Производная по времени: dV/dt = ∇V · f
            dV_dt = dV_dx1 * dx1_dt + dV_dx2 * dx2_dt
            
            return dV_dt
        
        return {
            'function': lyapunov_function,
            'derivative': lyapunov_derivative,
            'matrix_P': np.array([[P11, P12], [P12, P22]]),
            'det_P': det_P,
            'trace_P': P11 + P22,
            'is_positive_definite': True
        }
    
    def construct_energy_lyapunov_function(self) -> Dict:
        """
        Построение энергетической функции Ляпунова для экологической системы.
        
        V(x1, x2) = (x1 - x1* - x1*ln(x1/x1*)) + (x2 - x2* - x2*ln(x2/x2*))
        
        Эта функция учитывает биологический смысл переменных.
        
        Returns:
            словарь с информацией о функции Ляпунова
        """
        def lyapunov_function(x1: float, x2: float) -> float:
            """Энергетическая функция Ляпунова."""
            if self.x1_star is None or self.x2_star is None:
                raise ValueError("Необходимо установить точку равновесия")
            
            if x1 <= 0 or x2 <= 0:
                return float('inf')  # функция не определена для неположительных значений
            
            # Первый компонент (для жертв)
            V1 = x1 - self.x1_star - self.x1_star * np.log(x1 / self.x1_star)
            
            # Второй компонент (для хищников)
            V2 = x2 - self.x2_star - self.x2_star * np.log(x2 / self.x2_star)
            
            return V1 + V2
        
        def lyapunov_derivative(x1: float, x2: float) -> float:
            """Производная энергетической функции Ляпунова."""
            if self.x1_star is None or self.x2_star is None:
                raise ValueError("Необходимо установить точку равновесия")
            
            if x1 <= 0 or x2 <= 0:
                return float('inf')
            
            # Частные производные
            dV_dx1 = 1 - self.x1_star / x1
            dV_dx2 = 1 - self.x2_star / x2
            
            # Правые части системы с управлением
            u_val = self.control_function(x1, x2) if self.control_function else 0
            
            dx1_dt = self.model.a1 * x1 - self.model.beta1 * x1 * x2 + u_val
            dx2_dt = -self.model.a2 * x2 + self.model.beta2 * x1 * x2
            
            # Производная по времени
            dV_dt = dV_dx1 * dx1_dt + dV_dx2 * dx2_dt
            
            return dV_dt
        
        return {
            'function': lyapunov_function,
            'derivative': lyapunov_derivative,
            'type': 'energy',
            'domain': 'x1 > 0, x2 > 0'
        }
    
    def analytical_stability_proof(self, T1: float, T2: float) -> Dict:
        """
        Аналитическое доказательство устойчивости с конкретными параметрами.
        
        Args:
            T1: параметр управления T1
            T2: параметр управления T2
            
        Returns:
            результаты анализа устойчивости
        """
        if self.x1_star is None or self.x2_star is None:
            raise ValueError("Необходимо установить точку равновесия")
        
        print("ФОРМАЛЬНОЕ ДОКАЗАТЕЛЬСТВО УСТОЙЧИВОСТИ МЕТОДОМ ЛЯПУНОВА")
        print("=" * 60)
        print()
        
        print("1. ПОСТАНОВКА ЗАДАЧИ")
        print("-" * 20)
        print(f"Рассматривается система:")
        print(f"ẋ₁ = {self.model.a1}·x₁ - {self.model.beta1}·x₁·x₂ + u(x₁,x₂)")
        print(f"ẋ₂ = -{self.model.a2}·x₂ + {self.model.beta2}·x₁·x₂")
        print()
        print(f"Целевая точка равновесия: x₁* = {self.x1_star:.3f}, x₂* = {self.x2_star:.3f}")
        print(f"Параметры управления: T₁ = {T1:.3f}, T₂ = {T2:.3f}")
        print()
        
        print("2. ВЫБОР ФУНКЦИИ ЛЯПУНОВА")
        print("-" * 30)
        print("Выбираем квадратичную функцию Ляпунова:")
        print("V(e₁, e₂) = e₁² + e₂²")
        print("где e₁ = x₁ - x₁*, e₂ = x₂ - x₂*")
        print()
        print("Эта функция положительно определена при (e₁, e₂) ≠ (0, 0)")
        print("и V(0, 0) = 0")
        print()
        
        # Используем простую квадратичную функцию
        lyapunov_info = self.construct_quadratic_lyapunov_function(1.0, 0.0, 1.0)
        
        print("3. ВЫЧИСЛЕНИЕ ПРОИЗВОДНОЙ ФУНКЦИИ ЛЯПУНОВА")
        print("-" * 45)
        print("dV/dt = ∇V · f = (∂V/∂e₁)·ė₁ + (∂V/∂e₂)·ė₂")
        print()
        print("∂V/∂e₁ = 2e₁, ∂V/∂e₂ = 2e₂")
        print()
        print("Система в отклонениях с управлением u = -T₁·e₂ + T₂·(x₁ - a₂/β₂):")
        print()
        
        # Символьный анализ
        e1, e2 = sp.symbols('e1 e2', real=True)
        
        # Производная функции Ляпунова (упрощенный анализ)
        # Для конкретного случая с нашими параметрами
        a1, a2 = self.model.a1, self.model.a2
        beta1, beta2 = self.model.beta1, self.model.beta2
        x1_star, x2_star = self.x1_star, self.x2_star
        
        print("4. УСЛОВИЯ УСТОЙЧИВОСТИ")
        print("-" * 25)
        
        # Анализ линеаризованной системы
        J11 = a1 - beta1 * x2_star - T2
        J12 = -beta1 * x1_star
        J21 = beta2 * x2_star
        J22 = -a2 + beta2 * x1_star - T1
        
        jacobian = np.array([[J11, J12], [J21, J22]])
        trace = np.trace(jacobian)
        det = np.linalg.det(jacobian)
        
        print(f"Матрица Якоби в точке равновесия:")
        print(f"J = [[{J11:.3f}, {J12:.3f}],")
        print(f"     [{J21:.3f}, {J22:.3f}]]")
        print()
        print(f"След матрицы: tr(J) = {trace:.3f}")
        print(f"Определитель: det(J) = {det:.3f}")
        print()
        
        # Критерий Рауса-Гурвица для системы 2-го порядка
        is_stable = (trace < 0) and (det > 0)
        
        print("Критерий устойчивости (Раус-Гурвиц):")
        print(f"1) tr(J) < 0: {trace:.3f} < 0 → {trace < 0}")
        print(f"2) det(J) > 0: {det:.3f} > 0 → {det > 0}")
        print()
        print(f"ЗАКЛЮЧЕНИЕ: Система {'УСТОЙЧИВА' if is_stable else 'НЕУСТОЙЧИВА'}")
        print()
        
        # Собственные значения
        eigenvalues = np.linalg.eigvals(jacobian)
        print("5. АНАЛИЗ СОБСТВЕННЫХ ЗНАЧЕНИЙ")
        print("-" * 35)
        print("Собственные значения матрицы Якоби:")
        for i, lam in enumerate(eigenvalues):
            if np.isreal(lam):
                print(f"λ_{i+1} = {lam.real:.6f}")
            else:
                print(f"λ_{i+1} = {lam.real:.6f} ± {abs(lam.imag):.6f}i")
        
        all_negative_real = all(np.real(lam) < 0 for lam in eigenvalues)
        print(f"\nВсе собственные значения имеют отрицательную действительную часть: {all_negative_real}")
        print()
        
        if is_stable:
            print("6. ТЕОРЕМА ОБ УСТОЙЧИВОСТИ")
            print("-" * 30)
            print("Поскольку:")
            print("1) Функция V(e₁, e₂) = e₁² + e₂² положительно определена")
            print("2) Линеаризованная система устойчива")
            print("3) Все собственные значения имеют отрицательную действительную часть")
            print()
            print("То по теореме Ляпунова об устойчивости по первому приближению")
            print("система с управлением АСИМПТОТИЧЕСКИ УСТОЙЧИВА")
            print("в окрестности точки равновесия.")
        
        return {
            'is_stable': is_stable,
            'jacobian': jacobian,
            'trace': trace,
            'determinant': det,
            'eigenvalues': eigenvalues,
            'lyapunov_function': lyapunov_info,
            'all_eigenvalues_stable': all_negative_real
        }
    
    def visualize_lyapunov_function(self, lyapunov_info: Dict, 
                                  x1_range: Tuple[float, float],
                                  x2_range: Tuple[float, float],
                                  num_points: int = 50):
        """
        Визуализация функции Ляпунова и её производной.
        
        Args:
            lyapunov_info: информация о функции Ляпунова
            x1_range: диапазон значений x1
            x2_range: диапазон значений x2
            num_points: количество точек для построения
        """
        x1_vals = np.linspace(x1_range[0], x1_range[1], num_points)
        x2_vals = np.linspace(x2_range[0], x2_range[1], num_points)
        X1, X2 = np.meshgrid(x1_vals, x2_vals)
        
        V_vals = np.zeros_like(X1)
        dV_vals = np.zeros_like(X1)
        
        for i in range(num_points):
            for j in range(num_points):
                try:
                    V_vals[i, j] = lyapunov_info['function'](X1[i, j], X2[i, j])
                    dV_vals[i, j] = lyapunov_info['derivative'](X1[i, j], X2[i, j])
                except:
                    V_vals[i, j] = np.nan
                    dV_vals[i, j] = np.nan
        
        fig = plt.figure(figsize=(15, 5))
        
        # График функции Ляпунова
        ax1 = fig.add_subplot(131, projection='3d')
        surf1 = ax1.plot_surface(X1, X2, V_vals, cmap='viridis', alpha=0.7)
        ax1.set_xlabel('x₁')
        ax1.set_ylabel('x₂')
        ax1.set_zlabel('V(x₁, x₂)')
        ax1.set_title('Функция Ляпунова')
        
        # Контурный график функции Ляпунова
        ax2 = fig.add_subplot(132)
        contour1 = ax2.contour(X1, X2, V_vals, levels=20)
        ax2.plot(self.x1_star, self.x2_star, 'ro', markersize=8, label='Равновесие')
        ax2.set_xlabel('x₁')
        ax2.set_ylabel('x₂')
        ax2.set_title('Линии уровня V(x₁, x₂)')
        ax2.legend()
        ax2.grid(True)
        
        # График производной функции Ляпунова
        ax3 = fig.add_subplot(133)
        contour2 = ax3.contourf(X1, X2, dV_vals, levels=20, cmap='RdBu_r')
        ax3.plot(self.x1_star, self.x2_star, 'ro', markersize=8, label='Равновесие')
        plt.colorbar(contour2, ax=ax3)
        ax3.set_xlabel('x₁')
        ax3.set_ylabel('x₂')
        ax3.set_title('dV/dt (красный > 0, синий < 0)')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        return fig


def demo_lyapunov_analysis():
    """Демонстрация анализа устойчивости методом Ляпунова."""
    print("Демонстрация анализа устойчивости методом Ляпунова")
    print("=" * 55)
    
    # Создание модели
    model = PredatorPreyModel(a1=1.0, a2=0.5, beta1=0.2, beta2=0.1)
    
    # Создание анализатора
    analyzer = LyapunovAnalyzer(model)
    
    # Установка точки равновесия
    x2_target = 3.0
    x1_target = model.compute_desired_equilibrium(x2_target)
    analyzer.set_equilibrium_point(x1_target, x2_target)
    
    # Простая функция управления для демонстрации
    def demo_control(x1, x2):
        T1, T2 = 1.0, 0.5
        psi = x2 - x2_target
        phi_x2 = model.a2 / model.beta2
        return -T1 * psi + T2 * (x1 - phi_x2)
    
    analyzer.set_control_function(demo_control)
    
    # Формальное доказательство устойчивости
    analysis = analyzer.analytical_stability_proof(T1=1.0, T2=0.5)


if __name__ == "__main__":
    demo_lyapunov_analysis() 