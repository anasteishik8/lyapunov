"""
Синтез управления для модели жертва-хищник
=========================================

Реализация методики АКАР (Алгебраический критерий абсолютной устойчивости)
для синтеза стабилизирующего управления.
"""

import numpy as np
from typing import Tuple, Callable
import sympy as sp
from predator_prey_model import PredatorPreyModel


class ControlDesigner:
    """
    Класс для синтеза управления системой жертва-хищник.
    """
    
    def __init__(self, model: PredatorPreyModel):
        """
        Инициализация с моделью системы.
        
        Args:
            model: экземпляр PredatorPreyModel
        """
        self.model = model
        self.x2_star = None
        self.x1_star = None
        self.T1 = None  # параметр управления
        self.T2 = None  # параметр управления
        
    def set_target_point(self, x2_star: float):
        """
        Установка целевой точки стабилизации.
        
        Args:
            x2_star: целевое значение популяции хищников
        """
        self.x2_star = x2_star
        # Вычисляем соответствующее значение x1 из условия равновесия
        self.x1_star = self.model.compute_desired_equilibrium(x2_star)
        
        print(f"Целевая точка установлена:")
        print(f"x1* = {self.x1_star:.3f}")
        print(f"x2* = {self.x2_star:.3f}")
        
    def design_linear_control(self, T1: float, T2: float) -> Callable[[float, float], float]:
        """
        Синтез линейного управления на основе АКАР.
        
        Согласно выкладкам из изображений:
        u(x1, x2) = -T1*ψ + T2*(x1 - φ(x2))
        
        где:
        ψ = x2(t) - x2* 
        φ(x2) = (α2*x2 - T2*λ - μ*(x2 - x2*))/(β2*x2)
        
        Args:
            T1: параметр управления T1
            T2: параметр управления T2
            
        Returns:
            функция управления u(x1, x2)
        """
        self.T1 = T1
        self.T2 = T2
        
        def control_function(x1: float, x2: float) -> float:
            if self.x2_star is None:
                raise ValueError("Необходимо установить целевую точку")
            
            # Отклонение x2 от целевого значения
            psi = x2 - self.x2_star
            
            # Вычисляем φ(x2) согласно выкладкам
            # Упрощенная форма для начальной реализации
            if abs(x2) < 1e-10:  # избегаем деления на ноль
                phi_x2 = self.x1_star
            else:
                # φ(x2) = a2/β2 (из условия равновесия при x2 ≠ 0)
                phi_x2 = self.model.a2 / self.model.beta2
            
            # Закон управления
            u = -T1 * psi + T2 * (x1 - phi_x2)
            
            return u
        
        return control_function
    
    def design_nonlinear_control(self, k1: float, k2: float) -> Callable[[float, float], float]:
        """
        Синтез нелинейного управления для улучшенной стабилизации.
        
        Args:
            k1: коэффициент обратной связи по x1
            k2: коэффициент обратной связи по x2
            
        Returns:
            функция нелинейного управления
        """
        def nonlinear_control(x1: float, x2: float) -> float:
            if self.x2_star is None:
                raise ValueError("Необходимо установить целевую точку")
            
            # Отклонения от равновесия
            e1 = x1 - self.x1_star
            e2 = x2 - self.x2_star
            
            # Нелинейное управление с насыщением
            u = -k1 * e1 - k2 * e2 * x1
            
            # Ограничение управления для предотвращения слишком больших значений
            u_max = 10.0
            u = np.clip(u, -u_max, u_max)
            
            return u
        
        return nonlinear_control
    
    def analyze_stability_conditions(self) -> dict:
        """
        Анализ условий устойчивости для выбранных параметров управления.
        
        Returns:
            словарь с результатами анализа
        """
        if self.T1 is None or self.T2 is None:
            raise ValueError("Необходимо сначала задать параметры управления")
        
        # Характеристическое уравнение линеаризованной системы
        # в окрестности точки равновесия
        a1, a2 = self.model.a1, self.model.a2
        beta1, beta2 = self.model.beta1, self.model.beta2
        
        # Матрица Якоби в точке равновесия (с управлением)
        # J = [[a1 - β1*x2* - T2, -β1*x1*],
        #      [β2*x2*,           -a2 + β2*x1* - T1]]
        
        J11 = a1 - beta1 * self.x2_star - self.T2
        J12 = -beta1 * self.x1_star
        J21 = beta2 * self.x2_star
        J22 = -a2 + beta2 * self.x1_star - self.T1
        
        # Характеристический полином: λ² - trace*λ + det = 0
        trace = J11 + J22
        det = J11 * J22 - J12 * J21
        
        # Условия устойчивости по критерию Рауса-Гурвица:
        # 1) trace < 0 (сумма собственных значений отрицательна)
        # 2) det > 0 (произведение собственных значений положительно)
        
        is_stable = (trace < 0) and (det > 0)
        
        # Собственные значения
        discriminant = trace**2 - 4*det
        if discriminant >= 0:
            lambda1 = (trace + np.sqrt(discriminant)) / 2
            lambda2 = (trace - np.sqrt(discriminant)) / 2
            eigenvalues = [lambda1, lambda2]
            eigenvalue_type = "действительные"
        else:
            real_part = trace / 2
            imag_part = np.sqrt(-discriminant) / 2
            eigenvalues = [complex(real_part, imag_part), complex(real_part, -imag_part)]
            eigenvalue_type = "комплексные"
        
        return {
            'is_stable': is_stable,
            'trace': trace,
            'determinant': det,
            'eigenvalues': eigenvalues,
            'eigenvalue_type': eigenvalue_type,
            'stability_conditions': {
                'trace_negative': trace < 0,
                'determinant_positive': det > 0
            }
        }
    
    def optimal_parameters_search(self, T1_range: Tuple[float, float], 
                                T2_range: Tuple[float, float], 
                                num_points: int = 20) -> Tuple[float, float]:
        """
        Поиск оптимальных параметров управления в заданных диапазонах.
        
        Args:
            T1_range: диапазон поиска для T1
            T2_range: диапазон поиска для T2
            num_points: количество точек для поиска по каждому параметру
            
        Returns:
            (T1_opt, T2_opt): оптимальные параметры
        """
        T1_values = np.linspace(T1_range[0], T1_range[1], num_points)
        T2_values = np.linspace(T2_range[0], T2_range[1], num_points)
        
        best_T1, best_T2 = None, None
        best_real_part = float('inf')  # ищем наиболее отрицательную действительную часть
        
        stable_combinations = []
        
        for T1 in T1_values:
            for T2 in T2_values:
                # Временно устанавливаем параметры
                old_T1, old_T2 = self.T1, self.T2
                self.T1, self.T2 = T1, T2
                
                try:
                    analysis = self.analyze_stability_conditions()
                    
                    if analysis['is_stable']:
                        stable_combinations.append((T1, T2, analysis))
                        
                        # Находим максимальную действительную часть собственных значений
                        max_real_part = max(np.real(ev) for ev in analysis['eigenvalues'])
                        
                        if max_real_part < best_real_part:
                            best_real_part = max_real_part
                            best_T1, best_T2 = T1, T2
                
                except Exception:
                    pass  # игнорируем комбинации, приводящие к ошибкам
                
                # Восстанавливаем старые значения
                self.T1, self.T2 = old_T1, old_T2
        
        print(f"Найдено {len(stable_combinations)} стабильных комбинаций параметров")
        
        if best_T1 is not None:
            print(f"Оптимальные параметры: T1 = {best_T1:.3f}, T2 = {best_T2:.3f}")
            print(f"Максимальная действительная часть с.з.: {best_real_part:.6f}")
            return best_T1, best_T2
        else:
            print("Стабильные параметры не найдены в заданных диапазонах")
            return None, None


def demo_control_design():
    """Демонстрация синтеза управления."""
    print("Демонстрация синтеза управления")
    print("=" * 40)
    
    # Создание модели
    model = PredatorPreyModel(a1=1.0, a2=0.5, beta1=0.2, beta2=0.1)
    
    # Создание проектировщика управления
    designer = ControlDesigner(model)
    
    # Установка целевой точки
    x2_target = 3.0
    designer.set_target_point(x2_target)
    
    # Поиск оптимальных параметров
    T1_opt, T2_opt = designer.optimal_parameters_search(
        T1_range=(0.1, 2.0),
        T2_range=(0.1, 2.0),
        num_points=10
    )
    
    if T1_opt is not None:
        # Создание управления с оптимальными параметрами
        control = designer.design_linear_control(T1_opt, T2_opt)
        
        # Анализ устойчивости
        analysis = designer.analyze_stability_conditions()
        print("\nАнализ устойчивости:")
        print(f"Система устойчива: {analysis['is_stable']}")
        print(f"След матрицы: {analysis['trace']:.6f}")
        print(f"Определитель: {analysis['determinant']:.6f}")
        print(f"Собственные значения: {analysis['eigenvalues']}")


if __name__ == "__main__":
    demo_control_design() 