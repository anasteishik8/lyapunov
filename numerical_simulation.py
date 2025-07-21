"""
Численное моделирование системы жертва-хищник
===========================================

Реализация различных сценариев моделирования для анализа поведения системы
с управлением и без него.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import Tuple, List, Dict, Optional, Callable
import pandas as pd

from predator_prey_model import PredatorPreyModel, euler_method
from control_design import ControlDesigner
from lyapunov_analysis import LyapunovAnalyzer


class SimulationRunner:
    """
    Класс для запуска различных сценариев моделирования.
    """
    
    def __init__(self, model: PredatorPreyModel):
        """
        Инициализация с моделью системы.
        
        Args:
            model: экземпляр PredatorPreyModel
        """
        self.model = model
        self.results = {}
        
    def simulate_without_control(self, initial_conditions: List[Tuple[float, float]],
                                t_span: Tuple[float, float] = (0, 20),
                                dt: float = 0.01) -> Dict:
        """
        Моделирование системы без управления (классическая модель Лотки-Вольтерра).
        
        Args:
            initial_conditions: список начальных условий [(x1_0, x2_0), ...]
            t_span: временной интервал
            dt: шаг интегрирования
            
        Returns:
            результаты моделирования
        """
        print("Моделирование системы без управления...")
        
        trajectories = []
        
        for i, (x1_0, x2_0) in enumerate(initial_conditions):
            print(f"  Траектория {i+1}: x1(0) = {x1_0}, x2(0) = {x2_0}")
            
            # Численное решение методом Эйлера
            t_euler, y_euler = euler_method(
                self.model.dynamics_without_control,
                np.array([x1_0, x2_0]),
                t_span,
                dt
            )
            
            # Сравнение с scipy для проверки точности
            sol_scipy = solve_ivp(
                lambda t, y: self.model.dynamics_without_control(t, y),
                t_span,
                [x1_0, x2_0],
                t_eval=t_euler,
                rtol=1e-8
            )
            
            trajectories.append({
                'initial': (x1_0, x2_0),
                'time': t_euler,
                'euler_solution': y_euler,
                'scipy_solution': sol_scipy.y.T,
                'method_comparison': {
                    'max_error_x1': np.max(np.abs(y_euler[:, 0] - sol_scipy.y[0])),
                    'max_error_x2': np.max(np.abs(y_euler[:, 1] - sol_scipy.y[1]))
                }
            })
        
        # Анализ свойств системы без управления
        trivial, nontrivial = self.model.equilibrium_points()
        
        result = {
            'type': 'without_control',
            'trajectories': trajectories,
            'equilibrium_points': {
                'trivial': trivial,
                'nontrivial': nontrivial
            },
            'time_span': t_span,
            'dt': dt
        }
        
        self.results['without_control'] = result
        return result
    
    def simulate_with_control(self, control_function: Callable[[float, float], float],
                            initial_conditions: List[Tuple[float, float]],
                            target_point: Tuple[float, float],
                            t_span: Tuple[float, float] = (0, 20),
                            dt: float = 0.01) -> Dict:
        """
        Моделирование системы с управлением.
        
        Args:
            control_function: функция управления u(x1, x2)
            initial_conditions: список начальных условий
            target_point: целевая точка равновесия (x1*, x2*)
            t_span: временной интервал
            dt: шаг интегрирования
            
        Returns:
            результаты моделирования
        """
        print("Моделирование системы с управлением...")
        
        # Установка управления
        self.model.set_control(control_function)
        
        trajectories = []
        
        for i, (x1_0, x2_0) in enumerate(initial_conditions):
            print(f"  Траектория {i+1}: x1(0) = {x1_0}, x2(0) = {x2_0}")
            
            # Численное решение с управлением
            t, y = euler_method(
                self.model.dynamics_with_control,
                np.array([x1_0, x2_0]),
                t_span,
                dt
            )
            
            # Вычисление управляющего воздействия
            u_values = np.array([control_function(y[j, 0], y[j, 1]) for j in range(len(t))])
            
            # Анализ сходимости к целевой точке
            x1_target, x2_target = target_point
            errors_x1 = np.abs(y[:, 0] - x1_target)
            errors_x2 = np.abs(y[:, 1] - x2_target)
            
            # Время достижения 5% точности
            tolerance = 0.05
            settling_indices_x1 = np.where(errors_x1 < tolerance * x1_target)[0]
            settling_indices_x2 = np.where(errors_x2 < tolerance * x2_target)[0]
            
            settling_time_x1 = t[settling_indices_x1[0]] if len(settling_indices_x1) > 0 else None
            settling_time_x2 = t[settling_indices_x2[0]] if len(settling_indices_x2) > 0 else None
            
            trajectories.append({
                'initial': (x1_0, x2_0),
                'time': t,
                'solution': y,
                'control': u_values,
                'errors': {
                    'x1': errors_x1,
                    'x2': errors_x2
                },
                'settling_time': {
                    'x1': settling_time_x1,
                    'x2': settling_time_x2
                },
                'final_values': (y[-1, 0], y[-1, 1]),
                'control_effort': {
                    'max': np.max(np.abs(u_values)),
                    'mean': np.mean(np.abs(u_values)),
                    'rms': np.sqrt(np.mean(u_values**2))
                }
            })
        
        result = {
            'type': 'with_control',
            'trajectories': trajectories,
            'target_point': target_point,
            'time_span': t_span,
            'dt': dt
        }
        
        self.results['with_control'] = result
        return result
    
    def compare_control_strategies(self, control_strategies: Dict[str, Callable],
                                 initial_condition: Tuple[float, float],
                                 target_point: Tuple[float, float],
                                 t_span: Tuple[float, float] = (0, 15)) -> Dict:
        """
        Сравнение различных стратегий управления.
        
        Args:
            control_strategies: словарь {название: функция_управления}
            initial_condition: начальные условия
            target_point: целевая точка
            t_span: временной интервал
            
        Returns:
            результаты сравнения
        """
        print("Сравнение стратегий управления...")
        
        comparison_results = {}
        
        for strategy_name, control_func in control_strategies.items():
            print(f"  Стратегия: {strategy_name}")
            
            result = self.simulate_with_control(
                control_func, 
                [initial_condition], 
                target_point, 
                t_span
            )
            
            traj = result['trajectories'][0]
            
            comparison_results[strategy_name] = {
                'trajectory': traj,
                'performance_metrics': {
                    'settling_time_x1': traj['settling_time']['x1'],
                    'settling_time_x2': traj['settling_time']['x2'],
                    'max_control_effort': traj['control_effort']['max'],
                    'mean_control_effort': traj['control_effort']['mean'],
                    'final_error_x1': abs(traj['final_values'][0] - target_point[0]),
                    'final_error_x2': abs(traj['final_values'][1] - target_point[1])
                }
            }
        
        # Создание сводной таблицы
        metrics_df = pd.DataFrame({
            name: result['performance_metrics'] 
            for name, result in comparison_results.items()
        }).T
        
        comparison_results['summary_table'] = metrics_df
        self.results['comparison'] = comparison_results
        
        return comparison_results
    
    def parameter_sensitivity_analysis(self, base_control_designer: ControlDesigner,
                                     parameter_ranges: Dict[str, Tuple[float, float]],
                                     target_point: Tuple[float, float],
                                     num_samples: int = 10) -> Dict:
        """
        Анализ чувствительности к параметрам управления.
        
        Args:
            base_control_designer: базовый проектировщик управления
            parameter_ranges: диапазоны параметров {'T1': (min, max), 'T2': (min, max)}
            target_point: целевая точка
            num_samples: количество образцов для каждого параметра
            
        Returns:
            результаты анализа чувствительности
        """
        print("Анализ чувствительности к параметрам...")
        
        base_control_designer.set_target_point(target_point[1])
        
        sensitivity_results = {}
        
        for param_name, (min_val, max_val) in parameter_ranges.items():
            print(f"  Анализ параметра {param_name}")
            
            param_values = np.linspace(min_val, max_val, num_samples)
            results_for_param = []
            
            for param_val in param_values:
                if param_name == 'T1':
                    T1, T2 = param_val, 0.5  # фиксированное значение T2
                elif param_name == 'T2':
                    T1, T2 = 0.5, param_val  # фиксированное значение T1
                else:
                    continue
                
                try:
                    # Создание управления с текущими параметрами
                    control_func = base_control_designer.design_linear_control(T1, T2)
                    
                    # Анализ устойчивости
                    stability_analysis = base_control_designer.analyze_stability_conditions()
                    
                    # Краткое моделирование для оценки качества
                    sim_result = self.simulate_with_control(
                        control_func,
                        [(target_point[0] + 2, target_point[1] + 1)],  # небольшое отклонение
                        target_point,
                        t_span=(0, 10),
                        dt=0.01
                    )
                    
                    traj = sim_result['trajectories'][0]
                    
                    results_for_param.append({
                        'parameter_value': param_val,
                        'is_stable': stability_analysis['is_stable'],
                        'max_real_eigenvalue': max(np.real(stability_analysis['eigenvalues'])),
                        'settling_time_x2': traj['settling_time']['x2'],
                        'max_control_effort': traj['control_effort']['max'],
                        'final_error': np.sqrt(
                            (traj['final_values'][0] - target_point[0])**2 +
                            (traj['final_values'][1] - target_point[1])**2
                        )
                    })
                
                except Exception as e:
                    results_for_param.append({
                        'parameter_value': param_val,
                        'is_stable': False,
                        'max_real_eigenvalue': float('inf'),
                        'settling_time_x2': None,
                        'max_control_effort': float('inf'),
                        'final_error': float('inf'),
                        'error': str(e)
                    })
            
            sensitivity_results[param_name] = results_for_param
        
        self.results['sensitivity'] = sensitivity_results
        return sensitivity_results
    
    def export_results_to_csv(self, filename: str = "simulation_results.csv"):
        """
        Экспорт результатов в CSV файл.
        
        Args:
            filename: имя файла для сохранения
        """
        if 'comparison' in self.results:
            df = self.results['comparison']['summary_table']
            df.to_csv(filename)
            print(f"Результаты сравнения сохранены в {filename}")
        else:
            print("Нет результатов сравнения для экспорта")


def comprehensive_simulation_demo():
    """
    Комплексная демонстрация численного моделирования.
    """
    print("КОМПЛЕКСНОЕ ЧИСЛЕННОЕ МОДЕЛИРОВАНИЕ")
    print("=" * 50)
    
    # Создание модели с параметрами
    model = PredatorPreyModel(a1=1.0, a2=0.5, beta1=0.2, beta2=0.1)
    
    # Создание симулятора
    simulator = SimulationRunner(model)
    
    # 1. Моделирование без управления
    print("\n1. СИСТЕМА БЕЗ УПРАВЛЕНИЯ")
    print("-" * 30)
    
    initial_conditions = [
        (8.0, 4.0),  # далеко от равновесия
        (4.0, 6.0),  # другая начальная точка
        (6.0, 2.0)   # третья точка
    ]
    
    result_no_control = simulator.simulate_without_control(
        initial_conditions, t_span=(0, 15)
    )
    
    print(f"Точки равновесия:")
    print(f"  Тривиальная: {result_no_control['equilibrium_points']['trivial']}")
    print(f"  Нетривиальная: {result_no_control['equilibrium_points']['nontrivial']}")
    
    # 2. Создание управления
    print("\n2. ПРОЕКТИРОВАНИЕ УПРАВЛЕНИЯ")
    print("-" * 35)
    
    designer = ControlDesigner(model)
    target_x2 = 3.0
    designer.set_target_point(target_x2)
    
    # Поиск оптимальных параметров
    T1_opt, T2_opt = designer.optimal_parameters_search(
        T1_range=(0.1, 3.0), T2_range=(0.1, 3.0), num_points=15
    )
    
    if T1_opt is not None:
        # 3. Моделирование с управлением
        print("\n3. СИСТЕМА С УПРАВЛЕНИЕМ")
        print("-" * 30)
        
        control_func = designer.design_linear_control(T1_opt, T2_opt)
        target_point = (designer.x1_star, designer.x2_star)
        
        result_with_control = simulator.simulate_with_control(
            control_func,
            initial_conditions,
            target_point,
            t_span=(0, 15)
        )
        
        print("Результаты стабилизации:")
        for i, traj in enumerate(result_with_control['trajectories']):
            print(f"  Траектория {i+1}:")
            print(f"    Время установления x₂: {traj['settling_time']['x2']:.2f} с" 
                  if traj['settling_time']['x2'] else "    Не достигнуто")
            print(f"    Максимальное управление: {traj['control_effort']['max']:.3f}")
            print(f"    Финальные значения: ({traj['final_values'][0]:.3f}, {traj['final_values'][1]:.3f})")
        
        # 4. Сравнение стратегий управления
        print("\n4. СРАВНЕНИЕ СТРАТЕГИЙ УПРАВЛЕНИЯ")
        print("-" * 40)
        
        # Различные стратегии
        strategies = {
            'Оптимальное линейное': designer.design_linear_control(T1_opt, T2_opt),
            'Консервативное': designer.design_linear_control(T1_opt * 0.5, T2_opt * 0.5),
            'Агрессивное': designer.design_linear_control(T1_opt * 1.5, T2_opt * 1.5),
            'Нелинейное': designer.design_nonlinear_control(0.3, 0.1)
        }
        
        comparison = simulator.compare_control_strategies(
            strategies,
            initial_conditions[0],
            target_point
        )
        
        print("Сводная таблица сравнения:")
        print(comparison['summary_table'].round(3))
        
        # 5. Анализ чувствительности
        print("\n5. АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ")
        print("-" * 35)
        
        sensitivity = simulator.parameter_sensitivity_analysis(
            designer,
            {'T1': (0.1, 2.0), 'T2': (0.1, 2.0)},
            target_point,
            num_samples=8
        )
        
        # Экспорт результатов
        simulator.export_results_to_csv()
        
        print("\nМоделирование завершено успешно!")
        
        return simulator
    
    else:
        print("Не удалось найти стабильные параметры управления")
        return None


if __name__ == "__main__":
    simulator = comprehensive_simulation_demo() 