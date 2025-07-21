"""
Визуализация результатов моделирования
====================================

Создание графиков и диаграмм для анализа поведения системы жертва-хищник
с управлением и без него.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

from predator_prey_model import PredatorPreyModel
from numerical_simulation import SimulationRunner

# Настройка стиля графиков
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class Visualizer:
    """
    Класс для создания различных типов визуализации.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Инициализация визуализатора.
        
        Args:
            figsize: размер фигур по умолчанию
        """
        self.figsize = figsize
        self.colors = plt.cm.Set1(np.linspace(0, 1, 10))
        
    def plot_phase_portrait(self, model: PredatorPreyModel, 
                           simulation_results: Optional[Dict] = None,
                           x1_range: Tuple[float, float] = (0, 15),
                           x2_range: Tuple[float, float] = (0, 8),
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Построение фазового портрета системы.
        
        Args:
            model: модель системы
            simulation_results: результаты моделирования для наложения траекторий
            x1_range: диапазон по x1 (жертвы)
            x2_range: диапазон по x2 (хищники)
            save_path: путь для сохранения графика
            
        Returns:
            объект фигуры matplotlib
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Создание сетки для векторного поля
        X1, X2, U, V = model.phase_portrait_data(x1_range, x2_range, num_points=20)
        
        # Нормализация векторов для лучшей визуализации
        magnitude = np.sqrt(U**2 + V**2)
        magnitude[magnitude == 0] = 1  # избегаем деления на ноль
        U_norm = U / magnitude
        V_norm = V / magnitude
        
        # Векторное поле
        ax.quiver(X1, X2, U_norm, V_norm, magnitude, 
                 scale=30, alpha=0.6, cmap='viridis')
        
        # Точки равновесия
        trivial, nontrivial = model.equilibrium_points()
        ax.plot(trivial[0], trivial[1], 'ro', markersize=8, 
               label='Тривиальное равновесие')
        ax.plot(nontrivial[0], nontrivial[1], 'go', markersize=8, 
               label='Нетривиальное равновесие')
        
        # Наложение траекторий из результатов моделирования
        if simulation_results:
            if 'without_control' in simulation_results:
                for i, traj in enumerate(simulation_results['without_control']['trajectories']):
                    solution = traj['euler_solution']
                    ax.plot(solution[:, 0], solution[:, 1], 
                           '--', linewidth=2, alpha=0.8,
                           label=f'Без управления {i+1}')
                    # Начальная точка
                    ax.plot(solution[0, 0], solution[0, 1], 'o', 
                           markersize=6, color=self.colors[i])
            
            if 'with_control' in simulation_results:
                for i, traj in enumerate(simulation_results['with_control']['trajectories']):
                    solution = traj['solution']
                    ax.plot(solution[:, 0], solution[:, 1], 
                           '-', linewidth=2, alpha=0.9,
                           label=f'С управлением {i+1}')
                    # Начальная точка
                    ax.plot(solution[0, 0], solution[0, 1], 's', 
                           markersize=6, color=self.colors[i])
                    # Конечная точка
                    ax.plot(solution[-1, 0], solution[-1, 1], '*', 
                           markersize=8, color=self.colors[i])
                
                # Целевая точка
                target = simulation_results['with_control']['target_point']
                ax.plot(target[0], target[1], 'r*', markersize=12, 
                       label='Целевая точка')
        
        ax.set_xlabel('Популяция жертв (x₁)')
        ax.set_ylabel('Популяция хищников (x₂)')
        ax.set_title('Фазовый портрет системы жертва-хищник')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x1_range)
        ax.set_ylim(x2_range)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_time_series(self, simulation_results: Dict,
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Построение временных рядов популяций.
        
        Args:
            simulation_results: результаты моделирования
            save_path: путь для сохранения
            
        Returns:
            объект фигуры matplotlib
        """
        n_scenarios = len([k for k in simulation_results.keys() 
                          if k in ['without_control', 'with_control']])
        
        fig, axes = plt.subplots(n_scenarios, 2, figsize=(15, 6*n_scenarios))
        if n_scenarios == 1:
            axes = axes.reshape(1, -1)
        
        row_idx = 0
        
        # Графики без управления
        if 'without_control' in simulation_results:
            results = simulation_results['without_control']
            
            for i, traj in enumerate(results['trajectories']):
                t = traj['time']
                solution = traj['euler_solution']
                
                # График x1(t)
                axes[row_idx, 0].plot(t, solution[:, 0], 
                                    label=f'Траектория {i+1}', 
                                    linewidth=2, color=self.colors[i])
                
                # График x2(t)
                axes[row_idx, 1].plot(t, solution[:, 1], 
                                    label=f'Траектория {i+1}', 
                                    linewidth=2, color=self.colors[i])
            
            # Линии равновесия
            eq_points = results['equilibrium_points']['nontrivial']
            axes[row_idx, 0].axhline(y=eq_points[0], color='red', 
                                   linestyle='--', alpha=0.7, label='Равновесие')
            axes[row_idx, 1].axhline(y=eq_points[1], color='red', 
                                   linestyle='--', alpha=0.7, label='Равновесие')
            
            axes[row_idx, 0].set_title('Популяция жертв (без управления)')
            axes[row_idx, 1].set_title('Популяция хищников (без управления)')
            
            row_idx += 1
        
        # Графики с управлением
        if 'with_control' in simulation_results:
            results = simulation_results['with_control']
            
            for i, traj in enumerate(results['trajectories']):
                t = traj['time']
                solution = traj['solution']
                
                # График x1(t)
                axes[row_idx, 0].plot(t, solution[:, 0], 
                                    label=f'Траектория {i+1}', 
                                    linewidth=2, color=self.colors[i])
                
                # График x2(t)
                axes[row_idx, 1].plot(t, solution[:, 1], 
                                    label=f'Траектория {i+1}', 
                                    linewidth=2, color=self.colors[i])
            
            # Целевые значения
            target = results['target_point']
            axes[row_idx, 0].axhline(y=target[0], color='red', 
                                   linestyle='--', alpha=0.7, label='Цель')
            axes[row_idx, 1].axhline(y=target[1], color='red', 
                                   linestyle='--', alpha=0.7, label='Цель')
            
            axes[row_idx, 0].set_title('Популяция жертв (с управлением)')
            axes[row_idx, 1].set_title('Популяция хищников (с управлением)')
        
        # Оформление всех графиков
        for i in range(n_scenarios):
            for j in range(2):
                axes[i, j].set_xlabel('Время')
                axes[i, j].set_ylabel('Популяция')
                axes[i, j].legend()
                axes[i, j].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_control_signal(self, simulation_results: Dict,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Построение графика управляющего сигнала.
        
        Args:
            simulation_results: результаты моделирования с управлением
            save_path: путь для сохранения
            
        Returns:
            объект фигуры matplotlib
        """
        if 'with_control' not in simulation_results:
            raise ValueError("Нет данных о системе с управлением")
        
        results = simulation_results['with_control']
        
        fig, axes = plt.subplots(2, 1, figsize=self.figsize)
        
        for i, traj in enumerate(results['trajectories']):
            t = traj['time']
            u = traj['control']
            
            # График управляющего сигнала
            axes[0].plot(t, u, label=f'Траектория {i+1}', 
                        linewidth=2, color=self.colors[i])
            
            # График накопленной энергии управления
            u_energy = np.cumsum(u**2) * (t[1] - t[0])  # приближенный интеграл
            axes[1].plot(t, u_energy, label=f'Траектория {i+1}', 
                        linewidth=2, color=self.colors[i])
        
        axes[0].set_title('Управляющий сигнал u(t)')
        axes[0].set_xlabel('Время')
        axes[0].set_ylabel('Управление u')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_title('Накопленная энергия управления')
        axes[1].set_xlabel('Время')
        axes[1].set_ylabel('∫u²dt')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_error_dynamics(self, simulation_results: Dict,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Построение графиков ошибок регулирования.
        
        Args:
            simulation_results: результаты моделирования с управлением
            save_path: путь для сохранения
            
        Returns:
            объект фигуры matplotlib
        """
        if 'with_control' not in simulation_results:
            raise ValueError("Нет данных о системе с управлением")
        
        results = simulation_results['with_control']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for i, traj in enumerate(results['trajectories']):
            t = traj['time']
            errors_x1 = traj['errors']['x1']
            errors_x2 = traj['errors']['x2']
            
            # Ошибки по x1 и x2
            axes[0, 0].semilogy(t, errors_x1, label=f'Траектория {i+1}', 
                               linewidth=2, color=self.colors[i])
            axes[0, 1].semilogy(t, errors_x2, label=f'Траектория {i+1}', 
                               linewidth=2, color=self.colors[i])
            
            # Суммарная ошибка
            total_error = np.sqrt(errors_x1**2 + errors_x2**2)
            axes[1, 0].semilogy(t, total_error, label=f'Траектория {i+1}', 
                               linewidth=2, color=self.colors[i])
            
            # Фазовая ошибка
            axes[1, 1].plot(errors_x1, errors_x2, label=f'Траектория {i+1}', 
                           linewidth=2, color=self.colors[i])
            axes[1, 1].plot(errors_x1[0], errors_x2[0], 'o', 
                           markersize=6, color=self.colors[i])
        
        # Оформление графиков
        titles = [
            'Ошибка по популяции жертв |x₁ - x₁*|',
            'Ошибка по популяции хищников |x₂ - x₂*|',
            'Суммарная ошибка √(e₁² + e₂²)',
            'Фазовая плоскость ошибок'
        ]
        
        xlabels = ['Время', 'Время', 'Время', 'Ошибка x₁']
        ylabels = ['|e₁|', '|e₂|', '|e|', 'Ошибка x₂']
        
        for i, ax in enumerate(axes.flat):
            ax.set_title(titles[i])
            ax.set_xlabel(xlabels[i])
            ax.set_ylabel(ylabels[i])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Добавляем 5% зону толерантности на фазовой плоскости
        target = results['target_point']
        tolerance = 0.05
        circle = patches.Circle((0, 0), 
                              tolerance * np.sqrt(target[0]**2 + target[1]**2), 
                              fill=False, linestyle='--', color='red', alpha=0.5)
        axes[1, 1].add_patch(circle)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_comparison_dashboard(self, comparison_results: Dict,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Создание дашборда для сравнения стратегий управления.
        
        Args:
            comparison_results: результаты сравнения стратегий
            save_path: путь для сохранения
            
        Returns:
            объект фигуры matplotlib
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Извлечение данных
        strategies = list(comparison_results.keys())
        if 'summary_table' in strategies:
            strategies.remove('summary_table')
        
        # 1. Траектории в фазовом пространстве
        ax1 = plt.subplot(2, 3, 1)
        for i, strategy in enumerate(strategies):
            traj = comparison_results[strategy]['trajectory']
            solution = traj['solution']
            ax1.plot(solution[:, 0], solution[:, 1], 
                    linewidth=2, label=strategy, color=self.colors[i])
            ax1.plot(solution[0, 0], solution[0, 1], 'o', 
                    markersize=6, color=self.colors[i])
        
        # Целевая точка
        target = (traj['solution'][-1, 0], traj['solution'][-1, 1])  # примерно
        ax1.plot(target[0], target[1], 'r*', markersize=12, label='Цель')
        ax1.set_xlabel('x₁')
        ax1.set_ylabel('x₂')
        ax1.set_title('Фазовые траектории')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Временные ряды x₂
        ax2 = plt.subplot(2, 3, 2)
        for i, strategy in enumerate(strategies):
            traj = comparison_results[strategy]['trajectory']
            t = traj['time']
            solution = traj['solution']
            ax2.plot(t, solution[:, 1], linewidth=2, 
                    label=strategy, color=self.colors[i])
        
        ax2.set_xlabel('Время')
        ax2.set_ylabel('x₂ (хищники)')
        ax2.set_title('Динамика популяции хищников')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Управляющие сигналы
        ax3 = plt.subplot(2, 3, 3)
        for i, strategy in enumerate(strategies):
            traj = comparison_results[strategy]['trajectory']
            t = traj['time']
            u = traj['control']
            ax3.plot(t, u, linewidth=2, 
                    label=strategy, color=self.colors[i])
        
        ax3.set_xlabel('Время')
        ax3.set_ylabel('u(t)')
        ax3.set_title('Управляющие сигналы')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Барplot времени установления
        ax4 = plt.subplot(2, 3, 4)
        settling_times = []
        strategy_names = []
        for strategy in strategies:
            settling_time = comparison_results[strategy]['performance_metrics']['settling_time_x2']
            if settling_time is not None:
                settling_times.append(settling_time)
                strategy_names.append(strategy)
        
        bars = ax4.bar(range(len(settling_times)), settling_times, 
                      color=self.colors[:len(settling_times)])
        ax4.set_xlabel('Стратегия')
        ax4.set_ylabel('Время установления (с)')
        ax4.set_title('Скорость сходимости')
        ax4.set_xticks(range(len(strategy_names)))
        ax4.set_xticklabels(strategy_names, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        # 5. Барplot энергии управления
        ax5 = plt.subplot(2, 3, 5)
        control_efforts = [comparison_results[strategy]['performance_metrics']['max_control_effort'] 
                          for strategy in strategies]
        
        bars = ax5.bar(range(len(strategies)), control_efforts, 
                      color=self.colors[:len(strategies)])
        ax5.set_xlabel('Стратегия')
        ax5.set_ylabel('Макс. управление')
        ax5.set_title('Энергозатраты')
        ax5.set_xticks(range(len(strategies)))
        ax5.set_xticklabels(strategies, rotation=45, ha='right')
        ax5.grid(True, alpha=0.3)
        
        # 6. Радарная диаграмма
        ax6 = plt.subplot(2, 3, 6, projection='polar')
        
        # Нормализация метрик для радарной диаграммы
        metrics = ['settling_time_x2', 'max_control_effort', 'final_error_x2']
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # замыкание
        
        for i, strategy in enumerate(strategies):
            values = []
            perf = comparison_results[strategy]['performance_metrics']
            
            # Нормализация (инвертирование для лучшего отображения)
            st = perf['settling_time_x2'] if perf['settling_time_x2'] else 20
            values.append(1 / (1 + st/10))  # время установления
            values.append(1 / (1 + perf['max_control_effort']))  # энергия
            values.append(1 / (1 + perf['final_error_x2']*1000))  # точность
            
            values += values[:1]  # замыкание
            
            ax6.plot(angles, values, 'o-', linewidth=2, 
                    label=strategy, color=self.colors[i])
            ax6.fill(angles, values, alpha=0.1, color=self.colors[i])
        
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(['Скорость', 'Эффективность', 'Точность'])
        ax6.set_title('Комплексная оценка')
        ax6.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_sensitivity_analysis(self, sensitivity_results: Dict,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Визуализация анализа чувствительности.
        
        Args:
            sensitivity_results: результаты анализа чувствительности
            save_path: путь для сохранения
            
        Returns:
            объект фигуры matplotlib
        """
        n_params = len(sensitivity_results)
        fig, axes = plt.subplots(2, n_params, figsize=(6*n_params, 10))
        
        if n_params == 1:
            axes = axes.reshape(-1, 1)
        
        for j, (param_name, results) in enumerate(sensitivity_results.items()):
            param_values = [r['parameter_value'] for r in results]
            settling_times = [r['settling_time_x2'] if r['settling_time_x2'] else np.nan for r in results]
            max_eigenvalues = [r['max_real_eigenvalue'] for r in results]
            
            # График времени установления
            axes[0, j].plot(param_values, settling_times, 'o-', linewidth=2)
            axes[0, j].set_xlabel(param_name)
            axes[0, j].set_ylabel('Время установления (с)')
            axes[0, j].set_title(f'Влияние {param_name} на скорость')
            axes[0, j].grid(True, alpha=0.3)
            
            # График собственных значений
            axes[1, j].plot(param_values, max_eigenvalues, 'o-', linewidth=2)
            axes[1, j].axhline(y=0, color='red', linestyle='--', alpha=0.7)
            axes[1, j].set_xlabel(param_name)
            axes[1, j].set_ylabel('Макс. действ. часть с.з.')
            axes[1, j].set_title(f'Влияние {param_name} на устойчивость')
            axes[1, j].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def create_comprehensive_report(simulator: SimulationRunner, 
                              output_dir: str = "simulation_results"):
    """
    Создание комплексного отчета с визуализацией.
    
    Args:
        simulator: объект симулятора с результатами
        output_dir: директория для сохранения графиков
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = Visualizer()
    
    print("Создание визуализаций...")
    
    # 1. Фазовый портрет
    if 'without_control' in simulator.results or 'with_control' in simulator.results:
        model = simulator.model
        fig1 = visualizer.plot_phase_portrait(
            model, simulator.results, 
            save_path=f"{output_dir}/phase_portrait.png"
        )
        print("  ✓ Фазовый портрет сохранен")
    
    # 2. Временные ряды
    if simulator.results:
        fig2 = visualizer.plot_time_series(
            simulator.results,
            save_path=f"{output_dir}/time_series.png"
        )
        print("  ✓ Временные ряды сохранены")
    
    # 3. Управляющий сигнал
    if 'with_control' in simulator.results:
        fig3 = visualizer.plot_control_signal(
            simulator.results,
            save_path=f"{output_dir}/control_signal.png"
        )
        print("  ✓ График управления сохранен")
        
        # 4. Динамика ошибок
        fig4 = visualizer.plot_error_dynamics(
            simulator.results,
            save_path=f"{output_dir}/error_dynamics.png"
        )
        print("  ✓ График ошибок сохранен")
    
    # 5. Сравнение стратегий
    if 'comparison' in simulator.results:
        fig5 = visualizer.plot_comparison_dashboard(
            simulator.results['comparison'],
            save_path=f"{output_dir}/comparison_dashboard.png"
        )
        print("  ✓ Дашборд сравнения сохранен")
    
    # 6. Анализ чувствительности
    if 'sensitivity' in simulator.results:
        fig6 = visualizer.plot_sensitivity_analysis(
            simulator.results['sensitivity'],
            save_path=f"{output_dir}/sensitivity_analysis.png"
        )
        print("  ✓ Анализ чувствительности сохранен")
    
    print(f"\nВсе графики сохранены в директории: {output_dir}")


if __name__ == "__main__":
    from numerical_simulation import comprehensive_simulation_demo
    
    # Запуск демонстрации и создание отчета
    simulator = comprehensive_simulation_demo()
    if simulator:
        create_comprehensive_report(simulator) 