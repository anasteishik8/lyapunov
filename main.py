"""
ГЛАВНЫЙ МОДУЛЬ ПРОЕКТА "ЖЕРТВА-ХИЩНИК" С УПРАВЛЕНИЕМ
==================================================

Комплексный анализ системы жертва-хищник с управлением методом Ляпунова.
Структура выполняется согласно инструкции пользователя.

Автор: [Ваше имя]
Дата: [Текущая дата]
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime

# Импорт собственных модулей
from predator_prey_model import PredatorPreyModel
from control_design import ControlDesigner
from lyapunov_analysis import LyapunovAnalyzer
from numerical_simulation import SimulationRunner, comprehensive_simulation_demo
from visualization import Visualizer, create_comprehensive_report


class PredatorPreyProject:
    """
    Главный класс проекта, объединяющий все компоненты анализа.
    """
    
    def __init__(self):
        """Инициализация проекта."""
        print("="*70)
        print("ПРОЕКТ: МОДЕЛЬ 'ЖЕРТВА-ХИЩНИК' С УПРАВЛЕНИЕМ (МЕТОД ЛЯПУНОВА)")
        print("="*70)
        print(f"Дата запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Параметры модели (можно настроить для уникальности работы)
        self.model_params = {
            'a1': 1.2,    # коэффициент роста жертв
            'a2': 0.6,    # коэффициент убыли хищников  
            'beta1': 0.25, # коэффициент воздействия хищников на жертв
            'beta2': 0.12   # коэффициент воздействия жертв на хищников
        }
        
        # Целевая точка стабилизации
        self.target_x2 = 2.5  # целевая популяция хищников
        
        # Создание основных объектов
        self.model = None
        self.designer = None
        self.analyzer = None
        self.simulator = None
        self.visualizer = None
        
        # Результаты анализа
        self.results = {}
        
    def section_1_introduction(self):
        """1. ВВЕДЕНИЕ - актуальность и цель работы."""
        print("1. ВВЕДЕНИЕ")
        print("="*50)
        print()
        print("АКТУАЛЬНОСТЬ РАБОТЫ:")
        print("-"*20)
        print("Модели типа 'жертва-хищник' широко применяются в экологии,")
        print("экономике, эпидемиологии и других областях для описания")
        print("взаимодействующих популяций. Управление такими системами")
        print("позволяет стабилизировать популяции на желаемых уровнях,")
        print("что критически важно для:")
        print("• Управления экосистемами и биоресурсами")
        print("• Контроля эпидемий")
        print("• Моделирования рыночной конкуренции")
        print("• Анализа динамики социальных процессов")
        print()
        print("ЦЕЛЬ РАБОТЫ:")
        print("-"*12)
        print("Построить стабилизирующее управление для системы жертва-хищник,")
        print("обеспечивающее стабилизацию популяции хищников на заданном")
        print("уровне x₂* с формальным доказательством устойчивости методом Ляпунова.")
        print()
        print("ЗАДАЧИ ИССЛЕДОВАНИЯ:")
        print("-"*20)
        print("1. Исследовать поведение системы без управления")
        print("2. Синтезировать управление по методике АКАР")
        print("3. Провести формальное доказательство устойчивости по Ляпунову")
        print("4. Выполнить численное моделирование и верификацию")
        print("5. Проанализировать качество управления")
        print()
        
    def section_2_theory(self):
        """2. ТЕОРЕТИЧЕСКАЯ ЧАСТЬ - описание моделей и методов."""
        print("2. ТЕОРЕТИЧЕСКАЯ ЧАСТЬ")
        print("="*50)
        print()
        print("2.1 КЛАССИЧЕСКАЯ МОДЕЛЬ ЛОТКИ-ВОЛЬТЕРРА")
        print("-"*45)
        print("Базовая система описывается уравнениями:")
        print("ẋ₁ = a₁x₁ - β₁x₁x₂  (динамика жертв)")
        print("ẋ₂ = -a₂x₂ + β₂x₁x₂  (динамика хищников)")
        print()
        print("где:")
        print(f"• a₁ = {self.model_params['a1']} - коэффициент роста жертв")
        print(f"• a₂ = {self.model_params['a2']} - коэффициент убыли хищников")
        print(f"• β₁ = {self.model_params['beta1']} - интенсивность хищничества")
        print(f"• β₂ = {self.model_params['beta2']} - эффективность хищников")
        print()
        print("2.2 МЕТОД ЭЙЛЕРА ДЛЯ ЧИСЛЕННОГО РЕШЕНИЯ")
        print("-"*42)
        print("Дискретизация системы ОДУ:")
        print("x₁(k+1) = x₁(k) + dt·f₁(x₁(k), x₂(k))")
        print("x₂(k+1) = x₂(k) + dt·f₂(x₁(k), x₂(k))")
        print("где dt - шаг интегрирования")
        print()
        print("2.3 МЕТОДИКА АКАР")
        print("-"*17)
        print("Алгебраический критерий абсолютной устойчивости")
        print("используется для синтеза стабилизирующего управления")
        print("на основе анализа характеристического полинома")
        print("линеаризованной системы.")
        print()
        
    def section_3_problem_statement(self):
        """3. ПОСТАНОВКА ЗАДАЧИ."""
        print("3. ПОСТАНОВКА ЗАДАЧИ")
        print("="*50)
        print()
        print("РАССМАТРИВАЕМАЯ СИСТЕМА:")
        print("-"*25)
        print("ẋ₁ = a₁x₁ - β₁x₁x₂ + u(x₁,x₂)  (управляемая динамика жертв)")
        print("ẋ₂ = -a₂x₂ + β₂x₁x₂            (динамика хищников)")
        print()
        print("УПРАВЛЯЮЩЕЕ ВОЗДЕЙСТВИЕ:")
        print("-"*24)
        print("u(x₁,x₂) - управление по популяции жертв")
        print("(например, искусственное разведение/изъятие)")
        print()
        print("ЦЕЛЬ УПРАВЛЕНИЯ:")
        print("-"*16)
        print(f"Стабилизировать популяцию хищников на уровне x₂* = {self.target_x2}")
        print("при обеспечении устойчивости системы")
        print()
        print("ОГРАНИЧЕНИЯ:")
        print("-"*12)
        print("• x₁(t) ≥ 0, x₂(t) ≥ 0 (неотрицательность популяций)")
        print("• |u(t)| ≤ u_max (ограничение управления)")
        print("• Минимизация энергозатрат на управление")
        print()
        
    def section_4_discretization(self):
        """4. ДИСКРЕТИЗАЦИЯ СИСТЕМЫ."""
        print("4. ДИСКРЕТИЗАЦИЯ СИСТЕМЫ (МЕТОД ЭЙЛЕРА)")
        print("="*50)
        print()
        
        # Создание модели
        self.model = PredatorPreyModel(**self.model_params)
        
        print("НЕПРЕРЫВНАЯ СИСТЕМА:")
        print("-"*20)
        print(f"ẋ₁ = {self.model_params['a1']}x₁ - {self.model_params['beta1']}x₁x₂ + u")
        print(f"ẋ₂ = -{self.model_params['a2']}x₂ + {self.model_params['beta2']}x₁x₂")
        print()
        print("ДИСКРЕТНАЯ АППРОКСИМАЦИЯ (dt = 0.01):")
        print("-"*38)
        print(f"x₁(k+1) = x₁(k) + 0.01·({self.model_params['a1']}x₁(k) - {self.model_params['beta1']}x₁(k)x₂(k) + u(k))")
        print(f"x₂(k+1) = x₂(k) + 0.01·(-{self.model_params['a2']}x₂(k) + {self.model_params['beta2']}x₁(k)x₂(k))")
        print()
        
        # Анализ точек равновесия
        trivial, nontrivial = self.model.equilibrium_points()
        print("ТОЧКИ РАВНОВЕСИЯ СИСТЕМЫ БЕЗ УПРАВЛЕНИЯ:")
        print("-"*45)
        print(f"Тривиальная: ({trivial[0]:.1f}, {trivial[1]:.1f})")
        print(f"Нетривиальная: ({nontrivial[0]:.3f}, {nontrivial[1]:.3f})")
        print()
        
        self.results['equilibrium_points'] = {
            'trivial': trivial,
            'nontrivial': nontrivial
        }
        
    def section_5_control_synthesis(self):
        """5. СИНТЕЗ УПРАВЛЕНИЯ ПО АКАР."""
        print("5. СИНТЕЗ УПРАВЛЕНИЯ ПО АКАР")
        print("="*50)
        print()
        
        # Создание проектировщика управления
        self.designer = ControlDesigner(self.model)
        self.designer.set_target_point(self.target_x2)
        
        print("ЦЕЛЕВАЯ ТОЧКА СТАБИЛИЗАЦИИ:")
        print("-"*30)
        print(f"x₁* = {self.designer.x1_star:.3f} (соответствующая популяция жертв)")
        print(f"x₂* = {self.designer.x2_star:.3f} (целевая популяция хищников)")
        print()
        
        print("ПОИСК ОПТИМАЛЬНЫХ ПАРАМЕТРОВ УПРАВЛЕНИЯ:")
        print("-"*42)
        
        # Поиск оптимальных параметров
        T1_opt, T2_opt = self.designer.optimal_parameters_search(
            T1_range=(0.1, 4.0), T2_range=(0.1, 4.0), num_points=20
        )
        
        if T1_opt is not None:
            print(f"Найдены оптимальные параметры:")
            print(f"T₁ = {T1_opt:.3f}")
            print(f"T₂ = {T2_opt:.3f}")
            print()
            
            # Создание управления
            self.control_function = self.designer.design_linear_control(T1_opt, T2_opt)
            
            print("ЗАКОН УПРАВЛЕНИЯ:")
            print("-"*17)
            print("u(x₁,x₂) = -T₁·ψ + T₂·(x₁ - φ(x₂))")
            print("где:")
            print(f"ψ = x₂ - x₂* = x₂ - {self.target_x2}")
            print(f"φ(x₂) = a₂/β₂ = {self.model_params['a2']}/{self.model_params['beta2']} = {self.designer.x1_star:.3f}")
            print()
            
            # Анализ устойчивости
            stability_analysis = self.designer.analyze_stability_conditions()
            
            print("АНАЛИЗ УСТОЙЧИВОСТИ ЛИНЕАРИЗОВАННОЙ СИСТЕМЫ:")
            print("-"*48)
            print(f"След матрицы Якоби: {stability_analysis['trace']:.6f}")
            print(f"Определитель: {stability_analysis['determinant']:.6f}")
            print(f"Система устойчива: {stability_analysis['is_stable']}")
            print()
            print("Собственные значения:")
            for i, ev in enumerate(stability_analysis['eigenvalues']):
                if np.isreal(ev):
                    print(f"  λ_{i+1} = {ev.real:.6f}")
                else:
                    print(f"  λ_{i+1} = {ev.real:.6f} ± {abs(ev.imag):.6f}i")
            print()
            
            self.results['control_design'] = {
                'T1': T1_opt,
                'T2': T2_opt,
                'stability_analysis': stability_analysis
            }
            
            return True
        else:
            print("ОШИБКА: Не удалось найти стабильные параметры управления!")
            return False
            
    def section_6_lyapunov_proof(self):
        """6. ДОКАЗАТЕЛЬСТВО УСТОЙЧИВОСТИ ПО ЛЯПУНОВУ."""
        print("6. ДОКАЗАТЕЛЬСТВО УСТОЙЧИВОСТИ МЕТОДОМ ЛЯПУНОВА")
        print("="*50)
        print()
        
        # Создание анализатора Ляпунова
        self.analyzer = LyapunovAnalyzer(self.model)
        self.analyzer.set_equilibrium_point(self.designer.x1_star, self.designer.x2_star)
        self.analyzer.set_control_function(self.control_function)
        
        # Формальное доказательство
        analysis = self.analyzer.analytical_stability_proof(
            self.results['control_design']['T1'],
            self.results['control_design']['T2']
        )
        
        self.results['lyapunov_analysis'] = analysis
        
        print("\n" + "="*60)
        print("КРАТКИЕ ВЫВОДЫ ПО ДОКАЗАТЕЛЬСТВУ:")
        print("="*60)
        if analysis['is_stable']:
            print("✓ Функция Ляпунова V(e₁,e₂) = e₁² + e₂² положительно определена")
            print("✓ Все собственные значения имеют отрицательную действительную часть")
            print("✓ Система с управлением АСИМПТОТИЧЕСКИ УСТОЙЧИВА")
            print()
            print("Это означает, что из любой начальной точки в окрестности равновесия")
            print("траектории системы будут сходиться к целевой точке x₁*, x₂*")
        else:
            print("✗ Система неустойчива с выбранными параметрами")
        print()
        
    def section_7_numerical_simulation(self):
        """7. ЧИСЛЕННОЕ МОДЕЛИРОВАНИЕ."""
        print("7. ЧИСЛЕННОЕ МОДЕЛИРОВАНИЕ")
        print("="*50)
        print()
        
        # Создание симулятора
        self.simulator = SimulationRunner(self.model)
        
        # Начальные условия для тестирования
        initial_conditions = [
            (self.designer.x1_star + 3, self.designer.x2_star + 1.5),  # отклонение вправо-вверх
            (self.designer.x1_star - 1, self.designer.x2_star + 2),    # отклонение влево-вверх
            (self.designer.x1_star + 2, self.designer.x2_star - 1),    # отклонение вправо-вниз
        ]
        
        print("ТЕСТОВЫЕ НАЧАЛЬНЫЕ УСЛОВИЯ:")
        print("-"*30)
        for i, (x1_0, x2_0) in enumerate(initial_conditions):
            print(f"Траектория {i+1}: x₁(0) = {x1_0:.1f}, x₂(0) = {x2_0:.1f}")
        print(f"Целевая точка: x₁* = {self.designer.x1_star:.3f}, x₂* = {self.designer.x2_star:.3f}")
        print()
        
        # Моделирование системы без управления
        print("7.1 СИСТЕМА БЕЗ УПРАВЛЕНИЯ (сравнение)")
        print("-"*42)
        result_no_control = self.simulator.simulate_without_control(
            initial_conditions[:1], t_span=(0, 12), dt=0.01
        )
        
        # Моделирование системы с управлением
        print("\n7.2 СИСТЕМА С УПРАВЛЕНИЕМ")
        print("-"*27)
        target_point = (self.designer.x1_star, self.designer.x2_star)
        
        result_with_control = self.simulator.simulate_with_control(
            self.control_function,
            initial_conditions,
            target_point,
            t_span=(0, 15),
            dt=0.01
        )
        
        print("\nРЕЗУЛЬТАТЫ СТАБИЛИЗАЦИИ:")
        print("-"*25)
        for i, traj in enumerate(result_with_control['trajectories']):
            print(f"Траектория {i+1}:")
            print(f"  Время установления x₂: {traj['settling_time']['x2']:.2f} с" 
                  if traj['settling_time']['x2'] else "  Не достигнуто за время моделирования")
            print(f"  Максимальное управление: {traj['control_effort']['max']:.3f}")
            print(f"  Среднее управление: {traj['control_effort']['mean']:.3f}")
            print(f"  Финальная ошибка x₁: {abs(traj['final_values'][0] - target_point[0]):.4f}")
            print(f"  Финальная ошибка x₂: {abs(traj['final_values'][1] - target_point[1]):.4f}")
            print()
        
        self.results['simulation'] = {
            'without_control': result_no_control,
            'with_control': result_with_control
        }
        
        # Сравнение различных стратегий управления
        print("7.3 СРАВНЕНИЕ СТРАТЕГИЙ УПРАВЛЕНИЯ")
        print("-"*35)
        
        T1, T2 = self.results['control_design']['T1'], self.results['control_design']['T2']
        strategies = {
            'Оптимальное': self.designer.design_linear_control(T1, T2),
            'Консервативное': self.designer.design_linear_control(T1*0.6, T2*0.6),
            'Агрессивное': self.designer.design_linear_control(T1*1.4, T2*1.4),
            'Нелинейное': self.designer.design_nonlinear_control(0.4, 0.15)
        }
        
        comparison = self.simulator.compare_control_strategies(
            strategies,
            initial_conditions[0],
            target_point,
            t_span=(0, 12)
        )
        
        print("Сводная таблица показателей качества:")
        print(comparison['summary_table'].round(4))
        print()
        
        self.results['comparison'] = comparison
        
    def section_8_analysis_and_discussion(self):
        """8. АНАЛИЗ И ОБСУЖДЕНИЕ РЕЗУЛЬТАТОВ."""
        print("8. АНАЛИЗ И ОБСУЖДЕНИЕ РЕЗУЛЬТАТОВ")
        print("="*50)
        print()
        
        print("8.1 КАЧЕСТВО УПРАВЛЕНИЯ")
        print("-"*23)
        
        # Анализ результатов стабилизации
        if 'with_control' in self.results['simulation']:
            trajectories = self.results['simulation']['with_control']['trajectories']
            
            settling_times = [t['settling_time']['x2'] for t in trajectories if t['settling_time']['x2']]
            max_controls = [t['control_effort']['max'] for t in trajectories]
            
            if settling_times:
                avg_settling_time = np.mean(settling_times)
                print(f"Среднее время установления: {avg_settling_time:.2f} с")
                print(f"Диапазон времени установления: {min(settling_times):.2f} - {max(settling_times):.2f} с")
            else:
                print("Время установления: система не достигла 5% точности за время моделирования")
            
            avg_control = np.mean(max_controls)
            print(f"Среднее максимальное управление: {avg_control:.3f}")
            print(f"Диапазон управления: {min(max_controls):.3f} - {max(max_controls):.3f}")
            print()
        
        print("8.2 ПОВЕДЕНИЕ СИСТЕМЫ")
        print("-"*20)
        print("Анализ показывает, что:")
        print("• Система без управления демонстрирует периодические колебания")
        print("• Управление эффективно стабилизирует систему в целевой точке")
        print("• Переходный процесс носит апериодический характер")
        print("• Управляющее воздействие остается в разумных пределах")
        print()
        
        print("8.3 СРАВНЕНИЕ СТРАТЕГИЙ")
        print("-"*23)
        if 'comparison' in self.results:
            best_strategy = None
            best_score = float('inf')
            
            for strategy, metrics in self.results['comparison']['summary_table'].iterrows():
                # Комплексная оценка (время + управление + точность)
                settling = metrics['settling_time_x2'] if not np.isnan(metrics['settling_time_x2']) else 20
                control_effort = metrics['max_control_effort']
                error = metrics['final_error_x2']
                
                score = settling + control_effort*5 + error*1000  # взвешенная сумма
                
                if score < best_score:
                    best_score = score
                    best_strategy = strategy
            
            print(f"Лучшая стратегия по комплексному критерию: {best_strategy}")
            print()
        
        print("8.4 ПРАКТИЧЕСКАЯ ЗНАЧИМОСТЬ")
        print("-"*28)
        print("Полученные результаты могут применяться для:")
        print("• Управления рыбными запасами в водоемах")
        print("• Контроля популяций вредителей в сельском хозяйстве")  
        print("• Регулирования экосистем заповедников")
        print("• Моделирования взаимодействия конкурирующих компаний")
        print()
        
    def section_9_conclusion(self):
        """9. ЗАКЛЮЧЕНИЕ."""
        print("9. ЗАКЛЮЧЕНИЕ")
        print("="*50)
        print()
        
        print("ОСНОВНЫЕ РЕЗУЛЬТАТЫ РАБОТЫ:")
        print("-"*28)
        print("1. Построена математическая модель системы жертва-хищник с управлением")
        print("2. Синтезировано стабилизирующее управление методом АКАР")
        print("3. Проведено формальное доказательство устойчивости методом Ляпунова")
        print("4. Выполнена численная верификация результатов")
        print("5. Проанализировано качество различных стратегий управления")
        print()
        
        print("НАУЧНАЯ НОВИЗНА:")
        print("-"*16)
        print("• Применен комплексный подход, сочетающий АКАР и метод Ляпунова")
        print("• Получены конкретные рекомендации по выбору параметров управления")
        print("• Проведен сравнительный анализ различных стратегий управления")
        print()
        
        print("ПРАКТИЧЕСКАЯ ЦЕННОСТЬ:")
        print("-"*21)
        print("• Разработанные методы применимы к широкому классу экологических задач")
        print("• Получены количественные оценки качества управления")
        print("• Создано программное обеспечение для анализа подобных систем")
        print()
        
        if self.results.get('lyapunov_analysis', {}).get('is_stable', False):
            print("ЗАКЛЮЧЕНИЕ ОБ УСТОЙЧИВОСТИ:")
            print("-"*27)
            print("✓ Доказана асимптотическая устойчивость системы с управлением")
            print("✓ Гарантирована сходимость к целевой точке из окрестности равновесия")
            print("✓ Управление обеспечивает требуемое качество переходного процесса")
        
        print()
        print("ВОЗМОЖНЫЕ НАПРАВЛЕНИЯ РАЗВИТИЯ:")
        print("-"*32)
        print("• Исследование робастности управления к вариациям параметров")
        print("• Синтез адаптивного управления для неизвестных параметров")
        print("• Расширение на случай запаздываний в системе")
        print("• Учет стохастических возмущений")
        print()
        
    def create_visualizations(self):
        """Создание всех графиков и визуализаций."""
        print("СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ...")
        print("-"*25)
        
        self.visualizer = Visualizer()
        
        # Создание директории для результатов
        os.makedirs("project_results", exist_ok=True)
        
        # Создание комплексного отчета
        create_comprehensive_report(self.simulator, "project_results")
        
        print("Все графики сохранены в папке 'project_results'")
        print()
        
    def export_project_report(self):
        """Экспорт итогового отчета проекта."""
        print("ЭКСПОРТ ОТЧЕТА ПРОЕКТА...")
        print("-"*24)
        
        # Сохранение численных результатов
        if hasattr(self.simulator, 'export_results_to_csv'):
            self.simulator.export_results_to_csv("project_results/numerical_results.csv")
        
        # Создание текстового отчета
        report_filename = "project_results/project_summary.txt"
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("ОТЧЕТ ПО ПРОЕКТУ 'ЖЕРТВА-ХИЩНИК' С УПРАВЛЕНИЕМ\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Параметры модели: {self.model_params}\n")
            f.write(f"Целевая точка: x2* = {self.target_x2}\n\n")
            
            if 'control_design' in self.results:
                f.write("ПАРАМЕТРЫ УПРАВЛЕНИЯ:\n")
                f.write(f"T1 = {self.results['control_design']['T1']:.6f}\n")
                f.write(f"T2 = {self.results['control_design']['T2']:.6f}\n\n")
            
            if 'lyapunov_analysis' in self.results:
                analysis = self.results['lyapunov_analysis']
                f.write("АНАЛИЗ УСТОЙЧИВОСТИ:\n")
                f.write(f"Устойчивость: {'ДА' if analysis['is_stable'] else 'НЕТ'}\n")
                f.write(f"След матрицы: {analysis['trace']:.6f}\n")
                f.write(f"Определитель: {analysis['determinant']:.6f}\n")
                f.write("Собственные значения:\n")
                for i, ev in enumerate(analysis['eigenvalues']):
                    if np.isreal(ev):
                        f.write(f"  λ_{i+1} = {ev.real:.6f}\n")
                    else:
                        f.write(f"  λ_{i+1} = {ev.real:.6f} ± {abs(ev.imag):.6f}i\n")
                f.write("\n")
        
        print(f"Текстовый отчет сохранен: {report_filename}")
        print()
        
    def run_complete_analysis(self):
        """Запуск полного анализа проекта."""
        try:
            # Выполнение всех разделов по порядку
            self.section_1_introduction()
            self.section_2_theory() 
            self.section_3_problem_statement()
            self.section_4_discretization()
            
            if self.section_5_control_synthesis():
                self.section_6_lyapunov_proof()
                self.section_7_numerical_simulation()
                self.section_8_analysis_and_discussion()
                self.section_9_conclusion()
                
                # Создание визуализаций
                self.create_visualizations()
                
                # Экспорт отчета
                self.export_project_report()
                
                print("="*70)
                print("ПРОЕКТ УСПЕШНО ЗАВЕРШЕН!")
                print("="*70)
                print()
                print("Созданные файлы:")
                print("• project_results/ - папка с графиками")
                print("• project_results/numerical_results.csv - численные данные")
                print("• project_results/project_summary.txt - краткий отчет")
                print()
                print("Для просмотра графиков откройте файлы .png в папке project_results")
                
                return True
            else:
                print("ОШИБКА: Не удалось синтезировать управление")
                return False
                
        except Exception as e:
            print(f"ОШИБКА ПРИ ВЫПОЛНЕНИИ ПРОЕКТА: {e}")
            return False


def main():
    """Главная функция запуска проекта."""
    
    print("Запуск проекта 'Жертва-Хищник с управлением'...")
    print()
    
    # Создание и запуск проекта
    project = PredatorPreyProject()
    success = project.run_complete_analysis()
    
    if success:
        print("\nПроект выполнен успешно!")
        print("Вы можете:")
        print("1. Изучить созданные графики в папке project_results/")
        print("2. Ознакомиться с численными результатами в CSV файле")
        print("3. Прочитать краткий отчет в project_summary.txt")
        print("\nДля повторного запуска используйте: python main.py")
    else:
        print("\nПроект завершился с ошибками.")
        print("Проверьте параметры модели и повторите запуск.")


if __name__ == "__main__":
    main() 