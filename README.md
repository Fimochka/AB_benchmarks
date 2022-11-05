## Research #1 - algorithms_benchmarks

### Сравниваемые подходы:
1. Статистическая проверка гипотез (fixed-sized эксперимент)
2. Sequential AB (https://www.evanmiller.org/sequential-ab-testing.html)
3. Bayesian AB


### Цель исследования: 

сравнить различные методологии проведения экспериментов по трем параметрам:

1. Ошибка первого рода
2. Ошибка второго рода
3. Требуемый объем наблюдений

### Решение: 

серия экспериментов-симуляций на синтетических данных

### Вводные:

- Количество групп (вариантов) - 2
- Количество экспериментов (на каждое значение базовой конверсии) - 1000
- Объем трафика на каждое обновление (на вариант) - 500
- Базовые значения конверсии - из диапазона (0.05;0.95) с шагом 0.01
 
#### Критерий остановки для Байеса:

1. expected_loss(winner) < toc_th (здесь, toc_th - гиперпараметр, который варьировался)
2. вероятность победителя > prob_th (prob_th также рассматривался как гиперпараметр)
Верхней границей по трафику было рассчитанное заранее значение sample size (используя формулу для расчета размера выборки). Таким образом, в самом худшем случае Байесовский эксперимент по трафику не хуже “классического” статвывода.

## Сценарий 1 - отсутствие различий в базовой конверсии

Сравнение ошибок первого рода

На графике ниже по осям:
* по оси абсцисс (X) - значение базовой конверсии (BASE_CTR)
* по оси ординат (Y) - значение ошибки первого рода (I_TYPE_ERROR)

![alt text](https://github.com/Fimochka/AB_benchmarks/blob/main/Research1_algorithms_benchmark/reports/I_type_errors.png?raw=true)
Здесь:
* bayesian_toc_0.01 - байесовский подход с пороговым значением toc==0.01
* bayesian_toc_0.005 - байесовский подход с пороговым значением toc==0.01
* bayesian_toc_1e-05 - байесовский подход с пороговым значением toc==0.00005
* classic - “классический“ подход
* sequential - sequential подход

Наблюдения:
* ошибка первого рода для “классического“ подхода примерно на одном и том же уровне для всех значений базовой конверсии
* ошибка первого рода для sequential подхода уменьшается на средних и больших конверсиях практически до 0
* для байесовского подхода ошибка падает при увеличении конверсии (при этом значение toc==0.00005 кажется более оптимальным относительно 0.01/0.005)

### Трафик 

![alt text](https://github.com/Fimochka/AB_benchmarks/blob/main/Research1_algorithms_benchmark/reports/traffic.png?raw=true)

* по оси абсцисс (X) - значение базовой конверсии (BASE_CTR)
* по оси ординат (Y) - суммарный траффик всех экспериментов

## Сценарий 2 - наличие победителя (истинная конверсия больше)

В данном сценарии один из вариантов имеет истинную базовую конверсию больше на некоторый diff (в конкретном примере ниже diff==0.07 - относительное увеличение конверсии относительно базового значения),
Таким образом, у одной группы базовая конверсия равна значению BASE_CTR, а у другой - BASE_CTR(1+diff)
Сравнение ошибок второго рода

![alt text](https://github.com/Fimochka/AB_benchmarks/blob/main/Research1_algorithms_benchmark/reports/II_type_error.png?raw=true)

Суммарного траффика:

![alt text](https://github.com/Fimochka/AB_benchmarks/blob/main/Research1_algorithms_benchmark/reports/traffic_2_sc.png?raw=true)

### Полученные выводы:

1. на конверсиях больше ~0.3, sequential подход показывает себя хуже, чем классический (причем как в случае, когда победитель есть, так и когда конверсии одинаковые)
2. в случае, когда конверсии не отличаются (нет реального победителя), Байес требует меньше трафика чем Sequential при сопоставимых ошибках первого и второго рода
3. при больших значениях базовой конверсии (BASE_CTR>~0.5) Байес сопоставим по объему требуемого трафика с классическим выводом (но не лучше)
4. малые значения toc кажутся предпочтительнее (в районе 1e-05). Есть выигрыш по трафику и приемлемые ошибки первого и второго рода


## Research #2 - stattests_benchmark (in progress)

Идея для исследования: сравнение статистических критериев для оценки разницы средних на синтетических данных


