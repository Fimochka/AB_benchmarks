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

[Details](https://github.com/Fimochka/AB_benchmarks/tree/main/Research1_algorithms_benchmark#readme)

## Research #2 - stattests_benchmark (in progress)

### Цель исследования: 

сравнить статистических критериев для оценки разницы средних на синтетических данных

[Details](https://github.com/Fimochka/AB_benchmarks/tree/main/Research2_stattests_benchmark#readme)

## Research #3 - variance_reduction_methods_compare

### Цель исследования:

оценить влияние методов снижения дисперсии на ошибки для разных метрик (CTR/Global CTR/Average)

[Details](https://github.com/Fimochka/AB_benchmarks/tree/main/Research3_variance_reduction_methods_benchmark#readme)

## Research #4 - SD correction

### Цель исследования:

оценить влияние на смещенность корректировки для оценки выборочной дисперсии

[Details](https://github.com/Fimochka/AB_benchmarks/tree/main/Research4_SD_correction#readme)

## Research #6 - Multiple metrics research

### Цель исследования:

Проверить на синтетических данных идею ребят из Озона (https://habr.com/ru/companies/ozontech/articles/712306/) - пункт 6 про то как можно контролировать ошибку первого рода для множества метрик, не применяя поправку на множественное сравнение (не меняя alpha)

[Details](https://github.com/Fimochka/AB_benchmarks/tree/main/Research6_multiple_variants#readme)