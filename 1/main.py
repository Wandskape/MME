import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl

def task1():
    rng = np.random.default_rng()
    rng_array = rng.integers(0,21 , (4,5))
    print("rng_array\n" , rng_array)

    split_arrays = np.split(rng_array, 2, 0)
    print("\nSplit Arrays:")
    for part in split_arrays:
        print(part)

    nums = split_arrays[0][split_arrays[0] == 8]
    print("nums:" , nums, "\n")

    nums_count = len(nums)
    print("nums_count", nums_count, "\n")

def task2():
    rng = np.random.default_rng()


    series_array = pd.Series(np.arange(10))
    print("series_array", series_array)
    print("Выбор одного элемента")
    print(series_array[1])
    print("Выбор нескольких элементов")
    print(series_array[[2, 3]])
    print("Срез")
    print(series_array[3:])
    print("Поэлементное сложение")
    print(series_array + series_array)

    print("Добавление 2 к каждому элементу")
    print(series_array + 2)

    print("Умножение элементов на 3")
    print(series_array * 3)

    print("Сумма элементов")
    print(series_array.sum())

    print("Агрегация (sum, min, max):")
    print(series_array.agg(['sum', 'min', 'max']))

    print("Фильтрация, числа только больше 5:")
    print(series_array[series_array > 5])

    dataframe_array = pd.DataFrame(rng.integers(0,21 , (4,5)), columns=['A', 'B', 'C', 'D', 'E'])
    print("dataframe_array", dataframe_array)
    dataframe_array.drop(['B'], axis=1, inplace=True)
    print("dataframe_array", dataframe_array)
    dataframe_array.drop(index = 2, inplace=True)
    print("dataframe_array", dataframe_array)
    print("size dataframe_array", dataframe_array.size)

    nums = dataframe_array[dataframe_array == 3].values
    nums = nums[~np.isnan(nums)]
    print("nums: ", nums)

def task3():
    vids = pd.read_csv("Most popular 1000 Youtube videos.csv")
    print(vids.head())

    mpl.hist(vids["Likes"], label = "Likes and Dislikes")
    mpl.xlabel("Likes")
    mpl.ylabel("Dislikes")
    mpl.legend()
    mpl.show()

    mpl.hist(vids["Likes"], bins=50, edgecolor='black')
    mpl.xlabel("Number of Likes")
    mpl.ylabel("Frequency")
    mpl.title("Distribution of Likes")
    mpl.show()


    vids["Video views"] = vids["Video views"].str.replace(',', '').astype(float)
    vids["Likes"] = vids["Likes"].str.replace(',', '').astype(float)

    likes_mean = vids["Likes"].mean()
    likes_median = vids["Likes"].median()

    print("Mean of Likes:", likes_mean)
    print("Median of Likes:", likes_median)

    mpl.boxplot(vids["Likes"])
    mpl.xlabel("Likes")
    mpl.title("Box Plot of Likes")
    mpl.show()
    #Медиана (линия внутри ящика) показывает среднее значение.
    #Первый и третий квартили (границы ящика) показывают диапазон данных, где находится центральные 50% значений.
    #Усы показывают диапазон данных за пределами центральных 50%, исключая выбросы.
    #Выбросы (точки за пределами усов) показывают значения, значительно отличающиеся от остальных данных.

    description = vids["Likes"].describe()
    print("Description of Likes:")
    print(description)
    # count: Количество непустых значений.
    # mean: Среднее значение.
    # std: Стандартное отклонение, мера разброса значений.
    # min: Минимальное значение.
    # 25%: Первый квартиль.
    # 50%: Медиана или второй квартиль.
    # 75%: Третий квартиль.
    # max: Максимальное значение.

    category_group = vids.groupby("Category")["Likes"].mean()
    print("Average Likes by Category:")
    print(category_group)


if __name__ == '__main__':
    task3()
