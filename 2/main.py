import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as mpl
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Most popular 1000 Youtube videos.csv")

# Проверка на пропуски данных task1
missing_values = df.isnull().sum()
print(missing_values)

initial_row_count = df.shape[0]
print(f"Начальное количество строк: {initial_row_count}")

# for column in df.columns:
#     if df[column].dtype == np.number:
#         df[column].fillna(df[column].mean(), inplace=True)
#     else:
#         df[column].fillna('unknown', inplace=True)

# Удаление столбца с дизлайками task2
df.drop(columns=['Dislikes'], inplace=True)

# Удаление строк с пропусками в категории task2
df = df.dropna(subset=['Category'])

# Вывод итогового количества строк
final_row_count = df.shape[0]
print(f"Итоговое количество строк: {final_row_count}")

# Проверка на пропуски данных после заполнения
missing_values_after = df.isnull().sum()
print(missing_values_after)

# Выборка случайные 50 строк с просмотрами
random_indices = np.random.choice(df.index, size=50, replace=False)

# Сохранение этих строк до изменения
removed_views_rows = df.loc[random_indices].copy()

# Замена значений просмотров на NaN
df.loc[random_indices, 'Video views'] = np.nan

print("Строки с удалёнными просмотрами:")
print(removed_views_rows)

df['Video views'] = df['Video views'].replace({',': ''}, regex=True).astype(float)
df['Likes'] = df['Likes'].replace({',': ''}, regex=True).astype(float)

#task3
# Модель линейной регрессии для заполнения пропусков просмотров на основе лайков
train_data = df.dropna(subset=['Video views', 'Likes'])

# Обучение модели
X_train = train_data[['Likes']]  # Лайки
y_train = train_data['Video views']  # Просмотры

# Создание модели
model = LinearRegression()
model.fit(X_train, y_train)

# Заполнение пропусков в просмотрах
missing_data = df[df['Video views'].isna()]

# Предсказания для строк с пропусками
X_missing = missing_data[['Likes']]
predicted_views = model.predict(X_missing)

# Замена отрицательных значений на 0
predicted_views = np.maximum(predicted_views, 0)

# Заполнение просмотров
df.loc[missing_data.index, 'Video views'] = predicted_views

# Вывод значений, где была замена
filled_rows = df.loc[missing_data.index, ['rank', 'Video views']]
print("Видео с заполненными просмотрами:")
print(filled_rows)

# Проверка на пропуски данных после заполнения
missing_values_after_fill = df.isnull().sum()
print("Пропуски после заполнения:")
print(missing_values_after_fill)

# mpl.boxplot(df["Video views"])
# mpl.xlabel("Video views")
# mpl.title("Box Plot of Video views")
# mpl.show()

#task4

# Вычисление статистики для определения выбросов
df['Video views'] = df['Video views'].replace({',': ''}, regex=True).astype(float)

# Удаление строк с просмотрами 0
df = df[df['Video views'] > 0]

# Получаем топ-340 видео с самыми большими просмотрами
top_20_videos = df.nlargest(340, 'Video views')

# Удаляем эти топ-340 видео из данных
df = df[~df.index.isin(top_20_videos.index)]

# Выводим топ-340 видео, которые были удалены
print("Топ 340 видео с наибольшими просмотрами (удалены):")
print(top_20_videos[['rank', 'Video views']])

# Получаем 100 видео с наименьшими просмотрами
bottom_100_videos = df.nsmallest(100, 'Video views')

# Удаляем эти 100 видео из данных
df = df[~df.index.isin(bottom_100_videos.index)]

# Выводим 100 самых не популярных видео, которые были удалены
print("100 самых не популярных видео (удалены):")
print(bottom_100_videos[['rank', 'Video views']])

# Проверка на очень большие значения
print("Минимальное и максимальное значение просмотров до удаления выбросов:")
print(df['Video views'].min(), df['Video views'].max())

# Проверка на очень большие значения
print("Минимальное и максимальное значение просмотров до удаления выбросов:")
print(df['Video views'].min(), df['Video views'].max())

Q1 = df['Video views'].quantile(0.25)
Q3 = df['Video views'].quantile(0.75)
IQR = Q3 - Q1

# Определяем границы для выбросов
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Удаляем строки с выбросами по просмотрам
df_cleaned = df[(df['Video views'] >= lower_bound) & (df['Video views'] <= upper_bound)]

# Выводим количество строк до и после удаления выбросов
print(f"Начальное количество строк: {df.shape[0]}")
print(f"Количество строк после удаления выбросов: {df_cleaned.shape[0]}")

mpl.boxplot(df["Video views"])
mpl.xlabel("Video views")
mpl.title("Box Plot of Video views")
mpl.show()

#task5

# Удаляем столбец 'Video'
df.drop(columns=['Video'], inplace=True)

# Удаляем столбец 'rank'
df.drop(columns=['rank'], inplace=True)

# Преобразуем столбец 'Category' в числовой вид с помощью кодирования
df['Category'] = pd.Categorical(df['Category']).codes

# Выводим первые 5 строк с полным набором данных
print(df.head())

#task6

# Инициализируем скейлер
scaler = MinMaxScaler()
# Применяем нормализацию ко всем числовым столбцам
df[['Video views', 'Likes', 'Category']] = scaler.fit_transform(df[['Video views', 'Likes', 'Category']])
# Выводим первые 5 строк после нормализации
print(df.head())

# # Инициализируем стандартный скейлер
# scaler = StandardScaler()
#
# # Применяем стандартализацию ко всем числовым столбцам
# df[['Video views', 'Likes', 'Dislikes', 'Category']] = scaler.fit_transform(df[['Video views', 'Likes', 'Dislikes', 'Category']])
#
# # Выводим первые 5 строк после стандартизации
# print(df.head())

#task7
# Сохраняем текущий DataFrame в новый CSV файл

df.to_csv('normalized_dataset.csv', index=False)

