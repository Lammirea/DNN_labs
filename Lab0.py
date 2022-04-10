﻿# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 22:48:08 2022

@author: AM4
"""

# блок импорта пакетов (библиотек, модулей) сосредотачивается в начале файла,
 # но импорты работают в любом месте кода

# импорт может быть всей библиотеки
import time

# или отдельных функций (модулей, объектов и т.п.)
from random import randint

# часто библиотека импортируется под псевдонимом
import math as M


# к элементам импортированной библиотеки обращение проиходит через точку

time.time()

M.sqrt(16)

# к отдельным функциям напрямую
randint(0,10)


# значения переменным присваиваются через знак равенства

a = 5

b = randint(0,10)

t = time.time()

# допускается множественное присвоение переменных
c = d = e = 10

# существуют различные типы данных 

# целое число
a = int(5)
print(a)

# дробное число
a = float(.5)
print(a)

# строка
a = ".5"
print(a)

# приведение типов осуществляется прямым указанием типа перез значением
a = str(.5)
print(a)

# строки с обеих сторон ограничиваются одинарными или двойными кавычками
a = ".5"
print(a)

# конкатенация строк делатеся через сложение
b = " - это дробное число"
print(a+b)

# длина строки (и не только строки)
c = a + b
len(c)

# к элементам строки можно обратиться по индексу
print(c[10])

# можно делать срезы — бы получе­ние какой‑то час­ти строки, которая ограничена 
# индексами
print(c[10:15])

print(c[:15])

print(c[15:])

# в срезах можно задавать шаг
print(c[0:15:3])

# массивов по умолчанию нет, но есть списки
# к элементам списка также можно обрашаться по интексу и делать срезы
a = [7,5,0,2,3]

print(a[0])

print(a[3:])

# списки могут хранить любые значения
b = ['Пенза', 'Самара', 'Саратов', 12, 33]
print(b)

# в список можно добавить элемент
b.append(0.589)
print(b)

# или соединить с други списком
b.extend([1, 2, 3])
print(b)

# спи­сок мож­но отсорти­ровать 
a.sort()
print(a)


# циклы бывают нескольких видов

# для выполнения цикла заданное количество итераций
num = 0
for i in range(5):
    num=num + i       # вложенность задается табуляцией (4 пробела)
    print(num)
#    print(5)

for j in range(10):
    print(j)
    

# в цикле могут перебираться значения из списка
lst = [1,4,9,11,12]
for i in lst:
    print(i%2)

# цикл while
k = 10
while(k>5):
    print(k)
    k = k - 1


# оператор in может использоваться для проверки наличия значения в списке

if (5 in a):
    print('Есть')
else:
    print('Нет')


# функции обычно создаются вначале файла, после блока импортов
# объявление и описание кода функции дается одновременно
# функция может принимать или не принимать аргументы
# возвращать или не возвращать результат
# выполнение кода объявления функции до ее вызова является обязательным условием

def func1(a, b):
    c = a + b
    return c


print(func1(5, 3))

c = func1(6, 1)