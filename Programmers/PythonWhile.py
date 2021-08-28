"""
#0~4의 숫자)
LengthList = list(range(5))
print(LengthList)

#0~4의 숫자와 추가 : 5
LengthListAdd = list(range(5+1))
print(LengthListAdd)

#0, 2의 배열 (범위 0~10)
LengthInsideList = list(range(0,10,2))
print(LengthInsideList)

#Range은 반드시 정수를 입력해야 되기 때문에 INT 로 실수를 정수로 바꾸는 방법
#값은 0~4의 숫자가 나온다. 물론 여기에 범위 range(x, y+1) 해도 마지막자리는 나온다.
# / 연산자는 (float) 이다.
TotalNumber = 10
DivNumber = range(0, int(TotalNumber / 2)+1)
DivNumberList = list(DivNumber)
print(DivNumberList)

#허나 //은 나누기에서 정수값(int)만 나오기 때문에 // 해도 상관없다.
TotalNumber1 = 10
DivNumber1 = range(0, TotalNumber1 // 2)
DivNumberList1 = list(DivNumber1)
print(DivNumberList1)


#for 반복문
for i in range(5):
    print(str(i) + "=반복")  # i, "STRING" 도 된다.

#마찬가지로 range의 범위는 5시작 9까지 된다.
for i in range(5,10):
    print(i)

#range의 범위에서 마지막 숫자를 넣기 위해 +1 할 경우 0~5의 숫자가 출력된다.
for i in range(0,5+1):
    print(i)

#range의 범위 0~10에서 0과 2의 배수로 출력된다.
for i in range(0,10,2):
    print(i)

array = [20,18,14,34,74]

for i in array:
    print(i)



#array의 leng(array)는 총 길이를 뜻하며 5의 len을 가진다.
array = [20,18,14,34,74]

print(len(array))

#format은 {}안에 format(x,y)의 값을 순서대로 넣는다. 구구단이라고 생각하면 편함.
for i in range(len(array)):
    print("{}:{}".format(i,array[i]))


#역순으로 반복문 돌리는 방법. 4로 시작해서 -1하며 반복한다.
for i in range(4,-1, -1):
    print(i)

#또는 간단하게 reversed(range(number)) 해도 된다.
#그런데 reversed 함수는 의외로 문장에서 안되는 경우가 자주 생긴다.
for i in reversed(range(5)):
    print(i)
    
#무한반복이다.
while True:
    print("무한반복")


#0~10까지 반복해서 돌린다. for문하고 차이는 해당 코드에는 i+=1 이 추가되기 때문에 10까지 출력된다.
i = 0
while i < 10:
    i+= 1
    print(i)



#while에서 value의 값을 포함한 array 내의 정보를 삭제 할 수 있다.
#0~n 까지 반복이 아닌 value 처럼 특정 값을 제거 할 수 있다.
list_test = [1,2,3,2]
value = 2

while value in list_test:
    list_test.remove(value)

print(list_test)

#5초 동안의 시간으로 while 반복문을 할 수는 있으나 통신할 때 자주 사용하는 코드이다.
import time

number = 0

target_tick = time.time() + 5
while time.time() < target_tick:
    number += 1

print(number)


i = 0

#사용자가 직접 입력해서 종료할 수 있다.
while True:
    print("{}반복문".format(i))
    i = i + 1
    
    input_text = input("y or n :")
    if input_text in ["y","Y"]:
        print("Exit program")
        break

    
array = [5,15,6,20,7]

#10이하의 값은 continue로 다음 반복으로 넘어간다.
for number in array:
    if number < 10:
        continue
    print(number)
    


#파이썬 리스트에 사용할 수 있는 함수
#min(),max(),sum()
number = [111,234,512,1,612]
#print(min(number),",",max(number),",",sum(number))

#큰 순으로 정렬
numberReversed = sorted(number)
print(numberReversed)

#정렬 된 값을 큰 순이 0번째에 오게 reversed해서 뒤집어 준다.
numberReversed = reversed(numberReversed)
print(list(numberReversed))


#array을 생성 시킨 뒤 array.append로 값을 넣어서 배열을 만들 수 있다.
array = []

for i in range(0, 20, 2):
    array.append(i*i)

print(array)

#이것도 위와 같지만 한 줄로 작성할 수 있는 방법이다.
array = [i*i for i in range(0,20,2)]
print(array)

#array = [표현식 for 반복자 in 반복할 수 있는것 if 조건문] 도 할 수 있다.
"""

