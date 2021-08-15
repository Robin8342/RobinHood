#5개의 정수들의 최댓값과 최솟값을 구하는 프로그램
#정수는 한 줄에 하나씩 입력
#단 출력값은 첫째쭐에 최댓값 둘째줄에 최솟값을 출력한다.

ListTotal=[]
n = 5

for i in range(n):
    ListTotal.append(int(input()))


ListTotal.sort()

print(ListTotal[4])
print(ListTotal[0])





