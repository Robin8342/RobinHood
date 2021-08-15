'''#1~N까지의 숫자가 차례대로 적힌 N장의 카드 묶음
그 중 임의의 한개를 잃어버렸을때
빠진 카드 번호를 찾아서 출력한다.
단 잃어버리기 전의 카드의 총 수량은 3 <= N <= 50 이다.
이어지는 N-1개의 각 줄에는 한 장이 빠진 카드 묶음의 카드 숫자가 하나씩 순서 없이 나열되어 있다.
'''

CardTotal = []
#CardSumSave = []
while True:
    CardTotalMax = int(input())
    if CardTotalMax >= 3 and CardTotalMax <=50:
        break;
    else:
        print("Wrong Range(only 3~50)")

CardFindNumber = CardTotalMax
CardDecrease = CardTotalMax-1

for i in range(CardDecrease):
   CardTotal.append(int(input()))

CardTotal.sort(reverse=True)

for i in range(CardDecrease):
    if CardTotal[i] == CardFindNumber:
        CardFindNumber -= 1
    else:
        FindLostNumber = CardFindNumber

print(FindLostNumber)


'''
#각 숫자의 0번째 N번째의 두 합은 짝수 일때 N % 2 == 0이다.
#이를 이용해 CardSumSave에 두 합의 숫자를 더해 총 합을 저장해 놓는다.
#그 후 빠진 숫자는 (일정한 숫자의 합 - 임의의 숫자) 를 할 떄 빠진 숫자를 구할 수 있다.
CardTotalCheck = CardTotalMax-1

for i in range(CardTotalMax-1):
    if CardTotalMax % 2 == 0:
        Check = CardTotal[i] + CardTotal[CardTotalCheck]
        CardSumSave.append(Check)
        CardTotalCheck -= 1

    elif CardTotalMax % 2 == 1:  
        raise NotImplementedError
    else:
        raise NotImplementedError
'''
