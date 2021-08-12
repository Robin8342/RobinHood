#1. 해(year)가 4의 배수이면서 100의 배수가 아니면 윤년.
#2. 400의 배수이면 윤년.
#결과값이 2004 -> 윤년 , 2000년 -> 윤년, 1900-> 윤년아님 이면 된다.


while True:
    LeapYearInput = input("윤년인지 확인해 보세요.")
    LeapYearInput = int(LeapYearInput)
    if (LeapYearInput % 4)==0 and (LeapYearInput % 100) !=0:
        print(LeapYearInput, "년 =====> 윤년")
        print("Yes")
    elif (LeapYearInput % 400)==0:
        print(LeapYearInput, "년 =====> 윤년")
        print("Yes")
    else:
        print(LeapYearInput, "년 =====> 윤년 아님")
        print("No")
