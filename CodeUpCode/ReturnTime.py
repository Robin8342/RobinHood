#입력 된 시간에서 30분 전으로 되돌리는 프로그램 코드
#

while True:
    ReturnTimeInputHour = input()
    ReturnTimeInputMinute = input()

    ReturnTimeInputHour = int(ReturnTimeInputHour)
    ReturnTimeInputMinute = int(ReturnTimeInputMinute)

    if ReturnTimeInputHour >= 0 and ReturnTimeInputHour < 24 and ReturnTimeInputMinute < 60 and ReturnTimeInputMinute >=0:
        if ReturnTimeInputMinute < 30:
            ReturnTimeInputHour -= 1

            if ReturnTimeInputHour < 0:
                ReturnTimeInputHour = 23

            ReturnTimeInputMinute = ReturnTimeInputMinute+30
            print(ReturnTimeInputHour,"",ReturnTimeInputMinute)

        else:
            ReturnTimeInputMinute -= 30
            print(ReturnTimeInputHour,"",ReturnTimeInputMinute)

    elif ReturnTimeInputHour <= -1 or ReturnTimeInputMinute <= -1:
        raise NotImplementedError
        break

    else:
        raise NotImplementedError
