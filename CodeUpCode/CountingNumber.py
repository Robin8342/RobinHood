#어떤 숫자가 입력되면 그 숫자가 몇 자릿수 숫자인지 알아내는 프로그램

while True:
    CountingNumberInput = input()
    CountingNumberOutput = len(str(CountingNumberInput))
    print(CountingNumberOutput)