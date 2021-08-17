
while True:
    OddNumber = input()
    OddNumber = int(OddNumber)
    if (OddNumber % 2)==1:
        break
    else:
        print("Only Use OddNumber.")

MagixSquareArray = []

for i in range(OddNumber):
    MagixSquareArray.append([0]*OddNumber)

SquareWidth = 0
SquareLength = int(OddNumber//2)

MagixSquareArray[SquareWidth][SquareLength] =1


InitializationWidth = 0
InitializationLength = 0

for i in range(2,OddNumber*OddNumber+1):
    Width = SquareWidth
    Length = SquareLength
    SquareWidth-=1
    SquareLength+=1

    if(SquareWidth<0):
        SquareWidth=OddNumber-1

    if(SquareLength>OddNumber-1):
        SquareLength=0

    if(MagixSquareArray[SquareWidth][SquareLength]==0):
        MagixSquareArray[SquareWidth][SquareLength] = i

    else:
        SquareWidth=Width+1
        SquareLength=Length
        MagixSquareArray[SquareWidth][SquareLength] = i


#Print square.
for ArrayWidth in range(0,OddNumber):
    for ArrayLength in range(0,OddNumber):
        print(MagixSquareArray[ArrayWidth][ArrayLength],end=" ")
    print()