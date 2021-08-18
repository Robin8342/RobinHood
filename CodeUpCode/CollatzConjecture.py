CollatzCountSaveF = 0
CollatzMaxSave = 0
CollatzMinSave = 0
CollatzCountArray= []


for i in range(2):
    CollatzInput = int(input())
    CollatzCountArray.append(CollatzInput)

CollatzCountArrayLenth = len(CollatzCountArray)

for i in range(CollatzCountArrayLenth):
    CollatzCalculate = CollatzCountArray[i]
    while True:

        if CollatzCalculate % 2 == 0:
            CollatzCalculate = (CollatzCalculate//2)
            CollatzCountSaveF += 1

        elif CollatzCalculate % 2 == 1:
            CollatzCalculate = (3 * CollatzCalculate) + 1
            CollatzCountSaveF += 1
        else:
            raise NotImplementedError

        CollatzMinSave = CollatzCalculate

        if CollatzMaxSave <= CollatzMinSave:
            CollatzMaxSave = CollatzMinSave

        elif CollatzCalculate == 1:
            CollatzCountSaveF += 1
            break
    print("CollatzLenth :",CollatzCountSaveF,"CollatMax :", CollatzMaxSave)

