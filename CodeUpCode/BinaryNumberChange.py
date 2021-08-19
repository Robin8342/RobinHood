BinaryNumberInput = int(input())

BinaryNumber = bin(BinaryNumberInput)
OctalNumber = oct(BinaryNumberInput)
DecimalNumber = hex(BinaryNumberInput)

print ("BinaryNumber =",BinaryNumber[2:])
print ("OctalNumber =",OctalNumber[2:])
print ("DecimalNumber =",DecimalNumber[2:])
