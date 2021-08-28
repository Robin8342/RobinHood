#리스트 : list_a = [] ex) list_a[1]
#딕셔너리 : dict_a = {] ex) dict_a["name"] or for key in dict_a: dict_a[key]


dictionary = {
    "name" : "망고",
    "type" : "과일",
    "ingredient" : ["설탕","과당"],
    "origin" : "필리핀"
}

"""
#key -> name, type, ingredient, origin
for key in dictionary:
    print("=====")
    print(key)
    print("=====")

    print(key, ":", dictionary[key])
"""

key = input("접근 하고자 하는 키 ")

if key in dictionary:
    print(dictionary[key])
else:
    print("존재하지 않는 키")
