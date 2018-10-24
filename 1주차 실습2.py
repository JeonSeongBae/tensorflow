def mymethod(a):
    for i in range(1, a +1):
        for j in range(a - i):
            print(".", end="")
        for k in range(i):
            print("*", end="")
        print()

    for i in range(1, a):
        for j in range(i):
            print(".", end="")
        for k in range(a- i):
            print("*", end="")
        print()

print("mymethod3")
mymethod(3)
print("mymethod4")
mymethod(4)
