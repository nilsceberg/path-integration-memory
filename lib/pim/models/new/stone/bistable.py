
def bistable_neuron(Idown, Iup):
    state = 0
    def f(x):
        nonlocal state
        if x >= Iup:
            state = 1
        elif x <= Idown:
            state = 0
        return state
    return f