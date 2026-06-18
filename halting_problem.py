
def halts(function_to_check, input_for_function):
    pass

def paradox():
    if halts(paradox, None) == True:
        while True:
            pass 
    else:
        return
