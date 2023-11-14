curr_x = 2
rate = 0.01
precision = 0.000001
previous_step_size = 1
max_iterations = 1000
iterations = 0
df = lambda X:2*(X+3)


while previous_step_size>precision and iterations < max_iterations :
    prev_x = curr_x
    curr_x = curr_x-rate*df(prev_x)
    previous_step_size = abs(prev_x-curr_x)
    iterations+=1

print("Local minima occurs at: ", curr_x)
