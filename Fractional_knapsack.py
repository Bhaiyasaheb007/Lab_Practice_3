def fractional_knapsack():
	weights = []
	profits = []
	n = int(input("Enter number of items..."))
	for i in range(1,n+1):
		print("Enter element :", i)
		w = int(input("Enter weight :"))
		p = int(input("Enter profit :"))
		weights.append(w)
		profits.append(p)
	capacity = int(input("Enter capacity of knapsack..."))
	res = 0
	
	for pair in sorted(zip(weights, profits), key = lambda x: x[1]/x[0], reverse = True):
		if capacity <=0:
			break
		if pair[0] > capacity:
			res += int(capacity * (pair[1]/pair[0]))
			capacity = 0
		elif pair[0] <= capacity:
			res += pair[1]
			capacity -= pair[0]
	print("Maximum profit :",res)
	
if __name__=="__main__":
	fractional_knapsack()
