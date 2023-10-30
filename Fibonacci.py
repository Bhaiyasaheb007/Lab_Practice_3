def recursive_fibo(n):
	if n<=1:
		return n
	else:
		return recursive_fibo(n-1) + recursive_fibo(n-2)
		
def nonrecursive_fibo(n):
	first = 0
	second = 1
	print(first)
	print(second)
	while n-2>0:
		third = first + second
		first = second
		second = third
		print(third)
		n-=1
	
n = int(input("Enter value of n"))
for i in range(n):
	print(recursive_fibo(i))
nonrecursive_fibo(n)


