from hashlib import sha256
 
x = 5
y = 0
 
while sha256(str(x*y).encode()).hexdigest()[-1] != "0":
    y += 1
 
print('The solution is y = %d' % y)
print(sha256(str(x*y).encode()).hexdigest())