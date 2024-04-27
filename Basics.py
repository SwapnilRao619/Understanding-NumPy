import numpy as np

# a=[1,2,3,4,5]
# print(a)
# print(type(a))
# print(a[0])
# print(a[0:4])
# print(a[-1])

# a=np.array([1,2,3,4,5])
# print(a)
# print(type(a))
# print(a[0])
# print(a[0:4])
# print(a[-1])

# a1=np.array([[1,2,3],[4,5,6],[7,8,9]])
# print(a1)
# print(type(a1))
# print(a1[1])
# print(a1[0][0])
# print(a1.shape) dimensions

# a2=np.array([[[1,2,3],[4,5,6],[7,8,9]],[[10,11,12],[13,14,15],[16,17,18]]])
# print(a2)
# print(type(a2))
# print(a2[0])
# print(a2[0][0])
# print(a2[0][0][0])
# print(a2.shape)
# print(a2.ndim)
# print(a2.size)
# print(a2.dtype) numpy written with c

# a=np.array([[1,2,3],[4,"hello",6],[7,8,9]]) 
# a=np.array([[1,2,3],[4,"hello",6],[7,8,9]], dtype=np.int32) error 
# a=np.array([[1,2,3],[4,"5",6],[7,8,9]], dtype=np.int32) int32 typecasting
# a=np.array([[1,2,3],[4,"5",6],[7,8,9]]) <u11
# print(a)
# print(type(a[0][0])) numpystr
# print(type(a))
# print(a.dtype)

# d={"a":1,"b":2}
# a=np.array([[1,2,3],[4,d,6],[7,8,9]]) 
# print(type(a))
# print(a.dtype) object 
# print(type(a[1][1])) returns normal python based dt int dict etc when object or complicated 

# a=np.array([[1,2,3],[4,5,6],[7,8,9]])
# a=np.array([[1,2,3],[4,5,6],[7,8,9]], dtype="<U7")
# a=np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=np.float32)
# print(a)
# print(a.dtype)

# a1=np.full((2,3,4),9)
# print(a1) shape,number
# a2=np.zeros((8,7,9))
# print(a2) zeros
# a3=np.empty((7,8,9))
# print(a3) garbage value

# a1=np.linspace(0,1000,4)
# a2=np.arange(0,1000,4)
# print(a1)
# print(a2)

# print(np.isnan(np.nan))
# print(np.isinf(np.inf))
# print(np.sqrt(-1))
# print(np.array([10])/0)
# print(np.nan)
# print(np.inf)

# l1=[1,2,3,4,5]
# l2=[6,7,8,9,0]
# print(l1*5)
# print(l1+5) error 
# print(l1-5) error
# print(l1/5) error
# print(l1+l2)
# a1=np.array(l1) 
# a2=np.array(l2)
# print(a1*5)
# print(a1+5)
# print(a1+a2)
# print(a1-a2)
# print(a1/5)
# print((np.array([1,2,3]))+(np.array([[1],[2]]))) gives output 

# print(np.sqrt(9))
# print(np.sqrt(np.array([1,4,9,16])))
# print(np.sin(20))
# print(np.log10(20))
# print(np.exp(10))

# a=np.array([1,2,3,4])
# b=np.array([[1,2,3],[4,5,6]])
# np.append(a,[7,8,9]) gives full appended a
# print(a) gives 1234 only
# print(np.insert(a,3,[5,6,7,8,9])) same as above
# print(np.delete(b,1)) same as above and also, 01234... indexing even though [][]
# print(np.delete(b,0,0)) deletes first []
# print(np.delete(b,0)) vs deletes 1

# a=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
# print(a.shape)
# print(a.reshape(16,1))
# print(a.reshape(1,16)) instead of this, a.flatten->non perma change, a=a required vs or a.ravel->perma change, like resize from resize vs reshape
# print(a.reshape(2,2,4))
# print(a.reshape(2,4,2))
# print(a.reshape(4,2,2))
# print(a.reshape(1,4,4)) basically product should be as much as a.size
# a=a.reshape(1,2,2,4) instead
# a.resize(1,2,2,4)
# print(a)
# nv=[v for v in a.flat]
# print(nv)
# print(a.transpose()) or print(a.T)
# print(a.swapaxes(1,0)) or print(a.swapaxes(0,1)) name as a.T 

# a=np.array([[1,2,3,4],[9,10,11,12]])
# b=np.array([[5,6,7,8],[13,14,15,16]])
# print(np.concatenate((a,b),axis=1)) axis = 0 means rowwise and 1 means columnwise
# print(np.stack((a,b))) same as concatenate rowwise except here as 2 seperate arrays stacked
# print(np.split(a,2,axis=1))

# a=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
# print(a.max())
# print(a.min())
# print(a.mean())
# print(a.median())
# print(a.sum())
# print(a.std()) standard deviation

# r=np.random.randint(100)
# r=np.random.randint(100,size=(2,3,4))
# r=np.random.randint(50,100,size=(2,3,4)) start and end
# r=np.random.binomial(10,p=0.5,size=(5,10)) binomial distribution
# r=np.random.normal(loc=170,scale=15,size=(5,10))
# r=np.random.choice([1,2,3,4,5,6,7,8,9,10])
# r=np.random.choice([1,2,3,4,5],size=(1,2))
# print(r)

# a=np.array([[1,2,3,4,5],[6,7,8,9,10]])
# np.save("test.npy",a)
# t=np.load("test.npy")
# print(t)
# np.savetxt("forcsv.csv",a,delimiter=",")
# t=np.loadtxt("forcsv.csv",delimiter=",")
# print(t)