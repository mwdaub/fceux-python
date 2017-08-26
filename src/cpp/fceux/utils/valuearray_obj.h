#ifndef _VALUEARRAY_H_
#define _VALUEARRAY_H_

namespace FCEU {

template<typename T, int N>
struct ValueArray
{
	T data[N];
	T &operator[](int index) { return data[index]; }
	static const int size = N;
	bool operator!=(ValueArray<T,N> &other) { return !operator==(other); }
	bool operator==(ValueArray<T,N> &other)
	{
		for(int i=0;i<size;i++)
			if(data[i] != other[i])
				return false;
		return true;
	}
};

} // namespace FCEU

#endif // define _VALUEARRAY_H_
