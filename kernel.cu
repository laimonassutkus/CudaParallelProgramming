
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>
#include <type_traits>
#include <cassert>

#define LENGTH 10

template<class T> class DataContainer
{
public:
	DataContainer(T *_data, unsigned size);
	DataContainer(unsigned size);
	~DataContainer();

	T *getData() { return dataPtr; }
	unsigned getSize() { return dataSize; }

private:
	T *dataPtr;
	unsigned dataSize;
};

template<class T> DataContainer<T>::DataContainer(T *_data, unsigned size)
{
	dataPtr = _data;
	dataSize = size;
}

template<class T> DataContainer<T>::DataContainer(unsigned size)
{
	dataPtr = (T *)malloc(sizeof(T) * size);
	dataSize = size;
}

template<class T> DataContainer<T>::~DataContainer()
{
	free(dataPtr);
	dataPtr = nullptr;
}

template<typename T> cudaError_t addWithCuda(DataContainer<T> &data, DataContainer<T> &results);
template<typename T> T cudaMallocWrapper(T devPtr, unsigned count);
template<typename T> bool cudaMemcpyWrapper(T target, T source, unsigned count, cudaMemcpyKind cpyKind);
template<typename T> __global__ void addKernel(T dev_T, T result, unsigned size);

int *initSomeArray(unsigned size)
{
	int *a = (int *)malloc(sizeof(int) * size);
	std::fill(a, a + size, 1);
	return a;
}

int main()
{
	DataContainer<int> dataContainer(initSomeArray(LENGTH), LENGTH);
	DataContainer<int> resultContainer(LENGTH);

	cudaError_t cudaStatus = addWithCuda(dataContainer, resultContainer);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	puts("Results array:\n\n");
	for (unsigned i = 0; i < resultContainer.getSize(); i++)
	{
		printf("%d\n", resultContainer.getData()[i]);
	}

	return 0;
}

template<typename T> cudaError_t addWithCuda(DataContainer<T> &data, DataContainer<T> &results)
{
	unsigned size = data.getSize();
	T *dev_dataPtr = nullptr;
	T *dev_resultPtr = nullptr;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed.");
		goto Error;
	}

	if ((dev_dataPtr = cudaMallocWrapper(dev_dataPtr, size)) == nullptr)
		goto Error;

	if ((dev_resultPtr = cudaMallocWrapper(dev_resultPtr, size)) == nullptr)
		goto Error;

	if (!cudaMemcpyWrapper(dev_dataPtr, data.getData(), size, cudaMemcpyHostToDevice))
		goto Error;

	addKernel <<<1, size >>> (dev_dataPtr, dev_resultPtr, size);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	if (!cudaMemcpyWrapper(results.getData(), dev_resultPtr, size, cudaMemcpyDeviceToHost))
		goto Error;

Error:
	cudaFree(dev_dataPtr);
	cudaFree(dev_resultPtr);

	return cudaStatus;
}

template<typename T> T cudaMallocWrapper(T devPtr, unsigned count)
{
	cudaError_t cudaStatus = cudaMalloc((void**)&devPtr, count * sizeof(T));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return nullptr;
	}
	return devPtr;
}

template<typename T> bool cudaMemcpyWrapper(T target, T source, unsigned count, cudaMemcpyKind cpyKind)
{
	cudaError_t cudaStatus = cudaMemcpy(target, source, count * sizeof(T), cpyKind);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return false;
	}
	return true;
}

template<typename T> __global__ void addKernel(T dev_T, T result, unsigned size)
{
	int threadIndex = threadIdx.x;

	for (unsigned i = threadIndex; i < size; i++)
	{
		result[threadIndex] += dev_T[i];
	}
}

