dense_matvec: dense_matvec.cu dense_mat.cpp
	nvcc -O3 -o dense_matvec dense_matvec.cu dense_mat.cpp

debug: dense_matvec.cu dense_mat.cpp
	nvcc -g -G -o dense_matvec dense_matvec.cu dense_mat.cpp

clean:
	rm -f dense_matvec
