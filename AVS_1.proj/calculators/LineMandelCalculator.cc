/**
 * @file LineMandelCalculator.cc
 * @author FULL NAME <xlogin00@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over lines
 * @date DATE
 */
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>

#include <stdlib.h>

#include "LineMandelCalculator.h"

LineMandelCalculator::LineMandelCalculator(unsigned matrixBaseSize, unsigned limit) : BaseMandelCalculator(matrixBaseSize, limit, "LineMandelCalculator")
{
	data =      (int   *)(aligned_alloc(64, height * width * sizeof(int)));
	realField = (float *)(aligned_alloc(64, width * sizeof(float)));
	imagField = (float *)(aligned_alloc(64, width * sizeof(float)));
	realValues =(float *)(aligned_alloc(64, width * sizeof(float)));
}

LineMandelCalculator::~LineMandelCalculator()
{
	free(data);
	free(realField);
	free(imagField);
	data = NULL;
	realField = NULL;
	imagField = NULL;
}

int *LineMandelCalculator::calculateMandelbrot()
{
	int *pdata = data;
	float *dataReal = realField;
	float *dataImag = imagField;
	float *real_value = realValues;

	float xStart = (float)x_start;	// To work with same data types
	float yStart = (float)y_start;
	
	for (int i = 0; i < height / 2; i++)
	{
		float imag = yStart + i * dy; // current imaginary value
		int dataShift = i * width;

		#pragma omp simd simdlen(64) aligned(pdata, dataReal, dataImag, real_value : 64)
		for (int j = 0; j < width; j++)
		{
			pdata[dataShift + j] = limit;
			dataReal[j] = xStart + j * dx; // current real value
			dataImag[j] = imag;
			real_value[j] = xStart + j * dx; // current real value
		}

		int allAboveTwo = 0;	// If all values on the row are above 2, skip this line
		for (int cnt = 0; cnt < limit && allAboveTwo < width; ++cnt)
		{
			allAboveTwo = 0;
			// Reduction on allAboveTwo, because we run this for at once (vectorized)
			#pragma omp simd simdlen(64) reduction(+:allAboveTwo) aligned(pdata, dataReal, dataImag, real_value : 64) 
			for (int j = 0; j < width; j++)
			{
				float r2 = dataReal[j] * dataReal[j];
				float i2 = dataImag[j] * dataImag[j];
				dataImag[j] = 2.0f * dataReal[j] * dataImag[j] + imag;
				dataReal[j] = r2 - i2 + real_value[j];
				// If limit not on the position, it has been changed
				bool greaterThan2 = (pdata[dataShift + j] != limit);
				allAboveTwo += greaterThan2;
				// If value was not changed and should be, update it
				pdata[dataShift + j] = (!greaterThan2 && ((r2 + i2) > 4.0f)) ? cnt : pdata[dataShift + j];
			}
		
		}
		std::memcpy(&pdata[(height - i - 1) * width], &pdata[i * width], width * (sizeof(int)));
	}
	
	return data;
}
