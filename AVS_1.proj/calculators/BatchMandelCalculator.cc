/**
 * @file BatchMandelCalculator.cc
 * @author FULL NAME <xlogin00@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date DATE
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>

#include <stdlib.h>
#include <stdexcept>

#include "BatchMandelCalculator.h"

#define BlockWidth 64

BatchMandelCalculator::BatchMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "BatchMandelCalculator")
{
	data =      (int   *)(aligned_alloc(64, height * width * sizeof(int)));
	realField = (float *)(aligned_alloc(64, BlockWidth * sizeof(float)));
	imagField = (float *)(aligned_alloc(64, BlockWidth * sizeof(float)));
}

BatchMandelCalculator::~BatchMandelCalculator() {
	free(data);
	free(realField);
	free(imagField);
	data = NULL;
	realField = NULL;
	imagField = NULL;
}


int * BatchMandelCalculator::calculateMandelbrot () {
	int *pdata = data;
	float *dataReal = realField;
	float *dataImag = imagField;

	float xStart = (float)x_start;	// To work with same data types
	float yStart = (float)y_start;
	
	for (int i = 0; i < height / 2; i++)
	{
		float imag = yStart + i * dy; // current imaginary value (no. of the row)

		// Divide the row to the blocks
		int blocksOnRow = width/BlockWidth;
		for (int blkCnt = 0; blkCnt < blocksOnRow; blkCnt++)
		{
			int currBlkShift = blkCnt * BlockWidth;
			int dataShift = i * width;

			// Fill vectors with data
			#pragma omp simd simdlen(64) aligned(pdata, dataReal, dataImag: 64)
			for (int posInBlock = 0; posInBlock < BlockWidth; posInBlock++)
			{
				pdata[dataShift + currBlkShift + posInBlock] = limit;
				// We need to shift to the current block and position
				dataReal[posInBlock] = xStart + (currBlkShift + posInBlock) * dx; // current real value
				dataImag[posInBlock] = imag;
			}

			// Calculate the block
			int allAboveTwo = 0; // If all values on the row of a block are above 2, skip this line
			for (int cnt = 0; cnt < limit && allAboveTwo < BlockWidth; ++cnt)
			{
				allAboveTwo = 0;
				// Reduction on allAboveTwo, because we run this for at once (vectorized)
				// Cycle through the current block
				#pragma omp simd simdlen(64) reduction(+:allAboveTwo) aligned(pdata, dataReal, dataImag : 64)
				for (int posInBlk = currBlkShift; posInBlk < currBlkShift + BlockWidth; posInBlk++)
				{
					int posInArray = posInBlk - currBlkShift;
					float x = xStart + posInBlk * dx; // current real value
					float r2 = dataReal[posInArray] * dataReal[posInArray];
					float i2 = dataImag[posInArray] * dataImag[posInArray];
					dataImag[posInArray] = 2.0f * dataReal[posInArray] * dataImag[posInArray] + imag;
					dataReal[posInArray] = r2 - i2 + x;
					// If limit not on the position, it has been changed
					bool greaterThan2 = (pdata[dataShift + posInBlk] != limit);
					allAboveTwo += greaterThan2;
					// If value was not changed and should be, update it
					pdata[dataShift + posInBlk] = (!greaterThan2 && ((r2 + i2) > 4.0f)) ? cnt : pdata[dataShift + posInBlk];
				}
			}
		}
		
		std::memcpy(&pdata[(height - i - 1) * width], &pdata[i * width], width * (sizeof(int)));
	}
	return data;
}