#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <fstream>
#include <ctime>
#include <string>
#include <complex>
#include <iomanip>

//#include "kiss_fft130/kiss_fft.h"
#include "kiss_fft130/kissfft.hh"
//#include "kiss_fft130/_kiss_fft_guts.h"

#define FREQUENCY_HZ 400'000'000
#define ONE_PPM_IN_HZ 400

#define NP_64K 65536

#define M_PI 3.141592653589793238462


class Nucleus
{
public:
	Nucleus() {}
	Nucleus(int chemicalShift, double phase) : chemicalShift(chemicalShift), phase(phase)
	{
		frequency = FREQUENCY_HZ + (ONE_PPM_IN_HZ * chemicalShift);
		index = count++;
	}
	~Nucleus() {}

	int chemicalShift{ 0 };
	int frequency{ 0 };
	int index{ 0 };
	double phase{ 0 };
	std::vector<double> decay;
	static int count;

private:
};
int Nucleus::count = 0;


double fid_function(double t, int frequency, double phase) {
	return sin(2 * M_PI * static_cast<double>(frequency) * t + phase) * exp(-t * 2);
}

double fid_function_cosine(double t, int frequency, double phase) {
	return cos(2 * M_PI * static_cast<double>(frequency) * t + phase) * exp(-t * 2);
}

void log(std::string logstr) {
	std::cout << "* " << logstr << std::endl;
}

int main()
{
	std::random_device rd{};
	std::mt19937 gen{rd()};
	std::uniform_real_distribution<double> unif(0, 1);
	const auto random_0_1 = [&unif, &gen]() -> double { return unif(gen); };


	// ---------------- \\
	// --- Settings --- \\
	// ---------------- \\

	const int numberOfDecayPoints = NP_64K;
	const int spectralWidthPpm = 20;
	const double acqusitionTime = static_cast<double>(numberOfDecayPoints) / (2 * spectralWidthPpm * ONE_PPM_IN_HZ);
	const double deltaTime = acqusitionTime/numberOfDecayPoints;

	const int numberOfNuclei = 1;

	const bool samePhases = true;
	const bool makeNoise = true;
	const double noiseMagnitude = 0.1;

	std::fstream outfile;
	outfile.open("D:\\out.txt");

	outfile << "--- Settings ---\n";
	outfile << "* Base frequency = " << FREQUENCY_HZ << " Hz\n";
	outfile << "* Spectral width = " << spectralWidthPpm << " ppm\n";
	outfile << "* Spectral width = " << spectralWidthPpm * ONE_PPM_IN_HZ << " Hz\n";
	outfile << "* Acqisition time = " << acqusitionTime << " s\n";
	outfile << "* Digital resolution = " << std::setprecision(4) << static_cast<double>(1)/acqusitionTime << " Hz\n";
	//outfile << "* Sample rate = " << sampleRate << " points per second\n";
	outfile << "* Number of points in FID = " << numberOfDecayPoints << "\n\n\n";


	// ----------------------------------- \\
	// --- Nuclei system configuration --- \\
	// ----------------------------------- \\

	std::vector<double> timeVector(numberOfDecayPoints);
	std::vector<Nucleus> nucleiVector(numberOfNuclei);

	log("Generate vector with time values");
	std::generate(timeVector.begin(), timeVector.end(), [i = 0, &deltaTime]() mutable {return deltaTime * i++; });

	outfile << "--- Nuclei list ---\n";
	for (auto& it : nucleiVector) {
		double _phase = samePhases ? 0.0 : random_0_1() * 3.1415926;
		it = Nucleus(std::round(random_0_1() * static_cast<double>(spectralWidthPpm) - spectralWidthPpm/2.0), _phase);
		auto freq = it.frequency;

		log("Create nucleus #" + std::to_string(it.index));

		outfile << "* Nucleus #" << it.index\
			<< ": chemical shift (ppm) = " << it.chemicalShift\
			<< ", chemical shift (Hz) = " << it.frequency-FREQUENCY_HZ <<\
			", phase = " << it.phase << std::endl;

		for (const auto& time : timeVector) {
			it.decay.push_back(fid_function(time, freq, _phase) + (makeNoise ? (random_0_1() - 0.5)*noiseMagnitude : 0));
		}
	}

	log("Sum FID for all nuclei");
	std::vector<double> sineWavesSum(numberOfDecayPoints);
	for (const auto& it : nucleiVector) {
		std::transform(sineWavesSum.cbegin(), sineWavesSum.cend(), it.decay.cbegin(), sineWavesSum.begin(), std::plus<>{});
	}
	

	// ------------------------------------- \\
	// --- Simulate quadrature detection --- \\
	// ------------------------------------- \\
	
	log("Create pure sine and cosine with base frequency");
	std::vector<double> sineWithBaseFrequency;
	for (const auto& time : timeVector) sineWithBaseFrequency.push_back(fid_function(time, FREQUENCY_HZ, 0.0));

	std::vector<double> cosineWithBaseFrequency;
	for (const auto& time : timeVector) cosineWithBaseFrequency.push_back(fid_function_cosine(time, FREQUENCY_HZ, 0.0));

	log("Simulate quadrature detection: substract pure sine and cosine from FID");
	std::vector<double> FID_real(numberOfDecayPoints);
	std::transform(sineWavesSum.cbegin(), sineWavesSum.cend(), sineWithBaseFrequency.cbegin(), FID_real.begin(), std::multiplies<>{});

	std::vector<double> FID_imag(numberOfDecayPoints);
	std::transform(sineWavesSum.cbegin(), sineWavesSum.cend(), cosineWithBaseFrequency.cbegin(), FID_imag.begin(), std::multiplies<>{});


	// ------------------ \\
	// --- FFT of FID --- \\
	// ------------------ \\

	log("Make FFT of FID");

	if (FID_real.size() != numberOfDecayPoints) {
		log("ERROR! The size of FID_real vector is not equal to numberOfDecayPoints");
		return 0;
	}if (FID_imag.size() != numberOfDecayPoints) {
		log("ERROR! The size of FID_imag vector is not equal to numberOfDecayPoints");
		return 0;
	}

	kissfft<double> fft(numberOfDecayPoints, false);
	std::vector<std::complex<double>> fidComplexData(numberOfDecayPoints);
	std::vector<std::complex<double>> fidResult(numberOfDecayPoints);

	for (int i = 0; i < numberOfDecayPoints; ++i) {
		fidComplexData[i] = std::complex<double>(FID_real[i], FID_imag[i]);
	}

	fft.transform(&fidComplexData[0], &fidResult[0]);

	double deltaOmega = static_cast<double>(1) / acqusitionTime;
	std::vector<double> frequencies(numberOfDecayPoints);
	std::generate(frequencies.begin(), frequencies.end(), [i = -numberOfDecayPoints/2, &deltaOmega]() mutable {return deltaOmega * (i++); });

	// *** Shift FFT results
	auto centerIter = fidResult.begin() + fidResult.size()/2;
	std::vector<std::complex<double>> fidResult_shifted(centerIter, fidResult.end());
	fidResult_shifted.insert(fidResult_shifted.end(), fidResult.begin(), centerIter);


	// ---------------------------- \\
	// --- Write data to a file --- \\
	// ---------------------------- \\

	log("Export data to a file");
	auto timeIter = timeVector.begin();
	auto realFidIter = FID_real.begin();
	auto imagFidIter = FID_imag.begin();
	auto spectrumIter = fidResult_shifted.begin();
	auto freqIter = frequencies.begin();

	outfile << "\n\n--- Data ---\n";
	outfile << "time\tfid_real\tfid_imag\tfrequency\tspectrum_real\tspectrum_imag\n";

	while (timeIter != timeVector.end()
		|| realFidIter != FID_real.end()
		|| imagFidIter != FID_imag.end()
		|| spectrumIter != fidResult_shifted.end()
		|| freqIter != frequencies.end())
	{
		outfile << std::setprecision(10) << *timeIter << "\t"\
				<< *realFidIter << "\t"\
				<< *imagFidIter << "\t"\
				<< *freqIter << "\t"\
				<< (*spectrumIter).real() << "\t"\
				<< (*spectrumIter).imag() << "\n";

		++timeIter;
		++realFidIter;
		++imagFidIter;
		++freqIter;
		++spectrumIter;
	}
	outfile.close();


	log("Program is finished");
}