#ifndef ARRAYINIT_H
#define ARRAYINIT_H

// standard headers
#include <cstdio>
#include <random>

class ArrayInit
{
public:
    virtual float operator() (const std::size_t c, const std::size_t h, const std::size_t w) = 0;
};

class ZeroArrayInit : public ArrayInit
{
public:
    float operator() (const std::size_t, const std::size_t, const std::size_t) override
    {
        return 0.0;
    }
};

class RandArrayInit : public ArrayInit
{
private:
	std::mt19937 rng;
	std::uniform_real_distribution<float> uniform;

public:
	RandArrayInit() :
		rng(std::mt19937()),
		uniform(std::uniform_real_distribution<float>())
	{
		// ...
	}

	float operator() (const std::size_t, const std::size_t, const std::size_t) override
	{
		return uniform(rng);
	}
};

class WArrayInit : public ArrayInit
{
public:
	float operator() (const std::size_t, const std::size_t, const std::size_t w) override
	{
		return float(w);
	}
};

class HArrayInit : public ArrayInit
{
public:
	float operator() (const std::size_t, const std::size_t h, const std::size_t) override
	{
		return float(h);
	}
};

class CArrayInit : public ArrayInit
{
public:
	float operator() (const std::size_t c, const std::size_t, const std::size_t) override
	{
		return float(c);
	}
};

#endif
