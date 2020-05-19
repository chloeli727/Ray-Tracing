#pragma once

#define _USE_MATH_DEFINES
#include <math.h>

#include <atlas/core/Float.hpp>
#include <atlas/math/Math.hpp>
#include <atlas/math/Random.hpp>
#include <atlas/math/Ray.hpp>

#include <fmt/printf.h>
#include <stb_image.h>
#include <stb_image_write.h>

#include <limits>
#include <memory>
#include <vector>

const double PI = 3.1415926535897932384;

using atlas::core::areEqual;

using Colour = atlas::math::Vector;

void saveToFile(std::string const& filename,
	std::size_t width,
	std::size_t height,
	std::vector<Colour> const& image);

// Declarations
class BRDF;
class Camera;
class Material;
class Light;
class Shape;
class Sampler;

struct World
{
	std::size_t width, height;
	Colour background;
	std::shared_ptr<Sampler> sampler;
	std::vector<std::shared_ptr<Shape>> scene;
	std::vector<Colour> image;
	std::vector<std::shared_ptr<Light>> lights;
	std::shared_ptr<Light> ambient;
};

struct ShadeRec
{
	Colour color;
	float t;
	atlas::math::Normal normal;
	atlas::math::Ray<atlas::math::Vector> ray;
	std::shared_ptr<Material> material;
	std::shared_ptr<World> world;
};

// Abstract classes defining the interfaces for concrete entities

class Camera
{
public:
	Camera();

	virtual ~Camera() = default;

	virtual void renderScene(std::shared_ptr<World> world) const = 0;

	void setEye(atlas::math::Point const& eye);

	void setLookAt(atlas::math::Point const& lookAt);

	void setUpVector(atlas::math::Vector const& up);

	void computeUVW();

protected:
	atlas::math::Point mEye;
	atlas::math::Point mLookAt;
	atlas::math::Point mUp;
	atlas::math::Vector mU, mV, mW;
};

class Sampler
{
public:
	Sampler(int numSamples, int numSets);
	virtual ~Sampler() = default;

	int getNumSamples() const;

	void setupShuffledIndeces();

	virtual void generateSamples() = 0;

	void map_samples_to_unit_disk();

	atlas::math::Point sampleUnitSquare();
	atlas::math::Point sampleUnitDisk();
	void mapSamplesToHemisphere(const float e);
	virtual atlas::math::Point sampleHemisphere();

protected:
	std::vector<atlas::math::Point> mSamples;
	std::vector<atlas::math::Point> mDiskSamples;
	std::vector<atlas::math::Point> hemiPoints;
	std::vector<int> mShuffledIndeces;

	int mNumSamples;
	int mNumSets;
	unsigned long mCount;
	int mJump;
};

class Shape
{
public:
	Shape();
	virtual ~Shape() = default;

	// if t computed is less than the t in sr, it and the color should be
	// updated in sr
	virtual bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
		ShadeRec& sr) const = 0;

	void setColour(Colour const& col);

	Colour getColour() const;

	void setMaterial(std::shared_ptr<Material> const& material);

	std::shared_ptr<Material> getMaterial() const;

	virtual bool shadowHit(ShadeRec& sr, float& tmin) = 0;

	void setSampler(std::shared_ptr<Sampler> samp);

protected:
	virtual bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
		float& tMin) const = 0;

	Colour mColour;
	std::shared_ptr<Material> mMaterial;
	std::shared_ptr<Sampler> sampler;
};

class BRDF
{
public:
	virtual ~BRDF() = default;

	virtual Colour fn(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector const& incoming) const = 0;
	virtual Colour rho(ShadeRec const& sr,
		atlas::math::Vector const& reflected) const = 0;
};

class Material
{
public:
	virtual ~Material() = default;

	virtual Colour shade(ShadeRec& sr) = 0;
};

class Light
{
public:
	virtual atlas::math::Vector getDirection(ShadeRec& sr) = 0;

	virtual Colour L(ShadeRec& sr);

	void scaleRadiance(float b);

	void setColour(Colour const& c);

	bool shadow(ShadeRec& sr);

	void setSampler(std::shared_ptr<Sampler> s_ptr);

	std::shared_ptr<Sampler> getSampler();
protected:
	Colour mColour;
	float mRadiance;
	std::shared_ptr<Sampler> sampler;
};

// Concrete classes which we can construct and use in our ray tracer

class Sphere : public Shape
{
public:
	Sphere(atlas::math::Point center, float radius);

	bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
		ShadeRec& sr) const;

	bool shadowHit(ShadeRec& sr, float& tmin);

	void setSampler(std::shared_ptr<Sampler> samp);

private:
	bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
		float& tMin) const;

	atlas::math::Point mCentre;
	float mRadius;
	float mRadiusSqr;
};

//Plane
class Plane : public Shape
{
public:
	Plane(atlas::math::Point p, atlas::math::Vector normal);

	bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
		ShadeRec& sr) const;

	bool shadowHit(ShadeRec& sr, float& tmin);

private:
	bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
		float& tMin) const;

	atlas::math::Point mPoint;
	atlas::math::Vector mNormal;
};

//Triangle
class Triangle : public Shape
{
public:
	Triangle(atlas::math::Point v0, atlas::math::Point v1, atlas::math::Point v2);

	bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
		ShadeRec& sr) const;

	bool shadowHit(ShadeRec& sr, float& tmin);

private:
	bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
		float& tMin) const;

	atlas::math::Point mV0;
	atlas::math::Point mV1;
	atlas::math::Point mV2;
};

class Pinhole : public Camera
{
public:
	Pinhole();

	void setDistance(float distance);
	void setZoom(float zoom);

	atlas::math::Vector rayDirection(atlas::math::Point const& p) const;
	void renderScene(std::shared_ptr<World> world) const;

private:
	float mDistance;
	float mZoom;
};

class ThinLens : public Camera
{
public:
	ThinLens();

	void setLensRadius(float lens_radius);//lens radius
	void setDistance(float d);//view plane distance
	void setFocal(float f);//focal plane distance
	void setZoom(float zoom);//zoom factor

	void set_sampler(std::shared_ptr<Sampler> sp);

	atlas::math::Vector rayDirection(atlas::math::Point const& p, atlas::math::Point const& lens_p) const;
	void renderScene(std::shared_ptr<World> world) const;

private:
	float mLensRadius;
	float mDistance;
	float mF;
	float mZoom;
	std::shared_ptr<Sampler> sampler_ptr;
};


class Regular : public Sampler
{
public:
	Regular(int numSamples, int numSets);

	void generateSamples();
};

class Random : public Sampler
{
public:
	Random(int numSamples, int numSets);

	void generateSamples();
};

class Jittered : public Sampler
{
public:
	Jittered(int numSamples, int numSets);

	void generateSamples();
};

class Lambertian : public BRDF
{
public:
	Lambertian();
	Lambertian(Colour diffuseColor, float diffuseReflection);

	Colour fn(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector const& incoming) const override;

	Colour rho(ShadeRec const& sr,
		atlas::math::Vector const& reflected) const override;

	void setDiffuseReflection(float kd);

	void setDiffuseColour(Colour const& colour);

private:
	Colour mDiffuseColour;
	float mDiffuseReflection;
};

class Matte : public Material
{
public:
	Matte();
	Matte(float kd, float ka, Colour color);

	void setDiffuseReflection(float k);

	void setAmbientReflection(float k);

	void setDiffuseColour(Colour colour);

	Colour shade(ShadeRec& sr) override;

private:
	std::shared_ptr<Lambertian> mDiffuseBRDF;
	std::shared_ptr<Lambertian> mAmbientBRDF;
};

class PointLight : public Light
{
public:
	PointLight();
	PointLight(atlas::math::Vector const& p);

	void setDirection(atlas::math::Vector const& p);

	atlas::math::Vector getDirection(ShadeRec& sr) override;

private:
	atlas::math::Vector mDirection;
	bool shadows;
};

class Directional : public Light
{
public:
	Directional();
	Directional(atlas::math::Vector const& d);

	void setDirection(atlas::math::Vector const& d);

	atlas::math::Vector getDirection(ShadeRec& sr) override;

private:
	atlas::math::Vector mDirection;
};

class Ambient : public Light
{
public:
	Ambient();

	atlas::math::Vector getDirection(ShadeRec& sr) override;

private:
	atlas::math::Vector mDirection;
};

class AmbientOccluder : public Light
{
public:
	AmbientOccluder(const Colour& mc);
	void setMinColour(Colour mc);
	atlas::math::Vector getDirection(ShadeRec& sr) override;
	virtual Colour L(ShadeRec& sr);

	atlas::math::Vector u, v, w;
	Colour minAmount;
};
