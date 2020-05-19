#include "assignment.hpp"
// ******* Function Member Implementation *******

// ***** Shape function members *****
Shape::Shape() : mColour{ 0, 0, 0 }
{}

void Shape::setColour(Colour const& col)
{
	mColour = col;
}

Colour Shape::getColour() const
{
	return mColour;
}

void Shape::setMaterial(std::shared_ptr<Material> const& material)
{
	mMaterial = material;
}

std::shared_ptr<Material> Shape::getMaterial() const
{
	return mMaterial;
}

void Shape::setSampler(std::shared_ptr<Sampler> samp)
{
	sampler = samp;
}

// ***** Camera function members *****
Camera::Camera() :
	mEye{ 0.0f, 0.0f, 500.0f },
	mLookAt{ 0.0f },
	mUp{ 0.0f, 1.0f, 0.0f },
	mU{ 1.0f, 0.0f, 0.0f },
	mV{ 0.0f, 1.0f, 0.0f },
	mW{ 0.0f, 0.0f, 1.0f }
{}

void Camera::setEye(atlas::math::Point const& eye)
{
	mEye = eye;
}

void Camera::setLookAt(atlas::math::Point const& lookAt)
{
	mLookAt = lookAt;
}

void Camera::setUpVector(atlas::math::Vector const& up)
{
	mUp = up;
}

void Camera::computeUVW()
{
	mW = glm::normalize(mEye - mLookAt);
	mU = glm::normalize(glm::cross(mUp, mW));
	mV = glm::cross(mW, mU);

	if (areEqual(mEye.x, mLookAt.x) && areEqual(mEye.z, mLookAt.z) &&
		mEye.y > mLookAt.y)
	{
		mU = { 0.0f, 0.0f, 1.0f };
		mV = { 1.0f, 0.0f, 0.0f };
		mW = { 0.0f, 1.0f, 0.0f };
	}

	if (areEqual(mEye.x, mLookAt.x) && areEqual(mEye.z, mLookAt.z) &&
		mEye.y < mLookAt.y)
	{
		mU = { 1.0f, 0.0f, 0.0f };
		mV = { 0.0f, 0.0f, 1.0f };
		mW = { 0.0f, -1.0f, 0.0f };
	}
}

// ***** Sampler function members *****
Sampler::Sampler(int numSamples, int numSets) :
	mNumSamples{ numSamples }, mNumSets{ numSets }, mCount{ 0 }, mJump{ 0 }
{
	mSamples.reserve(mNumSets* mNumSamples);
	setupShuffledIndeces();
}

int Sampler::getNumSamples() const
{
	return mNumSamples;
}

void Sampler::setupShuffledIndeces()
{
	mShuffledIndeces.reserve(mNumSamples * mNumSets);
	std::vector<int> indices;

	std::random_device d;
	std::mt19937 generator(d());

	for (int j = 0; j < mNumSamples; ++j)
	{
		indices.push_back(j);
	}

	for (int p = 0; p < mNumSets; ++p)
	{
		std::shuffle(indices.begin(), indices.end(), generator);

		for (int j = 0; j < mNumSamples; ++j)
		{
			mShuffledIndeces.push_back(indices[j]);
		}
	}
}

void Sampler::map_samples_to_unit_disk()
{
	size_t size{ mSamples.size() };
	float r, phi;
	atlas::math::Point sp;
	mDiskSamples.reserve(size);

	for (int j = 0; j < size; j++) {
		sp.x = float(2.0 * mSamples[j].x - 1.0);
		sp.y = float(2.0 * mSamples[j].y - 1.0);

		if (sp.x > -sp.y) {
			if (sp.x > sp.y) {
				r = sp.x;
				phi = sp.y / sp.x;
			}
			else {
				r = sp.y;
				phi = 2 - sp.x / sp.y;
			}
		}
		else {
			if (sp.x < sp.y) {
				r = -sp.x;
				phi = 4 + sp.y / sp.x;
			}
			else {
				r = -sp.y;
				if (sp.y != 0.0)
					phi = 6 - sp.x / sp.y;
				else
					phi = 0.0;
			}
		}

		phi *= float(M_PI / 4.0);
		mDiskSamples[j].x = r * cos(phi);
		mDiskSamples[j].y = r * sin(phi);
	}
	mSamples.erase(mSamples.begin(), mSamples.end());
}

void Sampler::mapSamplesToHemisphere(const float e)
{
	std::size_t size = mSamples.size();
	hemiPoints.reserve(mNumSamples * mNumSets);

	for (std::size_t j{ 0 }; j < size; j++) {
		float cosPhi = cos(2.0f * glm::pi<float>() * mSamples[j].x);
		float sinPhi = sin(2.0f * glm::pi<float>() * mSamples[j].x);
		float cosTheta = pow((1.0f - mSamples[j].y), 1.0f / (e + 1.0f));
		float sinTheta = sqrt(1.0f - cosTheta * cosTheta);
		float pu = sinTheta * cosPhi;
		float pv = sinTheta * sinPhi;
		float pw = cosTheta;

		hemiPoints.push_back(atlas::math::Point(pu, pv, pw));
	}
}

atlas::math::Point Sampler::sampleHemisphere()
{
	if (mCount % mNumSamples == 0)
	{
		atlas::math::Random<int> engine;
		mJump = (engine.getRandomMax() % mNumSets) * mNumSamples;
	}

	return hemiPoints[mJump + mShuffledIndeces[mJump + mCount % mNumSamples]];
}

atlas::math::Point Sampler::sampleUnitSquare()
{
	if (mCount % mNumSamples == 0)
	{
		atlas::math::Random<int> engine;
		mJump = (engine.getRandomMax() % mNumSets) * mNumSamples;
	}

	return mSamples[mJump + mShuffledIndeces[mJump + mCount++ % mNumSamples]];
}

atlas::math::Point Sampler::sampleUnitDisk()
{
	if (mCount % mNumSamples == 0) {
		atlas::math::Random<int> engine;
		mJump = (engine.getRandomMax() % mNumSets) * mNumSamples;
	}
	return mDiskSamples[mJump + mShuffledIndeces[mJump + mCount++ % mNumSamples]];
}

// ***** Light function members *****
Colour Light::L([[maybe_unused]] ShadeRec& sr)
{
	return mRadiance * mColour;
}

void Light::scaleRadiance(float b)
{
	mRadiance = b;
}

void Light::setColour(Colour const& c)
{
	mColour = c;
}

bool Light::shadow(ShadeRec& sr)
{
	float e = 0.001f;
	atlas::math::Point hitP = sr.ray(sr.t);
	atlas::math::Vector lightD = getDirection(sr);
	atlas::math::Vector cor = e * lightD;
	atlas::math::Point corHitP = cor + hitP;

	atlas::math::Ray<atlas::math::Vector> ray{};
	ray.d = lightD;
	ray.o = corHitP;
	ShadeRec tmp1;
	for (auto const& obj : sr.world->scene) {
		if (obj->hit(ray, tmp1))
			return true;
	}
	return false;
}

void Light::setSampler(std::shared_ptr<Sampler> s_ptr)
{
	if (sampler) {
		sampler = NULL;
	}

	while (sampler != NULL) {
		sampler = s_ptr;
		sampler->mapSamplesToHemisphere(1.0f);
	}
}

// ***** Plane function members *****
Plane::Plane(atlas::math::Point p, atlas::math::Vector normal) :
	mPoint{ p }, mNormal{ normal }
{}

bool Plane::hit(atlas::math::Ray<atlas::math::Vector> const& ray,
	ShadeRec& sr) const
{
	atlas::math::Vector tmp = mPoint - ray.o;
	float t{ std::numeric_limits<float>::max() };
	bool intersect{ intersectRay(ray, t) };

	// update ShadeRec info about new closest hit
	if (intersect && t < sr.t)
	{
		sr.normal = glm::normalize(mNormal);
		sr.ray = ray;
		sr.color = mColour;
		sr.t = t;
		sr.material = mMaterial;
	}

	return intersect;
}

bool Plane::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
	float& tMin) const
{
	const auto tmp{ mPoint - ray.o };
	const auto top{ glm::dot(tmp, mNormal) };
	const auto btm{ glm::dot(ray.d, mNormal) };
	float t{ top / btm };

	if (t > 0) {
		tMin = t;
		return true;
	}

	return false;
}

bool Plane::shadowHit(ShadeRec& sr, float& tmin)
{
	float t = glm::dot((mPoint - sr.ray.o), mNormal) / glm::dot(sr.ray.d, mNormal);

	if (t > 0.0001f) {
		tmin = t;
		return true;
	}
	else {
		return false;
	}
}

// ***** Sphere function members *****
Sphere::Sphere(atlas::math::Point center, float radius) :
	mCentre{ center }, mRadius{ radius }, mRadiusSqr{ radius * radius }
{}

bool Sphere::hit(atlas::math::Ray<atlas::math::Vector> const& ray,
	ShadeRec& sr) const
{
	atlas::math::Vector tmp = ray.o - mCentre;
	float t{ std::numeric_limits<float>::max() };
	bool intersect{ intersectRay(ray, t) };

	// update ShadeRec info about new closest hit
	if (intersect && t < sr.t)
	{
		sr.normal = (tmp + t * ray.d) / mRadius;
		sr.ray = ray;
		sr.color = mColour;
		sr.t = t;
		sr.material = mMaterial;
	}

	return intersect;
}

bool Sphere::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
	float& tMin) const
{
	const auto tmp{ ray.o - mCentre };
	const auto a{ glm::dot(ray.d, ray.d) };
	const auto b{ 2.0f * glm::dot(ray.d, tmp) };
	const auto c{ glm::dot(tmp, tmp) - mRadiusSqr };
	const auto disc{ (b * b) - (4.0f * a * c) };

	if (atlas::core::geq(disc, 0.0f))
	{
		const float kEpsilon{ 0.01f };
		const float e{ std::sqrt(disc) };
		const float denom{ 2.0f * a };

		// Look at the negative root first
		float t = (-b - e) / denom;
		if (atlas::core::geq(t, kEpsilon))
		{
			tMin = t;
			return true;
		}

		// Now the positive root
		t = (-b + e);
		if (atlas::core::geq(t, kEpsilon))
		{
			tMin = t;
			return true;
		}
	}

	return false;
}

bool Sphere::shadowHit(ShadeRec& sr, float& tmin)
{
	atlas::math::Point mNormal = glm::normalize(mCentre - mCentre);

	float t = glm::dot((mCentre - sr.ray.o), mNormal) / glm::dot(sr.ray.d, mNormal);

	if (t > 0.0001f) {
		tmin = t;
		return true;
	}
	else {
		return false;
	}
}

// ***** Triangle function members *****
Triangle::Triangle(atlas::math::Point v0, atlas::math::Point v1, atlas::math::Point v2) :
	mV0{ v0 }, mV1{ v1 }, mV2{ v2 }
{}

bool Triangle::hit(atlas::math::Ray<atlas::math::Vector> const& ray,
	ShadeRec& sr) const
{
	atlas::math::Vector v0v1 = mV1 - mV0;
	atlas::math::Vector v0v2 = mV2 - mV0;
	atlas::math::Vector pvec{ glm::cross(ray.d, v0v2) };
	atlas::math::Vector tvec = ray.o - mV0;
	atlas::math::Vector qvec{ glm::cross(tvec, v0v1) };
	float t{ std::numeric_limits<float>::max() };
	bool intersect{ intersectRay(ray, t) };

	// update ShadeRec info about new closest hit
	if (intersect && t < sr.t)
	{
		sr.normal = glm::normalize(glm::cross(v0v1, v0v2));
		sr.ray = ray;
		sr.color = mColour;
		sr.t = t;
		sr.material = mMaterial;
	}

	return intersect;
}

bool Triangle::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
	float& tMin) const
{
	const auto v0v1{ mV1 - mV0 };
	const auto v0v2{ mV2 - mV0 };
	const auto pvec{ glm::cross(ray.d, v0v2) };
	const auto tvec{ ray.o - mV0 };
	float det{ glm::dot(v0v1, pvec) };
	float invDet{ 1 / det };
	float u{ glm::dot(tvec, pvec) * invDet };

	if (u < 0 || u > 1) {
		return false;
	}

	const auto qvec{ glm::cross(tvec, v0v1) };
	float v{ glm::dot(ray.d, qvec) * invDet };
	if (v < 0 || u + v > 1) {
		return false;
	}

	float t{ glm::dot(v0v2, qvec) * invDet };
	tMin = t;
	return true;
}

bool Triangle::shadowHit(ShadeRec& sr, float& tmin)
{
	atlas::math::Vector v0v1 = mV1 - mV0;
	atlas::math::Vector v0v2 = mV2 - mV0;

	atlas::math::Point mNormal = glm::normalize(glm::cross(v0v1, v0v2));
	float t = glm::dot((mV0 - sr.ray.o), mNormal) / glm::dot(sr.ray.d, mNormal);

	if (t > 0.0001f) {
		tmin = t;
		return true;
	}
	else {
		return false;
	}
}

// ***** Pinhole function members *****
Pinhole::Pinhole() : Camera{}, mDistance{ 500.0f }, mZoom{ 1.0f }
{}

void Pinhole::setDistance(float distance)
{
	mDistance = distance;
}

void Pinhole::setZoom(float zoom)
{
	mZoom = zoom;
}

atlas::math::Vector Pinhole::rayDirection(atlas::math::Point const& p) const
{
	const auto dir = p.x * mU + p.y * mV - mDistance * mW;
	return glm::normalize(dir);
}

void Pinhole::renderScene(std::shared_ptr<World> world) const
{
	using atlas::math::Point;
	using atlas::math::Ray;
	using atlas::math::Vector;

	Point samplePoint{}, pixelPoint{};
	Ray<atlas::math::Vector> ray{ {0, 0, 0}, {0, 0, -1} };

	ray.o = mEye;
	float avg{ 1.0f / world->sampler->getNumSamples() };

	for (int r{ 0 }; r < world->height; ++r)
	{
		for (int c{ 0 }; c < world->width; ++c)
		{
			Colour pixelAverage{ 0, 0, 0 };

			for (int j = 0; j < world->sampler->getNumSamples(); ++j)
			{
				ShadeRec trace_data{};
				trace_data.world = world;
				trace_data.t = std::numeric_limits<float>::max();
				samplePoint = world->sampler->sampleUnitSquare();
				pixelPoint.x = c - 0.5f * world->width + samplePoint.x;
				pixelPoint.y = r - 0.5f * world->height + samplePoint.y;
				ray.d = rayDirection(pixelPoint);
				bool hit{};

				for (auto obj : world->scene)
				{
					hit |= obj->hit(ray, trace_data);
				}

				if (hit)
				{
					pixelAverage += trace_data.material->shade(trace_data);
				}
			}

			//out of gamut
			float max_value = 0;
			pixelAverage.r *= avg;
			pixelAverage.g *= avg;
			pixelAverage.b *= avg;
			if (pixelAverage.r > 1.0f || pixelAverage.g > 1.0f || pixelAverage.b > 1.0f) {
				if (pixelAverage.r > max_value) {
					max_value = pixelAverage.r;
				}
				if (pixelAverage.g > max_value) {
					max_value = pixelAverage.g;
				}
				if (pixelAverage.b > max_value) {
					max_value = pixelAverage.b;
				}
				pixelAverage.r /= max_value;
				pixelAverage.g /= max_value;
				pixelAverage.b /= max_value;
			}
			world->image.push_back({ pixelAverage.r ,
									pixelAverage.g ,
									pixelAverage.b });
		}
	}
}

// ***** ThinLens function members *****
ThinLens::ThinLens() : Camera{}, mLensRadius{ 1.0f }, mDistance{ 200.0f }, mF{ 300.0f }, mZoom{ 1.0f }, sampler_ptr{ std::make_shared<Jittered>(9, 83) }
{}

void ThinLens::setLensRadius(float lens_radius)
{
	mLensRadius = lens_radius;
}

void ThinLens::setDistance(float d)
{
	mDistance = d;
}

void ThinLens::setFocal(float f)
{
	mF = f;
}

void ThinLens::setZoom(float zoom)
{
	mZoom = zoom;
}

atlas::math::Vector ThinLens::rayDirection(atlas::math::Point const& pixel_point, atlas::math::Point const& lens_point) const
{
	atlas::math::Point p; //hit point on focal plane
	p.x = pixel_point.x * mF / mDistance;
	p.y = pixel_point.y * mF / mDistance;
	const auto dir = (p.x - lens_point.x) * mU + (p.y - lens_point.y) * mV - mF * mW;
	return glm::normalize(dir);
}

void ThinLens::set_sampler(std::shared_ptr<Sampler> sp)
{
	if (sampler_ptr) {
		sampler_ptr = NULL;
	}
	sampler_ptr = sp;
	sampler_ptr->map_samples_to_unit_disk();
}

void ThinLens::renderScene(std::shared_ptr<World> world) const
{
	using atlas::math::Point;
	using atlas::math::Ray;
	using atlas::math::Vector;

	Point samplePoint{}, pixelPoint{}, diskPoint{}, lensPoint{};
	Ray<atlas::math::Vector> ray{ {0, 0, 0}, {0, 0, -1} };

	ray.o = mEye;
	float avg{ 1.0f / world->sampler->getNumSamples() };

	for (int r{ 0 }; r < world->height; ++r)//up
	{
		for (int c{ 0 }; c < world->width; ++c)//cross
		{
			Colour pixelAverage{ 0, 0, 0 };

			for (int j = 0; j < world->sampler->getNumSamples(); ++j)
			{
				ShadeRec trace_data{};
				trace_data.world = world;
				trace_data.t = std::numeric_limits<float>::max();
				samplePoint = world->sampler->sampleUnitSquare();
				pixelPoint.x = c - world->width / 2.0f + samplePoint.x;
				pixelPoint.y = r - world->height / 2.0f + samplePoint.y;

				diskPoint = world->sampler->sampleUnitDisk();
				lensPoint = diskPoint * mLensRadius;

				ray.o = mEye + lensPoint.x * mU + lensPoint.y * mV;
				ray.d = rayDirection(pixelPoint, lensPoint);
				bool hit{};

				for (auto obj : world->scene)
				{
					hit |= obj->hit(ray, trace_data);
				}

				if (hit)
				{
					pixelAverage += trace_data.material->shade(trace_data);
				}
			}

			//out of gamut
			float max_value = 0;
			pixelAverage.r *= avg;
			pixelAverage.g *= avg;
			pixelAverage.b *= avg;
			if (pixelAverage.r > 1.0f || pixelAverage.g > 1.0f || pixelAverage.b > 1.0f) {
				if (pixelAverage.r > max_value) {
					max_value = pixelAverage.r;
				}
				if (pixelAverage.g > max_value) {
					max_value = pixelAverage.g;
				}
				if (pixelAverage.b > max_value) {
					max_value = pixelAverage.b;
				}
				pixelAverage.r /= max_value;
				pixelAverage.g /= max_value;
				pixelAverage.b /= max_value;
			}
			world->image.push_back({ pixelAverage.r ,
									pixelAverage.g ,
									pixelAverage.b });
		}
	}
}


// ***** Regular function members *****
Regular::Regular(int numSamples, int numSets) : Sampler{ numSamples, numSets }
{
	generateSamples();
}

void Regular::generateSamples()
{
	int n = static_cast<int>(glm::sqrt(static_cast<float>(mNumSamples)));

	for (int j = 0; j < mNumSets; ++j)
	{
		for (int p = 0; p < n; ++p)
		{
			for (int q = 0; q < n; ++q)
			{
				mSamples.push_back(
					atlas::math::Point{ (q + 0.5f) / n, (p + 0.5f) / n, 0.0f });
			}
		}
	}
}

// ***** Jittered function members *****
Jittered::Jittered(int numSamples, int numSets) : Sampler{ numSamples, numSets }
{
	generateSamples();
}

void Jittered::generateSamples()
{
	int n = static_cast<int>(glm::sqrt(static_cast<float>(mNumSamples)));
	atlas::math::Random<float> engine;

	for (int j = 0; j < mNumSets; ++j)
	{
		for (int p = 0; p < n; ++p)
		{
			for (int q = 0; q < n; ++q)
			{
				mSamples.push_back(
					atlas::math::Point{ (q + engine.getRandomOne()) / n, (p + engine.getRandomOne()) / n, 0.0f });
			}
		}
	}
}

// ***** Random function members *****
Random::Random(int numSamples, int numSets) : Sampler{ numSamples, numSets }
{
	generateSamples();
}

void Random::generateSamples()
{
	atlas::math::Random<float> engine;
	for (int p = 0; p < mNumSets; ++p)
	{
		for (int q = 0; q < mNumSamples; ++q)
		{
			mSamples.push_back(atlas::math::Point{
				engine.getRandomOne(), engine.getRandomOne(), 0.0f });
		}
	}
}

// ***** Lambertian function members *****
Lambertian::Lambertian() : mDiffuseColour{}, mDiffuseReflection{}
{}

Lambertian::Lambertian(Colour diffuseColor, float diffuseReflection) :
	mDiffuseColour{ diffuseColor }, mDiffuseReflection{ diffuseReflection }
{}

Colour
Lambertian::fn([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected,
	[[maybe_unused]] atlas::math::Vector const& incoming) const
{
	return mDiffuseColour * mDiffuseReflection * glm::one_over_pi<float>();
}

Colour
Lambertian::rho([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected) const
{
	return mDiffuseColour * mDiffuseReflection;
}

void Lambertian::setDiffuseReflection(float kd)
{
	mDiffuseReflection = kd;
}

void Lambertian::setDiffuseColour(Colour const& colour)
{
	mDiffuseColour = colour;
}

// ***** Matte function members *****
Matte::Matte() :
	Material{},
	mDiffuseBRDF{ std::make_shared<Lambertian>() },
	mAmbientBRDF{ std::make_shared<Lambertian>() }
{}

Matte::Matte(float kd, float ka, Colour color) : Matte{}
{
	setDiffuseReflection(kd);
	setAmbientReflection(ka);
	setDiffuseColour(color);
}

void Matte::setDiffuseReflection(float k)
{
	mDiffuseBRDF->setDiffuseReflection(k);
}

void Matte::setAmbientReflection(float k)
{
	mAmbientBRDF->setDiffuseReflection(k);
}

void Matte::setDiffuseColour(Colour colour)
{
	mDiffuseBRDF->setDiffuseColour(colour);
	mAmbientBRDF->setDiffuseColour(colour);
}

Colour Matte::shade(ShadeRec& sr)
{
	using atlas::math::Ray;
	using atlas::math::Vector;

	Vector wo = -sr.ray.o;
	Colour L = mAmbientBRDF->rho(sr, wo) * sr.world->ambient->L(sr);
	size_t numLights = sr.world->lights.size();

	for (size_t i{ 0 }; i < numLights; ++i)
	{
		Vector wi = sr.world->lights[i]->getDirection(sr);
		float nDotWi = glm::dot(sr.normal, wi);

		if (nDotWi > 0.0f && !sr.world->lights[i]->shadow(sr))
		{
			L += mDiffuseBRDF->fn(sr, wo, wi) * sr.world->lights[i]->L(sr) *
				nDotWi;
		}
	}

	return L;
}

// ***** Directional function members *****
Directional::Directional() : Light{}
{}

Directional::Directional(atlas::math::Vector const& d) : Light{}
{
	setDirection(d);
}

void Directional::setDirection(atlas::math::Vector const& d)
{
	mDirection = glm::normalize(d);
}

atlas::math::Vector Directional::getDirection([[maybe_unused]] ShadeRec& sr)
{
	return mDirection;
}

// ***** Pointlight function members *****
PointLight::PointLight() : Light{}
{}

PointLight::PointLight(atlas::math::Vector const& p) : Light{}
{
	setDirection(p);
}

void PointLight::setDirection(atlas::math::Vector const& p)
{
	mDirection = p;
}

atlas::math::Vector PointLight::getDirection([[maybe_unused]] ShadeRec& sr)
{
	return glm::normalize(mDirection - (sr.ray.o + sr.t * sr.ray.d));
}

// ***** Ambient function members *****
Ambient::Ambient() : Light{}
{}

atlas::math::Vector Ambient::getDirection([[maybe_unused]] ShadeRec& sr)
{
	return atlas::math::Vector{ 0.0f };
}

// ***** AmbientOccluder members *****
AmbientOccluder::AmbientOccluder(const Colour& mc) :
	minAmount{ mc }
{}

atlas::math::Vector AmbientOccluder::getDirection([[maybe_unused]] ShadeRec& sr)
{
	//atlas::math::Random<unsigned int> engine;
	//unsigned int i = rand();
	
	atlas::math::Point sp = sampler->sampleHemisphere();

	return (sp.x * u + sp.y * v + sp.z * w);
}

Colour AmbientOccluder::L(ShadeRec& sr)
{
	w = sr.normal;
	v = glm::normalize(glm::cross(w, atlas::math::Vector(0.0f, 1.0f, 0.0f)));
	u = glm::cross(v, w);

	if (shadow(sr)) {
		return (minAmount * mRadiance * mColour);
	}
	else
		return (mRadiance * mColour);
}

inline std::shared_ptr<Sampler> Light::getSampler()
{
	return sampler;
}

// ******* Driver Code *******

int main()
{
	using atlas::math::Point;
	using atlas::math::Ray;
	using atlas::math::Vector;

	std::shared_ptr<World> world{ std::make_shared<World>() };

	world->width = 1000;
	world->height = 1000;
	world->background = { 0, 0, 0 };
	world->sampler = std::make_shared<Jittered>(9, 83);

	//Sphere1
	world->scene.push_back(
		std::make_shared<Sphere>(atlas::math::Point{ 0, 0, -600 }, 128.0f));
	world->scene[0]->setMaterial(
		std::make_shared<Matte>(0.50f, 0.05f, Colour{ 0.45, 0.30, 0.36 }));
	world->scene[0]->setColour({ 0.45, 0.30, 0.36 });

	//Plane
	world->scene.push_back(
		std::make_shared<Plane>(atlas::math::Point{ 0, 0, -800 }, atlas::math::Vector{ 0, 0, 1 }));
	world->scene[1]->setMaterial(
		std::make_shared<Matte>(0.50f, 0.05f, Colour{ 0.56, 0.706, 0.7411 }));
	world->scene[1]->setColour({ 0.56, 0.706, 0.7411 });

	//Triangle
	world->scene.push_back(
		std::make_shared<Triangle>(atlas::math::Point{ 150.0, 100.0, 0.0 }, atlas::math::Point{ 100.0, 150.0, 0.0 }, atlas::math::Point{ 0.0, 0.0, 150.0 }));
	world->scene[2]->setMaterial(
		std::make_shared<Matte>(0.50f, 0.05f, Colour{ 0.67, 0.65, 0.50 }));
	world->scene[2]->setColour({ 0.67, 0.65, 0.50 });

	//Sphere2
	world->scene.push_back(
		std::make_shared<Sphere>(atlas::math::Point{ -128, 32, -700 }, 64.0f));
	world->scene[3]->setMaterial(
		std::make_shared<Matte>(0.50f, 0.05f, Colour{ 0.788, 0.478, 0.4588 }));
	world->scene[3]->setColour({ 0.788, 0.478, 0.4588 });

	world->ambient = std::make_shared<Ambient>();
	world->lights.push_back(
		std::make_shared<Directional>(Directional{ {300, 300, 300} }));
	world->lights.push_back(
		std::make_shared<PointLight>(PointLight{ {-300, -300, -500} }));

	std::shared_ptr<Sampler> amSampler = std::make_shared<Jittered>(9, 80);
	world->ambient->setColour({ 1, 1, 1 });
	world->ambient->scaleRadiance(0.05f);
	world->ambient->setSampler(amSampler);

	//directional light
	world->lights[0]->setColour({ 1, 1, 1 });
	world->lights[0]->scaleRadiance(10.0f);
	//point light
	world->lights[1]->setColour({ 1, 1, 1 });
	world->lights[1]->scaleRadiance(4.0f);

	//AmbientOccular
	world->ambient = std::make_shared<AmbientOccluder>(Colour{ 0.1f, 0.1f, 0.1f });


	//Pinhole camera
	Pinhole camera{};

	// change camera position here
	//original 150, 150, 500
	//1: decrease, camera angle turning left; increase, camera angle turning right
	//2: decrease, camera angle turning up; increase, camera angle turning down
	//3: decrease, zoom in; increase, zoom out
	camera.setEye({ 0.0f, 0.0f, 300.0f });
	camera.computeUVW();
	camera.renderScene(world);
	saveToFile("pinhole.bmp", world->width, world->height, world->image);

	/*
	//Thinlens camera
	ThinLens tlCamera{};
	tlCamera.setEye({ 0.0f, 6.0f, 10.0f });
	tlCamera.setLookAt({ 0.0f, 6.0f, 0.0f });
	tlCamera.setUpVector({ -1.0f, -3.0f, 0.0f });
	tlCamera.set_sampler(world->sampler);
	tlCamera.computeUVW();
	tlCamera.renderScene(world);
	saveToFile("thinlens.bmp", world->width, world->height, world->image);*/

	/*
	//no camera
	Point samplePoint{}, pixelPoint{};
	Ray<atlas::math::Vector> ray{ {0, 0, 0}, {0, 0, -1} };

	float avg{ 1.0f / world->sampler->getNumSamples() };

	for (int r{ 0 }; r < world->height; ++r)
	{
		for (int c{ 0 }; c < world->width; ++c)
		{
			Colour pixelAverage{ 0, 0, 0 };

			for (int j = 0; j < world->sampler->getNumSamples(); ++j)
			{
				ShadeRec trace_data{};
				trace_data.world = world;
				trace_data.t = std::numeric_limits<float>::max();
				samplePoint = world->sampler->sampleUnitSquare();
				pixelPoint.x = c - 0.5f * world->width + samplePoint.x;
				pixelPoint.y = r - 0.5f * world->height + samplePoint.y;
				ray.o = atlas::math::Vector{ pixelPoint.x, pixelPoint.y, 0 };
				bool hit{};

				for (auto obj : world->scene)
				{
					hit |= obj->hit(ray, trace_data);
				}

				if (hit)
				{
					pixelAverage += trace_data.material->shade(trace_data);
				}
			}

			//out of gamut
			float max_value = 0;
			pixelAverage.r *= avg;
			pixelAverage.g *= avg;
			pixelAverage.b *= avg;
			if (pixelAverage.r > 1.0f || pixelAverage.g > 1.0f || pixelAverage.b > 1.0f) {
				if (pixelAverage.r > max_value) {
					max_value = pixelAverage.r;
				}
				if (pixelAverage.g > max_value) {
					max_value = pixelAverage.g;
				}
				if (pixelAverage.b > max_value) {
					max_value = pixelAverage.b;
				}
				pixelAverage.r /= max_value;
				pixelAverage.g /= max_value;
				pixelAverage.b /= max_value;
			}
			world->image.push_back({ pixelAverage.r ,
									pixelAverage.g ,
									pixelAverage.b });
		}
	}

	saveToFile("raytrace_noCam.bmp", world->width, world->height, world->image);*/

	return 0;
}

void saveToFile(std::string const& filename,
	std::size_t width,
	std::size_t height,
	std::vector<Colour> const& image)
{
	std::vector<unsigned char> data(image.size() * 3);

	for (std::size_t i{ 0 }, k{ 0 }; i < image.size(); ++i, k += 3)
	{
		Colour pixel = image[i];
		data[k + 0] = static_cast<unsigned char>(pixel.r * 255);
		data[k + 1] = static_cast<unsigned char>(pixel.g * 255);
		data[k + 2] = static_cast<unsigned char>(pixel.b * 255);
	}

	stbi_write_bmp(filename.c_str(),
		static_cast<int>(width),
		static_cast<int>(height),
		3,
		data.data());
}