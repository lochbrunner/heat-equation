#define Vector Vector3

#include <vector>

const float PI = 3.141592654f;

inline float degToRad(float deg) {return deg / 180.0f * PI; }
inline float radToDeg(float deg) {return deg * 180.0f / PI; }

template<typename T>
T min(T a, T b)
{
	return a < b ? a : b;
}

template<typename T>
T max(T a, T b)
{
	return a > b ? a : b;
}

template<typename T>
T clamp(T a, T minimum, T maximum)
{
	return max(minimum, min(a, maximum));
}


template<typename T>
class Vector2{

public:
	Vector2() : x(), y() {}
	Vector2(T &xx, T &yy) : x(xx), y(yy) {}


	T x,y;
};

struct Vector3
{
    float x, y, z;
    Vector3(float x, float y, float z)
    {
        this->x = x;
        this->y = y;
        this->z = z;
    }

	Vector3()
	{
		this->x = 0.0f;
		this->y = 0.0f;
		this->z = 0.0f;
	}

	float& operator[](int index)
	{
		switch(index)
		{
			case 0:
				return x;
			case 1:
				return y;
			case 2:
				return z;
		}
		return x;
	}

	Vector3 operator+(Vector3 v)
	{ 
		Vector3 result;
		result.x=this->x+v.x;
		result.y=this->y+v.y;
		result.z=this->z+v.z;
		return result;
	}

	Vector3 operator+=(Vector3 v)
	{ 
		this->x+=v.x;
		this->y+=v.y;
		this->z+=v.z;
		return *this;
	}

	Vector3 operator-(Vector3 v)
	{ 
		Vector3 result;
		result.x=this->x-v.x;
		result.y=this->y-v.y;
		result.z=this->z-v.z;
		return result;
	}

	Vector3 operator-=(Vector3 v)
	{ 
		this->x-=v.x;
		this->y-=v.y;
		this->z-=v.z;
		return *this;
	}

	Vector3 operator*(Vector3 v)
	{ 
		Vector3 result;
		result.x=this->x*v.x;
		result.y=this->y*v.y;
		result.z=this->z*v.z;
		return result;
	}

	Vector3 operator*(float scale)
	{ 
		Vector3 result;
		result.x=this->x*scale;
		result.y=this->y*scale;
		result.z=this->z*scale;
		return result;
	}

	Vector3 operator*=(float scale)
	{ 
		this->x *= scale;
		this->y *= scale;
		this->z *= scale;
		return *this;
	}

	Vector3 normalize(void)
	{
		float OneBylenght = 1.0f / sqrt(x*x+y*y+z*z);
		x *= OneBylenght;
		y *= OneBylenght;
		z *= OneBylenght;
		return *this;
	}

	float length(void)
	{
		return sqrt(x*x+y*y+z*z);
	}

	float sqLength(void)
	{
		return x*x+y*y+z*z;
	}

	static Vector3 CrossProduct(Vector3 &v1, Vector3 &v2)
	{
		Vector3 v;
		v.x = (v1.y * v2.z) - (v1.z * v2.y);
		v.y = (v1.z * v2.x) - (v1.x * v2.z);
		v.z = (v1.x * v2.y) - (v1.y * v2.x);

		return v;
	}

	static void CrossProduct(Vector3 &v1, Vector3 &v2, Vector3 &result)
	{
		Vector3 v;
		v.x = (v1.y * v2.z) - (v1.z * v2.y);
		v.y = (v1.z * v2.x) - (v1.x * v2.z);
		v.z = (v1.x * v2.y) - (v1.y * v2.x);

		result = v;
	}

	static float ScalarProduct(Vector3 &v1, Vector3 &v2)
	{
		return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
	}

	static void ScalarProduct(Vector3 &v1, Vector3 &v2, float &result)
	{
		result = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
	}

	static Vector3 RotX(Vector3 vec, float degreesOfAngle)
	{
		return  Vector3(vec.x, vec.y*cos(degToRad(degreesOfAngle)) - vec.z*sin(degToRad(degreesOfAngle)), vec.y*sin(degToRad(degreesOfAngle)) + vec.z*cos(degToRad(degreesOfAngle)));
	}

	static Vector3 RotY(Vector3 vec, float degreesOfAngle)
	{
		return Vector3(vec.x*cos(degToRad(degreesOfAngle)) + vec.z*sin(degToRad(degreesOfAngle)), vec.y, -vec.x*sin(degToRad(degreesOfAngle)) + vec.z*cos(degToRad(degreesOfAngle)));
	}

	static Vector3 RotZ(Vector3 vec, float degreesOfAngle)
	{
		return Vector3(vec.x*cos(degToRad(degreesOfAngle)) + vec.y*sin(degToRad(degreesOfAngle)), -vec.x*sin(degToRad(degreesOfAngle)) + vec.y*cos(degToRad(degreesOfAngle)), vec.z);
	}
};

struct Vector4
{
	float x,y,z,w;
	Vector4(float x, float y, float z, float w)
    {
        this->x = x;
        this->y = y;
        this->z = z;
        this->w = w;
    }

	Vector4()
    {
        this->x = 0.0f;
        this->y = 0.0f;
        this->z = 0.0f;
        this->w = 0.0f;
    }

	Vector4(Vector3 vec3, float w = 1.0f)
    {
		this->x = vec3.x;
        this->y = vec3.y;
        this->z = vec3.z;
        this->w = w;
    }

	static float ScalarProduct(Vector4 &v1, Vector4 &v2)
	{
		return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
	}
};


struct Color
{
    float r, g, b;
    Color(float r, float g, float b)
    {
        this->r = r;
        this->g = g;
        this->b = b;
    }
};



#define Mat4x4f Mat4x4<float>
#define Mat4x4d Mat4x4<double>

template<typename T>
struct Mat4x4{

	Mat4x4()
	{
		value[0] = 1.0f;	value[4] = 0.0f;	value[8] = 0.0f;	value[12] = 0.0f;
		value[1] = 0.0f;	value[5] = 1.0f;	value[9] = 0.0f;	value[13] = 0.0f;
		value[2] = 0.0f;	value[6] = 0.0f;	value[10] = 1.0f;	value[14] = 0.0f;
		value[3] = 0.0f;	value[7] = 0.0f;	value[11] = 0.0f;	value[15] = 1.0f;
	}

	struct ImpData{
		//	0	3	6	9
		//	1	4	7	10
		//	2	5	8	11	
		float value[12];
		Vector3 GetPosition()
		{
			return Vector3(value[9], value[10], value[11]);
		}
	};

	Mat4x4(ImpData &Data)
	{
		value[0] = Data.value[0];	value[4] = Data.value[3];	value[8] = Data.value[6];	value[12] = Data.value[9];
		value[1] = Data.value[1];	value[5] = Data.value[4];	value[9] = Data.value[7];	value[13] = Data.value[10];
		value[2] = Data.value[2];	value[6] = Data.value[5];	value[10] = Data.value[8];	value[14] = Data.value[11];
		value[3] = 0.0f;			value[7] = 0.0f;			value[11] = 0.0f;			value[15] = 1.0f;
	}

	ImpData GetImpData(){
		ImpData result;
		result.value[0] = value[0];	result.value[3] = value[4];	result.value[6] = value[8];	result.value[9] = value[12];
		result.value[1] = value[1];	result.value[4] = value[5];	result.value[7] = value[9];	result.value[10] = value[13];
		result.value[2] = value[2];	result.value[5] = value[6];	result.value[8] = value[10];result.value[11] = value[14];
		return result;
	}

	T value[16];

	Mat4x4 operator*(Mat4x4 &second)
	{
		Mat4x4 result;
		// Mache aus der Eins-Matrix eine Null-Matrix
		result.value[0] = 0.0f;	result.value[5] = 0.0f;	result.value[10] = 0.0f;	result.value[15] = 0.0f;
		for(int i = 0; i < 4; i++)
		{
			for(int j = 0; j < 4; j++)
			{
				for(int k = 0; k < 4; k++)
				{
					result.value[i*4+j] += second.value[i*4+k] * this->value[k*4+j];
				}
			}
		}
		return result;
	}
	Mat4x4 operator*=(Mat4x4 &second)
	{
		Mat4x4 result;
		// Mache aus der Eins-Matrix eine Null-Matrix
		result.value[0] = 0.0f;	result.value[5] = 0.0f;	result.value[10] = 0.0f;	result.value[15] = 0.0f;

		for(int i = 0; i < 4; i++)
		{
			for(int j = 0; j < 4; j++)
			{
				for(int k = 0; k < 4; k++)
				{
					result.value[i*4+j] += second.value[i*4+k] * this->value[k*4+j];
				}
			}
		}
		//memcpy(this->value, result.value, sizeof(Mat4x4::value));
		for(int i = 0; i < 16; ++i) this->value[i] = result.value[i];
		return *this;
	}

	Mat4x4 Rotate(float angle, float x, float y,  float z)
	{
		if(angle == 0.0f) return *this;

		// Siehe http://wiki.delphigl.com/index.php/glRotate
		float rad = degToRad(angle);
		float s = sin(rad);
		float c = cos(rad);
		float lenght = sqrt(x*x+y*y+z*z);
		if(lenght == 0.0f) return *this;
		float OneBylenght = 1.0f / lenght;

		x *= OneBylenght;
		y *= OneBylenght;
		z *= OneBylenght;
		
		Mat4x4 rotMat;

		rotMat.value[0] = x*x*(1.0f-c)+c;
		rotMat.value[1] = y*x*(1.0f-c)+z*s;
		rotMat.value[2] = z*x*(1.0f-c)-y*s;
		rotMat.value[3] = 0.0f;

		rotMat.value[4] = x*y*(1.0f-c)-z*s;
		rotMat.value[5] = y*y*(1.0f-c)+c;
		rotMat.value[6] = z*y*(1.0f-c)+x*s;
		rotMat.value[7] = 0.0f;

		rotMat.value[8] = x*z*(1.0f-c)+y*s;
		rotMat.value[9] = y*z*(1.0f-c)-x*s;
		rotMat.value[10] = z*z*(1.0f-c)+c;
		rotMat.value[11] = 0.0f;

		rotMat.value[12] = 0.0f;
		rotMat.value[13] = 0.0f;
		rotMat.value[14] = 0.0f;
		rotMat.value[15] = 1.0f;

		return ((*this) *= rotMat);
	}

	template<typename T>
	void swap(T &a, T &b)
	{
		T c(a);
		a = b;
		b = c;
	}

	Mat4x4 Transpose(void)
	{
		for(int i = 0; i < 4; ++i)
		{
			for(int j = 0; j < i; ++j)
			{
				swap(value[i*4+j], value[j*4+i]);
			}
		}
		return *this;
	}

	Vector3 GetPosition()
	{
		return Vector3(this->value[12], this->value[13], this->value[14]);
	}

	Mat4x4 Translate(float x, float y, float z)
	{
		Mat4x4 transMat;
		transMat.value[0] = 1.0f;
		transMat.value[1] = 0.0f;
		transMat.value[2] = 0.0f;
		transMat.value[3] = 0.0f;

		transMat.value[4] = 0.0f;
		transMat.value[5] = 1.0f;
		transMat.value[6] = 0.0f;
		transMat.value[7] = 0.0f;

		transMat.value[8] = 0.0f;
		transMat.value[9] = 0.0f;
		transMat.value[10] = 1.0f;
		transMat.value[11] = 0.0f;

		transMat.value[12] = x;
		transMat.value[13] = y;
		transMat.value[14] = z;
		transMat.value[15] = 1.0f;

		return ((*this) *= transMat);
	}

	Mat4x4 Translate(Vector3 &vec)
	{
		Mat4x4 transMat;
		transMat.value[0] = 1.0f;
		transMat.value[1] = 0.0f;
		transMat.value[2] = 0.0f;
		transMat.value[3] = 0.0f;

		transMat.value[4] = 0.0f;
		transMat.value[5] = 1.0f;
		transMat.value[6] = 0.0f;
		transMat.value[7] = 0.0f;

		transMat.value[8] = 0.0f;
		transMat.value[9] = 0.0f;
		transMat.value[10] = 1.0f;
		transMat.value[11] = 0.0f;

		transMat.value[12] = vec.x;
		transMat.value[13] = vec.y;
		transMat.value[14] = vec.z;
		transMat.value[15] = 1.0f;

		return ((*this) *= transMat);
	}

	static Mat4x4 Translation(const Vector3 &vec)
	{
		Mat4x4 transMat;
		transMat.value[0] = 1.0f;
		transMat.value[1] = 0.0f;
		transMat.value[2] = 0.0f;
		transMat.value[3] = 0.0f;

		transMat.value[4] = 0.0f;
		transMat.value[5] = 1.0f;
		transMat.value[6] = 0.0f;
		transMat.value[7] = 0.0f;

		transMat.value[8] = 0.0f;
		transMat.value[9] = 0.0f;
		transMat.value[10] = 1.0f;
		transMat.value[11] = 0.0f;

		transMat.value[12] = vec.x;
		transMat.value[13] = vec.y;
		transMat.value[14] = vec.z;
		transMat.value[15] = 1.0f;

		return transMat;
	}

	static Mat4x4 Rotation(float angle, Vector3 &axis)
	{
		if(angle == 0.0f) return Mat4x4::Identity();

		// Siehe http://wiki.delphigl.com/index.php/glRotate
		float rad = degToRad(angle);
		float s = sin(rad);
		float c = cos(rad);
		float lenght = sqrt(axis.x*axis.x+axis.y*axis.y+axis.z*axis.z);
		if(lenght == 0.0f) return Mat4x4::Identity();
		float OneBylenght = 1.0f / lenght;

		float x = axis.x;
		float y = axis.y;
		float z = axis.z;

		x *= OneBylenght;
		y *= OneBylenght;
		z *= OneBylenght;
		
		Mat4x4 rotMat;

		rotMat.value[0] = x*x*(1.0f-c)+c;
		rotMat.value[1] = y*x*(1.0f-c)+z*s;
		rotMat.value[2] = z*x*(1.0f-c)-y*s;
		rotMat.value[3] = 0.0f;

		rotMat.value[4] = x*y*(1.0f-c)-z*s;
		rotMat.value[5] = y*y*(1.0f-c)+c;
		rotMat.value[6] = z*y*(1.0f-c)+x*s;
		rotMat.value[7] = 0.0f;

		rotMat.value[8] = x*z*(1.0f-c)+y*s;
		rotMat.value[9] = y*z*(1.0f-c)-x*s;
		rotMat.value[10] = z*z*(1.0f-c)+c;
		rotMat.value[11] = 0.0f;

		rotMat.value[12] = 0.0f;
		rotMat.value[13] = 0.0f;
		rotMat.value[14] = 0.0f;
		rotMat.value[15] = 1.0f;
		return rotMat;
	}

	static	Mat4x4 Identity()
	{
		Mat4x4 IdentityMat;
		IdentityMat.value[0] = 1.0f;	IdentityMat.value[4] = 0.0f;	IdentityMat.value[8] = 0.0f;	IdentityMat.value[12] = 0.0f;
		IdentityMat.value[1] = 0.0f;	IdentityMat.value[5] = 1.0f;	IdentityMat.value[9] = 0.0f;	IdentityMat.value[13] = 0.0f;
		IdentityMat.value[2] = 0.0f;	IdentityMat.value[6] = 0.0f;	IdentityMat.value[10] = 1.0f;	IdentityMat.value[14] = 0.0f;
		IdentityMat.value[3] = 0.0f;	IdentityMat.value[7] = 0.0f;	IdentityMat.value[11] = 0.0f;	IdentityMat.value[15] = 1.0f;
		return IdentityMat;
	}

	Mat4x4 Scale(float x, float y, float z)
	{
		Mat4x4 scaleMat;
		scaleMat.value[0] = x;
		scaleMat.value[1] = 0.0f;
		scaleMat.value[2] = 0.0f;
		scaleMat.value[3] = 0.0f;

		scaleMat.value[4] = 0.0f;
		scaleMat.value[5] = y;
		scaleMat.value[6] = 0.0f;
		scaleMat.value[7] = 0.0f;

		scaleMat.value[8] = 0.0f;
		scaleMat.value[9] = 0.0f;
		scaleMat.value[10] = z;
		scaleMat.value[11] = 0.0f;

		scaleMat.value[12] = 0.0f;
		scaleMat.value[13] = 0.0f;
		scaleMat.value[14] = 0.0f;
		scaleMat.value[15] = 1.0f;

		return ((*this) *= scaleMat);
	}

	Mat4x4 Scale(Vector3 &vec)
	{
		Mat4x4 scaleMat;
		scaleMat.value[0] = vec.x;
		scaleMat.value[1] = 0.0f;
		scaleMat.value[2] = 0.0f;
		scaleMat.value[3] = 0.0f;

		scaleMat.value[4] = 0.0f;
		scaleMat.value[5] = vec.y;
		scaleMat.value[6] = 0.0f;
		scaleMat.value[7] = 0.0f;

		scaleMat.value[8] = 0.0f;
		scaleMat.value[9] = 0.0f;
		scaleMat.value[10] = vec.z;
		scaleMat.value[11] = 0.0f;

		scaleMat.value[12] = 0.0f;
		scaleMat.value[13] = 0.0f;
		scaleMat.value[14] = 0.0f;
		scaleMat.value[15] = 1.0f;

		return ((*this) *= scaleMat);
	}
};


//	0	3	6
//	1	4	7
//	2	5	8

struct Mat3x3
{
	Mat3x3()
	{
		value[0] = 1.0f;	value[3] = 0.0f;	value[6] = 0.0f;
		value[1] = 0.0f;	value[4] = 1.0f;	value[7] = 0.0f;
		value[2] = 0.0f;	value[5] = 0.0f;	value[8] = 1.0f;
	}

	Mat3x3 operator*(Mat3x3 &second)
	{
		Mat3x3 result;
		// Mache aus der Eins-Matrix eine Null-Matrix
		result.value[0] = 0.0f;	result.value[4] = 0.0f;	result.value[8] = 0.0f;
		for(int i = 0; i < 3; i++)
		{
			for(int j = 0; j < 3; j++)
			{
				for(int k = 0; k < 3; k++)
				{
					result.value[i*3+j] += second.value[i*3+k] * this->value[k*3+j];
				}
			}
		}
		return result;
	}
	Mat3x3 operator*=(Mat3x3 &second)
	{
		Mat3x3 result;
		// Mache aus der Eins-Matrix eine Null-Matrix
		result.value[0] = 0.0f;	result.value[4] = 0.0f;	result.value[8] = 0.0f;
		for(int i = 0; i < 3; i++)
		{
			for(int j = 0; j < 3; j++)
			{
				for(int k = 0; k < 3; k++)
				{
					result.value[i*3+j] += second.value[i*3+k] * this->value[k*3+j];
				}
			}
		}
		//memcpy(this->value, result.value, sizeof(Mat3x3::value));
		for(int i = 0; i < 9; ++i) this->value[i] = result.value[i];
		return *this;
	}
	Mat3x3 Scale(float x, float y)
	{
		Mat3x3 scaleMat;
		scaleMat.value[0] = x;
		scaleMat.value[1] = 0.0f;
		scaleMat.value[2] = 0.0f;

		scaleMat.value[3] = 0.0f;
		scaleMat.value[4] = y;
		scaleMat.value[5] = 0.0f;

		scaleMat.value[6] = 0.0f;
		scaleMat.value[7] = 0.0f;
		scaleMat.value[8] = 1.0f;
		
		return ((*this) *= scaleMat);
	}

	Mat3x3 Translate(float x, float y)
	{
		Mat3x3 scaleMat;
		scaleMat.value[0] = 1.0f;
		scaleMat.value[1] = 0.0f;
		scaleMat.value[2] = 0.0f;

		scaleMat.value[3] = 0.0f;
		scaleMat.value[4] = 1.0f;
		scaleMat.value[5] = 0.0f;

		scaleMat.value[6] = x;
		scaleMat.value[7] = y;
		scaleMat.value[8] = 1.0f;
		
		return ((*this) *= scaleMat);
	}

	Mat3x3 Rotate(float angle)	// Noch nicht getestet
	{
		float rad = degToRad(angle);
		float s = sin(rad);
		float c = cos(rad);

		Mat3x3 scaleMat;
		scaleMat.value[0] = c;
		scaleMat.value[1] = -s;
		scaleMat.value[2] = 0.0f;

		scaleMat.value[3] = s;
		scaleMat.value[4] = c;
		scaleMat.value[5] = 0.0f;

		scaleMat.value[6] = 0.0f;
		scaleMat.value[7] = 0.0f;
		scaleMat.value[8] = 1.0f;
		
		return ((*this) *= scaleMat);
	}


	float value[9];
};


struct TexCoord
{
    float s, t;
    TexCoord(float s, float t)
    {
        this->s = s;
        this->t = t;
    }
};


std::vector<float> calculateNormalMatrix(const float* modelviewMatrix)
{
    /*
        0   1   2
    0   0   3   6
    1   1   4   7
    2   2   5   8
    */

    //Grab the top 3x3 of the modelview matrix
    std::vector<float> M(3 * 3);
    M[0] = modelviewMatrix[0];
    M[1] = modelviewMatrix[1];
    M[2] = modelviewMatrix[2];
    M[3] = modelviewMatrix[4];
    M[4] = modelviewMatrix[5];
    M[5] = modelviewMatrix[6];
    M[6] = modelviewMatrix[8];
    M[7] = modelviewMatrix[9];
    M[8] = modelviewMatrix[10];

    //Work out the determinate
    float determinate = M[0] * M[4] * M[8] + M[1] * M[5] * M[6] + M[2] * M[3] * M[7];
    determinate -= M[2] * M[4] * M[6] + M[0] * M[5] * M[7] + M[1] * M[3] * M[8];

    //One division is faster than several
    float oneOverDet = 1.0f / determinate;

    std::vector<float> N(3 * 3);
    
    //Calculate the inverse and assign it to the transpose matrix positions
    N[0] = (M[4] * M[8] - M[5] * M[7]) * oneOverDet;
    N[3] = (M[2] * M[7] - M[1] * M[8]) * oneOverDet;
    N[6] = (M[1] * M[5] - M[2] * M[4]) * oneOverDet;

    N[1] = (M[5] * M[6] - M[3] * M[8]) * oneOverDet;
    N[4] = (M[0] * M[8] - M[2] * M[6]) * oneOverDet;
    N[7] = (M[2] * M[3] - M[0] * M[5]) * oneOverDet;

    N[2] = (M[3] * M[7] - M[4] * M[6]) * oneOverDet;
    N[5] = (M[1] * M[6] - M[8] * M[7]) * oneOverDet;
    N[8] = (M[0] * M[4] - M[1] * M[3]) * oneOverDet;

    return N;
}