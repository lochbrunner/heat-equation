#ifdef _WIN32
#include <windows.h>
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#pragma warning(disable:4996)
#endif

#include <GL/glew.h>
#include <math.h>


#include <GL/freeglut.h>

// CUDA utilities and system includes
#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
#include <cutil_gl_inline.h>
#include <cutil_gl_error.h>
#include <cudaGL.h>

// Shared Library Test Functions
#include <shrUtils.h>
#include <shrQATest.h>


#include <vector>
#include <string>
#include "lodepng.h"
#include "GLSL_Shader.h"
#include "AlgebraStuff.h"


char *SceneryNames[] = {
	"../scenes/Test0.png",
	"../scenes/Test1.png",
	"../scenes/Test5.png",
	"../scenes/Test6.png",
	"../scenes/Test7.png",
	"../scenes/Test8.png",
	"../scenes/CUDA.png",
	"../scenes/Test9.png"};

const float halfX = 2.0f;
const float halfZ = 2.0f;

Vector3 LightPosition = Vector3(0.0f, 4.0f, 4.0f);

unsigned int window_width = 512;
unsigned int window_height = 512;
int  mesh_width = 512;
int  mesh_height = 512;
int iGLUTWindowHandle = 0; 

Vector3 translation = Vector3(0.0f, 0.0f, -8.0f);
Vector3 rotation = Vector3(30.0f, 45.0f, 0.0f);

GLuint m_vertexBuffer = 0;
GLuint m_normalBuffer = 0;
GLuint m_biNormalBuffer = 0;
GLuint m_texCoordBuffer = 0;
GLuint m_indexBuffer = 0;

GLuint gradient_texture = 0;

int cVertices;
int cIndicies;

GLSLProgram* m_GLSLProgram;


// CUDA Stuff
GLuint opengl_tex;	
// <-->							//CUDA-Registration
struct cudaGraphicsResource *cuda_tex;

// --->							//CUDA-Prozess

float4* cuda_dest_resource;

GLuint opengl_tex_boundary_condition = 0;
struct cudaGraphicsResource *cuda_tex_boundary_condition;


#define REFRESH_DELAY	  10	// in Milliseconds
bool bAnimate = false;
float m_speed = 0.24f;

static int fpsCount = 0;
static int fpsLimit = 20;
unsigned int timer;

Vector4 *pImageVector = NULL;	// Globale Variable erspart kopierarbeit
Vector4 *pImageBoundary = NULL;	// Globale Variable erspart kopierarbeit

//CUDA-Kernel-Functions
extern "C"
void init_boundary_cuda(cudaArray *boundary_array);

extern "C" 
void launch_cudaProcess(dim3 grid, dim3 block, int sbytes, 
		   cudaArray *g_data_array, float4* g_odata, 
		   int imgw, int imgh, int tilew, float dw, float dh, float speed);


extern "C"
void launch_cudaPicking(dim3 grid, dim3 block, int sbytes, 
		   cudaArray *g_data_array, float4* g_odata, 
		   int imgw, int imgh, int tilew, float dw, float dh, float3 orig, float3  dir, float value);



float3 operator-(float3 &a, float3 &b)
{
	float3 result;
	result.x = a.x - b.x;
	result.y = a.y - b.y;
	result.z = a.z - b.z;
	return result;
}

void Cleanup(int iExitCode);
void display();
void idle();
void keyboard(unsigned char key, int x, int y);
void resize(int w, int h);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);
void setBoundaryToCuda()
{
	cudaArray *in_array;
	cutilSafeCall(cudaGraphicsMapResources(1, &cuda_tex_boundary_condition, 0));
	cutilSafeCall(cudaGraphicsSubResourceGetMappedArray(&in_array, cuda_tex_boundary_condition, 0, 0));

	init_boundary_cuda(in_array);

	cutilSafeCall(cudaGraphicsUnmapResources(1, &cuda_tex_boundary_condition, 0));
}
void processCuda( int width, int height, float speed) 
{
    cudaArray *in_array; 
    float4* out_data;

    out_data = cuda_dest_resource;

    // map buffer objects to get CUDA device pointers
    cutilSafeCall(cudaGraphicsMapResources(1, &cuda_tex, 0));
    cutilSafeCall(cudaGraphicsSubResourceGetMappedArray(&in_array, cuda_tex, 0, 0));

    // calculate grid size
    dim3 block(16, 16, 1);
    dim3 grid(width / block.x, height / block.y, 1);
    int sbytes = block.x*(block.y)*sizeof(unsigned int);

    // execute CUDA kernel
    launch_cudaProcess(grid, block, sbytes, 
                       in_array, out_data, width, height, 
					   block.x, (halfX*2.0f)/mesh_width, (2.0f*halfZ)/mesh_height, speed);

    cutilSafeCall(cudaGraphicsUnmapResources(1, &cuda_tex, 0));
}
void dragField(Vector2<int> new_mouse_pos, float value)
{
	// Calculating the ray from the mouse

	Mat4x4d modelviewMatrix = Mat4x4d::Identity();
	modelviewMatrix *= Mat4x4d::Translation(translation);
	modelviewMatrix *= Mat4x4d::Rotation(rotation.x, Vector(1.0, 0.0, 0.0));
	modelviewMatrix *= Mat4x4d::Rotation(rotation.y, Vector(0.0, 1.0, 0.0));
	GLdouble ProjectionMatrix[16];
	GLint ViewPort[4];

	glGetDoublev(GL_PROJECTION_MATRIX, ProjectionMatrix);
	glGetIntegerv(GL_VIEWPORT, ViewPort);

	double x,y,z;

	float3 vec0, vec1;
	// Near point
	gluUnProject(new_mouse_pos.x, ViewPort[3] - new_mouse_pos.y, 0.0f, modelviewMatrix.value, ProjectionMatrix, ViewPort, &x, &y, &z);
	vec0.x = static_cast<float>(x);
	vec0.y = static_cast<float>(y);
	vec0.z = static_cast<float>(z);
	// Far point
	gluUnProject(new_mouse_pos.x, ViewPort[3] - new_mouse_pos.y, 1.0f, modelviewMatrix.value, ProjectionMatrix, ViewPort, &x, &y, &z);
	vec1.x = static_cast<float>(x);
	vec1.y = static_cast<float>(y);
	vec1.z = static_cast<float>(z);

	float3 orig = vec0;
	float3 dir = vec1 - vec0;

	// Change the values of the are the ray passes by
	cudaArray *in_array; 
    float4* out_data;

    out_data = cuda_dest_resource;

    // map buffer objects to get CUDA device pointers
    cutilSafeCall(cudaGraphicsMapResources(1, &cuda_tex, 0));
    cutilSafeCall(cudaGraphicsSubResourceGetMappedArray(&in_array, cuda_tex, 0, 0));

    // calculate grid size
    dim3 block(16, 16, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    int sbytes = block.x*(block.y)*sizeof(unsigned int);


    // execute CUDA kernel
	launch_cudaPicking(grid, block, sbytes, 
                       in_array, out_data, mesh_width, mesh_height, 
					   block.x, (float)mesh_width, (float)mesh_height, orig, dir, value);

    cutilSafeCall(cudaGraphicsUnmapResources(1, &cuda_tex, 0));

}
Vector4 *LoadImage(unsigned int width, unsigned int height, char* filename)
{
	if(pImageVector) delete[]pImageVector;
	if(pImageBoundary) delete[]pImageBoundary;
	std::vector<unsigned char> decode_image;
		

	LodePNG::decode(decode_image, width, height, filename);

	if(mesh_height != height || mesh_width != height){
		printf("Resolutiobn of Image does not match!");
		return NULL;
	}
	
	pImageVector = new Vector4[mesh_height*mesh_width];
	pImageBoundary = new Vector4[mesh_height*mesh_width];

	for(unsigned int i = 0; i < width*height; i++)
	{
		pImageVector[i] = Vector4(static_cast<float>(decode_image[i*4])/255.0f, static_cast<float>(decode_image[i*4+1])/255.0f, 0.0f, 0.0f);
		pImageBoundary[i] = Vector4(static_cast<float>(decode_image[i*4])/255.0f, static_cast<float>(decode_image[i*4+1])/255.0f, static_cast<float>(decode_image[i*4+2])/255.0f, 0.0f);
	}

	decode_image.clear();
	printf("Scene %s was loades successfully.\n", filename);

	
	return pImageVector;
}
void createGradientTexture(){

	glEnable(GL_TEXTURE_1D);
	glGenTextures(1, &gradient_texture);
	glBindTexture(GL_TEXTURE_1D, gradient_texture);

	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);


	unsigned int width, height;

	std::vector<unsigned char> imgData;
	
	LodePNG::decode(imgData, width, height, "../scenes/gradient.png");

	glTexImage1D(GL_TEXTURE_1D,0, GL_RGBA8, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 
                  &imgData[0]);
}
void initCUDABuffers()
{
    // set up vertex data parameter
    int num_texels = mesh_width * mesh_height;
    int num_values = num_texels*4;
	int size_tex_data = sizeof(float) * num_values;
    cutilSafeCall(cudaMalloc((void**)&cuda_dest_resource, size_tex_data));
}
void copyBufferFormCudaResourceToTexure()
{
	cudaArray *texture_ptr;
    cutilSafeCall(cudaGraphicsMapResources(1, &cuda_tex, 0));
    cutilSafeCall(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex, 0, 0));

    int num_texels = mesh_width * mesh_height;
    int num_values = num_texels * 16;
    int size_tex_data = sizeof(GLubyte) * num_values;
    cutilSafeCall(cudaMemcpyToArray(texture_ptr, 0, 0, cuda_dest_resource, size_tex_data, cudaMemcpyDeviceToDevice));

    cutilSafeCall(cudaGraphicsUnmapResources(1, &cuda_tex, 0));
}
void createTexture(GLuint* pTexture, cudaGraphicsResource **ppCudaResource, unsigned int size_x, unsigned int size_y, GLvoid* pixels = NULL)
{
    // create a texture
    glGenTextures(1, pTexture);
    glBindTexture(GL_TEXTURE_2D, *pTexture);

    // set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, size_x, size_y, 0, GL_RGBA, GL_FLOAT, pixels);
    CUT_CHECK_ERROR_GL2();
    // register this texture with CUDA
    cutilSafeCall(cudaGraphicsGLRegisterImage(ppCudaResource, *pTexture, 
		GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));
}
void createTextureEx(GLuint* pTexture, cudaGraphicsResource **ppCudaResource, unsigned int size_x, unsigned int size_y, GLvoid* pixels = NULL)
{
	    // create a texture
    glGenTextures(1, pTexture);
    glBindTexture(GL_TEXTURE_2D, *pTexture);

    // set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F_ARB, size_x, size_y, 0, GL_RGB, GL_FLOAT, pixels);
    CUT_CHECK_ERROR_GL2();
    // register this texture with CUDA
    cutilSafeCall(cudaGraphicsGLRegisterImage(ppCudaResource, *pTexture, 
		GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));
}
bool initShader()
{
	m_GLSLProgram = new GLSLProgram("../shader/basic-mesh.vert", "../shader/basic-mesh.frag");
	if (!m_GLSLProgram->Initialize()) 
	{
		//Could not initialize the shaders.
		return false;
	}
	//Bind the attribute locations
	m_GLSLProgram->BindAttrib(0, "a_Vertex");
	m_GLSLProgram->BindAttrib(1, "a_Normal");
	m_GLSLProgram->BindAttrib(3, "a_BiNormal");
	m_GLSLProgram->BindAttrib(2, "a_TexCoord");
	
		
	//Re link the program
	m_GLSLProgram->LinkProgram();
	return true;
}
bool initVBOs()
{
	std::vector<Vector4> m_vertices;
	std::vector<GLuint> m_indices;
	std::vector<TexCoord> m_texCoords;
	std::vector<Vector4> m_normals;
	std::vector<Vector4> m_biNormals;

	const float dX = 2.0f*halfX/(static_cast<float>(mesh_width) -1.0f);
	const float dZ = 2.0f*halfZ/(static_cast<float>(mesh_width) -1.0f);

	const float dU =  1.0f/(static_cast<float>(mesh_width) -1.0f);
	const float dV =  1.0f/(static_cast<float>(mesh_height) -1.0f);

	float curX = -halfX;
	float curZ;
	float curU = 0.0;
	float curV;

	// Calculate the verticies
	for(int x = 0; x < mesh_width; x++)
	{
		curZ = -halfZ;
		curV = 0.0f;
		for(int z = 0; z < mesh_height; z++)
		{
			m_vertices.push_back(Vector3(curX, 0.0f, curZ));
			m_texCoords.push_back(TexCoord(curU, curV));
			m_normals.push_back(Vector3(0.0f, 1.0f, 0.0f));
			m_biNormals.push_back(Vector3(1.0f, 0.0f, 0.0f));

			curZ += dZ;
			curV += dV;
		}
		curX += dX;
		curU += dU;
	}
	cVertices = m_vertices.size();

	glGenBuffers(1, &m_vertexBuffer); //Generate a buffer for the vertices
	glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffer); //Bind the vertex buffer
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * m_vertices.size() * 4, &m_vertices[0], GL_STATIC_DRAW); //Send the data to OpenGL

	glGenBuffers(1, &m_texCoordBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, m_texCoordBuffer); //Bind the vertex buffer
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * m_texCoords.size() * 2, &m_texCoords[0], GL_STATIC_DRAW); //Send the data to OpenGL

	glGenBuffers(1, &m_normalBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, m_normalBuffer); //Bind the normal buffer
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * m_normals.size() * 4, &m_normals[0], GL_STATIC_DRAW); //Send the data to OpenGL

	glGenBuffers(1, &m_biNormalBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, m_biNormalBuffer); //Bind the Binormal buffer
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * m_biNormals.size() * 4, &m_biNormals[0], GL_STATIC_DRAW); //Send the data to OpenGL


	// Calculate the indicies
	for(int z = 0; z < mesh_height-1; z++)
	{
		for(int x = 0; x < mesh_width-1; x++)
		{
			m_indices.push_back(z+x*mesh_width);
			m_indices.push_back(z+(x+1)*mesh_width);
			m_indices.push_back(z+1+x*mesh_width);

			m_indices.push_back(z+(x+1)*mesh_width);
			m_indices.push_back(z+1+(x+1)*mesh_width);
			m_indices.push_back(z+1+x*mesh_width);
		}
	}
	cIndicies = m_indices.size();

	glGenBuffers(1, &m_indexBuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexBuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * m_indices.size(), &m_indices[0], GL_STATIC_DRAW);

	m_vertices.clear();
	m_indices.clear();
	m_texCoords.clear();
	m_normals.clear();
	m_biNormals.clear();

	return true;
}
void deInitVBOs()
{
	if(m_vertexBuffer) glDeleteBuffers(1, &m_vertexBuffer);
	if(m_normalBuffer) glDeleteBuffers(1, &m_normalBuffer);
	if(m_biNormalBuffer) glDeleteBuffers(1, &m_biNormalBuffer);
	if(m_texCoordBuffer) glDeleteBuffers(1, &m_texCoordBuffer);
	if(m_indexBuffer) glDeleteBuffers(1, &m_indexBuffer);
	

	m_GLSLProgram->Unload();
	delete m_GLSLProgram;

	glDeleteTextures(1, &opengl_tex);
    CUT_CHECK_ERROR_GL2();
}
bool initGL(int *argc, char **argv )
{
	// Create GL context
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(window_width, window_height);
    iGLUTWindowHandle = glutCreateWindow("HeatEquation");

    // initialize necessary OpenGL extensions
    glewInit();
    if (! glewIsSupported(
        "GL_VERSION_2_0 " 
        "GL_ARB_pixel_buffer_object "
        "GL_EXT_framebuffer_object "
        )) {
        //shrLog("ERROR: Support for necessary OpenGL extensions missing.");
        //fflush(stderr);
        return CUTFalse;
    }

    // default initialization
    glClearColor(0.5, 0.5, 0.5, 1.0);

	//glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);


    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.0625f, 512.0f);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glEnable(GL_LIGHT0);
    float red[] = { 1.0f, 0.1f, 0.1f, 1.0f };
    float white[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, red);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, white);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 60.0f);

    return CUTTrue;
}
int main(int argc, char** argv)
{
	
	initGL(&argc, argv);
	cudaGLSetGLDevice (cutGetMaxGflopsDeviceId() );
	initShader();
	initVBOs();


	LoadImage(mesh_width,mesh_height, SceneryNames[0]);

	createTexture(&opengl_tex, &cuda_tex, mesh_width, mesh_height, pImageVector);
	createTexture(&opengl_tex_boundary_condition, &cuda_tex_boundary_condition, mesh_width, mesh_height, pImageBoundary);
	setBoundaryToCuda();

	initCUDABuffers();
	//copyBufferFormCudaResourceToTexure();
	createGradientTexture();

	cutCreateTimer(&timer);
    cutResetTimer(timer);  

	glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(resize);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);


	glutMainLoop();
}
void display()
{
	cutStartTimer(timer);

	if(bAnimate)
	{
		for(int i =0; i < 24; i++)
		{
			processCuda(mesh_width, mesh_width, m_speed);
			copyBufferFormCudaResourceToTexure();
		}
	}

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	Mat4x4f modelviewMatrix = Mat4x4f::Identity();
	modelviewMatrix *= Mat4x4f::Translation(translation);
	modelviewMatrix *= Mat4x4f::Rotation(rotation.x, Vector(1.0, 0.0, 0.0));
	modelviewMatrix *= Mat4x4f::Rotation(rotation.y, Vector(0.0, 1.0, 0.0));

	// New Render method using own shader
	float projectionMatrix[16];
	
	glGetFloatv(GL_PROJECTION_MATRIX, projectionMatrix);
	std::vector<float> normalMatrix = calculateNormalMatrix(modelviewMatrix.value);

	m_GLSLProgram->BindShader();
	m_GLSLProgram->SendUniform4x4("modelview_matrix", modelviewMatrix.value);
	m_GLSLProgram->SendUniform4x4("projection_matrix", projectionMatrix);
	m_GLSLProgram->SendUniform3x3("normal_matrix", &normalMatrix[0]);
	m_GLSLProgram->SendUniform("texture0", 0);
	m_GLSLProgram->SendUniform("texture1", 1);
	// Set the Light
	m_GLSLProgram->SendUniform("light0.diffuse", 1.0f, 1.0f, 1.0f, 1.0f);
	m_GLSLProgram->SendUniform("light0.position", LightPosition.x, LightPosition.y, LightPosition.z, 1.0f);
	
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);
	glEnableVertexAttribArray(3);

	glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffer);
    glVertexAttribPointer((GLint)0, 4, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, m_normalBuffer);
	glVertexAttribPointer((GLint)1, 4, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, m_texCoordBuffer);
	glVertexAttribPointer((GLint)2, 2, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, m_biNormalBuffer);
	glVertexAttribPointer((GLint)3, 4, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexBuffer);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, opengl_tex);

	glEnable(GL_TEXTURE_1D);	
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_1D, gradient_texture);

	glDrawElements(GL_TRIANGLES, cIndicies, GL_UNSIGNED_INT, 0);

	glDisableVertexAttribArray(3);
	glDisableVertexAttribArray(2);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(0);
	glDisable(GL_TEXTURE_1D);

	cutStopTimer(timer);
    glutSwapBuffers();

	// Update fps counter, fps/title display and log
    if (++fpsCount == fpsLimit) {
        char cTitle[256];
        float fps = 1000.0f / cutGetAverageTimerValue(timer);
		sprintf(cTitle, "Heat Equation (%d x %d): %.1f fps", mesh_width, mesh_height, fps);  
        glutSetWindowTitle(cTitle);
        fpsCount = 0;
        //fpsLimit = (int)((fps > 1.0f) ? fps : 1.0f);
        cutResetTimer(timer);  
    }
}
void loadNewScenery(unsigned int id)
{
	if(id >= sizeof(SceneryNames)/4) return;

	glDeleteTextures(1, &opengl_tex);
    CUT_CHECK_ERROR_GL2();

	cutilSafeCall(cudaGraphicsUnregisterResource(cuda_tex));
		
	LoadImage(mesh_width, mesh_height, SceneryNames[id]);
	createTexture(&opengl_tex, &cuda_tex, mesh_width, mesh_height, pImageVector);
	createTexture(&opengl_tex_boundary_condition, &cuda_tex_boundary_condition, mesh_width, mesh_height, pImageBoundary);
	setBoundaryToCuda();
	processCuda(mesh_width, mesh_width, 0.0f);	// Compute Normals
	copyBufferFormCudaResourceToTexure();
	display();

}
void keyboard(unsigned char key, int x, int y)
{
	switch(key) {
		case(27) :
			Cleanup(EXIT_SUCCESS);
			break;
		case 'a':
			bAnimate ^= 1;
			if(bAnimate) printf("Animation started.\n");
			else printf("Animation stopped.\n");
			break;
		case '+':
			m_speed = clamp(m_speed+0.02f, 0.0f, 0.25f);
			printf("Current animationspeed: %f\n", m_speed);
			break;
		case '-':
			m_speed = clamp(m_speed-0.02f, 0.0f, 0.25f);
			printf("Current animationspeed: %f\n", m_speed);
			break;
		case '0':
			loadNewScenery(0);
			break;
		case '1':
			loadNewScenery(1);
			break;
		case '2':
			loadNewScenery(2);
			break;
		case '3':
			loadNewScenery(3);
			break;
		case '4':
			loadNewScenery(4);
			break;
		case '5':
			loadNewScenery(5);
			break;
		case '6':
			loadNewScenery(6);
			break;
		case '7':
			loadNewScenery(7);
			break;
		case '8':
			loadNewScenery(8);
			break;
		case '9':
			loadNewScenery(9);
			break;
		default:
			break;
	}
}
void resize(int w, int h)
{
    window_width = w;
    window_height = h;
	glViewport(0, 0, window_width, window_height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.0625f, 512.0f);

	glMatrixMode(GL_MODELVIEW);
}
int ox, oy;
int buttonState = 0;
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
	{
        buttonState  |= 1<<button;
		if(GLUT_ACTIVE_ALT != glutGetModifiers())
		{
			if(buttonState == 4)
			{
				dragField(Vector2<int>(x,y), -0.05f);
				copyBufferFormCudaResourceToTexure();
				processCuda(mesh_width, mesh_width, 0.0f);	// Compute Normals
				copyBufferFormCudaResourceToTexure();
				display();
			}
			else if(buttonState == 1)
			{
				dragField(Vector2<int>(x,y), +0.05f);
				copyBufferFormCudaResourceToTexure();
				processCuda(mesh_width, mesh_width, 0.0f);	// Compute Normals
				copyBufferFormCudaResourceToTexure();
				display();
			}
		}
	}
    else if (state == GLUT_UP)
        buttonState = 0;

    ox = x; oy = y;
    glutPostRedisplay();
}
void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);

	if(GLUT_ACTIVE_ALT == glutGetModifiers())
	{

		if(buttonState == 4) {
			// right = zoom
			translation.z *= powf(0.995f, dy);
		} 
		else if(buttonState == 2) 
		{
			// middle = translate
			translation.x += dx / 100.0f;
			translation.y -= dy / 100.0f;
		}
		else if(buttonState == 1)
		{
			// left = rotate
			rotation.x += dy / 5.0f;
			rotation.y += dx / 5.0f;
		}
	}
	else
	{
		if(buttonState == 4)
		{
			dragField(Vector2<int>(x,y), -0.05f);
			copyBufferFormCudaResourceToTexure();
			processCuda(mesh_width, mesh_width, 0.0f);	// Compute Normals
			copyBufferFormCudaResourceToTexure();
			display();
		}
		else if(buttonState == 1)
		{
			dragField(Vector2<int>(x,y), +0.05f);
			copyBufferFormCudaResourceToTexure();
			processCuda(mesh_width, mesh_width, 0.0f);	// Compute Normals
			copyBufferFormCudaResourceToTexure();
			display();
		}
	}
    ox = x; oy = y;
    glutPostRedisplay();
}
void timerEvent(int value)
{
    glutPostRedisplay();
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}
void Cleanup(int iExitCode)
{
    cutilCheckError(cutDeleteTimer(timer));
    // unregister this buffer object with CUDA
    cutilSafeCall(cudaGraphicsUnregisterResource(cuda_tex));

    cudaFree(cuda_dest_resource);
    cutilDeviceReset();
    if(iGLUTWindowHandle)glutDestroyWindow(iGLUTWindowHandle);

	deInitVBOs();

}