#pragma once

#include <map>
#include <string>


#ifndef GLuint		// Die Header-Datei wird hier nicht eingebunden um die Kompilierzeit zu verkürzen

typedef unsigned int GLuint;

#endif


using std::string;
using std::ifstream;
using std::map;

class GLSLProgram
{
public:
    struct GLSLShader
    {
        unsigned int id;
        string filename;
        string source;
    };

	GLSLProgram(const std::string& vertexShader, const std::string& fragmentShader);

    virtual ~GLSLProgram();

    void Unload();

    bool Initialize();
	void LinkProgram();

    GLuint GetUniformLocation(const string& name);
    GLuint GetAttribLocation(const string& name);


	void SendUniform(const string& name, const int id);
	void SendUniform4x4(const string& name, const float* matrix, bool transpose=false);
	void SendUniform3x3(const string& name, const float* matrix, bool transpose=false);
	void SendUniform2x2(const string& name, const float* matrix, bool transpose=false);
	void SendUniform(const string& name, const float red, const float green,
			 const float blue, const float alpha);
	void SendUniform(const string& name, const float x, const float y, const float z);
	void SendUniform(const string& name, const float x, const float y);
	void SendUniform(const string& name, const float scalar);

    void BindAttrib(unsigned int index, const string& attribName);
    void BindShader();

private:
    string ReadFile(const string& filename);
    bool CompileShader(const GLSLShader& shader);
    void OutputShaderLog(const GLSLShader& shader);

    GLSLShader m_vertexShader;
    GLSLShader m_fragmentShader;
    unsigned int m_programID;

    map<string, GLuint> m_uniformMap;
    map<string, GLuint> m_attribMap;

	//string m_ShaderName1;
	//string m_ShaderName2;
};
